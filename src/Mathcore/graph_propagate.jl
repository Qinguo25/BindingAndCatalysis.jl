#==================================================================================#
#           Calc H, H0 from graph propagation
#==================================================================================#

function _sparse_outer(
    c::SparseVector{Tc, Int}, s::SparseVector{Tc, Int}, scale::Tc
) where {Tc <: Real}
    nrow = length(c)
    ncol = length(s)
    Ic, Vc = findnz(c)
    Js, Vs = findnz(s)

    if isempty(Ic) || isempty(Js)
        return spzeros(Tc, nrow, ncol)
    end

    nnzA = length(Ic) * length(Js)
    I = Vector{Int}(undef, nnzA)
    J = Vector{Int}(undef, nnzA)
    V = Vector{Tc}(undef, nnzA)

    p = 0
    @inbounds for a in eachindex(Ic)
        ia = Ic[a]
        va = scale * Vc[a]
        for b in eachindex(Js)
            p += 1
            I[p] = ia
            J[p] = Js[b]
            V[p] = va * Vs[b]
        end
    end

    return sparse(I, J, V, nrow, ncol)
end

@inline _rank1_denominator_iszero(a::AbstractFloat, atol::Real) = abs(a) <= atol
@inline _rank1_denominator_iszero(a::Real, ::Real) = iszero(a)

function _rank1_step_update_from_regular(
    H::SparseMatrixCSC{Tc, Int}, # source H
    H0::AbstractVector{<:Real}, # source H0
    i::Int,
    c_c0::Hyperplane_perm,
    sign::Int8,
    atol::Float64=1e-12,
) where {Tc <: Real}
    c_qK = c_c0 * H .* sign
    c0_qK = c_c0 * H0 * sign
    dropzeros!(c_qK)
    a = 1 + c_qK[i]

    if _rank1_denominator_iszero(a, atol) # target is singular
        H_to = _sparse_outer(H[:, i], c_qK, -one(Tc))
        H0_to = -H[:, i] .* c0_qK
        nlt_to = 1
    else
        scale = inv(a)
        H_to = H - _sparse_outer(H[:, i], c_qK, scale)
        dropzeros!(H_to)
        H0_to = H0 - H[:, i] .* (scale * c0_qK)
        nlt_to = 0

        # if a < 0
        #     @show a
        #     @show inv(float.(H))
        #     @show inv(float.(H_to))
        #     @warn "H Determinant is changing sign in rank-1 update"
        # end

    end

    return H_to, H0_to, nlt_to, c_qK, c0_qK
end

"""
    _lowrank_update_H_H0(H, H0, U, V, δ0; kwargs...)

Woodbury-style affine update for

    M'  = M  + U V'
    M0' = M0 + U δ0

with

    H'  = H - H U (I + V' H U)^(-1) V' H
    H0' = H0 - H U (I + V' H U)^(-1) (V' H0 + δ0).
"""
function _lowrank_update_H_H0(
    H::SparseMatrixCSC{Float64, Int},
    H0::AbstractVector{<:Real},
    U::SparseMatrixCSC{Float64, Int},
    V::SparseMatrixCSC{Float64, Int},
    δ0::AbstractVector{<:Real};
    atol::Float64=1e-12,
)
    HU = Matrix(H * U)
    VtH = Matrix(transpose(V) * H)
    K = Matrix{Float64}(I, size(U, 2), size(U, 2)) + Matrix(transpose(V) * sparse(HU))
    abs(det(K)) <= atol && return nothing, nothing, K

    KVtH = K \ VtH
    H_new = sparse(Matrix(H) - HU * KVtH)

    rhs0 = Vector{Float64}(transpose(V) * Float64.(H0)) + Float64.(δ0)
    H0_new = Float64.(H0) - HU * (K \ rhs0)

    return H_new, vec(H0_new), K
end

mutable struct AffinePropagateWorkspace
    remaining::Vector{UInt8}
    claimed::Vector{Threads.Atomic{Int}}
    frontier::Vector{Int}
    next_frontier::Vector{Int}
    discovered::Vector{Int}
    next_locals::Vector{Vector{Int}}
    disc_locals::Vector{Vector{Int}}
end

function AffinePropagateWorkspace(n::Int; nt::Int=Threads.maxthreadid())
    return AffinePropagateWorkspace(
        fill(UInt8(0), n),
        [Threads.Atomic{Int}(0) for _ in 1:n],
        Int[],
        Int[],
        Int[],
        [Int[] for _ in 1:nt],
        [Int[] for _ in 1:nt],
    )
end

@inline function _try_claim!(claimed::Vector{Threads.Atomic{Int}}, idx::Int)
    return Threads.atomic_cas!(claimed[idx], 0, 1) == 0
end

@inline function _try_claim_component!(owners::Vector{Threads.Atomic{Int}}, cid::Int)
    return Threads.atomic_cas!(owners[cid], 0, Threads.threadid()) == 0
end

@inline function _set_vertices!(remaining::Vector{UInt8}, idxs::Vector{Int}, value::UInt8)
    @inbounds for idx in idxs
        remaining[idx] = value
    end
    return nothing
end

@inline function _append_locals!(dst::Vector{Int}, locals::Vector{Vector{Int}})
    for buf in locals
        append!(dst, buf)
        empty!(buf)
    end
    return dst
end

@inline function _reset_claims!(claimed::Vector{Threads.Atomic{Int}}, idxs::Vector{Int})
    @inbounds for idx in idxs
        Threads.atomic_xchg!(claimed[idx], 0)
    end
    return nothing
end

mutable struct SeedAnalysisState
    cache::NρCache
    cache_lock::ReentrantLock
    analyzed::Vector{Threads.Atomic{Int}}
    perm_keys::Vector{Any}
    perm_defs::Vector{Int}
    total_nullities::Vector{Int}
    statuses::Vector{Int8}
    component_owner::Vector{Threads.Atomic{Int}}
end

@inline _affine_info_ready(rgm::BindRegime) = !isnothing(rgm.H) && !isnothing(rgm.H0)

const _SEED_STATUS_UNKNOWN = Int8(-1)
const _SEED_STATUS_NOT_SEED = Int8(0)
const _SEED_STATUS_REGULAR = Int8(1)
const _SEED_STATUS_HIGH = Int8(2)

function SeedAnalysisState(n_vertices::Int, n_components::Int)
    return SeedAnalysisState(
        NρCache(),
        ReentrantLock(),
        [Threads.Atomic{Int}(0) for _ in 1:n_vertices],
        fill(nothing, n_vertices),
        fill(-1, n_vertices),
        fill(-1, n_vertices),
        fill(_SEED_STATUS_UNKNOWN, n_vertices),
        [Threads.Atomic{Int}(0) for _ in 1:n_components],
    )
end

function _store_Nρ_entry_threadsafe!(
    cache::NρCache,
    cache_lock::ReentrantLock,
    key::AbstractVector{<:Integer},
    built::NρCacheEntry,
)
    tkey = Tuple(Int.(key))
    lock(cache_lock)
    entry = get!(cache, tkey, built)
    unlock(cache_lock)
    return entry, tkey
end

function _get_or_build_Nρ_entry_threadsafe!(
    cache::NρCache,
    cache_lock::ReentrantLock,
    N::AbstractMatrix{Tv},
    key::AbstractVector{<:Integer};
    atol::Float64=1e-12,
    rtol::Float64=1e-10,
    drop_tol::Float64=1e-12,
) where {Tv <: Real}
    tkey = Tuple(Int.(key))

    lock(cache_lock)
    entry = get(cache, tkey, nothing)
    unlock(cache_lock)
    !isnothing(entry) && return entry, tkey

    built = _factor_Nρ(sparse(N[:, collect(key)]); atol=atol, rtol=rtol, drop_tol=drop_tol)
    return _store_Nρ_entry_threadsafe!(cache, cache_lock, key, built)
end

@inline function _load_seed_analysis(state::SeedAnalysisState, idx::Int)
    key = state.perm_keys[idx]
    entry = key === nothing ? nothing : get(state.cache, key, nothing)
    return state.statuses[idx], state.perm_defs[idx], key, entry
end

@inline function _store_seed_analysis!(
    state::SeedAnalysisState,
    idx::Int,
    status::Int8,
    pdef::Int,
    stored_key,
    total_nullity::Int,
)
    state.perm_defs[idx] = pdef
    state.perm_keys[idx] = stored_key
    state.total_nullities[idx] = total_nullity
    state.statuses[idx] = status
    Threads.atomic_xchg!(state.analyzed[idx], 2)
    return nothing
end

function _analyze_seed!(
    state::SeedAnalysisState,
    rgm::BindRegime,
    N::AbstractMatrix{Tv};
    drop_tol::Float64=1e-10,
) where {Tv <: Real}
    perm = rgm.perm
    n = size(N, 2)
    key, pdef = _get_Nρ_key_and_perm_nullity(perm, n)

    status = _SEED_STATUS_NOT_SEED
    stored_key = nothing
    total_nullity = pdef
    entry = nothing

    if pdef >= 2
        status = _SEED_STATUS_HIGH
    elseif pdef == 0
        _, total_nullity, key, _, entry = _calc_H_and_nullity_uncached(
            perm, N, rgm.network.direction; drop_tol=drop_tol
        )
        stored_key = Tuple(Int.(key))

        if entry.deficiency > 0
            entry, stored_key = _store_Nρ_entry_threadsafe!(
                state.cache, state.cache_lock, key, entry
            )
        end

        if total_nullity == 0
            status = _SEED_STATUS_REGULAR
        elseif total_nullity >= 2
            status = _SEED_STATUS_HIGH
        end
    else
        entry, stored_key = _get_or_build_Nρ_entry_threadsafe!(
            state.cache, state.cache_lock, N, key; drop_tol=drop_tol
        )
        total_nullity = _perm_total_nullity(pdef, entry)
        total_nullity >= 2 && (status = _SEED_STATUS_HIGH)
    end

    return status, pdef, stored_key, total_nullity, entry
end

function _initialize_regular_seed_affine!(
    rgm::BindRegime, state::SeedAnalysisState, idx::Int; drop_tol::Float64=1e-10
)
    _initialize_regime!(rgm)
    _affine_info_ready(rgm) && return nothing
    state.statuses[idx] == _SEED_STATUS_REGULAR ||
        error("Requested regular seed initialization for a non-regular regime.")

    H = let
        perm_key = state.perm_keys[idx]
        perm_key === nothing &&
            error("Missing complement key for exact regular seed initialization.")
        _build_regular_H_from_key_entry_exact(
            rgm.perm, rgm.network.N, collect(perm_key); drop_tol=drop_tol
        )
    end

    rgm.nullity = 0
    rgm.H = H
    rgm.H0 = vec(-(H * rgm.M0))
    return nothing
end

function _ensure_seed_analysis!(
    state::SeedAnalysisState,
    idx::Int,
    regimes::Vector{BindRegime},
    N::AbstractMatrix{Tv};
    drop_tol::Float64=1e-10,
) where {Tv <: Real}
    flag = state.analyzed[idx][]
    flag == 2 && return _load_seed_analysis(state, idx)

    if _try_claim!(state.analyzed, idx)
        status, pdef, stored_key, total_nullity, entry = _analyze_seed!(
            state, regimes[idx], N; drop_tol=drop_tol
        )
        _store_seed_analysis!(state, idx, status, pdef, stored_key, total_nullity)
        return status, pdef, stored_key, entry
    end

    while state.analyzed[idx][] != 2
        yield()
    end

    return _load_seed_analysis(state, idx)
end

function _initial_seed_ranges(n::Int, nt::Int)
    n == 0 && return UnitRange{Int}[]
    chunk = cld(n, max(nt, 1))
    return [
        ((k - 1) * chunk + 1):min(k * chunk, n) for k in 1:nt if (k - 1) * chunk + 1 <= n
    ]
end

function _prefill_affine_cache!(model::Bnc; ensure_built::Bool=true)
    ensure_built && find_all_regimes!(model)
    model._regimes_affine_ready && return nothing
    lock(model._regimes_affine_lock)
    try
        model._regimes_affine_ready && return nothing
        _prefill_affine_cache_core!(model)
    finally
        unlock(model._regimes_affine_lock)
    end

    return nothing
end

function _prefill_affine_cache_core!(model::Bnc)
    regimes = _bind_regimes_data(model)
    grh = model.vertices_graph
    isnothing(grh) && error("Regime graph is not initialized.")

    comps = connected_components(get_neighbor_graph_x(grh))
    comp_of = zeros(Int, length(regimes))
    for cid in eachindex(comps)
        @inbounds for idx in comps[cid]
            comp_of[idx] = cid
        end
    end

    state = SeedAnalysisState(length(regimes), length(comps))
    workspaces = [
        AffinePropagateWorkspace(length(regimes)) for _ in 1:Threads.maxthreadid()
    ]
    deferred_locals = [Int[] for _ in 1:Threads.maxthreadid()]

    ranges = _initial_seed_ranges(length(regimes), Threads.nthreads())
    Threads.@threads for rid in eachindex(ranges)
        tid = Threads.threadid()
        ws = workspaces[tid]
        deferred_local = deferred_locals[tid]

        for idx in ranges[rid]
            cid = comp_of[idx]
            state.component_owner[cid][] == 0 || continue

            status, _, _, _ = _ensure_seed_analysis!(state, idx, regimes, model.N)
            status == _SEED_STATUS_REGULAR || continue
            _try_claim_component!(state.component_owner, cid) || continue

            append!(
                deferred_local,
                _process_component_from_seed_scan!(
                    regimes, grh, comps[cid], ws, state; frontier_parallel_threshold=256
                ),
            )
        end
    end

    Threads.@threads :greedy for cid in eachindex(comps)
        _try_claim_component!(state.component_owner, cid) || continue
        tid = Threads.threadid()
        append!(
            deferred_locals[tid],
            _process_component_from_seed_scan!(
                regimes,
                grh,
                comps[cid],
                workspaces[tid],
                state;
                frontier_parallel_threshold=256,
            ),
        )
    end

    deferred_idxs = reduce(vcat, deferred_locals; init=Int[])
    model._vertices_Nρ_inv_dict = state.cache
    _finalize_deferred_affine_and_nullity!(model, deferred_idxs, state)

    model._regimes_affine_ready = true
    return nothing
end

@inline function _defer_high_nullity_seed!(
    deferred_idxs::Vector{Int},
    regimes::Vector{BindRegime},
    state::SeedAnalysisState,
    remaining::Vector{UInt8},
    idx::Int,
)
    remaining[idx] = 0x00
    regimes[idx].nullity = max(2, state.total_nullities[idx])
    push!(deferred_idxs, idx)
    return nothing
end

function _take_regular_seed!(
    regimes::Vector{BindRegime},
    comp::Vector{Int},
    remaining::Vector{UInt8},
    state::SeedAnalysisState,
    deferred_idxs::Vector{Int};
    drop_tol::Float64=1e-10,
)
    seed = 0

    @inbounds for idx in comp
        remaining[idx] == 0x01 || continue

        status, _, _, _ = _ensure_seed_analysis!(
            state, idx, regimes, regimes[idx].network.N; drop_tol=drop_tol
        )
        if status == _SEED_STATUS_HIGH
            _defer_high_nullity_seed!(deferred_idxs, regimes, state, remaining, idx)
        elseif status == _SEED_STATUS_REGULAR
            seed = idx
            _initialize_regular_seed_affine!(regimes[seed], state, seed; drop_tol=drop_tol)
            break
        end
    end

    return seed
end

function _defer_remaining_component!(
    regimes::Vector{BindRegime},
    comp::Vector{Int},
    remaining::Vector{UInt8},
    state::SeedAnalysisState,
    deferred_idxs::Vector{Int};
    drop_tol::Float64=1e-10,
)
    @inbounds for idx in comp
        remaining[idx] == 0x01 || continue
        status, _, _, _ = _ensure_seed_analysis!(
            state, idx, regimes, regimes[idx].network.N; drop_tol=drop_tol
        )
        remaining[idx] = 0x00
        status == _SEED_STATUS_HIGH &&
            (regimes[idx].nullity = max(2, state.total_nullities[idx]))
        push!(deferred_idxs, idx)
    end

    return deferred_idxs
end

function _process_component_from_seed_scan!(
    regimes::Vector{BindRegime},
    grh::RegimeGraph,
    comp::Vector{Int},
    ws::AffinePropagateWorkspace,
    state::SeedAnalysisState;
    frontier_parallel_threshold::Int=256,
    drop_tol::Float64=1e-10,
)
    remaining = ws.remaining
    _set_vertices!(remaining, comp, 0x01)

    deferred_idxs = Int[]

    while true
        seed = _take_regular_seed!(
            regimes, comp, remaining, state, deferred_idxs; drop_tol=drop_tol
        )
        seed == 0 && break

        remaining[seed] = 0x00
        discovered = _propagate_from_regular_seed!(
            regimes, grh, ws, seed; frontier_parallel_threshold=frontier_parallel_threshold
        )
        _set_vertices!(remaining, discovered, 0x00)
        _reset_claims!(ws.claimed, discovered)
    end

    return _defer_remaining_component!(
        regimes, comp, remaining, state, deferred_idxs; drop_tol=drop_tol
    )
end

function _finalize_deferred_affine_and_nullity!(
    model::Bnc,
    deferred_idxs::Vector{Int},
    state::SeedAnalysisState;
    drop_tol::Float64=1e-10,
)
    isempty(deferred_idxs) && return nothing

    regimes = _bind_regimes_data(model)

    @inbounds for idx in deferred_idxs
        regimes[idx].nullity = max(regimes[idx].nullity, state.total_nullities[idx])
    end

    for idx in deferred_idxs
        rgm = regimes[idx]
        rgm.nullity == 1 || continue
        _affine_info_ready(rgm) && continue

        _initialize_regime!(rgm)

        H, nlt = _build_singular_H_from_perm_exact(
            rgm.perm, rgm.network.N, rgm.network.direction
        )
        nlt == 1 || continue

        rgm.H = H
        rgm.H0 = vec(-(H * rgm.M0))
    end

    return nothing
end

@inline function _visit_edge!(
    regimes::Vector{BindRegime},
    remaining::Vector{UInt8},
    claimed::Vector{Threads.Atomic{Int}},
    from_idx::Int,
    edge::RegimeEdge,
    discovered::Vector{Int},
    next_frontier::Vector{Int},
)
    to_idx = edge.to
    remaining[to_idx] == 0x01 || return nothing
    _try_claim!(claimed, to_idx) || return nothing

    to_rgm = regimes[to_idx]
    propagate_regime!(regimes[from_idx], to_rgm, edge)
    push!(discovered, to_idx)
    to_rgm.nullity == 0 && push!(next_frontier, to_idx)
    return nothing
end

@inline function _clear_locals!(locals::Vector{Vector{Int}})
    for buf in locals
        empty!(buf)
    end
    return nothing
end

function _expand_frontier_serial!(
    regimes::Vector{BindRegime},
    grh::RegimeGraph,
    remaining::Vector{UInt8},
    claimed::Vector{Threads.Atomic{Int}},
    frontier::Vector{Int},
    discovered::Vector{Int},
    next_frontier::Vector{Int},
)
    for from_idx in frontier
        for edge in grh.neighbors[from_idx]
            _visit_edge!(
                regimes, remaining, claimed, from_idx, edge, discovered, next_frontier
            )
        end
    end
    return nothing
end

function _expand_frontier_parallel!(
    regimes::Vector{BindRegime},
    grh::RegimeGraph,
    remaining::Vector{UInt8},
    claimed::Vector{Threads.Atomic{Int}},
    frontier::Vector{Int},
    discovered::Vector{Int},
    next_frontier::Vector{Int},
    next_locals::Vector{Vector{Int}},
    disc_locals::Vector{Vector{Int}},
)
    _clear_locals!(next_locals)
    _clear_locals!(disc_locals)

    # Avoid `:static` here: this frontier propagation can run inside the
    # outer threaded component scan in `find_all_regimes!`, and nested
    # static scheduling throws `@threads :static cannot be used concurrently or nested`.
    Threads.@threads for pos in eachindex(frontier)
        tid = Threads.threadid()
        next_local = next_locals[tid]
        disc_local = disc_locals[tid]

        from_idx = frontier[pos]
        for edge in grh.neighbors[from_idx]
            _visit_edge!(
                regimes, remaining, claimed, from_idx, edge, disc_local, next_local
            )
        end
    end
    _append_locals!(discovered, disc_locals)
    _append_locals!(next_frontier, next_locals)
    return nothing
end

function _propagate_from_regular_seed!(
    regimes::Vector{BindRegime},
    grh::RegimeGraph,
    ws::AffinePropagateWorkspace,
    seed::Int;
    frontier_parallel_threshold::Int=256,
)
    remaining = ws.remaining
    claimed = ws.claimed
    nt = Threads.nthreads()

    frontier = ws.frontier
    next_frontier = ws.next_frontier
    discovered = ws.discovered
    next_locals = ws.next_locals
    disc_locals = ws.disc_locals

    empty!(frontier)
    empty!(next_frontier)
    empty!(discovered)
    push!(frontier, seed)

    while !isempty(frontier)
        empty!(next_frontier)

        if nt == 1 || length(frontier) < frontier_parallel_threshold
            _expand_frontier_serial!(
                regimes, grh, remaining, claimed, frontier, discovered, next_frontier
            )
        else
            _expand_frontier_parallel!(
                regimes,
                grh,
                remaining,
                claimed,
                frontier,
                discovered,
                next_frontier,
                next_locals,
                disc_locals,
            )
        end

        frontier, next_frontier = next_frontier, frontier
    end

    ws.frontier = frontier
    ws.next_frontier = next_frontier
    return discovered
end

@inline function propagate_regime!(rgm1::BindRegime, rgm2::BindRegime, edge::RegimeEdge)
    x_idx, x_sign = _edge_idx_sign(edge, 1)
    H_to, H0_to, nlt_to, _, _ = _rank1_step_update_from_regular(
        rgm1.H, rgm1.H0, edge.i, rgm1.network._L_helper.hyperplanes[x_idx], x_sign
    )

    rgm2.H = H_to
    rgm2.H0 = H0_to
    return rgm2.nullity = nlt_to
end
