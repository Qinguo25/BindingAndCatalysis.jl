
"""
    VertexEdge

Edge metadata connecting neighboring vertices in a regime graph.
"""
mutable struct VertexEdge
    to::Int
    i ::Int # different row index
    c_c0_x_idx::Int
    c_c0_x_sign::Int8
    qK_interface_idx::Int
    qK_interface_sign::Int8

    function VertexEdge(to::Int, i::Int, c_c0_x_idx::Int, c_c0_x_sign::Int8)
        return new(to, i, c_c0_x_idx, c_c0_x_sign, 0, 0)
    end

end

@inline _edge_has_qK_interface(edge::VertexEdge) =
    edge.qK_interface_idx != 0



struct RegimeHyperplane
    change_dir_qK::SparseVector{Float64, Int}
    intersect_qK::Float64
end


# Adjacency list + optional caches
"""
    VertexGraph

Adjacency structure for vertices with optional caches for change directions.
"""
mutable struct VertexGraph{Tv}
    bn::Union{AbstractBnc, Nothing}
    neighbors::Vector{Vector{VertexEdge}}

    edge_pos::Vector{Dict{Int, Int}}  # (u,v) -> (u,edge_pos[u][v]) to locate the VertexEdge.

    qK_interface_pool::Vector{RegimeHyperplane}
    x_interface_pool::Vector{Hyperplane_perm{Tv}}
    change_dir_qK_computed::Bool

    function VertexGraph(L_helper::MatrixHelper{Tv}, neighbors::Vector{Vector{VertexEdge}}) where {Tv}
        
        edge_pos = let
            edge_pos = Vector{Dict{Int,Int}}(undef, length(neighbors))
                for i in eachindex(neighbors)
                    edges = neighbors[i]
                    d = Dict{Int,Int}()
                    sizehint!(d, length(edges))
                    for (k, e) in enumerate(edges)
                        d[e.to] = k
                    end
                    edge_pos[i] = d
                end
                edge_pos
            end

        return new{Tv}(nothing, neighbors, edge_pos, RegimeHyperplane[], L_helper.hyperplanes, false)
    end
end




function Base.getproperty(grh::VertexGraph, sym::Symbol)
    if sym === :x_grh
        neighbors = getfield(grh, :neighbors)
        g = SimpleGraph(length(neighbors))
        for i in eachindex(neighbors)
            edges = neighbors[i]
            for e in edges
                add_edge!(g, i, e.to)
            end
        end
        return g
    end
    return getfield(grh, sym)
end
function Base.propertynames(grh::VertexGraph, private::Bool=false)
    names = Symbol[fieldnames(typeof(grh))..., :x_grh]
    return private ? Tuple(unique(names)) : Tuple(sym for sym in unique(names) if !startswith(String(sym), "_"))
end



#-----------------------------------------------------------------------------------------------
#This is graph associated functions for Bnc models and archetyple behaviors associated code
#-----------------------------------------------------------------------------------------------
"""
    _calc_regimes_graph(bnc::Bnc, perms) -> VertexGraph

Build a `VertexGraph` from regime permutations, connecting regimes that differ
in exactly one row.
"""
function _calc_regimes_graph(helper::MatrixHelper, perms::Vector{<:AbstractVector{T}}) where {T<:Integer}
    # n = helper.n
    n_vtxs = length(perms)
    d = length(perms[1])
    thread_edges = [Vector{Tuple{Int, VertexEdge}}() for _ in 1:Threads.maxthreadid()]

    # 按行分桶：key 为去掉该行后的签名（Tuple），值为该签名下的 (regime idx, row choice)
    @showprogress for i in 1:d
        buckets = Dict{Tuple{Vararg{T}}, Vector{Tuple{Int,T}}}()
        @inbounds for q in 1:n_vtxs
            v = perms[q]
            sig = if i == 1
                    Tuple(v[2:end])
                elseif i == d
                    Tuple(v[1:end-1])
                else
                    Tuple((v[1:i-1]..., v[i+1:end]...))
                end
            push!(get!(buckets, sig) do
                Vector{Tuple{Int,T}}()
            end, (q, v[i]))
        end

        groups = collect(values(buckets))

        # 同桶内两两相连：沿边方向表示“增加 target dominant term，减少 source dominant term”
        Threads.@threads for gi in eachindex(groups)
            tid = Threads.threadid()
            local_edges = thread_edges[tid]
            group = groups[gi]
            m = length(group)
            m <= 1 && continue

            @inbounds for a in 1:m-1
                from_idx, j_from = group[a]
                for b in a+1:m
                    to_idx, j_to = group[b]
                    j_from == j_to && continue

                    hp_id = choiceineq_between(helper, i, j_to, j_from)
                    push!(local_edges, (from_idx, VertexEdge(to_idx, i, hp_id.hid, hp_id.sign)))
                    push!(local_edges, (to_idx, VertexEdge(from_idx, i, hp_id.hid, -hp_id.sign)))
                end
            end
        end
    end

    all_edges = reduce(vcat, thread_edges; init=Tuple{Int, VertexEdge}[])
    neighbors = [Vector{VertexEdge}() for _ in 1:n_vtxs]
    for (from, e) in all_edges
        push!(neighbors[from], e)
    end
    return VertexGraph(helper, neighbors)
end






#==================================================================================#
#           Calc H,H0 from graph propagation
#==================================================================================#


mutable struct AffinePropagateWorkspace
    remaining::Vector{UInt8}
    claimed::Vector{Threads.Atomic{Int}}
    frontier::Vector{Int}
    next_frontier::Vector{Int}
    discovered::Vector{Int}
    next_locals::Vector{Vector{Int}}
    disc_locals::Vector{Vector{Int}}
end
function AffinePropagateWorkspace(n::Int; nt::Int=Threads.nthreads())
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

@inline function _clear_vertices!(remaining::Vector{UInt8}, idxs::Vector{Int})
    @inbounds for idx in idxs
        remaining[idx] = 0x00
    end
    return nothing
end
@inline function _mark_component_remaining!(remaining::Vector{UInt8}, comp::Vector{Int})
    @inbounds for idx in comp
        remaining[idx] = 0x01
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
@inline function _find_remaining_seed(comp::Vector{Int}, remaining::Vector{UInt8})
    @inbounds for idx in comp
        if remaining[idx] == 0x01
            return idx
        end
    end
    return 0
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
    regimes = model.vertices_data
    grh = model.vertices_graph
    isnothing(grh) && error("Regime graph is not initialized.")

    comps = connected_components(grh.x_grh)
    ws = AffinePropagateWorkspace(length(regimes))

    deferred = Vector{Vector{Int}}(undef, length(comps))
    for cid in eachindex(comps)
        deferred[cid] = _explore_component_and_collect_high!(regimes, grh, comps[cid], ws)
    end

    # the following are to calculate the nullity for regimes with nullity>=2
    deferred_idxs = reduce(vcat, deferred; init=Int[])
    if isempty(deferred_idxs)
        model._vertices_Nρ_inv_dict = nothing
    else
        deferred_perms = [regimes[idx].perm for idx in deferred_idxs]
        deferred_nullity, cache = _calc_nullity(deferred_perms, model.N)
        model._vertices_Nρ_inv_dict = cache
        for (idx, nlt) in zip(deferred_idxs, deferred_nullity)
            regimes[idx].nullity = nlt
        end
    end

    model._regimes_affine_ready = true

    return nothing
end



function _explore_component_and_collect_high!(
    regimes::Vector{BindRegime},
    grh::VertexGraph,
    comp::Vector{Int},
    ws::AffinePropagateWorkspace,
)
    remaining = ws.remaining

    _mark_component_remaining!(remaining, comp)

    high_idxs = Int[] # collect indices of regimes with nullity >= 2

    while true
        seed = _find_remaining_seed(comp, remaining)
        seed == 0 && break

        remaining[seed] = 0x00

        seed_rgm = regimes[seed]

        _direct_seed_affine_and_nullity!(seed_rgm)
        
        nlt_seed = seed_rgm.nullity

        if nlt_seed >= 2
            push!(high_idxs, seed)
            continue
        end

        if nlt_seed == 0
            discovered = _propagate_from_regular_seed!(regimes, grh, ws, seed)
            _clear_vertices!(remaining, discovered)
            _reset_claims!(ws.claimed, discovered)
        end
        # nlt_seed == 1:
        #   seed itself is already handled, but we do not propagate further
        #   from a singular seed.
    end

    return high_idxs
end




function _direct_seed_affine_and_nullity!(rgm::BindRegime; drop_tol::Float64=1e-10)
    _initialize_regime!(rgm)
    _affine_info_ready(rgm) && return nothing

    perm_nullity = _calc_perm_nullity(rgm.perm)
    if perm_nullity >= 2
        rgm.nullity = 2 # assign a placeholder nullity for regimes that are expected to be high-nullity, to avoid unnecessary calculations in the propagation step. The exact nullity will be calculated later by _calc_nullity.
        return nothing
    end

    H, nlt = direct_inverse_or_adjugate(rgm.M)

    rgm.nullity = nlt

    if nlt == 0
        drop_tol > 0 && droptol!(H, drop_tol)
        rgm.H = H
        rgm.H0 = vec(-(H * rgm.M0))
    elseif nlt == 1
        drop_tol > 0 && droptol!(H, drop_tol) 
        H = H .* rgm.network.direction
        rgm.H = H 
        rgm.H0 = vec(-(H * rgm.M0))
    end

    return nothing
end


@inline _affine_info_ready(rgm::BindRegime) = !isnothing(rgm.H) && !isnothing(rgm.H0)

function _masked_components(grh::SimpleGraph, mask::BitVector)
    n = nv(grh)
    seen = falses(n)
    comps = Vector{Vector{Int}}()

    for src in 1:n
        (!mask[src] || seen[src]) && continue
        comp = Int[]
        stack = [src]
        seen[src] = true
        while !isempty(stack)
            v = pop!(stack)
            push!(comp, v)
            for nb in neighbors(grh, v)
                if mask[nb] && !seen[nb]
                    seen[nb] = true
                    push!(stack, nb)
                end
            end
        end
        push!(comps, comp)
    end

    return comps
end


function _propagate_from_regular_seed!(
    regimes::Vector{BindRegime},
    grh::VertexGraph,
    ws::AffinePropagateWorkspace,
    seed::Int;
    frontier_parallel_threshold::Int = 256,
)
    # regimes = model.vertices_data
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
            for from_idx in frontier

                from_rgm = regimes[from_idx]
                for edge in grh.neighbors[from_idx]
                    to_idx = edge.to
                    remaining[to_idx] == 0x01 || continue
                    _try_claim!(claimed, to_idx) || continue

                    # to_rgm = _initialize_regime!(regimes[to_idx])
                    to_rgm = regimes[to_idx]
                    propagate_regime!(from_rgm, to_rgm, edge)
                    push!(discovered, to_idx)

                    to_rgm.nullity == 0 && push!(next_frontier, to_idx)
                end
            end
        else
            for buf in next_locals
                empty!(buf)
            end
            for buf in disc_locals
                empty!(buf)
            end

            Threads.@threads :static for pos in eachindex(frontier)
                tid = Threads.threadid()
                next_local = next_locals[tid]
                disc_local = disc_locals[tid]

                from_idx = frontier[pos]
                from_rgm = regimes[from_idx]
                for edge in grh.neighbors[from_idx]
                    to_idx = edge.to
                    remaining[to_idx] == 0x01 || continue
                    _try_claim!(claimed, to_idx) || continue

                    to_rgm = regimes[to_idx]
                    propagate_regime!(from_rgm, to_rgm, edge)
                    push!(disc_local, to_idx)
                    to_rgm.nullity == 0 && push!(next_local, to_idx)
                end
            end
            _append_locals!(discovered, disc_locals)
            _append_locals!(next_frontier, next_locals)
        end
        frontier, next_frontier = next_frontier, frontier
    end

    ws.frontier = frontier
    ws.next_frontier = next_frontier

    return discovered
end

@inline function propagate_regime!(rgm1::BindRegime, rgm2::BindRegime, edge::VertexEdge)
    H, H0 = rgm1.H, rgm1.H0
    i = edge.i
    c_c0 = edge.c_c0_x_idx
    sign = edge.c_c0_x_sign

    H_to, H0_to, nlt_to, _, _ = _rank1_step_update_from_regular(
        H,
        H0,
        i,
        rgm1.network._L_helper.hyperplanes[c_c0],
        sign
    )

    rgm2.H = H_to
    rgm2.H0 = H0_to
    rgm2.nullity = nlt_to
end


function _edge_qK_interface(grh::VertexGraph, edge::VertexEdge)
    edge.qK_interface_idx == 0 && return nothing

    hp = grh.qK_interface_pool[edge.qK_interface_idx]
    if edge.qK_interface_sign >= 0
        return hp.change_dir_qK, hp.intersect_qK
    else
        return -hp.change_dir_qK, -hp.intersect_qK
    end
end



# function _materialize_edge_qK_interface!(grh::VertexGraph, edge::VertexEdge)
#     return edge
# end

@inline function _dense_hyperplane_sign_and_scale(dir::SparseVector{Float64,Int}; atol::Float64=1e-10)
    I, V = findnz(dir)
    @inbounds for k in eachindex(V)
        if abs(V[k]) > atol
            sgn = V[k] >= 0 ? Int8(1) : Int8(-1)
            return sgn, abs(V[k])
        end
    end
    return Int8(1), 0.0
end

function _canonicalize_qK_interface(
    dir::SparseVector{Float64,Int},
    intersect::Real;
    atol::Float64=1e-10,
    round_digits::Int=10,
)
    droptol!(dir, atol)
    nnz(dir) == 0 && return nothing

    sign, scale = _dense_hyperplane_sign_and_scale(dir; atol=atol)
    scale <= atol && return nothing

    I, V = findnz(dir)
    Vnorm = (Float64.(V) .* sign) ./ scale
    dir_norm = SparseArrays.sparsevec(I, Vnorm, length(dir))
    droptol!(dir_norm, atol)
    I2, V2 = findnz(dir_norm)
    bnorm = Float64(intersect) * sign / scale
    key = (Tuple(Int.(I2)), Tuple(round.(Float64.(V2); digits=round_digits)), round(bnorm; digits=round_digits))
    return dir_norm, bnorm, sign, key
end

function _intern_qK_interface!(
    grh::VertexGraph,
    key_to_id::Dict,
    dir::SparseVector{Float64,Int},
    intersect::Real;
    atol::Float64=1e-10,
    round_digits::Int=10,
)
    canon = _canonicalize_qK_interface(dir, intersect; atol=atol, round_digits=round_digits)
    canon === nothing && return 0, Int8(0)
    dir_norm, bnorm, sign, key = canon
    hid = get!(key_to_id, key) do
        push!(grh.qK_interface_pool, RegimeHyperplane(dir_norm, bnorm))
        length(grh.qK_interface_pool)
    end
    return hid, sign
end




"""
    _fulfill_regimes_graph!(vtx_graph::VertexGraph) -> nothing

Compute qK-space change directions for edges in the vertex graph.
"""
function _fulfill_regimes_graph!(vtx_graph::VertexGraph)
    Bnc = vtx_graph.bn
    regimes = Bnc.vertices_data
    empty!(vtx_graph.qK_interface_pool)
    key_to_id = Dict{Any,Int}()

    for edges in vtx_graph.neighbors
        for e in edges
            e.qK_interface_idx = 0
            e.qK_interface_sign = 0
        end
    end

    @showprogress for p1 in eachindex(vtx_graph.neighbors)
        edges = vtx_graph.neighbors[p1]
        for e in edges
            p2 = e.to
            p1 < p2 || continue

            rev_pos = get(vtx_graph.edge_pos[p2], p1, nothing)
            rev_pos === nothing && continue
            e_rev = vtx_graph.neighbors[p2][rev_pos]

            src_idx, src_edge, dst_edge = let
                nlt1 = regimes[p1].nullity
                nlt2 = regimes[p2].nullity
                if nlt1 <= 1
                    (p1, e, e_rev)
                elseif nlt2 <= 1
                    (p2, e_rev, e)
                else
                    (0, nothing, nothing)
                end
            end
            src_idx == 0 && continue

            src_rgm = regimes[src_idx]
            c_c0 = vtx_graph.x_interface_pool[src_edge.c_c0_x_idx]
            dir_qK, intersect_qK = _calc_dir(
                src_rgm.nullity,
                src_rgm.H,
                src_rgm.H0,
                c_c0,
                src_edge.c_c0_x_sign,
            )

            hid, sign = _intern_qK_interface!(vtx_graph, key_to_id, dir_qK, intersect_qK)
            hid == 0 && continue

            src_edge.qK_interface_idx = hid
            src_edge.qK_interface_sign = sign
            dst_edge.qK_interface_idx = hid
            dst_edge.qK_interface_sign = Int8(-sign)
        end
    end
    return nothing
end


@inline function _calc_dir(
    nlt::Int,
    H::SparseMatrixCSC{Float64,Int},
    H0::AbstractVector{<:Real},

    c_c0::Hyperplane_perm,
    sign::Int8,

    drop_tol::Float64=1e-10,
)
   c_qK = c_c0*H .* sign 
    c0_qK = nlt ==0 ? c_c0 * H0 * sign : mul(c_c0, H0; with_c0=false) * sign 
    
    if drop_tol > 0 
        droptol!(c_qK, drop_tol) 
        zero_small!(c0_qK, drop_tol)
    end

    return c_qK, c0_qK
end
