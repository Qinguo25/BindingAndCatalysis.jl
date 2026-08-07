export match_regimes!, get_bnc_regime, get_bnc_regimes
export get_binding_regime, get_binding_perm, get_catalysis_perm, get_steady_state_perm
export get_C_C0_xk, get_C0_xk, get_C_xk
export get_C_C0_qKk, get_C0_qKk, get_C_qKk, get_C_C0_nullity_qKk
export get_C_C0_wKk, get_C0_wKk, get_C_wKk, get_C_C0_nullity_wKk
export get_H_bd, get_H_bd_numerically, get_qcat_F_F0
export get_affine_qKk2v, get_affine_x2Kk̃, get_affine_x2wKk̃, get_affine_xk2wKk̃k
export get_affine_wKk2wKk̃k, get_affine_wKk2wKk̃, get_affine_wKk̃2x, get_affine_wKk̃k2xk
export get_affine_wKk2x, get_affine_wKk2xk, get_affine_wKk2v, get_affine_wKk2qcat
export judge_stability!, is_stable, stability_code
export get_volume, get_volumes

function Base.getproperty(model::BncRegime, sym::Symbol)
    if sym === :perm
        return get_binding_perm(model), get_catalysis_perm(model)
    end
    return getfield(model, sym)
end

@inline _det_sign_exact(A::AbstractMatrix{<:Integer}) = begin
    detA = _bareiss_det_big(Matrix{Int}(A))
    if detA > 0
        1
    elseif detA < 0
        -1
    else
        0
    end
end

function get_fixed_point_perm(args...; kwargs...)
    let
        bindperm, catalysisperm = get_bnc_perm(args...; kwargs...)
        r_v = get_catalysis_network(args...; kwargs...).r_v
        return bindperm[(r_v + 1):end], catalysisperm
    end
end
is_fixed_point_singular(args...; kwargs...) = is_bnc_singular(args...; kwargs...)

get_H_bd(rgm::BncRegime) = rgm.H_bd

# Get an H_bd numerically if the inner binding regime is singular.
function get_H_bd_numerically(rgm::BncRegime)
    bind_rgm = get_binding_regime(rgm)
    cat_rgm = get_catalysis_regime(rgm)

    PΠ = cat_rgm.PΠ
    H_bind = get_H_numerically(bind_rgm)
    r_v = size(PΠ, 1)
    return sparse(Float64.(PΠ * H_bind[:, 1:r_v]))
end

# Determine the stability of a Binding-Catalysis regime.
function judge_stability!(rgm::BncRegime; kwargs...)
    isnothing(rgm.H_bd) && (rgm.H_bd = get_H_bd_numerically(rgm))
    H_bd = rgm.H_bd

    code = judge_dstable(H_bd; kwargs...)

    flag = if code == 1  # d-stable
        Int8(1)
    elseif code == 0 # d-unstable
        Int8(-1)
    else # undetermined
        Int8(2)
    end

    rgm.is_stable = flag

    return rgm.is_stable
end

function _stability_flag(rgm::BncRegime; recompute::Bool=false, kwargs...)
    _reject_stability_keywords(kwargs)
    return if recompute || rgm.is_stable == 0
        judge_stability!(rgm; kwargs...)
    else
        rgm.is_stable
    end
end

function stability_code(rgm::BncRegime; recompute::Bool=false, kwargs...)
    flag = _stability_flag(rgm; recompute=recompute, kwargs...)
    return if flag == 1
        1
    elseif flag == -1
        -1
    else
        0
    end
end

function stability_code(model::Bnc, bind, cat; kwargs...)
    return stability_code(get_bnc_regime(model, bind, cat; check=true); kwargs...)
end

function is_stable(rgm::BncRegime; recompute::Bool=false, kwargs...)
    flag = _stability_flag(rgm; recompute=recompute, kwargs...)
    return if flag == 1
        true
    elseif flag == -1
        false
    else
        missing
    end
end

function is_stable(model::Bnc, bind, cat; kwargs...)
    return is_stable(get_bnc_regime(model, bind, cat; check=true); kwargs...)
end

get_qcat_F_F0(rgm::BncRegime) = get_affine_wKk2qcat(rgm)

function get_polyhedron(rgm::BncRegime; chart::Symbol=:wKk, canonicalize::Bool=true)
    C, C0, nullity = _regime_C_C0_nullity(rgm, chart)
    return _build_polyhedron_from_C_C0(C, C0, nullity; canonicalize=canonicalize)
end

get_C_C0_nullity(rgm::BncRegime) = get_C_C0_nullity_wKk(rgm)
get_C_C0(rgm::BncRegime) = get_C_C0_wKk(rgm)
get_C(rgm::BncRegime) = get_C_wKk(rgm)
get_C0(rgm::BncRegime) = get_C0_wKk(rgm)

# The following functions is required to digging into to help fix the problem
#========================================================================================#
#  Functions for  Calcalating conditions.
#========================================================================================#

function _materialize_real_vector(v)
    vv = vec(v)
    T = isempty(vv) ? Int : foldl(promote_type, map(typeof, vv); init=Int)
    return T[convert(T, x) for x in vv]
end

function _drop_trivial_true_rows(C, C0, nlt::Integer; atol::Real=1.0e-10)
    keep = trues(size(C, 1))
    nlt_out = Int(nlt)
    for i in 1:size(C, 1)
        all(x -> _condition_scalar_iszero(x, atol), @view(C[i, :])) || continue

        if i <= nlt
            _condition_scalar_iszero(C0[i], atol) ||
                return _canonical_empty_condition(C, C0)
            keep[i] = false
            nlt_out -= 1
        elseif _condition_scalar_isnonnegative(C0[i], atol)
            keep[i] = false
        else
            return _canonical_empty_condition(C, C0)
        end
    end
    return C[keep, :], C0[keep], nlt_out
end

function _steady_state_affine(
    bind_rgm::BindRegime, perm, N_ss, N0_ss, r_v::Int, direction::Int, nlt::Int
)
    nlt > 1 && return nothing

    _, P0_bind = get_affine_x2q(bind_rgm)
    P0_ss = P0_bind[(r_v + 1):end]
    M0_ss = vcat(
        P0_ss, zeros(eltype(P0_ss), size(get_binding_network(bind_rgm).N, 1)), N0_ss
    )

    H_ss = if nlt == 0
        _exact_calc_H_regular(perm, N_ss)
    else
        _build_singular_H_from_perm_exact(perm, N_ss, direction)[1]
    end
    isnothing(H_ss) &&
        error("Failed to build steady-state affine map for a Bnc regime with nullity $nlt.")

    return sparse(H_ss), vec(-(H_ss * M0_ss))
end

function _rank1_inner_update_from_regular(
    H::AbstractMatrix{<:Real},
    H0::AbstractVector{<:Real},
    row::Int,
    c::SparseVector,
    c0;
    atol::Float64=1e-12,
)
    H = sparse(Float64.(H))
    H0 = Float64.(vec(H0))
    c_float = sparsevec(c.nzind, Float64.(c.nzval), length(c))
    cH_dense = zeros(Float64, size(H, 2))
    @inbounds for p in eachindex(c_float.nzind)
        cH_dense .+= c_float.nzval[p] .* Vector(H[c_float.nzind[p], :])
    end
    nz = findall(!iszero, cH_dense)
    cH = sparsevec(nz, cH_dense[nz], length(cH_dense))
    c0H = dot(c_float, H0) + Float64(c0)
    a = 1 + cH[row]

    if abs(a) <= atol
        H_to = _sparse_outer(H[:, row], cH, -1.0)
        H0_to = Vector(-H[:, row] .* c0H)
        return H_to, H0_to, 1, cH, c0H
    end

    scale = inv(a)
    H_to = H - _sparse_outer(H[:, row], cH, scale)
    dropzeros!(H_to)
    H0_to = H0 - H[:, row] .* (scale * c0H)
    return H_to, Vector(H0_to), 0, cH, c0H
end

function _catalysis_inner_update_from_regular(
    H::AbstractMatrix{<:Real},
    H0::AbstractVector{<:Real},
    edge::RegimeEdge,
    cat_grh::RegimeGraph,
    cn::CatalysisData,
    bn::Bnc,
)
    v_idx, v_sign = _edge_idx_sign(edge, cat_grh, :v)
    hp = get_hyperplane(cat_grh.hp_data[_space(cat_grh, :v)], v_idx)
    c_v, c0_v = _calc_c_c0(hp, cn.n_v, v_sign)
    c_x = _sparse_rational_vec(transpose(c_v[:, 1]) * cn.Π)

    flux_row = edge.i <= cn.r_v ? edge.i : edge.i - cn.r_v
    if edge.i > cn.r_v
        c_x = -c_x
        c0_v = -c0_v
    end

    row = cn.d_w + bn.r + flux_row
    return _rank1_inner_update_from_regular(H, H0, row, c_x, c0_v)
end

function _init_regular_or_nullity1_bnc_regime!(vtx::BncRegime)
    C_qKk, C0_qKk, nlt_qKk = _calc_C_C0_qKk(vtx.bind_rgm, vtx.catalysis_rgm)
    C_wKk, C0_wKk, nlt_wKk = _calc_C_C0_wKk(vtx)
    H_inner, H0_inner = get_affine_wKk̃2x(vtx)

    H_wKk, H0_wKk = let
        Z, Z0 = get_affine_wKk2wKk̃(vtx)
        H = H_inner * Z
        H0 = H_inner * Z0 + H0_inner
        droptol!(H, 1e-10), H0
    end

    vtx.H = H_wKk
    vtx.H0 = H0_wKk
    vtx.C_qKk_cat = C_qKk
    vtx.C0_qKk_cat = C0_qKk
    vtx.nlt_qKk_cat = nlt_qKk
    vtx.C_wKk = C_wKk
    vtx.C0_wKk = C0_wKk
    vtx.nlt_wKk = nlt_wKk
    return nothing
end

function _init_consistency_only_bnc_regime!(vtx::BncRegime)
    C_qKk_cat, C0_qKk_cat, nlt_qKk_cat = _calc_C_C0_qKk(vtx.bind_rgm, vtx.catalysis_rgm)
    C_wKk, C0_wKk, nlt_wKk = _calc_C_C0_wKk(vtx)

    vtx.H = nothing
    vtx.H0 = nothing
    vtx.C_qKk_cat = C_qKk_cat
    vtx.C0_qKk_cat = _materialize_real_vector(C0_qKk_cat)
    vtx.nlt_qKk_cat = nlt_qKk_cat
    vtx.C_wKk = C_wKk
    vtx.C0_wKk = _materialize_real_vector(C0_wKk)
    vtx.nlt_wKk = nlt_wKk
    return nothing
end

function _calc_C_C0_qKk(bind_rgm::BindRegime, cat_rgm::CatalysisRegime)
    C_xk_cat, C0_xk_cat = get_C_C0_xk(cat_rgm)
    C = C_xk_cat
    C0 = C0_xk_cat

    if is_singular(bind_rgm)
        M, M0 = get_affine_xk2qKk(bind_rgm)
        C_qKk, C0_qKk, nlt = _affine_mapping_polyhedra(C, C0, 0, M, M0)
        return C_qKk, C0_qKk, nlt
    else
        H, H0 = get_affine_qKk2xk(bind_rgm)
        C_qKk = C * H
        droptol!(C_qKk, 1e-10)
        C0_qKk = C0 + C * H0
        return C_qKk, C0_qKk, 0
    end
end

function _calc_C_C0_wKk(rgm::BncRegime)
    bind_rgm = get_binding_regime(rgm)
    cat_rgm = get_catalysis_regime(rgm)

    C_xk_bind, C0_xk_bind = get_C_C0_xk(bind_rgm)
    C_xk_cat, C0_xk_cat = get_C_C0_xk(cat_rgm)
    C = vcat(C_xk_bind, C_xk_cat)
    C0 = vcat(C0_xk_bind, C0_xk_cat)

    Z, Z0 = get_affine_wKk2wKk̃k(rgm) # not depends on initialization
    if is_singular(rgm)
        M, M0 = get_affine_xk2wKk̃k(rgm) # not depends on initialization
        C_wKkk, C0_wKkk, nlt = _affine_mapping_polyhedra(C, C0, 0, M, M0)
    else
        H, H0 = get_affine_wKk̃k2xk(rgm) # not depends on initialization
        C_wKkk = C * H
        C0_wKkk = C * H0 + C0
        nlt = 0
    end

    C_wKk = C_wKkk * Z
    droptol!(C_wKk, 1e-10)
    C0_wKk = C_wKkk * Z0 + C0_wKkk
    C_wKk, C0_wKk, nlt = _drop_trivial_true_rows(C_wKk, C0_wKk, nlt)
    return C_wKk, C0_wKk, nlt # change bases back to wKk
end

function _assign_inner_affine!(rgm::BncRegime, H, H0)
    rgm.H_inner = sparse(H)
    rgm.H0_inner = vec(H0)
    return rgm
end

function _direct_inner_affine(rgm::BncRegime)
    bn = get_binding_network(rgm)
    PΠ = get_PΠ(rgm.catalysis_rgm)
    _, P0 = get_affine_x2k̃(rgm.catalysis_rgm)
    N_ss = vcat(bn.N, PΠ)
    direction = _det_sign_exact(vcat(get_Lcat(bn), N_ss))
    perm = get_fixed_point_perm(rgm)[1]
    return _steady_state_affine(
        rgm.bind_rgm, perm, N_ss, P0, get_catalysis_network(rgm).r_v, direction, rgm.nlt
    )
end

function _same_ray(
    H1::AbstractMatrix{<:Real}, H2::AbstractMatrix{<:Real}; atol::Float64=1e-8
)
    v1 = vec(Float64.(Matrix(H1)))
    v2 = vec(Float64.(Matrix(H2)))
    i = findfirst(x -> abs(x) > atol, v1)
    isnothing(i) && return all(abs.(v2) .<= atol)
    scale = v2[i] / v1[i]
    scale > 0 || return false
    return all(abs.(v2 .- scale .* v1) .<= atol .* max(1, norm(v2, Inf)))
end

function _initialize_inner_affine_by_graph!(rgms::AbstractVector{BncRegime})
    isempty(rgms) && return NamedTuple[]

    bn = get_binding_network(first(rgms))
    cn = get_catalysis_network(bn)
    n_bind = n_bind_regimes(bn)
    n_cat = n_catalysis_regimes(bn)
    bind_grh = get_regimes_graph!(bn; full=false)
    cat_grh = get_catalysis_regimes_graph!(bn)
    r_v = cn.r_v

    for cat_idx in 1:n_cat
        cat_rgm = get_catalysis_regime(bn, cat_idx)
        N_ss = vcat(bn.N, get_PΠ(cat_rgm))
        perms = [
            get_fixed_point_perm(rgms[_bnc_linear_index(n_bind, bind_idx, cat_idx)])[1] for
            bind_idx in 1:n_bind
        ]
        nlt, _ = _calc_nullity(perms, N_ss)
        for bind_idx in 1:n_bind
            rgms[_bnc_linear_index(n_bind, bind_idx, cat_idx)].nlt = nlt[bind_idx]
        end
    end

    if _has_nontrivial_k_constraints(cn)
        for idx in eachindex(rgms)
            rgms[idx].nlt <= 1 || continue
            pair = _direct_inner_affine(rgms[idx])
            pair === nothing && continue
            _assign_inner_affine!(rgms[idx], pair[1], pair[2])
        end
        return NamedTuple[]
    end

    assigned = falses(length(rgms))
    inconsistencies = NamedTuple[]

    for seed in eachindex(rgms)
        rgms[seed].nlt == 0 || continue
        assigned[seed] && continue

        pair = _direct_inner_affine(rgms[seed])
        pair === nothing && continue
        _assign_inner_affine!(rgms[seed], pair[1], pair[2])
        assigned[seed] = true

        queue = [seed]
        while !isempty(queue)
            from = popfirst!(queue)
            from_rgm = rgms[from]
            from_rgm.nlt == 0 || continue
            bind_idx, cat_idx = _bnc_cart_index(n_bind, from)

            for edge in bind_grh.neighbors[bind_idx]
                to = _bnc_linear_index(n_bind, edge.to, cat_idx)
                rgms[to].nlt <= 1 || continue

                H_to, H0_to = if edge.i <= r_v
                    from_rgm.H_inner, from_rgm.H0_inner
                else
                    x_idx, x_sign = _edge_idx_sign(edge, bind_grh, :x)
                    hp = get_hyperplane(bind_grh.hp_data[_space(bind_grh, :x)], x_idx)
                    _rank1_step_update_from_regular(
                        from_rgm.H_inner, from_rgm.H0_inner, edge.i - r_v, hp, x_sign
                    )[1:2]
                end

                if assigned[to]
                    rgms[to].nlt == 1 &&
                        !_same_ray(rgms[to].H_inner, sparse(Float64.(H_to))) &&
                        push!(inconsistencies, (node=to, from=from, kind=:binding))
                    continue
                end

                _assign_inner_affine!(rgms[to], H_to, H0_to)
                assigned[to] = true
                rgms[to].nlt == 0 && push!(queue, to)
            end

            for edge in cat_grh.neighbors[cat_idx]
                to = _bnc_linear_index(n_bind, bind_idx, edge.to)
                rgms[to].nlt <= 1 || continue
                H_to, H0_to = _catalysis_inner_update_from_regular(
                    from_rgm.H_inner, from_rgm.H0_inner, edge, cat_grh, cn, bn
                )[1:2]

                if assigned[to]
                    rgms[to].nlt == 1 &&
                        !_same_ray(rgms[to].H_inner, sparse(Float64.(H_to))) &&
                        push!(inconsistencies, (node=to, from=from, kind=:catalysis))
                    continue
                end

                _assign_inner_affine!(rgms[to], H_to, H0_to)
                assigned[to] = true
                rgms[to].nlt == 0 && push!(queue, to)
            end
        end
    end

    for idx in eachindex(rgms)
        rgms[idx].nlt <= 1 || continue
        assigned[idx] && continue
        pair = _direct_inner_affine(rgms[idx])
        pair === nothing && continue
        _assign_inner_affine!(rgms[idx], pair[1], pair[2])
    end

    return inconsistencies
end

function _is_feasible_under_current_k_map(rgm::BncRegime)
    C, C0 = get_C_C0_xk(rgm)
    for i in 1:size(C, 1)
        if nnz(C[i, :]) == 0 && Float64(C0[i]) <= 1e-10
            return false
        end
    end
    poly = _build_polyhedron_from_C_C0(C, C0, 0; canonicalize=true)
    return _poly_is_full_dimensional(poly; canonicalize=false)
end

function _mark_feasible_bnc_regimes!(rgms::AbstractVector{BncRegime})
    isempty(rgms) && return 0
    cn = get_catalysis_network(first(rgms))
    if !_has_nontrivial_k_constraints(cn)
        for rgm in rgms
            rgm.is_feasible = true
        end
        return 0
    end

    n_removed = 0
    for rgm in rgms
        rgm.is_feasible = _is_feasible_under_current_k_map(rgm)
        n_removed += rgm.is_feasible ? 0 : 1
    end
    return n_removed
end

#========================================================================================#
#  Functions for displaying bnc regimes
#========================================================================================#

function summary_regime(rgm::BncRegime)
    rgm = get_bnc_regime(rgm)
    println(
        "bind_idx=$(get_idx(rgm.bind_rgm)), cat_idx=$(get_idx(rgm.catalysis_rgm)), nlt=$(rgm.nlt), stable=$(is_stable(rgm))",
    )
    println("Binding / catalysis conditions in (x, k):")
    display.(show_condition_xk(rgm; kind=:binding, log_space=false))
    display.(show_condition_xk(rgm; kind=:catalysis, log_space=false))
    println("Combined consistency in (w, K, k):")
    display.(show_condition_wKk(rgm; log_space=false))
    return nothing
end

summary(rgm::BncRegime) = summary_regime(rgm)
function summary_regime(model::Bnc, bind, cat)
    return summary_regime(get_bnc_regime(model, bind, cat; check=true))
end
summary(model::Bnc, bind, cat) = summary_regime(model, bind, cat)

@inline function _is_asymptotic(rgm::BncRegime)
    return is_asymptotic(rgm.bind_rgm) && is_asymptotic(rgm.catalysis_rgm)
end

@inline function _regime_display_dominant_mode(rgm::BncRegime)
    return "bind=$(get_binding_perm(rgm)), cat=$(get_catalysis_perm(rgm)), ss=$(get_perm(rgm))"
end

function Base.show(io::IO, rgm::BncRegime)
    return print(
        io,
        "BncRegime(",
        _regime_display_dominant_mode(rgm),
        ", nullity=",
        rgm.nlt,
        ", asymptotic=",
        _is_asymptotic(rgm),
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", rgm::BncRegime)
    println(io, "BncRegime")
    println(io, "  dominant mode: ", _regime_display_dominant_mode(rgm))
    println(io, "  nullity: ", rgm.nlt)
    return print(io, "  asymptotic: ", _is_asymptotic(rgm))
end

#=================================================THE main ENTRY=============================#

function match_regimes!(model::Bnc; warn_singular_propagation::Bool=true)
    return _with_regime_cache_lock(model) do
        if is_bnc_regimes_built(model)
            return nothing
        end

        find_all_regimes!(model)
        find_catalysis_regimes!(model)

        model.BncRegimes = _build_BncRegime(
            _catalysis_regimes_data(model), _bind_regimes_data(model)
        )

        model._diagnostics[:bnc_regime_initialization] = _initialize_regime!(
            model.BncRegimes; warn_singular_propagation=warn_singular_propagation
        ) # The real calculation

        return nothing
    end
end

function _build_BncRegime(cat_rgms::Regimes, bind_rgms::Regimes)
    return _build_BncRegime(cat_rgms.regimes_data, bind_rgms.regimes_data)
end

function _build_BncRegime(
    cat_rgms::AbstractVector{<:CatalysisRegime}, bind_rgms::AbstractVector{<:BindRegime}
)
    n_cat = length(cat_rgms)
    n_bind = length(bind_rgms)
    bncrgms = Vector{BncRegime}(undef, n_cat * n_bind)

    @info "Matching Catalysis Regimes and Binding Regimes to build BncRegimes..."
    Threads.@threads for cat_idx in 1:n_cat
        cat_rgm = cat_rgms[cat_idx]
        for bind_idx in 1:n_bind
            bind_rgm = bind_rgms[bind_idx]
            bncrgms[_bnc_linear_index(n_bind, bind_idx, cat_idx)] = BncRegime(
                bind_rgm, cat_rgm
            )
        end
    end
    @info "Finished matching BncRegimes."
    return bncrgms
end

function _bnc_regime_initialization_diagnostics(
    rgms::AbstractVector{BncRegime}, inconsistencies, n_removed::Integer
)
    return (;
        initialized=true,
        n_regimes=length(rgms),
        n_feasible=count(is_feasible, rgms),
        n_infeasible=count(rgm -> !is_feasible(rgm), rgms),
        n_singular=count(is_singular, rgms),
        n_nonsingular=count(rgm -> !is_singular(rgm), rgms),
        singular_propagation_inconsistencies=inconsistencies,
        n_singular_propagation_inconsistencies=length(inconsistencies),
        warning_affects_nonsingular=false,
        warning_scope=:singular_inner_affine_propagation,
        infeasible_removed=n_removed,
    )
end

function _initialize_regime!(
    rgms::AbstractVector{BncRegime}; warn_singular_propagation::Bool=true
)
    isempty(rgms) && return _empty_bnc_regime_diagnostics()

    @info "Initializing BncRegimes..."
    inconsistencies = _initialize_inner_affine_by_graph!(rgms)
    if warn_singular_propagation && !isempty(inconsistencies)
        @warn "Inconsistent singular BncRegime H_inner directions found during graph propagation: $(length(inconsistencies)) cases. These inconsistencies are confined to nlt == 1 singular inner-affine propagation and do not change nonsingular BNC regimes. Inspect `bnc_regime_diagnostics(model)` after `match_regimes!`, or pass `warn_singular_propagation=false` when scripted analyses later filter `singular=false`."
    end

    @showprogress Threads.@threads for idx in eachindex(rgms)
        vtx = rgms[idx]
        if vtx.nlt <= 1
            _init_regular_or_nullity1_bnc_regime!(vtx)
        else
            _init_consistency_only_bnc_regime!(vtx)
        end
    end

    n_removed = _mark_feasible_bnc_regimes!(rgms)
    n_removed == 0 ||
        @info "Removed $n_removed infeasible BncRegimes under affine k constraints."

    @info "Finished initializing BncRegimes."

    return _bnc_regime_initialization_diagnostics(rgms, inconsistencies, n_removed)
end
