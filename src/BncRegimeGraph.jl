export get_bnc_regimes_graph!, get_neighbor_graph_qKk, get_neighbor_graph_wKk

function _sparse_dot(c::SparseVector, v)
    s = zero(eltype(v))
    @inbounds for p in eachindex(c.nzind)
        s += c.nzval[p] * v[c.nzind[p]]
    end
    return s
end

function _extend_sparsevec(c::SparseVector, n_extra::Int)
    return sparsevec(copy(c.nzind), copy(c.nzval), length(c) + n_extra)
end

function _split_xk(c_xk::SparseVector, n_x::Int)
    I_x = Int[]
    V_x = eltype(c_xk.nzval)[]
    I_k = Int[]
    V_k = eltype(c_xk.nzval)[]
    @inbounds for p in eachindex(c_xk.nzind)
        idx = c_xk.nzind[p]
        val = c_xk.nzval[p]
        if idx <= n_x
            push!(I_x, idx); push!(V_x, val)
        else
            push!(I_k, idx - n_x); push!(V_k, val)
        end
    end
    return sparsevec(I_x, V_x, n_x), sparsevec(I_k, V_k, length(c_xk) - n_x)
end

function _combine_prefix_suffix(c_prefix::SparseVector, c_suffix::SparseVector)
    I = copy(c_prefix.nzind)
    V = copy(c_prefix.nzval)
    append!(I, c_suffix.nzind .+ length(c_prefix))
    append!(V, c_suffix.nzval)
    return sparsevec(I, V, length(c_prefix) + length(c_suffix))
end

function _xk_to_qKk_edge(c_xk::SparseVector, c0_xk, bind_rgm::BindRegime)
    H, H0 = get_affine_qK2x(bind_rgm)
    c_x, c_k = _split_xk(c_xk, size(H, 1))
    c_qK = _sparse_rational_vec(transpose(c_x) * H)
    c0_qK = c0_xk + _sparse_dot(c_x, H0)
    return _combine_prefix_suffix(c_qK, c_k), c0_qK
end

function _xk_to_wKk_edge(c_xk::SparseVector, c0_xk, rgm::BncRegime)
    H, H0 = get_affine_wKk2x(rgm)
    c_x, c_k = _split_xk(c_xk, size(H, 1))
    c_wK = _sparse_rational_vec(transpose(c_x) * H)
    k_offset = length(c_wK) - length(c_k)
    c_wKk = c_wK + sparsevec(c_k.nzind .+ k_offset, c_k.nzval, length(c_wK))
    c0_wK = c0_xk + _sparse_dot(c_x, H0)
    return c_wKk, c0_wK
end

function _add_bnc_edge_pair!(neighbors, from::Int, to::Int, i::Int)
    e = RegimeEdge(to, i, Tuple{Int,Int8}[(0, 0), (0, 0), (0, 0)])
    e_rev = RegimeEdge(from, i, Tuple{Int,Int8}[(0, 0), (0, 0), (0, 0)])
    push!(neighbors[from], e)
    push!(neighbors[to], e_rev)
    return e, e_rev
end

function _add_space_halfspace_pair!(db::RegimeToHyperplanePool, e::RegimeEdge, e_rev::RegimeEdge, space::Int, c, c0, sign::Integer)
    c0_exact = c0 isa ExactLogExpr ? c0 : ExactLogExpr(rationalize(Int, c0; tol=1e-10))
    hid, dir = add_halfspace!(db, _sparse_rational_vec(c), c0_exact, Int8(sign); canonicalize=true)
    hid == 0 && return nothing
    _set_edge_idx_sign!(e, space, hid, dir)
    _set_edge_idx_sign!(e_rev, space, hid, -dir)
    return hid
end

function _binding_xk_interface(bind_grh::RegimeGraph, edge::RegimeEdge, n_x::Int, n_k::Int)
    x_space = _space(bind_grh, :x)
    x_idx, x_sign = _edge_idx_sign(edge, x_space)
    hp = get_hyperplane(bind_grh.hp_data[x_space], x_idx)
    c_x, c0_x = _calc_c_c0(hp, n_x, x_sign)
    return _extend_sparsevec(c_x[:, 1], n_k), c0_x
end

function _copy_binding_edge!(
    neighbors,
    hp_xk::RegimeToHyperplanePool,
    hp_qKk::RegimeToHyperplanePool,
    hp_wKk::RegimeToHyperplanePool,
    bind_grh::RegimeGraph,
    rgms::AbstractVector{BncRegime},
    bind_edge::RegimeEdge,
    bind_idx::Int,
    cat_idx::Int,
    n_bind::Int,
    n_x::Int,
    n_k::Int,
)
    from = _bnc_linear_index(n_bind, bind_idx, cat_idx)
    to = _bnc_linear_index(n_bind, bind_edge.to, cat_idx)
    from < to || return nothing
    r_from = rgms[_bnc_linear_index(n_bind, bind_idx, cat_idx)]
    r_to = rgms[_bnc_linear_index(n_bind, bind_edge.to, cat_idx)]
    (is_feasible(r_from) && is_feasible(r_to)) || return nothing

    e, e_rev = _add_bnc_edge_pair!(neighbors, from, to, bind_edge.i)
    c_xk, c0_xk = _binding_xk_interface(bind_grh, bind_edge, n_x, n_k)
    _add_space_halfspace_pair!(hp_xk, e, e_rev, 1, c_xk, c0_xk, 1)

    if _edge_has_space(bind_edge, bind_grh, :qK)
        c_qK, c0_qK = _edge_qK_interface(bind_grh, bind_edge)
        c_qKk = _extend_sparsevec(c_qK, n_k)
        _add_space_halfspace_pair!(hp_qKk, e, e_rev, 2, c_qKk, c0_qK, 1)
    end

    if get_nullity(r_from) <= 1
        c_wKk, c0_wKk = _xk_to_wKk_edge(c_xk, c0_xk, r_from)
        _add_space_halfspace_pair!(hp_wKk, e, e_rev, 3, c_wKk, c0_wKk, 1)
    elseif get_nullity(r_to) <= 1
        c_wKk, c0_wKk = _xk_to_wKk_edge(-c_xk, -c0_xk, r_to)
        _add_space_halfspace_pair!(hp_wKk, e_rev, e, 3, c_wKk, c0_wKk, 1)
    end
    return nothing
end

function _copy_catalysis_edge!(
    neighbors,
    hp_xk::RegimeToHyperplanePool,
    hp_qKk::RegimeToHyperplanePool,
    hp_wKk::RegimeToHyperplanePool,
    cat_grh::RegimeGraph,
    rgms::AbstractVector{BncRegime},
    cat_edge::RegimeEdge,
    bind_idx::Int,
    cat_idx::Int,
    n_bind::Int,
)
    from = _bnc_linear_index(n_bind, bind_idx, cat_idx)
    to = _bnc_linear_index(n_bind, bind_idx, cat_edge.to)
    from < to || return nothing
    r_from = rgms[_bnc_linear_index(n_bind, bind_idx, cat_idx)]
    r_to = rgms[_bnc_linear_index(n_bind, bind_idx, cat_edge.to)]
    (is_feasible(r_from) && is_feasible(r_to)) || return nothing

    e, e_rev = _add_bnc_edge_pair!(neighbors, from, to, cat_edge.i)
    c_xk, c0_xk = _edge_interface(cat_grh, cat_edge, :xk)
    _add_space_halfspace_pair!(hp_xk, e, e_rev, 1, c_xk, c0_xk, 1)

    bind_rgm = get_binding_regime(r_from)
    if !is_singular(bind_rgm)
        c_qKk, c0_qKk = _xk_to_qKk_edge(c_xk, c0_xk, bind_rgm)
        _add_space_halfspace_pair!(hp_qKk, e, e_rev, 2, c_qKk, c0_qKk, 1)
    end

    if get_nullity(r_from) <= 1
        c_wKk, c0_wKk = _xk_to_wKk_edge(c_xk, c0_xk, r_from)
        _add_space_halfspace_pair!(hp_wKk, e, e_rev, 3, c_wKk, c0_wKk, 1)
    elseif get_nullity(r_to) <= 1
        c_wKk, c0_wKk = _xk_to_wKk_edge(-c_xk, -c0_xk, r_to)
        _add_space_halfspace_pair!(hp_wKk, e_rev, e, 3, c_wKk, c0_wKk, 1)
    end
    return nothing
end

function _finalize_bnc_hp_incidence!(grh::RegimeGraph, space::Int)
    db = grh.hp_data[space]
    I = Int[]
    J = Int[]
    V = Int8[]
    for (i, edges) in enumerate(grh.neighbors)
        for e in edges
            idx, sign = _edge_idx_sign(e, space)
            idx == 0 && continue
            push!(I, i); push!(J, idx); push!(V, -sign)
        end
    end
    db.hp_to_poly = FacetIncidence(
        sparse(I, J, V, length(grh.neighbors), length(db.hyperplanes)),
        sparse(J, I, V, length(db.hyperplanes), length(grh.neighbors)),
    )
    return nothing
end

function _bnc_xk_dominance_polyhedron(rgm::BncRegime)
    C, C0 = get_C_C0_xk(rgm; remove_h_redundancy=true)
    return get_polyhedron(C, C0, 0; canonicalize=true)
end

function _bnc_feasible_xk_polyhedra(rgms::AbstractVector{BncRegime})
    poly_by_idx = Dict{Int,Polyhedron}()
    for idx in eachindex(rgms)
        is_feasible(rgms[idx]) || continue
        poly = _bnc_xk_dominance_polyhedron(rgms[idx])
        isempty(poly) && continue
        dim(poly) == fulldim(poly) || continue
        poly_by_idx[idx] = poly
    end
    return poly_by_idx
end

function _try_add_qKk_edge_from_xk!(
    hp_qKk::RegimeToHyperplanePool,
    e::RegimeEdge,
    e_rev::RegimeEdge,
    c_xk,
    c0_xk,
    rgm::BncRegime,
)
    bind_rgm = get_binding_regime(rgm)
    is_singular(bind_rgm) && return nothing
    c_qKk, c0_qKk = _xk_to_qKk_edge(c_xk, c0_xk, bind_rgm)
    return _add_space_halfspace_pair!(hp_qKk, e, e_rev, 2, c_qKk, c0_qKk, 1)
end

function _try_add_wKk_edge_from_xk!(
    hp_wKk::RegimeToHyperplanePool,
    e::RegimeEdge,
    e_rev::RegimeEdge,
    c_xk,
    c0_xk,
    r_from::BncRegime,
    r_to::BncRegime,
)
    if get_nullity(r_from) <= 1
        c_wKk, c0_wKk = _xk_to_wKk_edge(c_xk, c0_xk, r_from)
        return _add_space_halfspace_pair!(hp_wKk, e, e_rev, 3, c_wKk, c0_wKk, 1)
    elseif get_nullity(r_to) <= 1
        c_wKk, c0_wKk = _xk_to_wKk_edge(-c_xk, -c0_xk, r_to)
        return _add_space_halfspace_pair!(hp_wKk, e_rev, e, 3, c_wKk, c0_wKk, 1)
    end
    return nothing
end

function _edge_sign_from_source_polyhedron(poly::Polyhedron, c, c0; tol::Float64=1.0e-8)
    c_dense = Float64.(collect(vec(c)))
    c0_float = Float64(c0)
    any(abs.(c_dense) .> tol) || return Int8(1)

    C, C0, _ = get_C_C0_nullity(poly)
    for row in axes(C, 1)
        a = Float64.(collect(vec(C[row, :])))
        scale = nothing
        is_parallel = true
        for idx in eachindex(c_dense)
            ci = c_dense[idx]
            ai = a[idx]
            if abs(ci) > tol
                ratio = ai / ci
                if isnothing(scale)
                    scale = ratio
                elseif abs(ratio - scale) > tol * max(1, abs(scale))
                    is_parallel = false
                    break
                end
            elseif abs(ai) > tol
                is_parallel = false
                break
            end
        end
        (is_parallel && !isnothing(scale) && abs(scale) > tol) || continue

        b = Float64(C0[row])
        if abs(b - scale * c0_float) <= tol * max(1, abs(b), abs(scale * c0_float))
            return scale > 0 ? Int8(-1) : Int8(1)
        end
    end

    point = get_one_inner_point(poly; rand_line=false, rand_ray=false)
    val = dot(c_dense, Float64.(point)) + c0_float
    return val > tol ? Int8(-1) : Int8(1)
end

function _try_add_reduced_xk_bnc_edge!(
    neighbors,
    hp_xk::RegimeToHyperplanePool,
    hp_qKk::RegimeToHyperplanePool,
    hp_wKk::RegimeToHyperplanePool,
    rgms::AbstractVector{BncRegime},
    poly_by_idx::Dict{Int,Polyhedron},
    from::Int,
    to::Int,
    ambient_dim::Int,
)
    ins_dim, ins = _poly_intersection_dim(poly_by_idx[from], poly_by_idx[to])
    ins_dim == ambient_dim - 1 || return false

    interface = _poly_interface_from_intersection(ins)
    isnothing(interface) && return false
    c_xk_raw, c0_xk_raw = interface
    edge_sign = _edge_sign_from_source_polyhedron(poly_by_idx[from], c_xk_raw, c0_xk_raw)

    e, e_rev = _add_bnc_edge_pair!(neighbors, from, to, 0)
    hid = _add_space_halfspace_pair!(hp_xk, e, e_rev, 1, c_xk_raw, c0_xk_raw, edge_sign)
    if isnothing(hid)
        pop!(neighbors[from])
        pop!(neighbors[to])
        return false
    end

    idx, sign = _edge_idx_sign(e, 1)
    hp = get_hyperplane(hp_xk, idx)
    c_xk, c0_xk = _calc_c_c0(hp, sign)

    r_from = rgms[from]
    r_to = rgms[to]
    _try_add_qKk_edge_from_xk!(hp_qKk, e, e_rev, c_xk, c0_xk, r_from)
    _try_add_wKk_edge_from_xk!(hp_wKk, e, e_rev, c_xk, c0_xk, r_from, r_to)
    return true
end

function _build_reduced_xk_bnc_regimes_graph!(model::Bnc)
    match_regimes!(model)
    rgms = model.BncRegimes
    n_nodes = length(rgms)
    cn = get_catalysis_network(model)
    dim_xk = model.n + cn.n_k

    poly_by_idx = _bnc_feasible_xk_polyhedra(rgms)
    hp_xk = RegimeToHyperplanePool(dim_xk)
    hp_qKk = RegimeToHyperplanePool(model.d + model.r + cn.n_k)
    hp_wKk = RegimeToHyperplanePool(cn.d_w + model.r + cn.n_k)
    neighbors = [RegimeEdge[] for _ in 1:n_nodes]

    feasible = sort!(collect(keys(poly_by_idx)))
    for i in 1:(length(feasible) - 1)
        from = feasible[i]
        for j in (i + 1):length(feasible)
            to = feasible[j]
            _try_add_reduced_xk_bnc_edge!(
                neighbors,
                hp_xk,
                hp_qKk,
                hp_wKk,
                rgms,
                poly_by_idx,
                from,
                to,
                dim_xk,
            )
        end
    end

    grh = RegimeGraph(
        neighbors,
        Any[hp_xk, hp_qKk, hp_wKk];
        bn=model,
        space_idx=Dict(:xk => 1, :qKk => 2, :wKk => 3),
    )
    _finalize_bnc_hp_incidence!(grh, 1)
    _finalize_bnc_hp_incidence!(grh, 2)
    _finalize_bnc_hp_incidence!(grh, 3)
    return grh
end

function get_bnc_regimes_graph!(model::Bnc)
    match_regimes!(model)
    rgms = model.BncRegimes
    cn = get_catalysis_network(model)
    _has_nontrivial_k_constraints(cn) && return _build_reduced_xk_bnc_regimes_graph!(model)

    n_bind = n_bind_regimes(model)
    n_cat = n_catalysis_regimes(model)
    n_nodes = length(rgms)

    bind_grh = get_regimes_graph!(model; full=true)
    cat_grh = get_catalysis_regimes_graph!(model)

    hp_xk = RegimeToHyperplanePool(model.n + cn.n_k)
    hp_qKk = RegimeToHyperplanePool(model.d + model.r + cn.n_k)
    hp_wKk = RegimeToHyperplanePool(cn.d_w + model.r + cn.n_k)
    neighbors = [RegimeEdge[] for _ in 1:n_nodes]

    for cat_idx in 1:n_cat
        for bind_idx in 1:n_bind
            for e in bind_grh.neighbors[bind_idx]
                _copy_binding_edge!(neighbors, hp_xk, hp_qKk, hp_wKk, bind_grh, rgms, e, bind_idx, cat_idx, n_bind, model.n, cn.n_k)
            end
            for e in cat_grh.neighbors[cat_idx]
                _copy_catalysis_edge!(neighbors, hp_xk, hp_qKk, hp_wKk, cat_grh, rgms, e, bind_idx, cat_idx, n_bind)
            end
        end
    end

    grh = RegimeGraph(
        neighbors,
        Any[hp_xk, hp_qKk, hp_wKk];
        bn=model,
        space_idx=Dict(:xk => 1, :qKk => 2, :wKk => 3),
    )
    _finalize_bnc_hp_incidence!(grh, 1)
    _finalize_bnc_hp_incidence!(grh, 2)
    _finalize_bnc_hp_incidence!(grh, 3)
    return grh
end

get_neighbor_graph_qKk(grh::RegimeGraph; kwargs...) = _neighbor_graph_by_space(grh, :qKk; kwargs...)
get_neighbor_graph_wKk(grh::RegimeGraph; kwargs...) = _neighbor_graph_by_space(grh, :wKk; kwargs...)

get_neighbor_graph_qKk(model::Bnc; kwargs...) = get_neighbor_graph_qKk(get_bnc_regimes_graph!(model); kwargs...)
get_neighbor_graph_wKk(model::Bnc; kwargs...) = get_neighbor_graph_wKk(get_bnc_regimes_graph!(model); kwargs...)
