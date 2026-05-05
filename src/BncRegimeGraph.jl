export get_bnc_regimes_graph!, get_neighbor_graph_qKk, get_neighbor_graph_wKk

@inline _bnc_linear_index(n_bind::Int, bind_idx::Int, cat_idx::Int) = bind_idx + (cat_idx - 1) * n_bind
@inline _bnc_cart_index(n_bind::Int, idx::Int) = ((idx - 1) % n_bind + 1, (idx - 1) ÷ n_bind + 1)

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
    c0_wK = c0_xk + _sparse_dot(c_x, H0)
    return _combine_prefix_suffix(c_wK, c_k), c0_wK
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
    x_space = _edge_space_index(bind_grh, :x)
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
    rgms::AbstractMatrix{BncRegime},
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

    e, e_rev = _add_bnc_edge_pair!(neighbors, from, to, bind_edge.i)
    c_xk, c0_xk = _binding_xk_interface(bind_grh, bind_edge, n_x, n_k)
    _add_space_halfspace_pair!(hp_xk, e, e_rev, _EDGE_SPACE_BNC_XK, c_xk, c0_xk, 1)

    if _edge_has_space(bind_edge, bind_grh, :qK)
        c_qK, c0_qK = _edge_qK_interface(bind_grh, bind_edge)
        c_qKk = _extend_sparsevec(c_qK, n_k)
        _add_space_halfspace_pair!(hp_qKk, e, e_rev, _EDGE_SPACE_QKK, c_qKk, c0_qK, 1)
    end

    r_from = rgms[cat_idx, bind_idx]
    r_to = rgms[cat_idx, bind_edge.to]
    if get_nullity(r_from) <= 1 && get_nullity(r_to) <= 1
        c_wKk, c0_wKk = _xk_to_wKk_edge(c_xk, c0_xk, r_from)
        _add_space_halfspace_pair!(hp_wKk, e, e_rev, _EDGE_SPACE_WKK, c_wKk, c0_wKk, 1)
    end
    return nothing
end

function _copy_catalysis_edge!(
    neighbors,
    hp_xk::RegimeToHyperplanePool,
    hp_qKk::RegimeToHyperplanePool,
    hp_wKk::RegimeToHyperplanePool,
    cat_grh::RegimeGraph,
    rgms::AbstractMatrix{BncRegime},
    cat_edge::RegimeEdge,
    bind_idx::Int,
    cat_idx::Int,
    n_bind::Int,
)
    from = _bnc_linear_index(n_bind, bind_idx, cat_idx)
    to = _bnc_linear_index(n_bind, bind_idx, cat_edge.to)
    from < to || return nothing

    e, e_rev = _add_bnc_edge_pair!(neighbors, from, to, cat_edge.i)
    c_xk, c0_xk = _edge_interface(cat_grh, cat_edge, :xk)
    _add_space_halfspace_pair!(hp_xk, e, e_rev, _EDGE_SPACE_BNC_XK, c_xk, c0_xk, 1)

    r_from = rgms[cat_idx, bind_idx]
    r_to = rgms[cat_edge.to, bind_idx]
    bind_rgm = get_binding_regime(r_from)
    if !is_singular(bind_rgm)
        c_qKk, c0_qKk = _xk_to_qKk_edge(c_xk, c0_xk, bind_rgm)
        _add_space_halfspace_pair!(hp_qKk, e, e_rev, _EDGE_SPACE_QKK, c_qKk, c0_qKk, 1)
    end

    if get_nullity(r_from) <= 1 && get_nullity(r_to) <= 1
        c_wKk, c0_wKk = _xk_to_wKk_edge(c_xk, c0_xk, r_from)
        _add_space_halfspace_pair!(hp_wKk, e, e_rev, _EDGE_SPACE_WKK, c_wKk, c0_wKk, 1)
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
            push!(I, i); push!(J, idx); push!(V, sign)
        end
    end
    db.hp_to_poly = FacetIncidence(
        sparse(I, J, V, length(grh.neighbors), length(db.hyperplanes)),
        sparse(J, I, V, length(db.hyperplanes), length(grh.neighbors)),
    )
    return nothing
end

function get_bnc_regimes_graph!(model::Bnc)
    match_regimes!(model)
    rgms = model.BncRegimes
    n_cat, n_bind = size(rgms)
    n_nodes = length(rgms)
    cn = get_catalysis_network(model)

    bind_grh = get_regimes_graph!(model; full=true)
    cat_grh = get_catalysis_regimes_graph!(model)

    hp_xk = RegimeToHyperplanePool(model.n + cn.n_v)
    hp_qKk = RegimeToHyperplanePool(model.n + cn.n_v)
    hp_wKk = RegimeToHyperplanePool(cn.d_w + model.r + cn.n_v)
    neighbors = [RegimeEdge[] for _ in 1:n_nodes]

    for cat_idx in 1:n_cat
        for bind_idx in 1:n_bind
            for e in bind_grh.neighbors[bind_idx]
                _copy_binding_edge!(neighbors, hp_xk, hp_qKk, hp_wKk, bind_grh, rgms, e, bind_idx, cat_idx, n_bind, model.n, cn.n_v)
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
        space_idx=Dict(:xk => _EDGE_SPACE_BNC_XK, :qKk => _EDGE_SPACE_QKK, :wKk => _EDGE_SPACE_WKK),
    )
    _finalize_bnc_hp_incidence!(grh, _EDGE_SPACE_BNC_XK)
    _finalize_bnc_hp_incidence!(grh, _EDGE_SPACE_QKK)
    _finalize_bnc_hp_incidence!(grh, _EDGE_SPACE_WKK)
    return grh
end

get_neighbor_graph_qKk(grh::RegimeGraph; kwargs...) = _neighbor_graph_by_space(grh, :qKk; kwargs...)
get_neighbor_graph_wKk(grh::RegimeGraph; kwargs...) = _neighbor_graph_by_space(grh, :wKk; kwargs...)

get_neighbor_graph_qKk(model::Bnc; kwargs...) = get_neighbor_graph_qKk(get_bnc_regimes_graph!(model); kwargs...)
get_neighbor_graph_wKk(model::Bnc; kwargs...) = get_neighbor_graph_wKk(get_bnc_regimes_graph!(model); kwargs...)
