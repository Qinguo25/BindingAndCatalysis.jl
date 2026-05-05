export get_catalysis_regimes_graph!, get_neighbor_graph_v, get_neighbor_graph_xk

function _sparse_rational_vec(v)
    vv = vec(v)
    I = Int[]
    V = Rational{Int}[]
    for i in eachindex(vv)
        iszero(vv[i]) && continue
        push!(I, i)
        push!(V, vv[i] isa Rational{Int} ? vv[i] : Rational{Int}(vv[i]))
    end
    return sparsevec(I, V, length(vv))
end

# There's no need to calculate each xk edge again. Will optimized further.
function _fulfill_catalysis_regimes_graph!(grh::RegimeGraph)
    cn = get_catalysis_network(grh.bn)
    z, z0 = get_affine_xk2v(cn)
    grh.hp_data[_EDGE_SPACE_XK] = RegimeToHyperplanePool(size(z, 2))
    db = grh.hp_data[_EDGE_SPACE_XK]

    I = Int[]
    J = Int[]
    V = Int8[]

    for p1 in eachindex(grh.neighbors)
        for e in grh.neighbors[p1]
            p2 = e.to
            p1 < p2 || continue

            rev_pos = grh.edge_pos[p2][p1]
            e_rev = grh.neighbors[p2][rev_pos]

            # The key is to fill the xk half space according to the v edge, which is already filled by the catalysis regime graph. We can directly use the v edge's hyperplane to get the c and c0 for xk half space.
            v_idx, dir_v = _edge_idx_sign(e, _EDGE_SPACE_V)
            hp = get_hyperplane(grh.hp_data[_EDGE_SPACE_V], v_idx) # hyperplane_perm
            c_xk = _sparse_rational_vec(hp * z)
            c0_xk = hp * z0

            hid, dir = add_halfspace!(db, c_xk, c0_xk, dir_v; canonicalize=true)
            hid == 0 && continue

            push!(I, p1); push!(J, hid); push!(V, -dir)
            push!(I, p2); push!(J, hid); push!(V, dir)

            _set_edge_idx_sign!(e, _EDGE_SPACE_XK, hid, dir)
            _set_edge_idx_sign!(e_rev, _EDGE_SPACE_XK, hid, -dir)
        end
    end

    M = sparse(I, J, V, length(grh.neighbors), length(db.hyperplanes))
    MT = sparse(J, I, V, length(db.hyperplanes), length(grh.neighbors))
    db.hp_to_poly = FacetIncidence(M, MT)
    return nothing
end

function get_catalysis_regimes_graph!(args...; kwargs...)
    cn = get_catalysis_network(args...; kwargs...)
    if !isnothing(cn.vertices_graph)
        return cn.vertices_graph
    end
    find_catalysis_regimes!(cn)
    perms = _catalysis_regimes_perms(cn)
    grh = _calc_regimes_graph(cn._S_helper, perms)
    grh.bn = cn
    _fulfill_catalysis_regimes_graph!(grh)
    cn.vertices_graph = grh
    return grh
end

function _neighbor_graph_by_space(grh::RegimeGraph, space::Int; both_side::Bool=false)
    n = length(grh.neighbors)
    g = SimpleDiGraph(n)
    for (i, edges) in enumerate(grh.neighbors)
        for e in edges
            _edge_has_space(e, space) || continue
            (!both_side && e.to < i) && continue
            add_edge!(g, i, e.to)
        end
    end
    return g
end

get_neighbor_graph_v(grh::RegimeGraph; kwargs...) = _neighbor_graph_by_space(grh, _EDGE_SPACE_V; kwargs...)
get_neighbor_graph_xk(grh::RegimeGraph; kwargs...) =
    _neighbor_graph_by_space(grh, grh.bn isa Bnc ? _EDGE_SPACE_BNC_XK : _EDGE_SPACE_XK; kwargs...)

get_neighbor_graph_v(args...; kwargs...) = get_neighbor_graph_v(get_catalysis_regimes_graph!(args...); kwargs...)
get_neighbor_graph_xk(args...; kwargs...) = get_neighbor_graph_xk(get_catalysis_regimes_graph!(args...); kwargs...)
get_neighbor_graph(model::CatalysisData; kwargs...) = get_neighbor_graph_xk(model; kwargs...)
