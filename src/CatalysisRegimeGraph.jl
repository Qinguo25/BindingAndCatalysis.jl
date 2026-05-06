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
    xk_space = _space(grh, :xk)
    v_space = _space(grh, :v)
    grh.hp_data[xk_space] = RegimeToHyperplanePool(size(z, 2))
    db = grh.hp_data[xk_space]
    src = grh.hp_data[v_space]

    I = Int[]
    J = Int[]
    V = Int8[]

    for hp in src.hyperplanes
        c_xk = _sparse_rational_vec(hp * z)
        c0_xk = hp * z0
        push!(db.hyperplanes, RegimeHyperplane(c_xk, c0_xk))
        db.hp_dict[get_hp_key(db.hyperplanes[end])] = length(db.hyperplanes)
    end

    for p1 in eachindex(grh.neighbors)
        for e in grh.neighbors[p1]
            p2 = e.to
            p1 < p2 || continue

            rev_pos = grh.edge_pos[p2][p1]
            e_rev = grh.neighbors[p2][rev_pos]

            hid, dir = _edge_idx_sign(e, v_space)
            hid == 0 && continue

            push!(I, p1); push!(J, hid); push!(V, -dir)
            push!(I, p2); push!(J, hid); push!(V, dir)

            _set_edge_idx_sign!(e, xk_space, hid, dir)
            _set_edge_idx_sign!(e_rev, xk_space, hid, -dir)
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
    grh = _calc_regimes_graph(cn._S_helper, perms; primary_space=:v, secondary_space=:xk)
    grh.bn = cn
    _fulfill_catalysis_regimes_graph!(grh)
    cn.vertices_graph = grh
    return grh
end

function _neighbor_graph_by_space(grh::RegimeGraph, space; both_side::Bool=false)
    space_idx = _space(grh, Symbol(space))
    n = length(grh.neighbors)
    g = SimpleDiGraph(n)
    for (i, edges) in enumerate(grh.neighbors)
        for e in edges
            _edge_has_space(e, space_idx) || continue
            (!both_side && e.to < i) && continue
            add_edge!(g, i, e.to)
        end
    end
    return g
end

get_neighbor_graph_v(grh::RegimeGraph; kwargs...) = _neighbor_graph_by_space(grh, :v; kwargs...)
get_neighbor_graph_xk(grh::RegimeGraph; kwargs...) = _neighbor_graph_by_space(grh, :xk; kwargs...)
get_neighbor_graph(grh::RegimeGraph; edge_space=nothing, kwargs...) =
    _neighbor_graph_by_space(grh, isnothing(edge_space) ? _first_space(grh, (:qK, :qKk, :xk, :wKk, :v, :x)) : edge_space; kwargs...)

get_neighbor_graph_v(args...; kwargs...) = get_neighbor_graph_v(get_catalysis_regimes_graph!(args...); kwargs...)
get_neighbor_graph_xk(args...; kwargs...) = get_neighbor_graph_xk(get_catalysis_regimes_graph!(args...); kwargs...)
get_neighbor_graph(model::CatalysisData; kwargs...) = get_neighbor_graph_xk(model; kwargs...)
