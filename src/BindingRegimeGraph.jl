export get_regimes_graph!, SIMOPaths, get_polyhedra, get_polyhedron, get_SIMO_graph
export get_path, get_edge, get_intersect
export get_neighbor_graph_x, get_neighbor_graph_qK, get_neighbor_graph
export get_sources, get_sinks, get_sources_sinks
export get_RO_path, group_sum, get_RO_paths, summary_RO_path
export get_volume

#---------------------------------------------------------------------------
#              Binding Network Graph
#-------------------------------------------------------------------------
"""
    get_binding_network_grh(bnc::Bnc) -> SimpleGraph

Build the bipartite binding network graph between q and x symbols.
"""
function get_binding_network_grh(Bnc::Bnc)::SimpleGraph
    g = SimpleGraph(Bnc.d + Bnc.n)
    for vi in eachindex(Bnc._valid_L_idx)
        for vj in Bnc._valid_L_idx[vi]
            add_edge!(g, vi, vj + Bnc.d)
        end
    end
    return g # get first d nodes as total, last n nodes as x
end

#------------------------------------------------------------------------------
#                  Getting the Graph of of regimes
#----------------------------------------------------------------------------
"""
    get_regimes_graph!(bnc::Bnc; full=false) -> RegimeGraph

Ensure the vertex graph is built and return the fulfilled graph.
`full` is kept for API compatibility.
"""
get_regimes_graph!(args...; kwargs...) =
    get_regimes_graph!(get_binding_network(args...); kwargs...)
function get_regimes_graph!(model::Bnc; full::Bool=false)::RegimeGraph
    return _with_regime_cache_lock(model) do
        if isnothing(model.vertices_graph)
            find_all_regimes!(model)
        end
        return model.vertices_graph
    end
end

"""
    get_edge(grh::RegimeGraph, from, to; full=false) -> Union{Nothing, RegimeEdge}

Return the edge between two vertices, optionally computing qK directions.
"""
function get_edge(grh::RegimeGraph, from, to; kwargs...)::Union{Nothing, RegimeEdge}
    from = get_idx(get_binding_network(grh), from)
    to = get_idx(get_binding_network(grh), to)

    pos = get(grh.edge_pos[from], to, nothing)
    if pos === nothing
        return nothing
    end
    edge = grh.neighbors[from][pos]
    return edge
end

"""
    get_edge(bnc, from, to; kwargs...) -> Union{Nothing, RegimeEdge}

Convenience wrapper to fetch an edge from a model.
"""
get_edge(Bnc, args...; kwargs...) =
    let
        bn = get_binding_network(Bnc)
        vtx_grh = get_regimes_graph!(bn)
        get_edge(vtx_grh, args...; kwargs...)
    end

"""
    get_binding_network(grh::RegimeGraph, args...) -> Bnc

Return the model backing a vertex graph.
"""
get_binding_network(grh::RegimeGraph, args...) = grh.bn

#-----------------------------------------------------------------------------------
"""
    get_neighbor_graph_x(grh::RegimeGraph) -> SimpleGraph

Return the x-space neighbor graph for a vertex graph.
"""
function get_neighbor_graph_x(grh::RegimeGraph)
    n = length(grh.neighbors)
    g = SimpleGraph(n)
    for (i, edges) in enumerate(grh.neighbors)
        for e in edges
            add_edge!(g, i, e.to)
        end
    end
    return g
end
"""
    get_neighbor_graph_qK(grh::RegimeGraph; both_side=false) -> SimpleDiGraph

Return the qK-space neighbor graph for a vertex graph.
"""
get_neighbor_graph_qK(grh::RegimeGraph; both_side::Bool=false)::SimpleDiGraph =
    let
        qK_grh = let # construct the qK_graph
            Bnc = get_binding_network(grh)
            n = length(grh.neighbors)
            g = SimpleDiGraph(n)
            for (i, edges) in enumerate(grh.neighbors)
                if get_nullity(Bnc, i) > 1
                    continue
                end
                for e in edges
                    if !_edge_has_qK_interface(grh, e) || (!both_side && e.to < i)
                        continue
                    end
                    add_edge!(g, i, e.to)
                end
            end
            g
        end

        return qK_grh
    end

"""
    get_neighbor_graph_x(bnc::Bnc) -> SimpleGraph

Return the x-space neighbor graph for a model.
"""
get_neighbor_graph_x(args...) = get_neighbor_graph_x(get_regimes_graph!(args...))
"""
    get_neighbor_graph_qK(bnc::Bnc; kwargs...) -> SimpleDiGraph

Return the qK neighbor graph for a model.
"""
get_neighbor_graph_qK(Bnc::Bnc; kwargs...) =
    get_neighbor_graph_qK(get_regimes_graph!(Bnc; full=true); kwargs...)
"""
    get_neighbor_graph(args...; kwargs...) -> SimpleDiGraph

Alias for `get_neighbor_graph_qK`.
"""
get_neighbor_graph(args...; kwargs...) = get_neighbor_graph_qK(args...; kwargs...)
