export get_regimes_graph!, SISOPaths, get_polyhedra, get_polyhedron, get_SISO_graph
export get_path, get_edge, get_intersect
export get_neighbor_graph_x, get_neighbor_graph_qK, get_neighbor_graph
export get_sources, get_sinks, get_sources_sinks
export get_RO_path, group_sum, get_RO_paths, summary_RO_path
export get_volume

#---------------------------------------------------------------------------------------------------
#             Helper functions: Functions for construct the regime graph paths
#----------------------------------------------------------------------------------------------------


"""
    get_sources(g::AbstractGraph) -> Set{Int}

Return source vertices with zero indegree.
"""
get_sources(g::AbstractGraph) = Set(v for v in vertices(g) if indegree(g, v) == 0)
"""
    get_sinks(g::AbstractGraph) -> Set{Int}

Return sink vertices with zero outdegree.
"""
get_sinks(g::AbstractGraph)   = Set(v for v in vertices(g) if outdegree(g, v) == 0)
"""
    get_sources_sinks(g::AbstractGraph) -> (Set{Int}, Set{Int})

Return sources and sinks for a graph.
"""
get_sources_sinks(g::AbstractGraph) = (get_sources(g), get_sinks(g))

"""
    get_sources_sinks(model::Bnc, g::AbstractGraph) -> (Vector{Int}, Vector{Int})

Return sources and sinks while excluding singular regimes.
"""
function get_sources_sinks(model::Bnc, g::AbstractGraph)
    sources_all = get_sources(g) 
    sinks_all   = get_sinks(g) 
    common_vs = intersect(sources_all, sinks_all)
    filter!(common_vs) do v
        get_nullity(model, v) > 0
    end
    sources = setdiff(sources_all, common_vs)
    sinks = setdiff(sinks_all, common_vs)
    return (collect(sources), collect(sinks))
end

# 只遍历子图：sources 可达 & 能到 sinks
"""
    _reachable_from_sources(g::AbstractGraph, sources) -> Vector{Bool}

Return a boolean mask of vertices reachable from sources.
"""
function _reachable_from_sources(g::AbstractGraph, sources::AbstractVector{Int})
    n = nv(g)
    seen = falses(n)
    stack = Int[]
    for s in sources
        if !seen[s]
            seen[s] = true
            push!(stack, s)
            while !isempty(stack)
                v = pop!(stack)
                for nb in outneighbors(g, v)
                    if !seen[nb]
                        seen[nb] = true
                        push!(stack, nb)
                    end
                end
            end
        end
    end
    return seen
end

"""
    _can_reach_sinks(g::AbstractGraph, sinks) -> Vector{Bool}

Return a boolean mask of vertices that can reach sinks.
"""
function _can_reach_sinks(g::AbstractGraph, sinks::AbstractVector{Int})
    n = nv(g)
    seen = falses(n)
    stack = Int[]
    for t in sinks
        if !seen[t]
            seen[t] = true
            push!(stack, t)
            while !isempty(stack)
                v = pop!(stack)
                for nb in inneighbors(g, v)   # 反向走
                    if !seen[nb]
                        seen[nb] = true
                        push!(stack, nb)
                    end
                end
            end
        end
    end
    return seen
end

"""
    _enumerate_paths(g; sources, sinks) -> Vector{Vector{Int}}

Enumerate all paths in a DAG from `sources` to `sinks`.
"""
function _enumerate_paths(
    g::AbstractGraph;
    sources::AbstractVector{Int},
    sinks::AbstractVector{Int},
)::Vector{Vector{Int}}

    @info "sources: $sources"
    @info "sinks: $sinks"
    n = nv(g)

    # 剪枝：只处理相关子图
    fromS = _reachable_from_sources(g, sources)
    toT   = _can_reach_sinks(g, sinks)
    active = fromS .& toT

    is_sink = falses(n)
    @inbounds for t in sinks
        is_sink[t] = true
    end

    # 拓扑排序（DAG）
    topo = topological_sort_by_dfs(g)   # Graphs.jl
    # memo[v] = Vector{Vector{Int}} 或 nothing
    memo = Vector{Union{Nothing, Vector{Vector{Int}}}}(undef, n)
    fill!(memo, nothing)

    @info "Start enumerating paths from sources to sinks. This may take a while if there are many paths."
    # 逆拓扑：先算子节点，再算父节点

    @info "Total vertices to process in topological order: $(length(topo))"
    @showprogress for v in Iterators.reverse(topo)
        active[v] || continue

        if is_sink[v]
            memo[v] = Vector{Vector{Int}}(undef, 1)
            memo[v][1] = [v]
            continue
        end

        # 收集所有 nb 的路径，并在前面加 v
        acc = Vector{Vector{Int}}()
        # 你也可以在这里做 sizehint!（需要先统计 path 数量，会多一次循环；看你取舍）
        for nb in outneighbors(g, v)
            active[nb] || continue
            paths_nb = memo[nb]
            paths_nb === nothing && continue
            for p in paths_nb
                L = length(p)
                np = Vector{Int}(undef, L + 1)
                np[1] = v
                @inbounds copyto!(np, 2, p, 1, L)
                push!(acc, np)
            end
        end

        memo[v] = isempty(acc) ? nothing : acc
    end

    # 汇总 sources 的结果
    @info "Finished enumerating paths. Now collecting paths from sources. Total sources: $(length(sources))"
    out = Vector{Vector{Int}}()
    @showprogress for s in sources
        active[s] || continue
        ps = memo[s]
        ps === nothing && continue
        append!(out, ps)
    end

    sort!(out)
    return out
end

"""
    _ensure_full_regimes_graph!(grh::VertexGraph) -> nothing

Ensure qK change directions are computed for a vertex graph.
"""
function _ensure_full_regimes_graph!(grh::VertexGraph)
    if !grh.change_dir_qK_computed
        @info "Calculating vertices neighbor graph with qK change dir"
        _fulfill_regimes_graph!(grh)
        grh.change_dir_qK_computed = true
    end
    return nothing
end

_ensure_full_regimes_graph!(model::Bnc) = _ensure_full_regimes_graph!(get_regimes_graph!(model; full=false))




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
            add_edge!(g, vi, vj+Bnc.d)
        end
    end
    return g # get first d nodes as total, last n nodes as x
end




#------------------------------------------------------------------------------
#                  Getting the Graph of of regimes
#----------------------------------------------------------------------------
"""
    get_regimes_graph!(bnc::Bnc; full=false) -> VertexGraph

Ensure the vertex graph is built; when `full=true`, also compute qK change directions.
"""
function get_regimes_graph!(model::Bnc; full::Bool=false)::VertexGraph
    if isnothing(model.vertices_graph)
        find_all_regimes!(model)
    end
    return model.vertices_graph
end


"""
    get_edge(grh::VertexGraph, from, to; full=false) -> Union{Nothing, VertexEdge}

Return the edge between two vertices, optionally computing qK directions.
"""
function get_edge(grh::VertexGraph, from, to; kwargs...)::Union{Nothing, VertexEdge}
    
    from = get_idx(get_binding_network(grh), from)
    to = get_idx(get_binding_network(grh), to)
    
    pos = get(grh.edge_pos[from], to, nothing)
    if pos === nothing
        return nothing
    end
    edge = grh.neighbors[from][pos]
    # full && _materialize_edge_qK_interface!(grh, edge)
    return edge
end


"""
    get_edge(bnc, from, to; kwargs...) -> Union{Nothing, VertexEdge}

Convenience wrapper to fetch an edge from a model.
"""
get_edge(Bnc, from, to; kwargs...)= let
    vtx_grh = get_regimes_graph!(Bnc; full=false)
    bn = get_binding_network(Bnc)
    from = get_idx(Bnc, from)
    to = get_idx(Bnc, to)
    get_edge(vtx_grh, from, to; kwargs...)
end

"""
    get_binding_network(grh::VertexGraph, args...) -> Bnc

Return the model backing a vertex graph.
"""
get_binding_network(grh::VertexGraph,args...) = grh.bn
# get_regimes_graph!(grh::VertexGraph,args...; kwargs...) = grh






#-----------------------------------------------------------------------------------
"""
    get_neighbor_graph_x(grh::VertexGraph) -> SimpleGraph

Return the x-space neighbor graph for a vertex graph.
"""
get_neighbor_graph_x(grh::VertexGraph) = grh.x_grh
"""
    get_neighbor_graph_x(bnc::Bnc) -> SimpleGraph

Return the x-space neighbor graph for a model.
"""
get_neighbor_graph_x(Bnc::Bnc) = get_neighbor_graph_x(get_regimes_graph!(Bnc; full=false))

"""
    get_neighbor_graph_qK(grh::VertexGraph; both_side=false) -> SimpleDiGraph

Return the qK-space neighbor graph for a vertex graph.
"""
get_neighbor_graph_qK(grh::VertexGraph; both_side::Bool=false)::SimpleDiGraph = let

    qK_grh = let # construct the qK_graph
        Bnc = get_binding_network(grh)
        n = length(grh.neighbors)
        g = SimpleDiGraph(n)
        for (i, edges) in enumerate(grh.neighbors)
            if get_nullity(Bnc,i) >1
                continue
            end
            for e in edges
                if !_edge_has_qK_interface(e) || (!both_side && e.to < i)
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
    get_neighbor_graph_qK(bnc::Bnc; kwargs...) -> SimpleDiGraph

Return the qK neighbor graph for a model.
"""
get_neighbor_graph_qK(Bnc::Bnc; kwargs...) = get_neighbor_graph_qK(get_regimes_graph!(Bnc; full=true); kwargs...)

"""
    get_neighbor_graph(args...; kwargs...) -> SimpleDiGraph

Alias for `get_neighbor_graph_qK`.
"""
get_neighbor_graph(args...; kwargs...) = get_neighbor_graph_qK(args...; kwargs...)

    






