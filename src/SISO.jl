"""
Axis-Aligned SISO Path-Condition Backend
========================================

This file contains the shared path-condition backend used by
`SISOPaths.get_polyhedra`.

High-level idea
----------------
1. Build a directed graph from regime adjacency and the chosen `change_qK`
   coordinate.
2. Precompute graph reachability summaries:
   - `sources`, `sinks`
   - `upstream[i]`: vertices that can reach `i`
   - `downstream[i]`: vertices reachable from `i`
3. Answer pair query `(from, to)` with `_find_pair_path_conditions!` using
   recursive, memoized decomposition.
4. Propagate path conditions using intersections of edge-interface prisms, with
   vertex prisms only used for trivial length-0 paths.

Why `SISO.get_polyhedra` now uses this backend
----------------------------------------------
The old suffix-DAG implementation eagerly materialized a large shared state over
all enumerated source-to-sink paths. The current implementation uses a shared
pair solver instead:

- reachability caches cut impossible subproblems early
- memoization avoids repeated pair solves
- vertex/interface prism caches avoid repeated projections
- infeasible intersections are pruned immediately

This backend is intentionally axis-aligned: the varying direction is always one
of the qK coordinates, so path conditions are computed directly in reduced
qK-space by eliminating `change_qK_idx`.
"""


export SISOPaths, get_polyhedra, get_polyhedron, get_SISO_graph
export get_sources, get_sinks, get_sources_sinks
export get_regimes_graph!
export get_path, get_edge, get_intersect
export get_neighbor_graph_x, get_neighbor_graph_qK, get_neighbor_graph
export get_RO_path, group_sum, get_RO_paths, summary_RO_path
export get_volume

#---------------------------------------------------------------------------------------------------
#             Helper functions: Functions for construct the regime graph paths
#----------------------------------------------------------------------------------------------------






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




# #---------------------------------------------------------------------------
# #              Binding Network Graph
# #-------------------------------------------------------------------------
# """
#     get_binding_network_grh(bnc::Bnc) -> SimpleGraph

# Build the bipartite binding network graph between q and x symbols.
# """
# function get_binding_network_grh(Bnc::Bnc)::SimpleGraph
#     g = SimpleGraph(Bnc.d + Bnc.n)
#     for vi in eachindex(Bnc._valid_L_idx)
#         for vj in Bnc._valid_L_idx[vi]
#             add_edge!(g, vi, vj+Bnc.d)
#         end
#     end
#     return g # get first d nodes as total, last n nodes as x
# end




#------------------------------------------------------------------------------
#                  Getting the Graph of of regimes
#----------------------------------------------------------------------------
"""
    get_regimes_graph!(bnc::Bnc; full=false) -> VertexGraph

Ensure the vertex graph is built; when `full=true`, also compute qK change directions.
"""
function get_regimes_graph!(Bnc::Bnc; full::Bool=false)::VertexGraph

    if full
        vtx_graph = get_regimes_graph!(Bnc; full=false)
        _ensure_full_regimes_graph!(vtx_graph)
    else
        if isnothing(Bnc.vertices_graph)
            find_all_regimes!(Bnc)
        end
    end

    return Bnc.vertices_graph
end


"""
    get_edge(grh::VertexGraph, from, to; full=false) -> Union{Nothing, VertexEdge}

Return the edge between two vertices, optionally computing qK directions.
"""
function get_edge(grh::VertexGraph, from, to; kwargs...)::Union{Nothing, VertexEdge}
    
    from = get_idx(get_binding_network(grh), from)
    to = get_idx(get_binding_network(grh), to)
    
    # if full
    #     _ensure_full_regimes_graph!(grh)
    # end
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
    _ensure_full_regimes_graph!(grh)

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



"""
    get_sources_sinks(model::Bnc, connectome::AbstractMatrix{Bool}) -> (Vector{Int}, Vector{Int})

Return sources and sinks for a boolean adjacency matrix while excluding
singular vertices that are isolated in the directed graph.
"""
function get_sources_sinks(model::Bnc, connectome::AbstractMatrix{Bool})
    n = size(connectome, 1)
    size(connectome, 2) == n || error("connectome must be square, got size $(size(connectome)).")

    sources_all = Set(v for v in 1:n if !any(@view connectome[:, v]))
    sinks_all = Set(v for v in 1:n if !any(@view connectome[v, :]))
    common_vs = intersect(sources_all, sinks_all)
    regimes = _bind_regimes_data(model)
    filter!(v -> regimes[v].nullity > 0, common_vs)
    sources = sort!(collect(setdiff(sources_all, common_vs)))
    sinks = sort!(collect(setdiff(sinks_all, common_vs)))
    return sources, sinks
end

@inline function _direction_score(dir::SparseVector{Float64,Int}, change_qK_idx::Integer)
    return get(dir, Int(change_qK_idx), 0.0)
end

@inline function _direction_score(
    dir::SparseVector{Float64,Int},
    weights::AbstractVector{<:Real},
    d::Integer,
)
    I, V = findnz(dir)
    acc = 0.0
    @inbounds for k in eachindex(I)
        idx = I[k]
        idx <= d || continue
        acc += V[k] * Float64(weights[idx])
    end
    return acc
end

function _collect_oriented_edge_pairs(
    grh::VertexGraph,
    score_fn::F;
    tol::Float64=1e-6,
) where {F}
    _ensure_full_regimes_graph!(grh)
    bn = get_binding_network(grh)
    regimes = _bind_regimes_data(bn)
    thread_edges = [Tuple{Int,Int}[] for _ in 1:Threads.maxthreadid()]

    Threads.@threads for i in eachindex(grh.neighbors)
        regimes[i].nullity > 1 && continue
        local_edges = thread_edges[Threads.threadid()]
        for e in grh.neighbors[i]
            (!_edge_has_qK_interface(e) || e.to < i) && continue
            iface = _edge_qK_interface(grh, e)
            iface === nothing && continue
            score = score_fn(iface[1])
            if score > tol
                push!(local_edges, (i, e.to))
            elseif score < -tol
                push!(local_edges, (e.to, i))
            end
        end
    end

    return reduce(vcat, thread_edges; init=Tuple{Int,Int}[])
end

function _collect_oriented_edge_pairs(
    grh::VertexGraph,
    change_qK_idx::Integer;
    tol::Float64=1e-6,
)
    idx = Int(change_qK_idx)
    return _collect_oriented_edge_pairs(grh, dir -> _direction_score(dir, idx); tol=tol)
end

function _collect_oriented_edge_pairs(
    grh::VertexGraph,
    v::AbstractVector{<:Real};
    tol::Float64=1e-6,
)
    bn = get_binding_network(grh)
    length(v) == bn.d || error("Length of v must be $(bn.d), got $(length(v)).")
    weights = Float64.(v)
    return _collect_oriented_edge_pairs(grh, dir -> _direction_score(dir, weights, bn.d); tol=tol)
end

function _edge_pairs_to_connectome(
    n_vertices::Integer,
    edge_pairs::AbstractVector{<:Tuple{Int,Int}},
)::Matrix{Bool}
    connectome = fill(false, Int(n_vertices), Int(n_vertices))
    @inbounds for (from, to) in edge_pairs
        connectome[from, to] = true
    end
    return connectome
end

function _edge_pairs_to_digraph(
    n_vertices::Integer,
    edge_pairs::AbstractVector{<:Tuple{Int,Int}},
)::SimpleDiGraph
    g = SimpleDiGraph(Int(n_vertices))
    @inbounds for (from, to) in edge_pairs
        add_edge!(g, from, to)
    end
    return g
end

function _oriented_connectome(
    grh::VertexGraph,
    dir_spec;
    tol::Float64=1e-6,
)::Matrix{Bool}
    edge_pairs = _collect_oriented_edge_pairs(grh, dir_spec; tol=tol)
    return _edge_pairs_to_connectome(length(grh.neighbors), edge_pairs)
end

function _oriented_digraph(
    grh::VertexGraph,
    dir_spec;
    tol::Float64=1e-6,
)::SimpleDiGraph
    edge_pairs = _collect_oriented_edge_pairs(grh, dir_spec; tol=tol)
    return _edge_pairs_to_digraph(length(grh.neighbors), edge_pairs)
end




export get_sources, get_sinks, get_sources_sinks







_clean_polyhedron!(p::Polyhedron) = (detecthlinearity!(p); removehredundancy!(p); p)

"""
    _get_interface_prism(model, from, to, change_qK_idx) -> Polyhedron

Project the interface `poly(from) ∩ poly(to)` by eliminating `change_qK_idx`.
"""
function _get_interface_prism(
    bnc_sys::Bnc,
    vertex_idx_from::Int,
    vertex_idx_to::Int,
    change_qK_idx::Int,
)::Polyhedra.Polyhedron
    p = intersect(get_polyhedron(bnc_sys, vertex_idx_from), get_polyhedron(bnc_sys, vertex_idx_to))
    detecthlinearity!(p)
    removehredundancy!(p)
    isempty(p) && return p
    p = eliminate(p, change_qK_idx)
    removehredundancy!(p)
    return p
end

"""
    _get_polyhedron_prism(model, vertex_idx, change_qK_idx) -> Polyhedron

Project a single regime polyhedron by eliminating `change_qK_idx`.
"""
function _get_polyhedron_prism(
    bnc_sys::Bnc,
    vertex_idx::Int,
    change_qK_idx::Int,
)::Polyhedra.Polyhedron
    p = get_polyhedron(bnc_sys, vertex_idx)
    detecthlinearity!(p)
    removehredundancy!(p)
    isempty(p) && return p
    p = eliminate(p, change_qK_idx)
    removehredundancy!(p)
    return p
end

mutable struct RegimePath
    path::Vector{Int}
    condition::Polyhedra.Polyhedron

    function RegimePath(path::Vector{Int}, condition::Polyhedra.Polyhedron)
        new(path, condition)
    end
end

mutable struct SISOHelper
    bnc_system::Bnc
    change_qK_idx::Int
    connectome::Matrix{Bool}
    predecessors::Vector{Set{Int}}
    successors::Vector{Set{Int}}
    upstream::Vector{Set{Int}}
    downstream::Vector{Set{Int}}
    paths::Matrix{Union{Vector{RegimePath},Nothing}}
    path_found::BitMatrix
    sources::Vector{Int}
    sinks::Vector{Int}
    vertex_prisms::Vector{Union{Nothing,Polyhedra.Polyhedron}}
    vertex_prism_found::BitVector
    interface_prisms::Matrix{Union{Nothing,Polyhedra.Polyhedron}}
    interface_prism_found::BitMatrix
end

function _build_predecessor_successor_sets(
    connectome::AbstractMatrix{Bool},
)::Tuple{Vector{Set{Int}},Vector{Set{Int}}}
    n_vtx = size(connectome, 1)
    predecessors = Vector{Set{Int}}(undef, n_vtx)
    successors = Vector{Set{Int}}(undef, n_vtx)
    for i in 1:n_vtx
        predecessors[i] = Set{Int}(findall(@view connectome[:, i]))
        successors[i] = Set{Int}(findall(@view connectome[i, :]))
    end
    return predecessors, successors
end

function SISOHelper(
    bnc_sys::Bnc,
    change_qK;
    connectome=nothing,
 )::SISOHelper
    change_qK_idx = change_qK isa Integer ? Int(change_qK) : locate_sym_qK(bnc_sys, change_qK)
    connectome_bool = if isnothing(connectome)
        vtx_grh = get_vertices_graph!(bnc_sys; full=true)
        _oriented_connectome(vtx_grh, change_qK_idx)
    else
        Matrix{Bool}(connectome)
    end
    n_vtx = size(connectome_bool, 1)
    predecessors, successors = _build_predecessor_successor_sets(connectome_bool)
    paths = Matrix{Union{Vector{RegimePath},Nothing}}(undef, n_vtx, n_vtx)
    fill!(paths, nothing)
    vertex_prisms = Vector{Union{Nothing,Polyhedra.Polyhedron}}(undef, n_vtx)
    fill!(vertex_prisms, nothing)
    interface_prisms = Matrix{Union{Nothing,Polyhedra.Polyhedron}}(undef, n_vtx, n_vtx)
    fill!(interface_prisms, nothing)

    return SISOHelper(
        bnc_sys,
        change_qK_idx,
        connectome_bool,
        predecessors,
        successors,
        [Set{Int}() for _ in 1:n_vtx],
        [Set{Int}() for _ in 1:n_vtx],
        paths,
        falses(n_vtx, n_vtx),
        Int[],
        Int[],
        vertex_prisms,
        falses(n_vtx),
        interface_prisms,
        falses(n_vtx, n_vtx),
    )
end

@inline function _edge_exists(helper::SISOHelper, from::Int, to::Int)::Bool
    return helper.connectome[from, to]
end

function _get_vertex_prism!(
    helper::SISOHelper,
    vertex_idx::Int,
)::Polyhedra.Polyhedron
    if helper.vertex_prism_found[vertex_idx]
        prism = helper.vertex_prisms[vertex_idx]
        prism === nothing && error("Cached vertex prism for $vertex_idx is missing.")
        return prism
    end

    prism = _get_polyhedron_prism(
        helper.bnc_system,
        vertex_idx,
        helper.change_qK_idx,
    ) |> _clean_polyhedron!
    helper.vertex_prisms[vertex_idx] = prism
    helper.vertex_prism_found[vertex_idx] = true
    return prism
end

function _get_interface_prism!(
    helper::SISOHelper,
    vertex_idx_from::Int,
    vertex_idx_to::Int,
)::Polyhedra.Polyhedron
    if helper.interface_prism_found[vertex_idx_from, vertex_idx_to]
        prism = helper.interface_prisms[vertex_idx_from, vertex_idx_to]
        prism === nothing && error("Cached interface prism for ($(vertex_idx_from), $(vertex_idx_to)) is missing.")
        return prism
    end

    prism = _get_interface_prism(
        helper.bnc_system,
        vertex_idx_from,
        vertex_idx_to,
        helper.change_qK_idx,
    ) |> _clean_polyhedron!

    helper.interface_prisms[vertex_idx_from, vertex_idx_to] = prism
    helper.interface_prisms[vertex_idx_to, vertex_idx_from] = prism
    helper.interface_prism_found[vertex_idx_from, vertex_idx_to] = true
    helper.interface_prism_found[vertex_idx_to, vertex_idx_from] = true
    return prism
end

function _intersect_nonempty(polys::Polyhedra.Polyhedron...)::Union{Nothing,Polyhedra.Polyhedron}
    poly = intersect(polys...) |> _clean_polyhedron!
    return isempty(poly) ? nothing : poly
end

function _dfs_upstream!(
    helper::SISOHelper,
    visited::Vector{Bool},
    current::Int,
    up_stream_done::Vector{Bool},
    n_vtx::Int,
)::Nothing
    visited[current] = true
    if up_stream_done[current]
        visited[current] = false
        return
    end
    for neighbor in 1:n_vtx
        if helper.connectome[neighbor, current] && !visited[neighbor]
            _dfs_upstream!(helper, visited, neighbor, up_stream_done, n_vtx)
            push!(helper.upstream[current], neighbor)
            union!(helper.upstream[current], helper.upstream[neighbor])
        end
    end
    up_stream_done[current] = true
    visited[current] = false
    return
end

function _dfs_downstream!(
    helper::SISOHelper,
    visited::Vector{Bool},
    current::Int,
    down_stream_done::Vector{Bool},
    n_vtx::Int,
)::Nothing
    visited[current] = true
    if down_stream_done[current]
        visited[current] = false
        return
    end
    for neighbor in 1:n_vtx
        if helper.connectome[current, neighbor] && !visited[neighbor]
            _dfs_downstream!(helper, visited, neighbor, down_stream_done, n_vtx)
            push!(helper.downstream[current], neighbor)
            union!(helper.downstream[current], helper.downstream[neighbor])
        end
    end
    down_stream_done[current] = true
    visited[current] = false
    return
end

function _trace_reachability!(
    helper::SISOHelper;
    recompute_sources_sinks::Bool=true,
)::Nothing
    n_vtx = size(helper.connectome, 1)
    helper.upstream = [Set{Int}() for _ in 1:n_vtx]
    helper.downstream = [Set{Int}() for _ in 1:n_vtx]
    if recompute_sources_sinks
        helper.sources, helper.sinks = get_sources_sinks(helper.bnc_system, helper.connectome)
    end

    upstream_done = fill(false, n_vtx)
    for i in helper.sinks
        visited = fill(false, n_vtx)
        _dfs_upstream!(helper, visited, i, upstream_done, n_vtx)
    end

    downstream_done = fill(false, n_vtx)
    for i in helper.sources
        visited = fill(false, n_vtx)
        _dfs_downstream!(helper, visited, i, downstream_done, n_vtx)
    end

    return
end

@inline function _cache_pair_paths!(
    helper::SISOHelper,
    vertex_idx_from::Int,
    vertex_idx_to::Int,
    paths::Vector{RegimePath},
)::Bool
    helper.paths[vertex_idx_from, vertex_idx_to] = isempty(paths) ? nothing : paths
    helper.path_found[vertex_idx_from, vertex_idx_to] = true
    return !isempty(paths)
end

function _maybe_push_direct_path!(
    paths::Vector{RegimePath},
    helper::SISOHelper,
    vertex_idx_from::Int,
    vertex_idx_to::Int,
)::Nothing
    _edge_exists(helper, vertex_idx_from, vertex_idx_to) || return nothing
    condition = _get_interface_prism!(helper, vertex_idx_from, vertex_idx_to)
    isempty(condition) && return nothing
    push!(paths, RegimePath([vertex_idx_from, vertex_idx_to], condition))
    return nothing
end

function _find_pair_path_conditions!(
    helper::SISOHelper,
    vertex_idx_from::Int,
    vertex_idx_to::Int,
)::Bool
    if helper.path_found[vertex_idx_from, vertex_idx_to]
        return !isnothing(helper.paths[vertex_idx_from, vertex_idx_to])
    end

    if vertex_idx_from == vertex_idx_to
        condition = _get_vertex_prism!(helper, vertex_idx_from)
        helper.paths[vertex_idx_from, vertex_idx_to] = [RegimePath([vertex_idx_from], condition)]
        helper.path_found[vertex_idx_from, vertex_idx_to] = true
        return true
    end

    paths = RegimePath[]
    _maybe_push_direct_path!(paths, helper, vertex_idx_from, vertex_idx_to)

    pass_by = intersect(helper.downstream[vertex_idx_from], helper.upstream[vertex_idx_to])
    isempty(pass_by) && return _cache_pair_paths!(helper, vertex_idx_from, vertex_idx_to, paths)

    successors = intersect(pass_by, helper.successors[vertex_idx_from])
    predecessors = intersect(pass_by, helper.predecessors[vertex_idx_to])

    isempty(successors) && error("Invariant violated: `pass_by` is non-empty but `pass_by ∩ successors[from]` is empty for (from=$(vertex_idx_from), to=$(vertex_idx_to)).")
    isempty(predecessors) && error("Invariant violated: `pass_by` is non-empty but `pass_by ∩ predecessors[to]` is empty for (from=$(vertex_idx_from), to=$(vertex_idx_to)).")

    num_successor_calculated = sum(helper.path_found[successor, vertex_idx_to] for successor in successors)
    num_predecessor_calculated = sum(helper.path_found[vertex_idx_from, predecessor] for predecessor in predecessors)
    percentage_successor_calculated = num_successor_calculated / length(successors)
    percentage_predecessor_calculated = num_predecessor_calculated / length(predecessors)

    if num_predecessor_calculated == 0 && num_successor_calculated == 0
        for successor in successors
            left_condition = _get_interface_prism!(helper, vertex_idx_from, successor)
            isempty(left_condition) && continue
            for predecessor in predecessors
                right_condition = _get_interface_prism!(helper, predecessor, vertex_idx_to)
                isempty(right_condition) && continue
                if _find_pair_path_conditions!(helper, successor, predecessor)
                    for middle_path in helper.paths[successor, predecessor]
                        full_condition = _intersect_nonempty(left_condition, middle_path.condition, right_condition)
                        isnothing(full_condition) && continue
                        push!(paths, RegimePath([vertex_idx_from; middle_path.path; vertex_idx_to], full_condition))
                    end
                end
            end
        end
        return _cache_pair_paths!(helper, vertex_idx_from, vertex_idx_to, paths)
    end

    if percentage_successor_calculated > percentage_predecessor_calculated
        for successor in successors
            if _find_pair_path_conditions!(helper, successor, vertex_idx_to)
                left_condition = _get_interface_prism!(helper, vertex_idx_from, successor)
                isempty(left_condition) && continue
                for suffix_path in helper.paths[successor, vertex_idx_to]
                    full_condition = _intersect_nonempty(left_condition, suffix_path.condition)
                    isnothing(full_condition) && continue
                    push!(paths, RegimePath([vertex_idx_from; suffix_path.path], full_condition))
                end
            end
        end
        return _cache_pair_paths!(helper, vertex_idx_from, vertex_idx_to, paths)
    end

    for predecessor in predecessors
        if _find_pair_path_conditions!(helper, vertex_idx_from, predecessor)
            right_condition = _get_interface_prism!(helper, predecessor, vertex_idx_to)
            isempty(right_condition) && continue
            for prefix_path in helper.paths[vertex_idx_from, predecessor]
                full_condition = _intersect_nonempty(prefix_path.condition, right_condition)
                isnothing(full_condition) && continue
                push!(paths, RegimePath([prefix_path.path; vertex_idx_to], full_condition))
            end
        end
    end
    return _cache_pair_paths!(helper, vertex_idx_from, vertex_idx_to, paths)
end

"""
    _find_all_path_conditions!(helper) -> SISOHelper

Solve all source-to-sink pair conditions stored in a helper, with progress.
"""
function _find_all_path_conditions!(helper::SISOHelper)::SISOHelper
    _trace_reachability!(helper)
    pair_queries = [(source, sink) for source in helper.sources for sink in helper.sinks]
    isempty(pair_queries) && return helper

    if length(pair_queries) == 1
        source, sink = only(pair_queries)
        _find_pair_path_conditions!(helper, source, sink)
        return helper
    end

    @info "Start finding all possible path conditions across $(length(pair_queries)) source-sink pairs."
    @showprogress dt=0.1 desc="Finding path conditions" for (source, sink) in pair_queries
        _find_pair_path_conditions!(helper, source, sink)
    end
    return helper
end

mutable struct SISOPaths{T}
    bn::Bnc{T}
    qK_grh::SimpleDiGraph
    change_qK_idx::Int
    sources::Vector{Int}
    sinks::Vector{Int}
    paths_dict::Union{Nothing,Dict{Vector{Int},Int}}
    rgm_paths::Vector{Vector{Int}}
    condition_helper::Union{Nothing,SISOHelper}
    path_polys::Vector{Polyhedron}
    path_volume::Vector{Volume}
    path_volume_is_calc::BitVector
    path_polys_is_calc::BitVector

    function SISOPaths(model::Bnc{T}, qK_grh, change_qK_idx, sources, sinks, rgm_paths) where T
        rgm_paths_int = [Int.(path) for path in rgm_paths]
        path_polys = Vector{Polyhedron}(undef, length(rgm_paths_int))
        path_volume = Vector{Volume}(undef, length(rgm_paths_int))
        path_volume_is_calc = falses(length(rgm_paths_int))
        path_polys_is_calc = falses(length(rgm_paths_int))
        new{T}(
            model,
            qK_grh,
            Int(change_qK_idx),
            Int.(collect(sources)),
            Int.(collect(sinks)),
            nothing,
            rgm_paths_int,
            nothing,
            path_polys,
            path_volume,
            path_volume_is_calc,
            path_polys_is_calc,
        )
    end
end

function _build_paths_dict(rgm_paths::AbstractVector{<:AbstractVector{<:Integer}})
    paths_dict = Dict{Vector{Int},Int}()
    sizehint!(paths_dict, length(rgm_paths))
    for (i, p) in enumerate(rgm_paths)
        paths_dict[Int.(p)] = i
    end
    return paths_dict
end

function _ensure_paths_dict!(grh::SISOPaths)
    isnothing(grh.paths_dict) || return grh.paths_dict
    grh.paths_dict = _build_paths_dict(grh.rgm_paths)
    return grh.paths_dict
end

function _connectome_matrix(g::SimpleDiGraph)::Matrix{Bool}
    n = nv(g)
    connectome = falses(n, n)
    for edge in edges(g)
        connectome[src(edge), dst(edge)] = true
    end
    return connectome
end

function _ensure_condition_helper!(grh::SISOPaths)::SISOHelper
    if isnothing(grh.condition_helper)
        helper = SISOHelper(
            grh.bn,
            grh.change_qK_idx;
            connectome=_connectome_matrix(grh.qK_grh),
        )
        helper.sources = copy(grh.sources)
        helper.sinks = copy(grh.sinks)
        _trace_reachability!(helper; recompute_sources_sinks=false)
        grh.condition_helper = helper
    end
    return grh.condition_helper
end

function _ensure_path_polyhedra!(
    grh::SISOPaths,
    path_idxs::AbstractVector{<:Integer},
)::Nothing
    helper = _ensure_condition_helper!(grh)
    paths_by_pair = Dict{Tuple{Int,Int},Vector{Int}}()
    for idx in Int.(path_idxs)
        path = grh.rgm_paths[idx]
        push!(get!(paths_by_pair, (first(path), last(path)), Int[]), idx)
    end

    pair_entries = collect(paths_by_pair)
    isempty(pair_entries) && return nothing

    process_entry(entry) = begin
        (from, to), idxs = entry
        _find_pair_path_conditions!(helper, from, to)
        pair_paths = helper.paths[from, to]
        pair_paths === nothing && error("No feasible condition found for requested path pair ($(from), $(to)).")

        pair_map = Dict{Tuple{Vararg{Int}},Polyhedron}()
        sizehint!(pair_map, length(pair_paths))
        for regime_path in pair_paths
            pair_map[Tuple(regime_path.path)] = regime_path.condition
        end

        for idx in idxs
            key = Tuple(grh.rgm_paths[idx])
            poly = get(pair_map, key, nothing)
            poly === nothing && error("Requested path $(collect(key)) is missing from the shared path-condition backend.")
            grh.path_polys[idx] = poly
            grh.path_polys_is_calc[idx] = true
        end
        return nothing
    end

    if length(pair_entries) == 1
        process_entry(only(pair_entries))
        return nothing
    end

    @info "Start finding path conditions for $(length(path_idxs)) paths across $(length(pair_entries)) source-sink pairs."
    @showprogress dt=0.1 desc="Finding path conditions" for entry in pair_entries
        process_entry(entry)
    end
    return nothing
end

"""
    _calc_polyhedra_for_path(model, paths, change_qK_idx) -> Vector{Polyhedron}

Compute qK-space polyhedra for each regime path using the shared recursive
path-condition backend.
"""
function _calc_polyhedra_for_path(
    model::Bnc,
    paths::AbstractVector{<:AbstractVector{<:Integer}},
    change_qK_idx::Integer,
)::Vector{Polyhedron}
    siso = SISOPaths(model, Int(change_qK_idx); rgm_paths=[Int.(path) for path in paths])
    return get_polyhedra(siso)
end

function _calc_polyhedra_for_path(
    model::Bnc,
    path::AbstractVector{<:Integer},
    change_qK,
)::Polyhedron
    change_qK_idx = change_qK isa Integer ? Int(change_qK) : locate_sym_qK(model, change_qK)
    return _calc_polyhedra_for_path(model, [Int.(path)], change_qK_idx)[1]
end
"""
    Polyhedra.intersect(p::Polyhedron) -> Polyhedron

Identity overload for single-polyhedron intersections.
"""
Polyhedra.intersect(p::Polyhedron)= p # a fix for above function for if only one edge, no need to intersect



"""
    get_neighbor_graph_qK(grh::SISOPaths; kwargs...) -> SimpleDiGraph

Return the qK neighbor graph for a SISO path object.
"""
get_neighbor_graph_qK(grh::SISOPaths; kwargs...) = grh.qK_grh



"""
    get_SISO_graph(grh::SISOPaths) -> SimpleDiGraph

Return the SISO graph stored in a `SISOPaths` object.
"""

get_SISO_graph(grh::SISOPaths) = grh.qK_grh
"""
    get_SISO_graph(model::Bnc, change_qK) -> SimpleDiGraph

Return a SISO graph for a chosen qK coordinate.
"""
get_SISO_graph(model::Bnc, change_qK) = get_SISO_graph(get_regimes_graph!(model; full=true), change_qK)
"""
    get_SISO_graph(grh::VertexGraph, change_qK) -> SimpleDiGraph

Build a SISO graph from a vertex graph for a chosen qK coordinate.
"""
function get_SISO_graph(grh::VertexGraph, change_qK)::SimpleDiGraph
    bn = get_binding_network(grh)
    change_qK_idx = locate_sym_qK(bn, change_qK)
    return _oriented_digraph(grh, change_qK_idx)
end



#------------------------------------------------------------------------------
# Higher wrapper for regime graph paths
#------------------------------------------------------------------------------------------

"""
    SISOPaths(model::Bnc, change_qK; rgm_paths=nothing) -> SISOPaths

Construct a `SISOPaths` object for a chosen qK coordinate.
"""
function SISOPaths(model::Bnc{T}, change_qK; rgm_paths=nothing) where {T}
    change_qK_idx = locate_sym_qK(model, change_qK)

    if rgm_paths === nothing
        qK_grh = get_SISO_graph(model, change_qK)
        sources, sinks = get_sources_sinks(model, qK_grh)
        rgm_paths = _enumerate_paths(qK_grh; sources, sinks)
    else
        qK_grh = graph_from_paths(rgm_paths, n_regimes(model))
        sources, sinks = get_sources_sinks(qK_grh)
    end

    return SISOPaths(model, qK_grh, change_qK_idx, sources, sinks, rgm_paths)
end

"""
    get_path(grh::SISOPaths, pth_idx; return_idx=false) -> Vector

Return a path by index, optionally as vertex indices.
"""
function get_path(grh::SISOPaths, pth_idx::Integer; return_idx::Bool=false)
    rgm_idxs = grh.rgm_paths[pth_idx]
    if return_idx
        return rgm_idxs
    else
        bn = get_binding_network(grh)
        return get_perm.(Ref(bn), rgm_idxs)
    end
    return perms
end
"""
    get_path(grh::SISOPaths, pth::AbstractVector; return_idx=false) -> Vector

Normalize a path representation to indices or permutations.
"""
function get_path(grh::SISOPaths, pth::AbstractVector; return_idx::Bool=false)
    bn = get_binding_network(grh)
    return return_idx ? get_idx.(Ref(bn), pth) : get_perm.(Ref(bn), pth)
end

"""
    get_binding_network(grh::SISOPaths, args...) -> Bnc

Return the model backing a SISO path object.
"""
get_binding_network(grh::SISOPaths,args...)= grh.bn
"""
    get_C_C0_nullity_qK(grh::SISOPaths, pth_idx) -> (Matrix, Vector, Int)

Return constraints for a SISO path polyhedron.
"""
get_C_C0_nullity_qK(grh::SISOPaths, pth_idx) = get_polyhedron(grh, pth_idx) |> get_C_C0_nullity



"""
    get_idx(grh::SISOPaths, pth) -> Int

Return the index for a SISO path specification.
"""
get_idx(grh::SISOPaths, pth::AbstractVector) = let
    bn = get_binding_network(grh)
    idxs = get_idx.(Ref(bn), pth)
    _ensure_paths_dict!(grh)[idxs] 
end
"""
    get_idx(grh::SISOPaths, pth::Integer) -> Int

Return the provided path index.
"""
get_idx(grh::SISOPaths, pth::Integer) = pth





"""
    get_polyhedra(grh::SISOPaths, pth_idx=nothing) -> Vector{Polyhedron}

Return polyhedra for selected SISO paths.
"""
function get_polyhedra(grh::SISOPaths, pth_idx::Union{AbstractVector,Nothing} = nothing)::Vector{Polyhedron}
    pth_idx = let 
            if isnothing(pth_idx)
                1:length(grh.rgm_paths)
            else
                get_idx.(Ref(grh), pth_idx)
            end
        end
    
    pth_poly_to_calc = filter(x -> !grh.path_polys_is_calc[x], pth_idx)
    
    isempty(pth_poly_to_calc) || _ensure_path_polyhedra!(grh, pth_poly_to_calc)

    return grh.path_polys[pth_idx]
end
"""
    get_polyhedron(grh::SISOPaths, pth) -> Polyhedron

Return the polyhedron for a single SISO path.
"""
get_polyhedron(grh::SISOPaths, pth)= get_polyhedra(grh, [get_idx(grh, pth)])[1]



"""
    get_volumes(grh::SISOPaths, pth_idx=nothing; asymptotic=true, recalculate=false, kwargs...) -> Vector{Volume}

Compute volumes for SISO paths.
"""
function get_volumes(grh::SISOPaths, pth_idx::Union{AbstractVector,Nothing}=nothing; 
    rebase_K = false,
    rebase_mat = nothing,
    recalculate=false, kwargs...)

    pth_idx = let 
            if isnothing(pth_idx)
                1:length(grh.rgm_paths)
            else
                get_idx.(Ref(grh), pth_idx)
            end
        end
    
    idxes_to_calculate = recalculate ? pth_idx : filter(x -> !grh.path_volume_is_calc[x], pth_idx)
    
    if !isempty(idxes_to_calculate)

        rebase_mat = if  !isnothing(rebase_mat)
                    @assert !rebase_K "Cannot specify both rebase_K and providing rebase_mat"
                    rebase_mat
                elseif rebase_K
                    Bnc = get_binding_network(grh) 
                    Q = rebase_mat_lgK(Bnc.N)
                    blockdiag(spdiagm(fill(Rational(1), Bnc.d-1)), Q)
                else
                    nothing
                end

        polys = get_polyhedra(grh, idxes_to_calculate)

        rlts = calc_volume(polys; rebase_mat=rebase_mat, kwargs...)
        for (i, idx) in enumerate(idxes_to_calculate)
            grh.path_volume[idx] = rlts[i]
            grh.path_volume_is_calc[idx] = true
        end
    end
    return grh.path_volume[pth_idx]
end

"""
    get_volume(grh::SISOPaths, pth; kwargs...) -> Volume

Return the volume for a single SISO path.
"""
get_volume(grh::SISOPaths, pth; kwargs...) = get_volumes(grh, [get_idx(grh, pth)]; kwargs...)[1]



#-------------------------------------------------------------------------------------
# Regime shifting associated functions
#-------------------------------------------------------------------------------------

"""
    show_regime_path(grh::SISOPaths, pth) -> nothing

Print a formatted regime path with optional volume.
"""
function show_regime_path(grh::SISOPaths, pth)
    pth_idx = get_idx(grh, pth)
    pth = get_path(grh, pth_idx; return_idx=true)
    vol_is_calc = grh.path_volume_is_calc[pth_idx]
    volume = vol_is_calc ? grh.path_volume[pth_idx] : nothing
    print_path(pth; prefix="#",id = pth_idx,volume=volume)
    return nothing
end


"""
    get_expression_path(grh::SISOPaths, pth; observe_x=nothing) -> (Vector, Vector)

Return expression coefficients and interfaces along a SISO path.
"""
function get_expression_path(grh::SISOPaths, pth; observe_x=nothing)
    
    bn = get_binding_network(grh)
    rgm_pth = get_path(grh, pth; return_idx=true)
    # @show rgm_pth
    rgm_nlt = get_nullities(bn, rgm_pth)
    
    change_qK_idx = grh.change_qK_idx
    observe_x_idx = isnothing(observe_x) ? (1:bn.n) : locate_sym_x.(Ref(bn), observe_x)
    
    rgm_interface = get_interface.(Ref(bn),rgm_pth[1:end-1], rgm_pth[2:end])
    
    H_H0 = Vector{Any}(undef, length(rgm_pth))
    for i in eachindex(rgm_pth)
        rgm = rgm_pth[i]
        nlt = rgm_nlt[i]
        if nlt == 0 # for non-singular regime, we care about the expression, tells by the H[i，：]
            H,H0 = get_H_H0(bn, rgm)
            # @show H,H0, observe_x_idx
            H_H0[i] = (H[observe_x_idx, :], H0[observe_x_idx]) 
        elseif nlt == 1 # for singular regime, we care about the contiuity, tells by the H[i,j]
            H = get_H(bn,rgm)
            H_H0[i] = (H[observe_x_idx, change_qK_idx], nothing)
        else
            error("Nullity > 1 is not supported for expression path.") # should ne change if under constrain.
        end
    end
    return H_H0, rgm_interface
end



#-------------------------------------------------------------------------------------------
# 
"""
    _calc_RO_for_single_path(model, path, change_qK_idx, observe_x_idx) -> Vector

Compute the reaction-order profile along a single path.
"""
function _calc_RO_for_single_path(model, path::AbstractVector{<:Integer}, change_qK_idx, observe_x_idx)::Vector{<:Real}
    r_ord = Vector{Float64}(undef, length(path))
    for i in eachindex(path)
        if !is_singular(model, path[i])
            r_ord[i] = round(Float64(get_H(model, path[i])[observe_x_idx, change_qK_idx]); digits=3)
        else
            ord = get_H(model, path[i])[observe_x_idx, change_qK_idx]
            if abs(ord) < 1e-6
                r_ord[i] = NaN  # We use NaN to denote continuous singular, if reaction order not same before and after, means discontinuity
            else 
                r_ord[i] = Float64(ord) * Inf
            end     
        end
    end
    return r_ord
end
"""
    _dedup(ord_path) -> Vector

Deduplicate consecutive reaction-order values while preserving discontinuities.
"""
function _dedup(ord_path::AbstractVector{T})::Vector{T} where T<:Real
    isempty(ord_path) && return T[]
    out = T[ord_path[1]]
    pending_nan = false
    last_out = out[1]  
    @assert !isnan(last_out) "The first element cannot be NaN for deduplication."

    for x in @view ord_path[2:end]
        if isnan(x)
            pending_nan = true
            continue
        end
        if x != last_out
            if pending_nan
                push!(out, NaN)
                pending_nan = false
            end
            push!(out, x)
            last_out = x
        else
            pending_nan = false
        end
    end
    return out
end





"""
    get_RO_path(model::Bnc, rgm_idx_shift_pth; change_qK, observe_x,
        deduplicate=false, keep_singular=true, keep_nonasymptotic=true) -> Vector

Calculate the reaction-order profile for a single regime path.
"""
function get_RO_path(
    model::Bnc,rgm_idx_shift_pth::AbstractVector; 
    change_qK, observe_x,
    
    deduplicate::Bool=false,
    keep_singular::Bool=true,
    keep_nonasymptotic::Bool=true
    )::Vector{<:Real}

    
    # get reaction order along the path
    rgm_idx_shift_pth = get_idx.(Ref(model), rgm_idx_shift_pth)

    ord_path = let 
        change_qK_idx = locate_sym_qK(model, change_qK)
        observe_x_idx = locate_sym_x(model, observe_x)
        _calc_RO_for_single_path(model, rgm_idx_shift_pth, change_qK_idx, observe_x_idx)
    end
    

    # apply the regime filter
    mask = _get_mask(model, rgm_idx_shift_pth;
        singular=keep_singular ? nothing : false,
        asymptotic=keep_nonasymptotic ? nothing : true)
    
    ord_path = ord_path[mask]

    # remove redundency
    if deduplicate
        ord_path = _dedup(ord_path)
    end

    return ord_path
end




function _ensure_ro_regimes_materialized!(
    model::Bnc,
    rgm_idx_for_each_paths::AbstractVector{<:AbstractVector{<:Integer}},
)
    seen = Set{Int}()
    ordered_idxs = Int[]

    for path in rgm_idx_for_each_paths
        for idx in path
            idx = Int(idx)
            if !(idx in seen)
                push!(ordered_idxs, idx)
                push!(seen, idx)
            end
        end
    end

    for idx in ordered_idxs
        get_regime(model, idx; inv_info=true)
    end

    return nothing
end



"""
    get_RO_paths(model::Bnc, rgm_paths, args...; kwargs...) -> Vector{Vector}

Calculate reaction-order profiles for multiple regime paths.
"""
function get_RO_paths(model::Bnc, rgm_paths::AbstractVector{<:AbstractVector}, args...; kwargs...)::Vector{Vector{<:Real}}
    
    rgm_idx_for_each_paths = rgm_paths .|> x -> get_idx.(Ref(model), x)
    # Different paths may share the same regime. Pre-materialize once so the
    # threaded loop below only reads cached affine/qK data.
    _ensure_ro_regimes_materialized!(model, rgm_idx_for_each_paths)

    ord_for_each_paths = Vector{Vector{<:Real}}(undef, length(rgm_idx_for_each_paths))
    Threads.@threads for i in eachindex(rgm_idx_for_each_paths)
        ord_for_each_paths[i] = get_RO_path(model, rgm_idx_for_each_paths[i], args...; kwargs...)
    end
    return ord_for_each_paths
end
"""
    get_RO_paths(model::SISOPaths, pth_idx=nothing; observe_x, kwargs...) -> Vector{Vector}

Calculate reaction-order profiles for paths in a `SISOPaths` object.
"""
function get_RO_paths(model::SISOPaths, pth_idx::Union{Nothing, AbstractVector}=nothing ; observe_x, kwargs...)
    rgm_paths = isnothing(pth_idx) ? model.rgm_paths : get_path.(Ref(model), pth_idx; return_idx=true)
    observe_x_idx = locate_sym_x(model.bn, observe_x)
    return get_RO_paths(model.bn, rgm_paths; 
        change_qK=model.change_qK_idx, observe_x=observe_x_idx, kwargs...)
end
"""
    get_RO_path(model::SISOPaths, pth_idx, args...; kwargs...) -> Vector

Single-path wrapper for `get_RO_paths`.
"""
get_RO_path(model::SISOPaths, pth_idx, args...; kwargs...) = get_RO_paths(model, [get_idx(model,pth_idx)], args... ; kwargs...)[1]



"""
    summary(grh::SISOPaths; show_volume=true, prefix="#", kwargs...) -> nothing

Print the paths stored in `SISOPaths`, optionally with volumes.
"""
function summary(grh::SISOPaths; show_volume::Bool=true, prefix::AbstractString="#", kwargs...)
    paths = grh.rgm_paths
    if show_volume
        vols = get_volumes(grh; kwargs...)
        print_paths(paths; prefix=prefix, volumes = vols, ids = 1:length(paths))
    else
        print_paths(paths; prefix=prefix, ids = 1:length(paths))
    end
    return nothing
end



"""
    summary_RO_path(grh::SISOPaths; observe_x, show_volume=true, deduplicate=true,
        keep_singular=true, keep_nonasymptotic=true, kwargs...) -> nothing

Summarize reaction-order paths grouped by profile.
"""
function summary_RO_path(grh::SISOPaths;observe_x, show_volume::Bool=true,

    deduplicate::Bool=true,keep_singular::Bool=true,keep_nonasymptotic::Bool=true,kwargs...)

    ord_pth = get_RO_paths(grh; observe_x=observe_x, 
        deduplicate=deduplicate,
        keep_singular=keep_singular,
        keep_nonasymptotic=keep_nonasymptotic)

    volumes = if show_volume
        get_volumes(grh; kwargs...)
    else
        fill(nothing, length(grh.rgm_paths))
    end



    rsts = group_sum(ord_pth, volumes)
    # for (id, pth, volume) in rsts
    #      print_path(pth; prefix="",id=id, volume=volume)
    # end

    # print 
    ids = getindex.(rsts, 1)
    ords = getindex.(rsts, 2)
    vols = getindex.(rsts, 3)
    print_paths(ords; prefix="", ids=ids, volumes=vols)
    return nothing
end
