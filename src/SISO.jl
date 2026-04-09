"""
Axis-Aligned SISO Path-Condition Backend
========================================

This file contains the SISO-specific graph, path, polyhedron, and reaction-order
helpers used by `SISOPaths`.

High-level idea
----------------
1. Build a directed regime graph by orienting each qK interface along one chosen
   `change_qK` coordinate.
2. Enumerate all source-to-sink regime paths in that DAG.
3. Reuse a memoized pair solver (`SISOHelper`) to compute path conditions in
   reduced qK-space.
4. Expose higher-level APIs for path polyhedra, volumes, expression tracing, and
   reaction-order summaries.
"""

export SISOPaths, get_polyhedra, get_polyhedron, get_SISO_graph
export get_sources, get_sinks, get_sources_sinks
export get_regimes_graph!
export get_path, get_edge, get_intersect
export get_neighbor_graph_x, get_neighbor_graph_qK, get_neighbor_graph
export get_RO_path, group_sum, get_RO_paths, summary_RO_path
export get_volume


# ============================================================================
# Regime Graph Access
# ============================================================================

"""
    get_binding_network(grh::VertexGraph, args...) -> Bnc

Return the model backing a vertex graph.
"""
get_binding_network(grh::VertexGraph, args...) = grh.bn

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

"""
    get_regimes_graph!(bnc::Bnc; full=false) -> VertexGraph

Ensure the vertex graph is built; when `full=true`, also compute qK change
directions.
"""
function get_regimes_graph!(bnc::Bnc; full::Bool=false)::VertexGraph
    if full
        vtx_graph = get_regimes_graph!(bnc; full=false)
        _ensure_full_regimes_graph!(vtx_graph)
    elseif isnothing(bnc.vertices_graph)
        find_all_regimes!(bnc)
    end
    return bnc.vertices_graph
end

"""
    get_edge(grh::VertexGraph, from, to; kwargs...) -> Union{Nothing, VertexEdge}

Return the edge between two vertices.
"""
function get_edge(grh::VertexGraph, from, to; kwargs...)::Union{Nothing,VertexEdge}
    from_idx = get_idx(get_binding_network(grh), from)
    to_idx = get_idx(get_binding_network(grh), to)
    pos = get(grh.edge_pos[from_idx], to_idx, nothing)
    pos === nothing && return nothing
    return grh.neighbors[from_idx][pos]
end

"""
    get_edge(bnc, from, to; kwargs...) -> Union{Nothing, VertexEdge}

Convenience wrapper to fetch an edge from a model.
"""
function get_edge(bnc, from, to; kwargs...)
    vtx_grh = get_regimes_graph!(bnc; full=false)
    return get_edge(vtx_grh, from, to; kwargs...)
end

"""
    get_neighbor_graph_x(grh::VertexGraph) -> SimpleGraph

Return the x-space neighbor graph for a vertex graph.
"""
get_neighbor_graph_x(grh::VertexGraph) = grh.x_grh

"""
    get_neighbor_graph_x(bnc::Bnc) -> SimpleGraph

Return the x-space neighbor graph for a model.
"""
get_neighbor_graph_x(bnc::Bnc) = get_neighbor_graph_x(get_regimes_graph!(bnc; full=false))

"""
    get_neighbor_graph_qK(grh::VertexGraph; both_side=false) -> SimpleDiGraph

Return the raw qK-space neighbor graph for a vertex graph.
"""
function get_neighbor_graph_qK(grh::VertexGraph; both_side::Bool=false)::SimpleDiGraph
    _ensure_full_regimes_graph!(grh)

    bn = get_binding_network(grh)
    g = SimpleDiGraph(length(grh.neighbors))
    for (i, edges) in enumerate(grh.neighbors)
        get_nullity(bn, i) > 1 && continue
        for e in edges
            if !_edge_has_qK_interface(e) || (!both_side && e.to < i)
                continue
            end
            add_edge!(g, i, e.to)
        end
    end
    return g
end

"""
    get_neighbor_graph_qK(bnc::Bnc; kwargs...) -> SimpleDiGraph

Return the qK neighbor graph for a model.
"""
get_neighbor_graph_qK(bnc::Bnc; kwargs...) = get_neighbor_graph_qK(get_regimes_graph!(bnc; full=true); kwargs...)

"""
    get_neighbor_graph(args...; kwargs...) -> SimpleDiGraph

Alias for `get_neighbor_graph_qK`.
"""
get_neighbor_graph(args...; kwargs...) = get_neighbor_graph_qK(args...; kwargs...)


# ============================================================================
# Graph Utilities
# ============================================================================

"""
    get_sources(g::AbstractGraph) -> Set{Int}

Return source vertices with zero indegree.
"""
get_sources(g::AbstractGraph) = Set(v for v in vertices(g) if indegree(g, v) == 0)

"""
    get_sinks(g::AbstractGraph) -> Set{Int}

Return sink vertices with zero outdegree.
"""
get_sinks(g::AbstractGraph) = Set(v for v in vertices(g) if outdegree(g, v) == 0)

"""
    get_sources_sinks(g::AbstractGraph) -> (Set{Int}, Set{Int})

Return sources and sinks for a graph.
"""
get_sources_sinks(g::AbstractGraph) = (get_sources(g), get_sinks(g))

function _filter_singular_isolated_vertices!(
    sources_all::Set{Int},
    sinks_all::Set{Int},
    is_singular::F,
) where {F}
    common_vs = intersect(sources_all, sinks_all)
    filter!(is_singular, common_vs)
    sources = sort!(collect(setdiff(sources_all, common_vs)))
    sinks = sort!(collect(setdiff(sinks_all, common_vs)))
    return sources, sinks
end

"""
    get_sources_sinks(model::Bnc, g::AbstractGraph) -> (Vector{Int}, Vector{Int})

Return sources and sinks while excluding singular isolated regimes.
"""
function get_sources_sinks(model::Bnc, g::AbstractGraph)
    return _filter_singular_isolated_vertices!(
        get_sources(g),
        get_sinks(g),
        v -> get_nullity(model, v) > 0,
    )
end

"""
    get_sources_sinks(model::Bnc, connectome::AbstractMatrix{Bool}) -> (Vector{Int}, Vector{Int})

Return sources and sinks for a boolean adjacency matrix while excluding
singular isolated regimes.
"""
function get_sources_sinks(model::Bnc, connectome::AbstractMatrix{Bool})
    n = size(connectome, 1)
    size(connectome, 2) == n || error("connectome must be square, got size $(size(connectome)).")

    sources_all = Set(v for v in 1:n if !any(@view connectome[:, v]))
    sinks_all = Set(v for v in 1:n if !any(@view connectome[v, :]))
    regimes = _bind_regimes_data(model)
    return _filter_singular_isolated_vertices!(
        sources_all,
        sinks_all,
        v -> regimes[v].nullity > 0,
    )
end

"""
    _reachable_from_sources(g, sources) -> Vector{Bool}

Return a boolean mask of vertices reachable from `sources`.
"""
function _reachable_from_sources(g::AbstractGraph, sources::AbstractVector{Int})
    seen = falses(nv(g))
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
    _can_reach_sinks(g, sinks) -> Vector{Bool}

Return a boolean mask of vertices that can reach `sinks`.
"""
function _can_reach_sinks(g::AbstractGraph, sinks::AbstractVector{Int})
    seen = falses(nv(g))
    stack = Int[]
    for t in sinks
        if !seen[t]
            seen[t] = true
            push!(stack, t)
            while !isempty(stack)
                v = pop!(stack)
                for nb in inneighbors(g, v)
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

    active = _reachable_from_sources(g, sources) .& _can_reach_sinks(g, sinks)
    is_sink = falses(nv(g))
    for t in sinks
        is_sink[t] = true
    end

    topo = topological_sort_by_dfs(g)
    memo = Vector{Union{Nothing,Vector{Vector{Int}}}}(undef, nv(g))
    fill!(memo, nothing)

    @info "Start enumerating paths from sources to sinks. This may take a while if there are many paths."
    @info "Total vertices to process in topological order: $(length(topo))"
    @showprogress for v in Iterators.reverse(topo)
        active[v] || continue
        if is_sink[v]
            memo[v] = [[v]]
            continue
        end

        acc = Vector{Vector{Int}}()
        for nb in outneighbors(g, v)
            active[nb] || continue
            paths_nb = memo[nb]
            paths_nb === nothing && continue
            for p in paths_nb
                np = Vector{Int}(undef, length(p) + 1)
                np[1] = v
                copyto!(np, 2, p, 1, length(p))
                push!(acc, np)
            end
        end
        memo[v] = isempty(acc) ? nothing : acc
    end

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


# ============================================================================
# SISO-Oriented Graph Construction
# ============================================================================

@inline _direction_score(dir::SparseVector{Float64,Int}, change_qK_idx::Integer) = get(dir, Int(change_qK_idx), 0.0)

function _collect_oriented_edge_pairs(
    grh::VertexGraph,
    change_qK_idx::Integer;
    tol::Float64=1e-6,
)
    _ensure_full_regimes_graph!(grh)
    regimes = _bind_regimes_data(get_binding_network(grh))
    thread_edges = [Tuple{Int,Int}[] for _ in 1:Threads.maxthreadid()]
    idx = Int(change_qK_idx)

    Threads.@threads for i in eachindex(grh.neighbors)
        regimes[i].nullity > 1 && continue
        local_edges = thread_edges[Threads.threadid()]
        for e in grh.neighbors[i]
            (!_edge_has_qK_interface(e) || e.to < i) && continue
            iface = _edge_qK_interface(grh, e)
            iface === nothing && continue
            score = _direction_score(iface[1], idx)
            if score > tol
                push!(local_edges, (i, e.to))
            elseif score < -tol
                push!(local_edges, (e.to, i))
            end
        end
    end

    return reduce(vcat, thread_edges; init=Tuple{Int,Int}[])
end

function _edge_pairs_to_connectome(
    n_vertices::Integer,
    edge_pairs::AbstractVector{<:Tuple{Int,Int}},
)::Matrix{Bool}
    connectome = falses(Int(n_vertices), Int(n_vertices))
    for (from, to) in edge_pairs
        connectome[from, to] = true
    end
    return connectome
end

function _edge_pairs_to_digraph(
    n_vertices::Integer,
    edge_pairs::AbstractVector{<:Tuple{Int,Int}},
)::SimpleDiGraph
    g = SimpleDiGraph(Int(n_vertices))
    for (from, to) in edge_pairs
        add_edge!(g, from, to)
    end
    return g
end

function _oriented_connectome(
    grh::VertexGraph,
    change_qK_idx::Integer;
    tol::Float64=1e-6,
)::Matrix{Bool}
    edge_pairs = _collect_oriented_edge_pairs(grh, change_qK_idx; tol=tol)
    return _edge_pairs_to_connectome(length(grh.neighbors), edge_pairs)
end

function _oriented_digraph(
    grh::VertexGraph,
    change_qK_idx::Integer;
    tol::Float64=1e-6,
)::SimpleDiGraph
    edge_pairs = _collect_oriented_edge_pairs(grh, change_qK_idx; tol=tol)
    return _edge_pairs_to_digraph(length(grh.neighbors), edge_pairs)
end

"""
    get_SISO_graph(grh::VertexGraph, change_qK) -> SimpleDiGraph

Build a SISO graph from a vertex graph for a chosen qK coordinate.
"""
function get_SISO_graph(grh::VertexGraph, change_qK)::SimpleDiGraph
    change_qK_idx = locate_sym_qK(get_binding_network(grh), change_qK)
    return _oriented_digraph(grh, change_qK_idx)
end

"""
    get_SISO_graph(model::Bnc, change_qK) -> SimpleDiGraph

Return a SISO graph for a chosen qK coordinate.
"""
get_SISO_graph(model::Bnc, change_qK) = get_SISO_graph(get_regimes_graph!(model; full=true), change_qK)


# ============================================================================
# Polyhedron Utilities
# ============================================================================

_clean_polyhedron!(p::Polyhedron) = (detecthlinearity!(p); removehredundancy!(p); p)

"""
    Polyhedra.intersect(p::Polyhedron) -> Polyhedron

Identity overload for single-polyhedron intersections.
"""
Polyhedra.intersect(p::Polyhedron) = p

function _project_polyhedron(poly::Polyhedron, change_qK_idx::Int)::Polyhedron
    detecthlinearity!(poly)
    removehredundancy!(poly)
    isempty(poly) && return poly
    poly = eliminate(poly, change_qK_idx)
    removehredundancy!(poly)
    return poly
end

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
    poly = intersect(get_polyhedron(bnc_sys, vertex_idx_from), get_polyhedron(bnc_sys, vertex_idx_to))
    return _project_polyhedron(poly, change_qK_idx)
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
    return _project_polyhedron(get_polyhedron(bnc_sys, vertex_idx), change_qK_idx)
end

function _intersect_nonempty(polys::Polyhedra.Polyhedron...)::Union{Nothing,Polyhedra.Polyhedron}
    poly = intersect(polys...) |> _clean_polyhedron!
    return isempty(poly) ? nothing : poly
end


# ============================================================================
# SISO Helper
# ============================================================================

mutable struct RegimePath
    path::Vector{Int}
    condition::Polyhedra.Polyhedron
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

function _init_path_condition_storage(n_vtx::Int)
    paths = Matrix{Union{Vector{RegimePath},Nothing}}(undef, n_vtx, n_vtx)
    fill!(paths, nothing)

    vertex_prisms = Vector{Union{Nothing,Polyhedra.Polyhedron}}(undef, n_vtx)
    fill!(vertex_prisms, nothing)

    interface_prisms = Matrix{Union{Nothing,Polyhedra.Polyhedron}}(undef, n_vtx, n_vtx)
    fill!(interface_prisms, nothing)

    return paths, vertex_prisms, interface_prisms
end

function SISOHelper(
    bnc_sys::Bnc,
    change_qK;
    connectome=nothing,
)::SISOHelper
    change_qK_idx = change_qK isa Integer ? Int(change_qK) : locate_sym_qK(bnc_sys, change_qK)
    connectome_bool = isnothing(connectome) ?
        _oriented_connectome(get_regimes_graph!(bnc_sys; full=true), change_qK_idx) :
        Matrix{Bool}(connectome)

    n_vtx = size(connectome_bool, 1)
    predecessors, successors = _build_predecessor_successor_sets(connectome_bool)
    paths, vertex_prisms, interface_prisms = _init_path_condition_storage(n_vtx)

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

@inline _edge_exists(helper::SISOHelper, from::Int, to::Int) = helper.connectome[from, to]

function _get_vertex_prism!(
    helper::SISOHelper,
    vertex_idx::Int,
)::Polyhedra.Polyhedron
    if helper.vertex_prism_found[vertex_idx]
        prism = helper.vertex_prisms[vertex_idx]
        prism === nothing && error("Cached vertex prism for $vertex_idx is missing.")
        return prism
    end

    prism = _get_polyhedron_prism(helper.bnc_system, vertex_idx, helper.change_qK_idx) |> _clean_polyhedron!
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

function _dfs_upstream!(
    helper::SISOHelper,
    visited::Vector{Bool},
    current::Int,
    upstream_done::Vector{Bool},
    n_vtx::Int,
)::Nothing
    visited[current] = true
    if upstream_done[current]
        visited[current] = false
        return
    end

    for neighbor in 1:n_vtx
        if helper.connectome[neighbor, current] && !visited[neighbor]
            _dfs_upstream!(helper, visited, neighbor, upstream_done, n_vtx)
            push!(helper.upstream[current], neighbor)
            union!(helper.upstream[current], helper.upstream[neighbor])
        end
    end

    upstream_done[current] = true
    visited[current] = false
    return
end

function _dfs_downstream!(
    helper::SISOHelper,
    visited::Vector{Bool},
    current::Int,
    downstream_done::Vector{Bool},
    n_vtx::Int,
)::Nothing
    visited[current] = true
    if downstream_done[current]
        visited[current] = false
        return
    end

    for neighbor in 1:n_vtx
        if helper.connectome[current, neighbor] && !visited[neighbor]
            _dfs_downstream!(helper, visited, neighbor, downstream_done, n_vtx)
            push!(helper.downstream[current], neighbor)
            union!(helper.downstream[current], helper.downstream[neighbor])
        end
    end

    downstream_done[current] = true
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
    for sink in helper.sinks
        _dfs_upstream!(helper, fill(false, n_vtx), sink, upstream_done, n_vtx)
    end

    downstream_done = fill(false, n_vtx)
    for source in helper.sources
        _dfs_downstream!(helper, fill(false, n_vtx), source, downstream_done, n_vtx)
    end
    return nothing
end

@inline function _cache_pair_paths!(
    helper::SISOHelper,
    from::Int,
    to::Int,
    paths::Vector{RegimePath},
)::Bool
    helper.paths[from, to] = isempty(paths) ? nothing : paths
    helper.path_found[from, to] = true
    return !isempty(paths)
end

function _maybe_push_direct_path!(
    paths::Vector{RegimePath},
    helper::SISOHelper,
    from::Int,
    to::Int,
)::Nothing
    _edge_exists(helper, from, to) || return nothing
    condition = _get_interface_prism!(helper, from, to)
    isempty(condition) && return nothing
    push!(paths, RegimePath([from, to], condition))
    return nothing
end

function _find_pair_path_conditions!(
    helper::SISOHelper,
    from::Int,
    to::Int,
)::Bool
    if helper.path_found[from, to]
        return !isnothing(helper.paths[from, to])
    end

    if from == to
        condition = _get_vertex_prism!(helper, from)
        helper.paths[from, to] = [RegimePath([from], condition)]
        helper.path_found[from, to] = true
        return true
    end

    paths = RegimePath[]
    _maybe_push_direct_path!(paths, helper, from, to)

    pass_by = intersect(helper.downstream[from], helper.upstream[to])
    isempty(pass_by) && return _cache_pair_paths!(helper, from, to, paths)

    successors = intersect(pass_by, helper.successors[from])
    predecessors = intersect(pass_by, helper.predecessors[to])
    isempty(successors) && error("Invariant violated: `pass_by` is non-empty but `pass_by ∩ successors[from]` is empty for (from=$(from), to=$(to)).")
    isempty(predecessors) && error("Invariant violated: `pass_by` is non-empty but `pass_by ∩ predecessors[to]` is empty for (from=$(from), to=$(to)).")

    n_solved_successors = sum(helper.path_found[successor, to] for successor in successors)
    n_solved_predecessors = sum(helper.path_found[from, predecessor] for predecessor in predecessors)
    solved_successor_ratio = n_solved_successors / length(successors)
    solved_predecessor_ratio = n_solved_predecessors / length(predecessors)

    if n_solved_successors == 0 && n_solved_predecessors == 0
        for successor in successors
            left_condition = _get_interface_prism!(helper, from, successor)
            isempty(left_condition) && continue
            for predecessor in predecessors
                right_condition = _get_interface_prism!(helper, predecessor, to)
                isempty(right_condition) && continue
                if _find_pair_path_conditions!(helper, successor, predecessor)
                    for middle_path in helper.paths[successor, predecessor]
                        full_condition = _intersect_nonempty(left_condition, middle_path.condition, right_condition)
                        isnothing(full_condition) && continue
                        push!(paths, RegimePath([from; middle_path.path; to], full_condition))
                    end
                end
            end
        end
        return _cache_pair_paths!(helper, from, to, paths)
    end

    if solved_successor_ratio > solved_predecessor_ratio
        for successor in successors
            if _find_pair_path_conditions!(helper, successor, to)
                left_condition = _get_interface_prism!(helper, from, successor)
                isempty(left_condition) && continue
                for suffix_path in helper.paths[successor, to]
                    full_condition = _intersect_nonempty(left_condition, suffix_path.condition)
                    isnothing(full_condition) && continue
                    push!(paths, RegimePath([from; suffix_path.path], full_condition))
                end
            end
        end
        return _cache_pair_paths!(helper, from, to, paths)
    end

    for predecessor in predecessors
        if _find_pair_path_conditions!(helper, from, predecessor)
            right_condition = _get_interface_prism!(helper, predecessor, to)
            isempty(right_condition) && continue
            for prefix_path in helper.paths[from, predecessor]
                full_condition = _intersect_nonempty(prefix_path.condition, right_condition)
                isnothing(full_condition) && continue
                push!(paths, RegimePath([prefix_path.path; to], full_condition))
            end
        end
    end

    return _cache_pair_paths!(helper, from, to, paths)
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


# ============================================================================
# SISOPaths
# ============================================================================

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

    function SISOPaths(model::Bnc{T}, qK_grh, change_qK_idx, sources, sinks, rgm_paths) where {T}
        rgm_paths_int = [Int.(path) for path in rgm_paths]
        n_paths = length(rgm_paths_int)
        return new{T}(
            model,
            qK_grh,
            Int(change_qK_idx),
            Int.(collect(sources)),
            Int.(collect(sinks)),
            nothing,
            rgm_paths_int,
            nothing,
            Vector{Polyhedron}(undef, n_paths),
            Vector{Volume}(undef, n_paths),
            falses(n_paths),
            falses(n_paths),
        )
    end
end

"""
    get_binding_network(grh::SISOPaths, args...) -> Bnc

Return the model backing a SISO path object.
"""
get_binding_network(grh::SISOPaths, args...) = grh.bn

function _build_paths_dict(rgm_paths::AbstractVector{<:AbstractVector{<:Integer}})
    paths_dict = Dict{Vector{Int},Int}()
    sizehint!(paths_dict, length(rgm_paths))
    for (i, path) in enumerate(rgm_paths)
        paths_dict[Int.(path)] = i
    end
    return paths_dict
end

function _ensure_paths_dict!(grh::SISOPaths)
    isnothing(grh.paths_dict) || return grh.paths_dict
    grh.paths_dict = _build_paths_dict(grh.rgm_paths)
    return grh.paths_dict
end

function _connectome_matrix(g::SimpleDiGraph)::Matrix{Bool}
    connectome = falses(nv(g), nv(g))
    for edge in edges(g)
        connectome[src(edge), dst(edge)] = true
    end
    return connectome
end

function _normalize_path_indices(
    grh::SISOPaths,
    pth_idx::Union{Nothing,AbstractVector},
)::Vector{Int}
    return isnothing(pth_idx) ? collect(1:length(grh.rgm_paths)) : Int.(get_idx.(Ref(grh), pth_idx))
end

function _group_path_indices_by_endpoints(
    grh::SISOPaths,
    path_idxs::AbstractVector{<:Integer},
)
    groups = Dict{Tuple{Int,Int},Vector{Int}}()
    for idx in Int.(path_idxs)
        path = grh.rgm_paths[idx]
        push!(get!(groups, (first(path), last(path)), Int[]), idx)
    end
    return collect(groups)
end

function _build_pair_condition_map(pair_paths::AbstractVector{RegimePath})
    pair_map = Dict{Tuple{Vararg{Int}},Polyhedron}()
    sizehint!(pair_map, length(pair_paths))
    for regime_path in pair_paths
        pair_map[Tuple(regime_path.path)] = regime_path.condition
    end
    return pair_map
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

function _store_pair_polyhedra!(
    grh::SISOPaths,
    helper::SISOHelper,
    from::Int,
    to::Int,
    idxs::AbstractVector{<:Integer},
)::Nothing
    _find_pair_path_conditions!(helper, from, to)
    pair_paths = helper.paths[from, to]
    pair_paths === nothing && error("No feasible condition found for requested path pair ($(from), $(to)).")

    pair_map = _build_pair_condition_map(pair_paths)
    for idx in idxs
        key = Tuple(grh.rgm_paths[idx])
        poly = get(pair_map, key, nothing)
        poly === nothing && error("Requested path $(collect(key)) is missing from the shared path-condition backend.")
        grh.path_polys[idx] = poly
        grh.path_polys_is_calc[idx] = true
    end
    return nothing
end

function _ensure_path_polyhedra!(
    grh::SISOPaths,
    path_idxs::AbstractVector{<:Integer},
)::Nothing
    helper = _ensure_condition_helper!(grh)
    pair_entries = _group_path_indices_by_endpoints(grh, path_idxs)
    isempty(pair_entries) && return nothing

    if length(pair_entries) == 1
        ((from, to), idxs) = only(pair_entries)
        _store_pair_polyhedra!(grh, helper, from, to, idxs)
        return nothing
    end

    @info "Start finding path conditions for $(length(path_idxs)) paths across $(length(pair_entries)) source-sink pairs."
    @showprogress dt=0.1 desc="Finding path conditions" for ((from, to), idxs) in pair_entries
        _store_pair_polyhedra!(grh, helper, from, to, idxs)
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
    SISOPaths(model::Bnc, change_qK; rgm_paths=nothing) -> SISOPaths

Construct a `SISOPaths` object for a chosen qK coordinate.
"""
function SISOPaths(model::Bnc{T}, change_qK; rgm_paths=nothing) where {T}
    change_qK_idx = locate_sym_qK(model, change_qK)

    if isnothing(rgm_paths)
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
    get_path(grh::SISOPaths, pth_idx; return_idx=false) -> Vector

Return a path by index, optionally as vertex indices.
"""
function get_path(grh::SISOPaths, pth_idx::Integer; return_idx::Bool=false)
    rgm_idxs = grh.rgm_paths[pth_idx]
    return return_idx ? rgm_idxs : get_perm.(Ref(get_binding_network(grh)), rgm_idxs)
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
    get_C_C0_nullity_qK(grh::SISOPaths, pth_idx) -> (Matrix, Vector, Int)

Return constraints for a SISO path polyhedron.
"""
get_C_C0_nullity_qK(grh::SISOPaths, pth_idx) = get_C_C0_nullity(get_polyhedron(grh, pth_idx))

"""
    get_idx(grh::SISOPaths, pth) -> Int

Return the index for a SISO path specification.
"""
function get_idx(grh::SISOPaths, pth::AbstractVector)
    idxs = get_idx.(Ref(get_binding_network(grh)), pth)
    return _ensure_paths_dict!(grh)[idxs]
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
function get_polyhedra(grh::SISOPaths, pth_idx::Union{AbstractVector,Nothing}=nothing)::Vector{Polyhedron}
    selected_idxs = _normalize_path_indices(grh, pth_idx)
    pending = filter(idx -> !grh.path_polys_is_calc[idx], selected_idxs)
    isempty(pending) || _ensure_path_polyhedra!(grh, pending)
    return grh.path_polys[selected_idxs]
end

"""
    get_polyhedron(grh::SISOPaths, pth) -> Polyhedron

Return the polyhedron for a single SISO path.
"""
get_polyhedron(grh::SISOPaths, pth) = get_polyhedra(grh, [get_idx(grh, pth)])[1]

function _prepare_rebase_matrix(grh::SISOPaths; rebase_K::Bool=false, rebase_mat=nothing)
    if !isnothing(rebase_mat)
        @assert !rebase_K "Cannot specify both rebase_K and providing rebase_mat"
        return rebase_mat
    end
    if rebase_K
        bn = get_binding_network(grh)
        Q = rebase_mat_lgK(bn.N)
        return blockdiag(spdiagm(fill(Rational(1), bn.d - 1)), Q)
    end
    return nothing
end

"""
    get_volumes(grh::SISOPaths, pth_idx=nothing; kwargs...) -> Vector{Volume}

Compute volumes for SISO paths.
"""
function get_volumes(
    grh::SISOPaths,
    pth_idx::Union{AbstractVector,Nothing}=nothing;
    rebase_K=false,
    rebase_mat=nothing,
    recalculate=false,
    kwargs...,
)
    selected_idxs = _normalize_path_indices(grh, pth_idx)
    pending = recalculate ? selected_idxs : filter(idx -> !grh.path_volume_is_calc[idx], selected_idxs)

    if !isempty(pending)
        polys = get_polyhedra(grh, pending)
        rebasing = _prepare_rebase_matrix(grh; rebase_K=rebase_K, rebase_mat=rebase_mat)
        volumes = calc_volume(polys; rebase_mat=rebasing, kwargs...)
        for (i, idx) in enumerate(pending)
            grh.path_volume[idx] = volumes[i]
            grh.path_volume_is_calc[idx] = true
        end
    end

    return grh.path_volume[selected_idxs]
end

"""
    get_volume(grh::SISOPaths, pth; kwargs...) -> Volume

Return the volume for a single SISO path.
"""
get_volume(grh::SISOPaths, pth; kwargs...) = get_volumes(grh, [get_idx(grh, pth)]; kwargs...)[1]


# ============================================================================
# Path Inspection
# ============================================================================

"""
    show_regime_path(grh::SISOPaths, pth) -> nothing

Print a formatted regime path with optional volume.
"""
function show_regime_path(grh::SISOPaths, pth)
    pth_idx = get_idx(grh, pth)
    path = get_path(grh, pth_idx; return_idx=true)
    volume = grh.path_volume_is_calc[pth_idx] ? grh.path_volume[pth_idx] : nothing
    print_path(path; prefix="#", id=pth_idx, volume=volume)
    return nothing
end

"""
    get_expression_path(grh::SISOPaths, pth; observe_x=nothing) -> (Vector, Vector)

Return expression coefficients and interfaces along a SISO path.
"""
function get_expression_path(grh::SISOPaths, pth; observe_x=nothing)
    bn = get_binding_network(grh)
    rgm_path = get_path(grh, pth; return_idx=true)
    rgm_nullities = get_nullities(bn, rgm_path)

    change_qK_idx = grh.change_qK_idx
    observe_x_idx = isnothing(observe_x) ? (1:bn.n) : locate_sym_x.(Ref(bn), observe_x)
    rgm_interface = get_interface.(Ref(bn), rgm_path[1:end-1], rgm_path[2:end])

    H_H0 = Vector{Any}(undef, length(rgm_path))
    for i in eachindex(rgm_path)
        rgm = rgm_path[i]
        nlt = rgm_nullities[i]
        if nlt == 0
            H, H0 = get_H_H0(bn, rgm)
            H_H0[i] = (H[observe_x_idx, :], H0[observe_x_idx])
        elseif nlt == 1
            H_H0[i] = (get_H(bn, rgm)[observe_x_idx, change_qK_idx], nothing)
        else
            error("Nullity > 1 is not supported for expression path.")
        end
    end
    return H_H0, rgm_interface
end


# ============================================================================
# Reaction Orders
# ============================================================================

"""
    _calc_RO_for_single_path(model, path, change_qK_idx, observe_x_idx) -> Vector

Compute the reaction-order profile along a single path.
"""
function _calc_RO_for_single_path(
    model,
    path::AbstractVector{<:Integer},
    change_qK_idx,
    observe_x_idx,
)::Vector{<:Real}
    r_ord = Vector{Float64}(undef, length(path))
    for i in eachindex(path)
        if !is_singular(model, path[i])
            r_ord[i] = round(Float64(get_H(model, path[i])[observe_x_idx, change_qK_idx]); digits=3)
        else
            ord = get_H(model, path[i])[observe_x_idx, change_qK_idx]
            r_ord[i] = abs(ord) < 1e-6 ? NaN : Float64(ord) * Inf
        end
    end
    return r_ord
end

"""
    _dedup(ord_path) -> Vector

Deduplicate consecutive reaction-order values while preserving discontinuities.
"""
function _dedup(ord_path::AbstractVector{T})::Vector{T} where {T<:Real}
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
    get_RO_path(model::Bnc, rgm_idx_shift_pth; change_qK, observe_x, kwargs...) -> Vector

Calculate the reaction-order profile for a single regime path.
"""
function get_RO_path(
    model::Bnc,
    rgm_idx_shift_pth::AbstractVector;
    change_qK,
    observe_x,
    deduplicate::Bool=false,
    keep_singular::Bool=true,
    keep_nonasymptotic::Bool=true,
)::Vector{<:Real}
    rgm_idx_shift_pth = get_idx.(Ref(model), rgm_idx_shift_pth)

    ord_path = _calc_RO_for_single_path(
        model,
        rgm_idx_shift_pth,
        locate_sym_qK(model, change_qK),
        locate_sym_x(model, observe_x),
    )

    mask = _get_mask(
        model,
        rgm_idx_shift_pth;
        singular=keep_singular ? nothing : false,
        asymptotic=keep_nonasymptotic ? nothing : true,
    )
    ord_path = ord_path[mask]

    return deduplicate ? _dedup(ord_path) : ord_path
end

function _ensure_ro_regimes_materialized!(
    model::Bnc,
    rgm_idx_for_each_paths::AbstractVector{<:AbstractVector{<:Integer}},
)
    seen = Set{Int}()
    ordered_idxs = Int[]
    for path in rgm_idx_for_each_paths, idx in path
        idx = Int(idx)
        if !(idx in seen)
            push!(ordered_idxs, idx)
            push!(seen, idx)
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
function get_RO_paths(
    model::Bnc,
    rgm_paths::AbstractVector{<:AbstractVector},
    args...;
    kwargs...,
)::Vector{Vector{<:Real}}
    rgm_idx_for_each_paths = rgm_paths .|> path -> get_idx.(Ref(model), path)
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
function get_RO_paths(
    model::SISOPaths,
    pth_idx::Union{Nothing,AbstractVector}=nothing;
    observe_x,
    kwargs...,
)
    selected_idxs = _normalize_path_indices(model, pth_idx)
    rgm_paths = model.rgm_paths[selected_idxs]
    observe_x_idx = locate_sym_x(model.bn, observe_x)
    return get_RO_paths(
        model.bn,
        rgm_paths;
        change_qK=model.change_qK_idx,
        observe_x=observe_x_idx,
        kwargs...,
    )
end

"""
    get_RO_path(model::SISOPaths, pth_idx, args...; kwargs...) -> Vector

Single-path wrapper for `get_RO_paths`.
"""
get_RO_path(model::SISOPaths, pth_idx, args...; kwargs...) = get_RO_paths(model, [get_idx(model, pth_idx)], args...; kwargs...)[1]


# ============================================================================
# Summaries
# ============================================================================

"""
    summary(grh::SISOPaths; show_volume=true, prefix="#", kwargs...) -> nothing

Print the paths stored in `SISOPaths`, optionally with volumes.
"""
function summary(grh::SISOPaths; show_volume::Bool=true, prefix::AbstractString="#", kwargs...)
    if show_volume
        print_paths(grh.rgm_paths; prefix=prefix, volumes=get_volumes(grh; kwargs...), ids=1:length(grh.rgm_paths))
    else
        print_paths(grh.rgm_paths; prefix=prefix, ids=1:length(grh.rgm_paths))
    end
    return nothing
end

"""
    summary_RO_path(grh::SISOPaths; observe_x, show_volume=true, kwargs...) -> nothing

Summarize reaction-order paths grouped by profile.
"""
function summary_RO_path(
    grh::SISOPaths;
    observe_x,
    show_volume::Bool=true,
    deduplicate::Bool=true,
    keep_singular::Bool=true,
    keep_nonasymptotic::Bool=true,
    kwargs...,
)
    ord_paths = get_RO_paths(
        grh;
        observe_x=observe_x,
        deduplicate=deduplicate,
        keep_singular=keep_singular,
        keep_nonasymptotic=keep_nonasymptotic,
    )

    volumes = show_volume ? get_volumes(grh; kwargs...) : fill(nothing, length(grh.rgm_paths))
    grouped = group_sum(ord_paths, volumes)

    ids = getindex.(grouped, 1)
    ords = getindex.(grouped, 2)
    vols = getindex.(grouped, 3)
    print_paths(ords; prefix="", ids=ids, volumes=vols)
    return nothing
end
