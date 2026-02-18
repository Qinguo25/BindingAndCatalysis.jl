"""
Path-finding workflow for directed regime graphs
================================================

This file implements path discovery between pairs of regime vertices under a
direction vector `v`, together with a geometric feasibility condition for each
path. The feasibility condition is represented as a polyhedron in
`(d-1)`-dimensional prism space (obtained by eliminating the direction axis).

High-level idea
----------------
1. Build a directed graph from regime adjacency and the sign of edge movement
     along `v` (`_graph_init`).
2. Precompute graph reachability summaries:
     - `sources`, `sinks`
     - `upstream[i]`: vertices that can reach `i`
     - `downstream[i]`: vertices reachable from `i`
     using DFS (`_path_tracing!`, `_dfs_upstream!`, `_dfs_downstream!`).
3. Answer pair query `(from, to)` with `_better_path_finder!` using recursive,
     memoized decomposition.
4. For each discovered path, compute and propagate a polyhedral condition by
     intersections of prisms/interface-prisms.

Data structures
---------------
- `RegimePath`
    - `path::Vector{Int}`: ordered vertex sequence.
    - `condition::Polyhedra.Polyhedron`: feasible region for this sequence.
- `RegimeGraph`
    - `connectome[i,j] == true` means a directed edge `i -> j`.
    - `predecessors`/`successors`: local graph neighborhoods.
    - `upstream`/`downstream`: transitive reachability caches.
    - `paths[i,j]`: `nothing` (no feasible path) or `Vector{RegimePath}`.
    - `path_found[i,j]`: memoization flag indicating `(i,j)` already solved.

Graph construction (`_graph_init`)
----------------------------------
- Uses undirected vertex-neighbor relations from `get_vertices_graph!(...;
    full=true)`.
- For each undirected pair, computes `dir = dot(change_dir_qK[1:d], v)`.
    - `dir > tol`: orient edge as `i -> j`.
    - `dir < -tol`: orient edge as `j -> i`.
    - near zero: no directed edge.
- Vertices with nullity > 1 are skipped for orientation in this pass.

Reachability preprocessing (`_path_tracing!`)
--------------------------------------------
- `sources`: vertices with no incoming edges.
- `sinks`: vertices with no outgoing edges.
- DFS from sinks (reverse direction) fills `upstream`.
- DFS from sources (forward direction) fills `downstream`.
- These sets provide aggressive pruning for pairwise path search.

Core recursive solver (`_better_path_finder!`)
----------------------------------------------
For a query `(from, to)`:

1. Memoization check
     - If `path_found[from,to]` is true, return cached existence immediately.

2. Base case: `from == to`
     - Build vertex prism via `_get_polyhedron_prism(...)`.
     - Store one trivial path `[from]` with that condition.

3. Candidate bridge set
     - `pass_by = downstream[from] ∩ upstream[to]`.
     - If empty, only direct-interface attempt is possible:
         - Compute `_get_interface_prism(from, to, ...)`.
         - If empty after redundancy removal: cache `nothing` and fail.
         - Else cache one length-2 path `[from, to]`.

4. Recursive decomposition when `pass_by` is non-empty
     - Define
         - `_successors = pass_by ∩ successors[from]`
         - `_predecessors = pass_by ∩ predecessors[to]`
     - Compute solved ratios for both frontiers and choose expansion direction
         based on which side already has more memoized subproblems.

5. Three expansion modes
     - Mode A: neither side has solved subproblems yet
         - Intersect endpoint self-conditions (`from->from`, `to->to`) to create a
             shared gate condition.
         - Enumerate `(successor, predecessor)` pairs, solve recursively, and
             append/prepend endpoints to each middle path.
     - Mode B: successor side is better cached
         - Solve `successor -> to`, then prepend `from` and intersect with
             `from->from` condition.
     - Mode C: predecessor side is better cached
         - Solve `from -> predecessor`, then append `to` and intersect with
             `to->to` condition.

6. Feasibility propagation
     - Every composed path condition is built by polyhedron intersection.
     - Redundancy is removed; empty intersections are discarded.
     - If no feasible composed path remains, cache `nothing`.

Why this is efficient
---------------------
- `upstream/downstream` cuts impossible pairs early.
- `path_found` + `paths` memoization avoids recomputation.
- Directional expansion heuristic uses already-solved frontier side first,
    reducing recursion fanout in practice.
- Polyhedral intersections enforce geometric validity at each composition step,
    so infeasible branches are pruned immediately.
"""

include(joinpath(@__DIR__,"get_prism.jl"))

mutable struct RegimePath
    path::Vector{Int} # the path of vertices, represented by their indices in the vertices array of Bnc
    condition::Polyhedra.Polyhedron # the path condition, represented as a polyhedron in H-representation

    # constructor
    function RegimePath(path::Vector{Int}, condition::Polyhedra.Polyhedron)
        new(path, condition)
    end
end


mutable struct RegimeGraph
    bnc_system::Bnc
    dir_vec::Vector{Float64} # the vector v used to determine the direction
    connectome::Matrix{Bool}    # the connectome of the graph, where connectome[i,j] is true if there is an edge from vertex i to vertex j
    predecessors::Vector{Set{Int}} # the predecessors of each vertex, where predecessors[i] is the set of indices of the vertices that have edges to vertex i
    successors::Vector{Set{Int}}   # the successors of each vertex, where successors[i] is the set of indices of the vertices that have edges from vertex i
    upstream::Vector{Set{Int}}   # the achievable upstream vertices for each vertex, where upstream[i] is the list of indices of the vertices that can reach vertex i
    downstream::Vector{Set{Int}} # the achievable downstream vertices for each vertex, where downstream[i] is the list of indices of the vertices that can be reached from vertex i
    paths::Matrix{Union{Vector{RegimePath},Nothing}} # the paths of the graph, where paths[i,j] is the paths from vertex i to vertex j. Nothing if the paths have not been calculated yet, or the condition does not exist.
    path_found::Matrix{Bool} # a boolean matrix to indicate whether the paths from vertex i to vertex j have been calculated, where path_found[i,j] is true if the paths from vertex i to vertex j have been calculated
    sources::Vector{Int} # the indices of the source vertices, which have no upstream vertices
    sinks::Vector{Int}   # the indices of the sink vertices, which have no downstream vertices
    householder_matrix::Matrix{Float64} # the Householder transformation matrix, which is used to transform the polyhedra to eliminate the dimension along the vector v
    axis_to_eliminate::Int # the index of the coordinate axis to eliminate, which is the axis that is aligned with the vector v. If v is not aligned with any coordinate axis, it is -1.

    # constructor
    function RegimeGraph(bnc_sys::Bnc, v::Vector{Float64})
        # Build directed adjacency from regime neighbor graph and direction vector `v`.
        connectome = _graph_init(bnc_sys, v)
        n_vtx = size(connectome, 1)

        # Build one-step predecessor/successor sets from adjacency matrix.
        predecessors = Vector{Set{Int}}(undef, n_vtx)
        successors = Vector{Set{Int}}(undef, n_vtx)
        for i in eachindex(predecessors)
            predecessors[i] = Set{Int}(findall(@view connectome[:, i]))
            successors[i] = Set{Int}(findall(@view connectome[i, :]))
        end
        new(
            bnc_sys,
            v,
            connectome,
            predecessors,
            successors,
            Vector{Set{Int}}(), 
            Vector{Set{Int}}(), 
            Matrix{Union{Vector{RegimePath}, Nothing}}(nothing, size(connectome)), 
            fill(false, size(connectome)),
            Int[],
            Int[],
            _get_Householder_transformation(v),
            _get_axis_to_eliminate(v))
    end
end

"""
    _graph_init(
        bnc_sys::Bnc,
        v::Vector{Float64}
        )::Matrix{Bool}

based on the BnC system and the vector v, evaluate the direction of each edge in the graph, and generate the basic connectome to fill in.
"""
function _graph_init(
    bnc_sys::Bnc,
    v::Vector{Float64},
    )::Matrix{Bool}

    # ensure that the vertices have been calculated
    n_vtx = length(bnc_sys.vertices_perm)
    n_vtx == 0 && find_all_vertices!(bnc_sys)
    n_vtx = length(bnc_sys.vertices_perm)

    # ensure that the length of v is correct
    length(v) == bnc_sys.d || error("Length of v must be $(bnc_sys.d), got $(length(v)).")

    
    vtx_grh = get_vertices_graph!(bnc_sys; full=true)
    connectome = fill(false, n_vtx, n_vtx)

    tol = 1e-6
    for (i, edges) in enumerate(vtx_grh.neighbors)
        # Skip singular/high-nullity vertices for orientation in this pass.
        if get_nullity(bnc_sys, i) > 1
            continue
        end
        for e in edges
            # process each undirected pair once
            if isnothing(e.change_dir_qK) || e.to < i
                continue
            end

            # use only q-space components (first d coordinates in qK space)
            dir = dot(e.change_dir_qK[1:bnc_sys.d], v)
            if dir > tol
                connectome[i, e.to] = true
            elseif dir < -tol
                connectome[e.to, i] = true
            end
        end
    end

    return connectome
end

"""
    _dfs_upstream!(
        regime_graph::RegimeGraph,
        visited::Vector{Bool},
        current::Int,
        up_stream_done::Vector{Bool},
        n_vtx::Int
    )::Nothing

A helper function for depth-first search to calculate the upstream vertices for each vertex.
It is called by _path_tracing! function. It updates the upstream field of the regime graph in place.
"""
function _dfs_upstream!(
    regime_graph::RegimeGraph,
    visited::Vector{Bool},
    current::Int,
    up_stream_done::Vector{Bool},
    n_vtx::Int
    )::Nothing

    # Standard DFS bookkeeping to avoid cycles on current recursion stack.
    visited[current] = true
    if up_stream_done[current]
        visited[current] = false
        return
    end
    for neighbor in 1:n_vtx
        if regime_graph.connectome[neighbor, current] && !visited[neighbor]
            # Reverse-edge DFS: collect all vertices that can reach `current`.
            _dfs_upstream!(regime_graph, visited, neighbor, up_stream_done, n_vtx)
            push!(regime_graph.upstream[current], neighbor)
            union!(regime_graph.upstream[current], regime_graph.upstream[neighbor])
        end
    end
    up_stream_done[current] = true
    visited[current] = false
    return
end

"""
    _dfs_downstream!(
        regime_graph::RegimeGraph,
        visited::Vector{Bool},
        current::Int,
        down_stream_done::Vector{Bool},
        n_vtx::Int
    )::Nothing

A helper function for depth-first search to calculate the downstream vertices for each vertex.
It is called by _path_tracing! function. It updates the downstream field of the regime graph in place.
"""
function _dfs_downstream!(
    regime_graph::RegimeGraph,
    visited::Vector{Bool},
    current::Int,
    down_stream_done::Vector{Bool},
    n_vtx::Int
    )::Nothing

    # Standard DFS bookkeeping to avoid cycles on current recursion stack.
    visited[current] = true
    if down_stream_done[current]
        visited[current] = false
        return
    end
    for neighbor in 1:n_vtx
        if regime_graph.connectome[current, neighbor] && !visited[neighbor]
            # Forward-edge DFS: collect all vertices reachable from `current`.
            _dfs_downstream!(regime_graph, visited, neighbor, down_stream_done, n_vtx)
            push!(regime_graph.downstream[current], neighbor)
            union!(regime_graph.downstream[current], regime_graph.downstream[neighbor])
        end
    end
    down_stream_done[current] = true
    visited[current] = false
    return
end

"""
    _path_tracing!(
        regime_graph::RegimeGraph,
    )::Nothing

Based on the connectome of the graph, 
calculate the sources/sinks of the graph and the upstream and downstream vertices for each vertex.
"""
function _path_tracing!(
    regime_graph::RegimeGraph,
    )::Nothing
    
    n_vtx = size(regime_graph.connectome, 1)
    regime_graph.upstream = [Set{Int}() for _ in 1:n_vtx]
    regime_graph.downstream = [Set{Int}() for _ in 1:n_vtx]
    regime_graph.sources = Int[]
    regime_graph.sinks = Int[]

    # Source: no incoming edge. Sink: no outgoing edge.
    for i in 1:n_vtx
        has_upstream = any(regime_graph.connectome[:, i])
        has_downstream = any(regime_graph.connectome[i, :])
        if !has_upstream
            push!(regime_graph.sources, i)
        end
        if !has_downstream
            push!(regime_graph.sinks, i)
        end
    end

    # calculate upstream and downstream vertices using depth-first search
    upstream_done = fill(false, n_vtx)
    for i in regime_graph.sinks
        visited = fill(false, n_vtx)
        _dfs_upstream!(regime_graph, visited, i, upstream_done, n_vtx)
    end
    downstream_done = fill(false, n_vtx)
    for i in regime_graph.sources
        visited = fill(false, n_vtx)
        _dfs_downstream!(regime_graph, visited, i, downstream_done, n_vtx)
    end

    return
end

"""
    _better_path_finder!(
        regime_graph::RegimeGraph,
        vertex_idx_from::Int,
        vertex_idx_to::Int,
    )::Bool

Find the paths from vertex_idx_from to vertex_idx_to in the regime graph, and calculate the path conditions.
It updates the paths and path_found fields of the regime graph in place.
It returns true if there is at least one path from vertex_idx_from to vertex_idx_to, and false otherwise.
"""
function _better_path_finder!(
    regime_graph::RegimeGraph,
    vertex_idx_from::Int,
    vertex_idx_to::Int,
    )::Bool

    # Memoization: if solved before, just return cached existence.
    if regime_graph.path_found[vertex_idx_from, vertex_idx_to]
        if isnothing(regime_graph.paths[vertex_idx_from, vertex_idx_to])
            return false
        else
            return true
        end
    end

    # Base case: trivial self-path with its own prism condition.
    if vertex_idx_from == vertex_idx_to
        _condition = _get_polyhedron_prism(
            regime_graph.bnc_system,
            vertex_idx_from,
            regime_graph.axis_to_eliminate,
            regime_graph.householder_matrix,
        )
        _path = RegimePath([vertex_idx_from], _condition)
        regime_graph.paths[vertex_idx_from, vertex_idx_to] = [_path]
        regime_graph.path_found[vertex_idx_from, vertex_idx_to] = true
        return true
    end
    
    # Candidate middle vertices must be reachable from `from` and can reach `to`.
    pass_by = intersect(regime_graph.downstream[vertex_idx_from], regime_graph.upstream[vertex_idx_to])
    if isempty(pass_by)
        # No intermediate bridge candidate: attempt direct interface feasibility.
        _condition = _get_interface_prism(
            regime_graph.bnc_system,
            vertex_idx_from,
            vertex_idx_to,
            regime_graph.axis_to_eliminate,
            regime_graph.householder_matrix,
        )
        removehredundancy!(_condition)
        if isempty(_condition)
            regime_graph.paths[vertex_idx_from, vertex_idx_to] = nothing
            regime_graph.path_found[vertex_idx_from, vertex_idx_to] = true
            return false
        end
        _path = RegimePath([vertex_idx_from, vertex_idx_to], _condition)
        regime_graph.paths[vertex_idx_from, vertex_idx_to] = [_path]
        regime_graph.path_found[vertex_idx_from, vertex_idx_to] = true
        return true
    end

    # Collect feasible composed paths for `(from, to)`.
    _paths = RegimePath[]
    # Frontier choices for decomposition from each side.
    _successors = intersect(pass_by, regime_graph.successors[vertex_idx_from])
    _predecessors = intersect(pass_by, regime_graph.predecessors[vertex_idx_to])

    # Invariant check:
    # if `pass_by` is non-empty, both frontiers should be non-empty as well.
    # Otherwise reachability caches (`upstream`/`downstream`) are inconsistent.
    if isempty(_successors)
        error("Invariant violated: `pass_by` is non-empty but `pass_by ∩ successors[from]` is empty for (from=$(vertex_idx_from), to=$(vertex_idx_to)).")
    end
    if isempty(_predecessors)
        error("Invariant violated: `pass_by` is non-empty but `pass_by ∩ predecessors[to]` is empty for (from=$(vertex_idx_from), to=$(vertex_idx_to)).")
    end

    # Heuristic: prefer expanding from the side with more solved subproblems.
    num_successor_calculated = sum(regime_graph.path_found[successor, vertex_idx_to] for successor in _successors)
    percentage_successor_calculated = num_successor_calculated / length(_successors)
    # Same solved-ratio estimate on predecessor side.
    num_predecessor_calculated = sum(regime_graph.path_found[vertex_idx_from, predecessor] for predecessor in _predecessors)
    percentage_predecessor_calculated = num_predecessor_calculated / length(_predecessors)


    if num_predecessor_calculated == 0 && num_successor_calculated == 0
        # Cold-start mode: no cached subproblem on either side yet.
        _better_path_finder!(regime_graph, vertex_idx_from, vertex_idx_from)
        _better_path_finder!(regime_graph, vertex_idx_to, vertex_idx_to)
        _condition = intersect(
            regime_graph.paths[vertex_idx_from, vertex_idx_from][1].condition,
            regime_graph.paths[vertex_idx_to, vertex_idx_to][1].condition)
        removehredundancy!(_condition)
        if isempty(_condition)
            regime_graph.paths[vertex_idx_from, vertex_idx_to] = nothing
            regime_graph.path_found[vertex_idx_from, vertex_idx_to] = true
            return false
        end

        for successor in _successors
            for predecessor in _predecessors
                if _better_path_finder!(regime_graph, successor, predecessor)
                    for path1 in regime_graph.paths[successor, predecessor]
                        full_condition = intersect(_condition, path1.condition)
                        removehredundancy!(full_condition)
                        if isempty(full_condition)
                            continue
                        end

                        _path = RegimePath([vertex_idx_from; path1.path; vertex_idx_to], full_condition)
                        push!(_paths, _path)
                    end
                end
            end
        end
        if isempty(_paths)
            regime_graph.paths[vertex_idx_from, vertex_idx_to] = nothing
            regime_graph.path_found[vertex_idx_from, vertex_idx_to] = true
            return false
        else
            regime_graph.paths[vertex_idx_from, vertex_idx_to] = _paths
            regime_graph.path_found[vertex_idx_from, vertex_idx_to] = true
            return true
        end
    end

    if percentage_successor_calculated > percentage_predecessor_calculated
        # Expand from successor side: solve `successor -> to`, then prepend `from`.
        _better_path_finder!(regime_graph, vertex_idx_from, vertex_idx_from)
        for successor in _successors
            if _better_path_finder!(regime_graph, successor, vertex_idx_to)
                for path1 in regime_graph.paths[successor, vertex_idx_to]
                    _condition = regime_graph.paths[vertex_idx_from, vertex_idx_from][1].condition
                    full_condition = intersect(_condition, path1.condition)
                    removehredundancy!(full_condition)
                    if isempty(full_condition)
                        continue
                    end
                    _path = RegimePath([vertex_idx_from; path1.path], full_condition)
                    push!(_paths, _path)
                end
            end
        end
        
        if isempty(_paths)
            regime_graph.paths[vertex_idx_from, vertex_idx_to] = nothing
            regime_graph.path_found[vertex_idx_from, vertex_idx_to] = true
            return false
        else
            regime_graph.paths[vertex_idx_from, vertex_idx_to] = _paths
            regime_graph.path_found[vertex_idx_from, vertex_idx_to] = true
            return true
        end
    else
        # Expand from predecessor side: solve `from -> predecessor`, then append `to`.
        _better_path_finder!(regime_graph, vertex_idx_to, vertex_idx_to)
        for predecessor in _predecessors
            if _better_path_finder!(regime_graph, vertex_idx_from, predecessor)
                for path1 in regime_graph.paths[vertex_idx_from, predecessor]
                    _condition = regime_graph.paths[vertex_idx_to, vertex_idx_to][1].condition
                    full_condition = intersect(_condition, path1.condition)
                    removehredundancy!(full_condition)
                    if isempty(full_condition)
                        continue
                    end
                    _path = RegimePath([path1.path; vertex_idx_to], full_condition)
                    push!(_paths, _path)
                end
            end
        end
        
        if isempty(_paths)
            regime_graph.paths[vertex_idx_from, vertex_idx_to] = nothing
            regime_graph.path_found[vertex_idx_from, vertex_idx_to] = true
            return false
        else
            regime_graph.paths[vertex_idx_from, vertex_idx_to] = _paths
            regime_graph.path_found[vertex_idx_from, vertex_idx_to] = true
            return true
        end
    end
end

"""
    better_path_finder(
        bnc_sys::Bnc,
        v::Vector{Float64}
    )::RegimeGraph

The main function to find the paths from all sources to sinks, and calculate the path conditions.
"""
function better_path_finder(
    bnc_sys::Bnc,
    v::Vector{Float64}
    )::RegimeGraph

    println("Initializing regime graph...")

    r_g = RegimeGraph(bnc_sys, v)
    _path_tracing!(r_g)

    for source in r_g.sources
        for sink in r_g.sinks
            _better_path_finder!(r_g, source, sink)
        end
    end

    return r_g
end