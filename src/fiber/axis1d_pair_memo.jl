function _axis1d_project_condition(poly::Polyhedron, change_qK_idx::Int)::Polyhedron
    projected = _poly_eliminate(poly, BitSet((change_qK_idx,)); canonicalize=false)
    return _clean_polyhedron!(projected)
end

"""
    _axis1d_interface_condition(model, from, to, change_qK_idx) -> Polyhedron

Project the interface `poly(from) ∩ poly(to)` by eliminating `change_qK_idx`.
"""
function _axis1d_interface_condition(
    bnc_sys::Bnc, vertex_idx_from::Int, vertex_idx_to::Int, change_qK_idx::Int
)::Polyhedra.Polyhedron
    projected = _poly_intersect_eliminate(
        get_polyhedron(bnc_sys, vertex_idx_from),
        get_polyhedron(bnc_sys, vertex_idx_to),
        BitSet((change_qK_idx,));
        canonicalize=false,
    )
    return _clean_polyhedron!(projected)
end

"""
    _axis1d_regime_condition(model, vertex_idx, change_qK_idx) -> Polyhedron

Project a single regime polyhedron by eliminating `change_qK_idx`.
"""
function _axis1d_regime_condition(
    bnc_sys::Bnc, vertex_idx::Int, change_qK_idx::Int
)::Polyhedra.Polyhedron
    return _axis1d_project_condition(get_polyhedron(bnc_sys, vertex_idx), change_qK_idx)
end

function _axis1d_intersect_nonempty(
    polys::Polyhedra.Polyhedron...
)::Union{Nothing, Polyhedra.Polyhedron}
    poly = _clean_polyhedron!(_poly_intersect_many(collect(polys); canonicalize=false))
    return isempty(poly) ? nothing : poly
end

# ============================================================================
# Axis-aligned one-dimensional pair-memo backend
# ============================================================================

const Axis1DPathKey = Tuple{Vararg{Int}}
const Axis1DPairKey = NTuple{2, Int}
const Axis1DPathConditionMap = Dict{Axis1DPathKey, Polyhedron}
struct Axis1DDAG
    graph::SimpleDiGraph
    sources::Vector{Int}
    sinks::Vector{Int}
    reachable::BitMatrix
end

struct Axis1DProblem{T}
    bn::Bnc{T}
    change_qK_idx::Int
    dag::Axis1DDAG
end

mutable struct Axis1DDAGProfile
    planning_ns::UInt64
    pair_solve_ns::UInt64
    middle_collect_ns::UInt64
    middle_compute_ns::UInt64
    middle_merge_ns::UInt64
    pair_solve_calls::Int
    planned_pairs::Int
    middle_parallel_nodes::Int
    middle_serial_nodes::Int
    middle_join_pairs::Int
    queue_pair_tasks::Int
    queue_chunk_tasks::Int
    queue_chunked_pairs::Int
    queue_finalize_tasks::Int
    queue_max_chunks_per_pair::Int
    queue_max_chunk_estimated_entries::Int
    queue_total_chunk_estimated_entries::Int
    queue_max_chunk_seconds::Float64
    queue_total_chunk_seconds::Float64
    queue_finalize_ns::UInt64
    queue_chunk_candidate_pairs::Int
    queue_chunk_size_gate_skips::Int
    queue_chunk_width_gate_skips::Int
    queue_chunk_thread_gate_skips::Int
    queue_estimator_entries_per_second::Float64
    queue_estimator_target_entries::Int
    queue_estimator_min_parallel_entries::Float64
    queue_estimator_target_seconds::Float64
    weighted_work_done::Float64
    weighted_work_total::Float64
    weighted_progress_units::Int
    largest_pair_seconds::Float64
    largest_pair_from::Int
    largest_pair_to::Int
    current_pair_from::Int
    current_pair_to::Int
    current_pair_branch::Symbol
    current_pair_weight::Float64
    current_pair_start_ns::UInt64
    current_pair_elapsed_seconds::Float64
    current_pair_output_entries::Int
end

function Axis1DDAGProfile()
    return Axis1DDAGProfile(
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.0,
        0.0,
        0,
        0,
        0,
        0,
        0,
        _axis1d_dag_fallback_entries_per_second(),
        0,
        0.0,
        _axis1d_dag_target_chunk_seconds(),
        0.0,
        0.0,
        0,
        0.0,
        0,
        0,
        0,
        0,
        :none,
        0.0,
        UInt64(0),
        0.0,
        0,
    )
end

mutable struct Axis1DPairMemoBackend{T}
    problem::Axis1DProblem{T}
    vertex_prisms::Vector{Union{Nothing, Polyhedron}}
    interface_prisms::Dict{Axis1DPairKey, Polyhedron}
    pair_conditions::Dict{Axis1DPairKey, Axis1DPathConditionMap}
    cache_lock::ReentrantLock
    dag_profile::Union{Nothing, Axis1DDAGProfile}
end

@inline _pair_key(from::Int, to::Int)::Axis1DPairKey = (from, to)
@inline _undirected_pair_key(a::Int, b::Int)::Axis1DPairKey = a <= b ? (a, b) : (b, a)
@inline _path_key(path::AbstractVector{<:Integer})::Axis1DPathKey = Tuple(Int.(path))
@inline _prepend_vertex(v::Int, key::Axis1DPathKey)::Axis1DPathKey = (v, key...)
@inline _append_vertex(key::Axis1DPathKey, v::Int)::Axis1DPathKey = (key..., v)
@inline _wrap_vertices(left::Int, key::Axis1DPathKey, right::Int)::Axis1DPathKey =
    (left, key..., right)

function _build_reachability(g::SimpleDiGraph)::BitMatrix
    n = nv(g)
    reachable = falses(n, n)
    topo = topological_sort_by_dfs(g)

    for v in Iterators.reverse(topo)
        row_v = @view reachable[v, :]
        for nb in outneighbors(g, v)
            row_v[nb] = true
            row_nb = @view reachable[nb, :]
            @inbounds for j in 1:n
                row_v[j] |= row_nb[j]
            end
        end
    end
    return reachable
end

function _build_axis1d_problem(
    bnc_sys::Bnc{T},
    change_qK_idx::Integer,
    qK_grh::SimpleDiGraph,
    sources::AbstractVector{<:Integer},
    sinks::AbstractVector{<:Integer},
) where {T}
    dag = Axis1DDAG(
        qK_grh,
        sort!(Int.(collect(sources))),
        sort!(Int.(collect(sinks))),
        _build_reachability(qK_grh),
    )
    return Axis1DProblem{T}(bnc_sys, Int(change_qK_idx), dag)
end

function Axis1DPairMemoBackend(problem::Axis1DProblem{T}) where {T}
    n_vtx = nv(problem.dag.graph)
    vertex_prisms = Vector{Union{Nothing, Polyhedron}}(undef, n_vtx)
    fill!(vertex_prisms, nothing)
    return Axis1DPairMemoBackend{T}(
        problem,
        vertex_prisms,
        Dict{Axis1DPairKey, Polyhedron}(),
        Dict{Axis1DPairKey, Axis1DPathConditionMap}(),
        ReentrantLock(),
        nothing,
    )
end

get_binding_network(problem::Axis1DProblem, args...) = problem.bn
function get_binding_network(helper::Axis1DPairMemoBackend, args...)
    return get_binding_network(helper.problem)
end
_axis1d_graph(problem::Axis1DProblem) = problem.dag.graph
_axis1d_graph(helper::Axis1DPairMemoBackend) = _axis1d_graph(helper.problem)
get_sources(problem::Axis1DProblem) = copy(problem.dag.sources)
get_sources(helper::Axis1DPairMemoBackend) = get_sources(helper.problem)
get_sinks(problem::Axis1DProblem) = copy(problem.dag.sinks)
get_sinks(helper::Axis1DPairMemoBackend) = get_sinks(helper.problem)
_axis1d_change_axis(problem::Axis1DProblem) = problem.change_qK_idx
_axis1d_change_axis(helper::Axis1DPairMemoBackend) = _axis1d_change_axis(helper.problem)
_axis1d_dag_profile(helper::Axis1DPairMemoBackend) = helper.dag_profile

@inline _edge_exists(helper::Axis1DPairMemoBackend, from::Int, to::Int) =
    has_edge(_axis1d_graph(helper), from, to)
@inline _pair_is_cached(helper::Axis1DPairMemoBackend, from::Int, to::Int) =
    lock(helper.cache_lock) do
        haskey(helper.pair_conditions, _pair_key(from, to))
    end
@inline _pair_conditions(helper::Axis1DPairMemoBackend, from::Int, to::Int) =
    lock(helper.cache_lock) do
        get(helper.pair_conditions, _pair_key(from, to), nothing)
    end
@inline _can_reach(helper::Axis1DPairMemoBackend, from::Int, to::Int) =
    helper.problem.dag.reachable[from, to]

function _axis1d_vertex_condition!(
    helper::Axis1DPairMemoBackend, vertex_idx::Int
)::Polyhedra.Polyhedron
    prism = helper.vertex_prisms[vertex_idx]
    if !isnothing(prism)
        return prism
    end

    prism = _clean_polyhedron!(
        _axis1d_regime_condition(
            helper.problem.bn, vertex_idx, helper.problem.change_qK_idx
        ),
    )
    helper.vertex_prisms[vertex_idx] = prism
    return prism
end

function _axis1d_interface_condition!(
    helper::Axis1DPairMemoBackend, vertex_idx_from::Int, vertex_idx_to::Int
)::Polyhedra.Polyhedron
    key = _undirected_pair_key(vertex_idx_from, vertex_idx_to)
    prism = get(helper.interface_prisms, key, nothing)
    if !isnothing(prism)
        return prism
    end

    prism = _clean_polyhedron!(
        _axis1d_interface_condition(
            helper.problem.bn, vertex_idx_from, vertex_idx_to, helper.problem.change_qK_idx
        ),
    )

    helper.interface_prisms[key] = prism
    return prism
end

function _bridge_successors(helper::Axis1DPairMemoBackend, from::Int, to::Int)::Vector{Int}
    out = Int[]
    for successor in outneighbors(_axis1d_graph(helper), from)
        successor == to && continue
        _can_reach(helper, successor, to) || continue
        push!(out, successor)
    end
    return out
end

function _bridge_predecessors(
    helper::Axis1DPairMemoBackend, from::Int, to::Int
)::Vector{Int}
    out = Int[]
    for predecessor in inneighbors(_axis1d_graph(helper), to)
        predecessor == from && continue
        _can_reach(helper, from, predecessor) || continue
        push!(out, predecessor)
    end
    return out
end

function _cache_pair_conditions!(
    helper::Axis1DPairMemoBackend, from::Int, to::Int, conditions::Axis1DPathConditionMap
)::Axis1DPathConditionMap
    lock(helper.cache_lock) do
        helper.pair_conditions[_pair_key(from, to)] = conditions
    end
    return conditions
end

function _maybe_store_direct_path!(
    conditions::Axis1DPathConditionMap, helper::Axis1DPairMemoBackend, from::Int, to::Int
)::Nothing
    _edge_exists(helper, from, to) || return nothing
    condition = _axis1d_interface_condition!(helper, from, to)
    isempty(condition) && return nothing
    conditions[(from, to)] = condition
    return nothing
end

function _find_pair_path_conditions!(
    helper::Axis1DPairMemoBackend, from::Int, to::Int
)::Axis1DPathConditionMap
    cached = _pair_conditions(helper, from, to)
    !isnothing(cached) && return cached

    conditions = Axis1DPathConditionMap()
    if from == to
        condition = _axis1d_vertex_condition!(helper, from)
        isempty(condition) || (conditions[(from,)] = condition)
        return _cache_pair_conditions!(helper, from, to, conditions)
    end

    _maybe_store_direct_path!(conditions, helper, from, to)

    successors = _bridge_successors(helper, from, to)
    predecessors = _bridge_predecessors(helper, from, to)
    if isempty(successors) || isempty(predecessors)
        return _cache_pair_conditions!(helper, from, to, conditions)
    end

    n_solved_successors = count(
        successor -> _pair_is_cached(helper, successor, to), successors
    )
    n_solved_predecessors = count(
        predecessor -> _pair_is_cached(helper, from, predecessor), predecessors
    )
    solved_successor_ratio = n_solved_successors / length(successors)
    solved_predecessor_ratio = n_solved_predecessors / length(predecessors)

    if n_solved_successors == 0 && n_solved_predecessors == 0
        for successor in successors
            left_condition = _axis1d_interface_condition!(helper, from, successor)
            isempty(left_condition) && continue
            for predecessor in predecessors
                right_condition = _axis1d_interface_condition!(helper, predecessor, to)
                isempty(right_condition) && continue
                middle_conditions = _find_pair_path_conditions!(
                    helper, successor, predecessor
                )
                isempty(middle_conditions) && continue
                for (middle_path, middle_condition) in middle_conditions
                    full_condition = _axis1d_intersect_nonempty(
                        left_condition, middle_condition, right_condition
                    )
                    isnothing(full_condition) && continue
                    conditions[_wrap_vertices(from, middle_path, to)] = full_condition
                end
            end
        end
        return _cache_pair_conditions!(helper, from, to, conditions)
    end

    if solved_successor_ratio > solved_predecessor_ratio
        for successor in successors
            suffix_conditions = _find_pair_path_conditions!(helper, successor, to)
            isempty(suffix_conditions) && continue
            left_condition = _axis1d_interface_condition!(helper, from, successor)
            isempty(left_condition) && continue
            for (suffix_path, suffix_condition) in suffix_conditions
                full_condition = _axis1d_intersect_nonempty(
                    left_condition, suffix_condition
                )
                isnothing(full_condition) && continue
                conditions[_prepend_vertex(from, suffix_path)] = full_condition
            end
        end
        return _cache_pair_conditions!(helper, from, to, conditions)
    end

    for predecessor in predecessors
        prefix_conditions = _find_pair_path_conditions!(helper, from, predecessor)
        isempty(prefix_conditions) && continue
        right_condition = _axis1d_interface_condition!(helper, predecessor, to)
        isempty(right_condition) && continue
        for (prefix_path, prefix_condition) in prefix_conditions
            full_condition = _axis1d_intersect_nonempty(prefix_condition, right_condition)
            isnothing(full_condition) && continue
            conditions[_append_vertex(prefix_path, to)] = full_condition
        end
    end

    return _cache_pair_conditions!(helper, from, to, conditions)
end

"""
    _find_all_path_conditions!(helper) -> Axis1DPairMemoBackend

Solve all source-to-sink pair conditions stored in a helper, with progress.
"""
function _find_all_path_conditions!(helper::Axis1DPairMemoBackend)::Axis1DPairMemoBackend
    pair_queries = [
        (source, sink) for source in helper.problem.dag.sources for
        sink in helper.problem.dag.sinks
    ]
    isempty(pair_queries) && return helper

    if length(pair_queries) == 1
        source, sink = only(pair_queries)
        _find_pair_path_conditions!(helper, source, sink)
        return helper
    end

    @info "Start finding all possible path conditions across $(length(pair_queries)) source-sink pairs."
    @showprogress dt = 0.1 desc = "Finding path conditions" for (source, sink) in
                                                                pair_queries
        _find_pair_path_conditions!(helper, source, sink)
    end
    return helper
end
