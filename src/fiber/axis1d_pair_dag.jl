struct Axis1DPairPlan
    branch::Symbol
    successors::Vector{Int}
    predecessors::Vector{Int}
    dependencies::Vector{Axis1DPairKey}
end

const AXIS1D_DAG_MIDDLE_PARALLEL_THRESHOLD = 8
const AXIS1D_DAG_PROGRESS_UNITS = 10_000

_axis1d_dag_scheduler() = Symbol(lowercase(get(ENV, "BNC_SISO_DAG_SCHEDULER", "auto")))
function _axis1d_dag_layer_parallel_enabled()
    return parse(Bool, get(ENV, "BNC_SISO_DAG_LAYER_PARALLEL", "false"))
end
_axis1d_dag_pair_queue_enabled() = parse(Bool, get(ENV, "BNC_SISO_DAG_PAIR_QUEUE", "false"))
function _axis1d_dag_chunk_queue_enabled()
    return parse(Bool, get(ENV, "BNC_SISO_DAG_CHUNK_QUEUE", "true"))
end
function _axis1d_dag_layer_inner_parallel_width()
    return max(0, parse(Int, get(ENV, "BNC_SISO_DAG_LAYER_INNER_PARALLEL_WIDTH", "1")))
end
function _axis1d_dag_inner_parallel_min_weight()
    return parse(Float64, get(ENV, "BNC_SISO_DAG_INNER_PARALLEL_MIN_WEIGHT", "50000"))
end
function _axis1d_dag_chunk_size_gate_enabled()
    return parse(Bool, get(ENV, "BNC_SISO_DAG_CHUNK_SIZE_GATE", "true"))
end
function _axis1d_dag_chunk_width_gate_enabled()
    return parse(Bool, get(ENV, "BNC_SISO_DAG_CHUNK_WIDTH_GATE", "true"))
end
function _axis1d_dag_chunk_thread_gate_enabled()
    return parse(Bool, get(ENV, "BNC_SISO_DAG_CHUNK_THREAD_GATE", "true"))
end
function _axis1d_dag_target_chunk_seconds()
    return max(0.1, parse(Float64, get(ENV, "BNC_SISO_DAG_TARGET_CHUNK_SECONDS", "40")))
end
function _axis1d_dag_fallback_entries_per_second()
    return max(
        1.0, parse(Float64, get(ENV, "BNC_SISO_DAG_FALLBACK_ENTRIES_PER_SECOND", "125"))
    )
end
function _axis1d_dag_chunk_rate_alpha()
    return min(
        1.0, max(0.0, parse(Float64, get(ENV, "BNC_SISO_DAG_CHUNK_RATE_ALPHA", "0.2")))
    )
end
function _axis1d_dag_chunk_width_factor()
    return max(0.0, parse(Float64, get(ENV, "BNC_SISO_DAG_CHUNK_WIDTH_FACTOR", "2")))
end
function _axis1d_dag_inner_parallel_max_pairs_per_layer()
    return max(
        0, parse(Int, get(ENV, "BNC_SISO_DAG_INNER_PARALLEL_MAX_PAIRS_PER_LAYER", "2"))
    )
end
function _axis1d_dag_inner_parallel_target_entries()
    return if haskey(ENV, "BNC_SISO_DAG_INNER_PARALLEL_TARGET_ENTRIES")
        max(1, parse(Int, ENV["BNC_SISO_DAG_INNER_PARALLEL_TARGET_ENTRIES"]))
    else
        max(
            1,
            round(
                Int,
                _axis1d_dag_fallback_entries_per_second() *
                _axis1d_dag_target_chunk_seconds(),
            ),
        )
    end
end
function _axis1d_dag_inner_parallel_max_chunks()
    return max(
        1,
        parse(
            Int,
            get(
                ENV,
                "BNC_SISO_DAG_INNER_PARALLEL_MAX_CHUNKS",
                string(4 * Threads.nthreads()),
            ),
        ),
    )
end

function _axis1d_dag_use_queue_scheduler()::Bool
    scheduler = _axis1d_dag_scheduler()
    scheduler === :serial && return false
    scheduler === :queue && return Threads.nthreads() > 1
    scheduler === :auto && return Threads.nthreads() > 1
    scheduler === :layer && return false
    return error(
        "Unsupported BNC_SISO_DAG_SCHEDULER=$(scheduler). Use auto, serial, or queue."
    )
end

function _axis1d_dag_use_layer_scheduler()::Bool
    _axis1d_dag_scheduler() === :layer && return Threads.nthreads() > 1
    return _axis1d_dag_layer_parallel_enabled() && Threads.nthreads() > 1
end

mutable struct Axis1DDAGProgressState
    pair_total::Int
    pair_done::Int
    static_pair_weight::Dict{Axis1DPairKey, Float64}
    weighted_done::Float64
    weighted_total::Float64
    displayed_units::Int
    largest_pair_seconds::Float64
    largest_pair::Axis1DPairKey
    cached_condition_entries::Int
end

mutable struct Axis1DPairSolveStats
    pair_solve_ns::UInt64
    middle_collect_ns::UInt64
    middle_compute_ns::UInt64
    middle_merge_ns::UInt64
    middle_join_pairs::Int
    middle_parallel_nodes::Int
    middle_serial_nodes::Int
end

Axis1DPairSolveStats() = Axis1DPairSolveStats(0, 0, 0, 0, 0, 0, 0)

function _add_pair_solve_stats!(
    profile::Axis1DDAGProfile, stats::Axis1DPairSolveStats
)::Nothing
    profile.pair_solve_ns += stats.pair_solve_ns
    profile.middle_collect_ns += stats.middle_collect_ns
    profile.middle_compute_ns += stats.middle_compute_ns
    profile.middle_merge_ns += stats.middle_merge_ns
    profile.middle_join_pairs += stats.middle_join_pairs
    profile.middle_parallel_nodes += stats.middle_parallel_nodes
    profile.middle_serial_nodes += stats.middle_serial_nodes
    return nothing
end

function _static_pair_weight(plan::Axis1DPairPlan)::Float64
    if plan.branch === :diagonal || plan.branch === :no_bridge
        return 1.0
    end
    return Float64(max(1, length(plan.dependencies)))
end

function Axis1DDAGProgressState(
    scheduled_pairs::AbstractVector{Axis1DPairKey},
    plans::Dict{Axis1DPairKey, Axis1DPairPlan},
)::Axis1DDAGProgressState
    static_pair_weight = Dict{Axis1DPairKey, Float64}()
    weighted_total = 0.0
    for pair in scheduled_pairs
        weight = _static_pair_weight(plans[pair])
        static_pair_weight[pair] = weight
        weighted_total += weight
    end
    return Axis1DDAGProgressState(
        length(scheduled_pairs),
        0,
        static_pair_weight,
        0.0,
        max(weighted_total, 1.0),
        0,
        0.0,
        (0, 0),
        0,
    )
end

@inline function _cached_pair_condition_count(
    helper::Axis1DPairMemoBackend, pair::Axis1DPairKey
)::Int
    cached = lock(helper.cache_lock) do
        get(helper.pair_conditions, pair, nothing)
    end
    return isnothing(cached) ? 0 : length(cached)
end

function _adaptive_pair_weight(helper::Axis1DPairMemoBackend, plan::Axis1DPairPlan)::Float64
    if plan.branch === :diagonal || plan.branch === :no_bridge
        return 1.0
    end
    child_entries = 0
    for dependency in plan.dependencies
        child_entries += _cached_pair_condition_count(helper, dependency)
    end
    return Float64(max(1, length(plan.dependencies) + child_entries))
end

function _progress_showvalues(state::Axis1DDAGProgressState, profile::Axis1DDAGProfile)
    weighted_pct = 100 * state.weighted_done / max(state.weighted_total, 1.0)
    pair_pct = 100 * state.pair_done / max(state.pair_total, 1)
    current_pair = (profile.current_pair_from, profile.current_pair_to)
    largest_pair = state.largest_pair
    return [
        (:weighted, Printf.@sprintf("%.1f%%", weighted_pct)),
        (
            :pairs,
            "$(state.pair_done)/$(state.pair_total) ($(Printf.@sprintf("%.1f%%", pair_pct)))",
        ),
        (
            :current,
            "$(current_pair) $(profile.current_pair_branch) $(Printf.@sprintf("%.1fs", profile.current_pair_elapsed_seconds))",
        ),
        (
            :largest,
            "$(largest_pair) $(Printf.@sprintf("%.1fs", state.largest_pair_seconds))",
        ),
        (:cached_entries, state.cached_condition_entries),
    ]
end

function _begin_weighted_pair!(
    state::Axis1DDAGProgressState,
    profile::Axis1DDAGProfile,
    helper::Axis1DPairMemoBackend,
    pair::Axis1DPairKey,
    plan::Axis1DPairPlan,
)::Float64
    adaptive_weight = _adaptive_pair_weight(helper, plan)
    state.weighted_total += adaptive_weight - state.static_pair_weight[pair]
    from, to = pair
    profile.current_pair_from = from
    profile.current_pair_to = to
    profile.current_pair_branch = plan.branch
    profile.current_pair_weight = adaptive_weight
    profile.current_pair_start_ns = time_ns()
    profile.current_pair_elapsed_seconds = 0.0
    profile.current_pair_output_entries = 0
    return adaptive_weight
end

function _finish_weighted_pair!(
    state::Axis1DDAGProgressState,
    profile::Axis1DDAGProfile,
    progress::ProgressMeter.Progress,
    pair::Axis1DPairKey,
    weight::Float64,
    pair_seconds::Float64,
    output_entries::Int,
)::Nothing
    state.pair_done += 1
    state.weighted_done += weight
    state.cached_condition_entries += output_entries
    if pair_seconds > state.largest_pair_seconds
        state.largest_pair_seconds = pair_seconds
        state.largest_pair = pair
    end

    from, to = pair
    profile.weighted_work_done = state.weighted_done
    profile.weighted_work_total = state.weighted_total
    profile.current_pair_from = from
    profile.current_pair_to = to
    profile.current_pair_start_ns = UInt64(0)
    profile.current_pair_elapsed_seconds = pair_seconds
    profile.current_pair_output_entries = output_entries
    profile.largest_pair_seconds = state.largest_pair_seconds
    profile.largest_pair_from = state.largest_pair[1]
    profile.largest_pair_to = state.largest_pair[2]

    units = floor(
        Int,
        AXIS1D_DAG_PROGRESS_UNITS * state.weighted_done / max(state.weighted_total, 1.0),
    )
    state.displayed_units = max(
        state.displayed_units, min(AXIS1D_DAG_PROGRESS_UNITS, units)
    )
    profile.weighted_progress_units = state.displayed_units
    ProgressMeter.update!(
        progress, state.displayed_units; showvalues=_progress_showvalues(state, profile)
    )
    return nothing
end

function _build_pair_plan!(
    helper::Axis1DPairMemoBackend,
    plans::Dict{Axis1DPairKey, Axis1DPairPlan},
    from::Int,
    to::Int,
)::Axis1DPairPlan
    key = _pair_key(from, to)
    cached = get(plans, key, nothing)
    !isnothing(cached) && return cached

    if from == to
        plan = Axis1DPairPlan(:diagonal, Int[], Int[], Axis1DPairKey[])
        plans[key] = plan
        return plan
    end

    successors = _bridge_successors(helper, from, to)
    predecessors = _bridge_predecessors(helper, from, to)
    if isempty(successors) || isempty(predecessors)
        plan = Axis1DPairPlan(:no_bridge, successors, predecessors, Axis1DPairKey[])
        plans[key] = plan
        return plan
    end

    n_solved_successors = count(
        successor ->
            haskey(plans, _pair_key(successor, to)) ||
                _pair_is_cached(helper, successor, to),
        successors,
    )
    n_solved_predecessors = count(
        predecessor ->
            haskey(plans, _pair_key(from, predecessor)) ||
                _pair_is_cached(helper, from, predecessor),
        predecessors,
    )
    solved_successor_ratio = n_solved_successors / length(successors)
    solved_predecessor_ratio = n_solved_predecessors / length(predecessors)

    if n_solved_successors == 0 && n_solved_predecessors == 0
        dependencies = Axis1DPairKey[]
        plan = Axis1DPairPlan(:middle, successors, predecessors, dependencies)
        plans[key] = plan
        for successor in successors
            for predecessor in predecessors
                child_key = _pair_key(successor, predecessor)
                push!(dependencies, child_key)
                _build_pair_plan!(helper, plans, successor, predecessor)
            end
        end
        return plan
    end

    if solved_successor_ratio > solved_predecessor_ratio
        dependencies = Axis1DPairKey[]
        plan = Axis1DPairPlan(:suffix, successors, predecessors, dependencies)
        plans[key] = plan
        for successor in successors
            child_key = _pair_key(successor, to)
            push!(dependencies, child_key)
            _build_pair_plan!(helper, plans, successor, to)
        end
        return plan
    end

    dependencies = Axis1DPairKey[]
    plan = Axis1DPairPlan(:prefix, successors, predecessors, dependencies)
    plans[key] = plan
    for predecessor in predecessors
        child_key = _pair_key(from, predecessor)
        push!(dependencies, child_key)
        _build_pair_plan!(helper, plans, from, predecessor)
    end
    return plan
end

function _append_pair_postorder!(
    pair::Axis1DPairKey,
    plans::Dict{Axis1DPairKey, Axis1DPairPlan},
    seen::Set{Axis1DPairKey},
    out::Vector{Axis1DPairKey},
)::Nothing
    pair in seen && return nothing
    push!(seen, pair)
    plan = plans[pair]
    for dependency in plan.dependencies
        _append_pair_postorder!(dependency, plans, seen, out)
    end
    push!(out, pair)
    return nothing
end

function _collect_pair_plan(
    helper::Axis1DPairMemoBackend, roots::AbstractVector{<:Tuple{<:Integer, <:Integer}}
)::Tuple{Dict{Axis1DPairKey, Axis1DPairPlan}, Vector{Axis1DPairKey}}
    plans = Dict{Axis1DPairKey, Axis1DPairPlan}()
    roots_int = Axis1DPairKey[(Int(from), Int(to)) for (from, to) in roots]
    for (from, to) in roots_int
        _build_pair_plan!(helper, plans, from, to)
    end

    order = Axis1DPairKey[]
    seen = Set{Axis1DPairKey}()
    for root in roots_int
        _append_pair_postorder!(root, plans, seen, order)
    end
    return plans, order
end

function _pair_plan_depth!(
    depths::Dict{Axis1DPairKey, Int},
    pair::Axis1DPairKey,
    plans::Dict{Axis1DPairKey, Axis1DPairPlan},
)::Int
    cached = get(depths, pair, nothing)
    cached === nothing || return cached
    plan = plans[pair]
    depth = if isempty(plan.dependencies)
        1
    else
        1 + maximum(_pair_plan_depth!(depths, dep, plans) for dep in plan.dependencies)
    end
    depths[pair] = depth
    return depth
end

function _pair_plan_layers(
    scheduled_pairs::AbstractVector{Axis1DPairKey},
    plans::Dict{Axis1DPairKey, Axis1DPairPlan},
)::Vector{Vector{Axis1DPairKey}}
    depths = Dict{Axis1DPairKey, Int}()
    max_depth = 0
    for pair in scheduled_pairs
        max_depth = max(max_depth, _pair_plan_depth!(depths, pair, plans))
    end

    layers = [Axis1DPairKey[] for _ in 1:max_depth]
    scheduled_set = Set(scheduled_pairs)
    for pair in scheduled_pairs
        push!(layers[depths[pair]], pair)
    end
    for layer in layers
        filter!(pair -> pair in scheduled_set, layer)
    end
    return layers
end

function _prewarm_pair_plan_prisms!(
    helper::Axis1DPairMemoBackend, pair::Axis1DPairKey, plan::Axis1DPairPlan
)::Nothing
    from, to = pair
    if plan.branch === :diagonal
        _axis1d_vertex_condition!(helper, from)
        return nothing
    end

    if _edge_exists(helper, from, to)
        _axis1d_interface_condition!(helper, from, to)
    end

    if plan.branch === :middle
        for successor in plan.successors
            _axis1d_interface_condition!(helper, from, successor)
        end
        for predecessor in plan.predecessors
            _axis1d_interface_condition!(helper, predecessor, to)
        end
    elseif plan.branch === :suffix
        for successor in plan.successors
            _axis1d_interface_condition!(helper, from, successor)
        end
    elseif plan.branch === :prefix
        for predecessor in plan.predecessors
            _axis1d_interface_condition!(helper, predecessor, to)
        end
    end
    return nothing
end

function _prewarm_pair_plan_layer_prisms!(
    helper::Axis1DPairMemoBackend,
    layer::AbstractVector{Axis1DPairKey},
    plans::Dict{Axis1DPairKey, Axis1DPairPlan},
)::Nothing
    for pair in layer
        _prewarm_pair_plan_prisms!(helper, pair, plans[pair])
    end
    return nothing
end

function _prewarm_pair_plan_prisms!(
    helper::Axis1DPairMemoBackend,
    scheduled_pairs::AbstractVector{Axis1DPairKey},
    plans::Dict{Axis1DPairKey, Axis1DPairPlan},
)::Nothing
    for pair in scheduled_pairs
        _prewarm_pair_plan_prisms!(helper, pair, plans[pair])
    end
    return nothing
end

function _merge_middle_join_local!(
    conditions::Axis1DPathConditionMap,
    helper::Axis1DPairMemoBackend,
    from::Int,
    successor::Int,
    predecessor::Int,
    to::Int,
    stats::Axis1DPairSolveStats,
)::Axis1DPathConditionMap
    stats.middle_join_pairs += 1
    start_ns = time_ns()
    left_condition = _axis1d_interface_condition!(helper, from, successor)
    isempty(left_condition) && return conditions
    right_condition = _axis1d_interface_condition!(helper, predecessor, to)
    isempty(right_condition) && return conditions
    middle_conditions = _pair_conditions(helper, successor, predecessor)
    middle_conditions === nothing &&
        error("Missing cached middle condition for pair ($(successor), $(predecessor)).")
    isempty(middle_conditions) && return conditions

    for (middle_path, middle_condition) in middle_conditions
        full_condition = _axis1d_intersect_nonempty(
            left_condition, middle_condition, right_condition
        )
        isnothing(full_condition) && continue
        conditions[_wrap_vertices(from, middle_path, to)] = full_condition
    end
    stats.middle_compute_ns += time_ns() - start_ns
    return conditions
end

function _merge_middle_join_chunk_indices_local!(
    conditions::Axis1DPathConditionMap,
    helper::Axis1DPairMemoBackend,
    from::Int,
    child_pairs::AbstractVector{<:Tuple{Int, Int}},
    chunk_indices::AbstractVector{Int},
    to::Int,
    stats::Axis1DPairSolveStats,
)::Axis1DPathConditionMap
    for idx in chunk_indices
        successor, predecessor = child_pairs[idx]
        _merge_middle_join_local!(
            conditions, helper, from, successor, predecessor, to, stats
        )
    end
    return conditions
end

function _middle_join_weighted_chunks(
    helper::Axis1DPairMemoBackend,
    child_pairs::AbstractVector{<:Tuple{Int, Int}},
    target_entries::Int=_axis1d_dag_inner_parallel_target_entries(),
    max_chunks::Int=_axis1d_dag_inner_parallel_max_chunks(),
)::Vector{Vector{Int}}
    n_items = length(child_pairs)
    n_items == 0 && return Vector{Int}[]

    weights = [max(1, _cached_pair_condition_count(helper, pair)) for pair in child_pairs]
    total_weight = sum(weights; init=0)
    target_entries = max(1, target_entries)
    max_chunks = min(n_items, max(1, max_chunks))
    n_chunks = min(max_chunks, max(1, cld(total_weight, target_entries)))

    chunks = [Int[] for _ in 1:n_chunks]
    chunk_loads = zeros(Int, n_chunks)
    for idx in sortperm(weights; rev=true)
        chunk_idx = argmin(chunk_loads)
        push!(chunks[chunk_idx], idx)
        chunk_loads[chunk_idx] += weights[idx]
    end
    return filter!(!isempty, chunks)
end

function _middle_join_chunk_entry_loads(
    helper::Axis1DPairMemoBackend,
    child_pairs::AbstractVector{<:Tuple{Int, Int}},
    chunks::AbstractVector{<:AbstractVector{Int}},
)::Vector{Int}
    weights = [max(1, _cached_pair_condition_count(helper, pair)) for pair in child_pairs]
    return [sum((weights[idx] for idx in chunk); init=0) for chunk in chunks]
end

function _queue_estimator_target_entries(profile::Axis1DDAGProfile)::Int
    if haskey(ENV, "BNC_SISO_DAG_INNER_PARALLEL_TARGET_ENTRIES")
        return _axis1d_dag_inner_parallel_target_entries()
    end
    target_entries = round(
        Int, profile.queue_estimator_entries_per_second * _axis1d_dag_target_chunk_seconds()
    )
    return max(1, target_entries)
end

function _queue_estimator_min_parallel_entries(target_entries::Int)::Float64
    return max(_axis1d_dag_inner_parallel_min_weight(), 4.0 * target_entries)
end

function _update_queue_chunk_rate!(
    profile::Axis1DDAGProfile, estimated_entries::Int, chunk_seconds::Float64
)::Nothing
    estimated_entries > 0 || return nothing
    chunk_seconds > 0 || return nothing
    sample_rate = estimated_entries / chunk_seconds
    alpha = _axis1d_dag_chunk_rate_alpha()
    profile.queue_estimator_entries_per_second =
        (1 - alpha) * profile.queue_estimator_entries_per_second + alpha * sample_rate
    return nothing
end

function _collect_middle_join_pairs(
    helper::Axis1DPairMemoBackend,
    from::Int,
    to::Int,
    successors::AbstractVector{Int},
    predecessors::AbstractVector{Int},
)::Vector{Tuple{Int, Int}}
    child_pairs = Tuple{Int, Int}[]
    for successor in successors
        left_condition = _axis1d_interface_condition!(helper, from, successor)
        isempty(left_condition) && continue
        for predecessor in predecessors
            right_condition = _axis1d_interface_condition!(helper, predecessor, to)
            isempty(right_condition) && continue
            middle_conditions = _pair_conditions(helper, successor, predecessor)
            middle_conditions === nothing && error(
                "Missing cached middle condition for pair ($(successor), $(predecessor)).",
            )
            isempty(middle_conditions) && continue
            push!(child_pairs, (successor, predecessor))
        end
    end
    return child_pairs
end

function _compute_pair_plan_conditions!(
    helper::Axis1DPairMemoBackend,
    from::Int,
    to::Int,
    plan::Axis1DPairPlan,
    stats::Axis1DPairSolveStats;
    use_inner_parallel::Bool=true,
)::Axis1DPathConditionMap
    conditions = Axis1DPathConditionMap()
    if plan.branch === :diagonal
        condition = _axis1d_vertex_condition!(helper, from)
        isempty(condition) || (conditions[(from,)] = condition)
        return conditions
    end

    _maybe_store_direct_path!(conditions, helper, from, to)
    if plan.branch === :no_bridge
        return conditions
    end

    if plan.branch === :middle
        collect_start_ns = time_ns()
        child_pairs = _collect_middle_join_pairs(
            helper, from, to, plan.successors, plan.predecessors
        )
        stats.middle_collect_ns += time_ns() - collect_start_ns

        if use_inner_parallel &&
            Threads.nthreads() > 1 &&
            length(child_pairs) >= AXIS1D_DAG_MIDDLE_PARALLEL_THRESHOLD
            stats.middle_parallel_nodes += 1
            chunks = _middle_join_weighted_chunks(helper, child_pairs)
            local_maps = [Axis1DPathConditionMap() for _ in eachindex(chunks)]
            local_stats = [Axis1DPairSolveStats() for _ in eachindex(chunks)]
            merge_start_ns = time_ns()
            Threads.@threads :dynamic for chunk_idx in eachindex(chunks)
                _merge_middle_join_chunk_indices_local!(
                    local_maps[chunk_idx],
                    helper,
                    from,
                    child_pairs,
                    chunks[chunk_idx],
                    to,
                    local_stats[chunk_idx],
                )
            end
            for local_map in local_maps
                merge!(conditions, local_map)
            end
            stats.middle_merge_ns += time_ns() - merge_start_ns
            for chunk_stats in local_stats
                stats.middle_compute_ns += chunk_stats.middle_compute_ns
                stats.middle_join_pairs += chunk_stats.middle_join_pairs
            end
        else
            stats.middle_serial_nodes += 1
            for (successor, predecessor) in child_pairs
                _merge_middle_join_local!(
                    conditions, helper, from, successor, predecessor, to, stats
                )
            end
        end
        return conditions
    end

    if plan.branch === :suffix
        for successor in plan.successors
            suffix_conditions = _pair_conditions(helper, successor, to)
            suffix_conditions === nothing &&
                error("Missing cached suffix condition for pair ($(successor), $(to)).")
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
        return conditions
    end

    if plan.branch === :prefix
        for predecessor in plan.predecessors
            prefix_conditions = _pair_conditions(helper, from, predecessor)
            prefix_conditions === nothing &&
                error("Missing cached prefix condition for pair ($(from), $(predecessor)).")
            isempty(prefix_conditions) && continue
            right_condition = _axis1d_interface_condition!(helper, predecessor, to)
            isempty(right_condition) && continue
            for (prefix_path, prefix_condition) in prefix_conditions
                full_condition = _axis1d_intersect_nonempty(
                    prefix_condition, right_condition
                )
                isnothing(full_condition) && continue
                conditions[_append_vertex(prefix_path, to)] = full_condition
            end
        end
        return conditions
    end

    return error(
        "Unsupported axis-1D pair plan branch $(plan.branch) for pair ($(from), $(to))."
    )
end

function _solve_pair_plan!(
    helper::Axis1DPairMemoBackend, from::Int, to::Int, plan::Axis1DPairPlan
)::Axis1DPathConditionMap
    pair_start_ns = time_ns()
    cached = _pair_conditions(helper, from, to)
    !isnothing(cached) && return cached
    profile = helper.dag_profile
    profile !== nothing && (profile.pair_solve_calls += 1)

    stats = Axis1DPairSolveStats()
    conditions = _compute_pair_plan_conditions!(helper, from, to, plan, stats)
    stats.pair_solve_ns += time_ns() - pair_start_ns
    profile !== nothing && _add_pair_solve_stats!(profile, stats)
    return _cache_pair_conditions!(helper, from, to, conditions)
end

function _pair_plan_dependents(
    scheduled_pairs::AbstractVector{Axis1DPairKey},
    plans::Dict{Axis1DPairKey, Axis1DPairPlan},
)::Tuple{Dict{Axis1DPairKey, Int}, Vector{Vector{Int}}, Vector{Int}}
    pair_index = Dict{Axis1DPairKey, Int}(
        pair => idx for (idx, pair) in enumerate(scheduled_pairs)
    )
    dependents = [Int[] for _ in eachindex(scheduled_pairs)]
    remaining_deps = zeros(Int, length(scheduled_pairs))
    for (idx, pair) in enumerate(scheduled_pairs)
        for dependency in plans[pair].dependencies
            dep_idx = get(pair_index, dependency, nothing)
            dep_idx === nothing &&
                error("Missing scheduled dependency $(dependency) for pair $(pair).")
            push!(dependents[dep_idx], idx)
            remaining_deps[idx] += 1
        end
    end
    return pair_index, dependents, remaining_deps
end

function _solve_pair_plan_queue!(
    helper::Axis1DPairMemoBackend,
    plans::Dict{Axis1DPairKey, Axis1DPairPlan},
    scheduled_pairs::AbstractVector{Axis1DPairKey},
    progress_state::Axis1DDAGProgressState,
    progress::ProgressMeter.Progress,
)::Nothing
    profile = helper.dag_profile
    profile === nothing && error("DAG profile must be initialized before queue solving.")

    _prewarm_pair_plan_prisms!(helper, scheduled_pairs, plans)
    _, dependents, remaining_deps = _pair_plan_dependents(scheduled_pairs, plans)
    n_pairs = length(scheduled_pairs)
    n_workers = Threads.nthreads()
    # A worker may enqueue a complete chunk fan-out before returning to
    # `take!`. Reserve enough space for every worker to do so concurrently,
    # plus dependency and finalization/stop notifications.
    ready_capacity = n_pairs + n_workers * (_axis1d_dag_inner_parallel_max_chunks() + 2)
    ready = Channel{Tuple{Symbol, Int, Int}}(ready_capacity)
    scheduler_lock = ReentrantLock()
    progress_lock = ReentrantLock()
    completed = Ref(0)
    ready_pair_count = Ref(0)
    pair_weights = zeros(Float64, n_pairs)
    pair_start_ns = zeros(UInt64, n_pairs)
    pair_base_conditions = Vector{Union{Nothing, Axis1DPathConditionMap}}(nothing, n_pairs)
    pair_stats = [Axis1DPairSolveStats() for _ in 1:n_pairs]
    chunk_pairs_by_pair = Vector{Any}(nothing, n_pairs)
    chunk_indices_by_pair = Vector{Any}(nothing, n_pairs)
    chunk_loads_by_pair = Vector{Any}(nothing, n_pairs)
    chunk_maps_by_pair = Vector{Any}(nothing, n_pairs)
    chunk_stats_by_pair = Vector{Any}(nothing, n_pairs)
    chunks_remaining = zeros(Int, n_pairs)

    for idx in eachindex(scheduled_pairs)
        if remaining_deps[idx] == 0
            ready_pair_count[] += 1
            put!(ready, (:pair, idx, 0))
        end
    end

    function enqueue_dependents_or_stop!(idx::Int)::Nothing
        newly_ready = Int[]
        should_stop = false
        lock(scheduler_lock) do
            completed[] += 1
            for dependent_idx in dependents[idx]
                remaining_deps[dependent_idx] -= 1
                if remaining_deps[dependent_idx] == 0
                    ready_pair_count[] += 1
                    push!(newly_ready, dependent_idx)
                end
            end
            should_stop = completed[] == n_pairs
        end
        for ready_idx in newly_ready
            put!(ready, (:pair, ready_idx, 0))
        end
        if should_stop
            for _ in 1:n_workers
                put!(ready, (:stop, 0, 0))
            end
        end
        return nothing
    end

    function finish_queue_pair!(
        idx::Int,
        pair_seconds::Float64,
        output_entries::Int,
        stats::Axis1DPairSolveStats;
        conditions::Union{Nothing, Axis1DPathConditionMap}=nothing,
        was_cached::Bool=false,
    )::Nothing
        pair = scheduled_pairs[idx]
        from, to = pair
        if !was_cached
            conditions === nothing &&
                error("Missing computed conditions for queued pair ($(from), $(to)).")
            _cache_pair_conditions!(helper, from, to, conditions)
        end
        lock(progress_lock) do
            if !was_cached
                profile.pair_solve_calls += 1
                _add_pair_solve_stats!(profile, stats)
            end
            _finish_weighted_pair!(
                progress_state,
                profile,
                progress,
                pair,
                pair_weights[idx],
                pair_seconds,
                output_entries,
            )
        end
        enqueue_dependents_or_stop!(idx)
        return nothing
    end

    function chunk_estimator_params()::Tuple{Int, Float64}
        lock(progress_lock) do
            target_entries = _queue_estimator_target_entries(profile)
            profile.queue_estimator_target_entries = target_entries
            profile.queue_estimator_target_seconds = _axis1d_dag_target_chunk_seconds()
            profile.queue_estimator_min_parallel_entries = _queue_estimator_min_parallel_entries(
                target_entries
            )
            return target_entries, profile.queue_estimator_min_parallel_entries
        end
    end

    function should_chunk_pair(
        idx::Int, plan::Axis1DPairPlan, target_entries::Int, min_entries::Float64
    )::Bool
        _axis1d_dag_chunk_queue_enabled() || return false
        plan.branch === :middle || return false
        lock(progress_lock) do
            profile.queue_chunk_candidate_pairs += 1
        end
        if _axis1d_dag_chunk_size_gate_enabled() && pair_weights[idx] < min_entries
            lock(progress_lock) do
                profile.queue_chunk_size_gate_skips += 1
            end
            return false
        end
        if _axis1d_dag_chunk_width_gate_enabled()
            queued_pairs = lock(scheduler_lock) do
                ready_pair_count[]
            end
            if queued_pairs > _axis1d_dag_chunk_width_factor() * n_workers
                lock(progress_lock) do
                    profile.queue_chunk_width_gate_skips += 1
                end
                return false
            end
        end
        if _axis1d_dag_chunk_thread_gate_enabled() &&
            cld(max(1, round(Int, pair_weights[idx])), target_entries) <= 1
            lock(progress_lock) do
                profile.queue_chunk_thread_gate_skips += 1
            end
            return false
        end
        return true
    end

    @info "Solving DAG pair plans with a global dependency queue across $(n_workers) worker threads."
    @sync for _ in 1:n_workers
        Threads.@spawn begin
            while true
                task = take!(ready)
                kind, idx, chunk_idx = task
                kind === :stop && break

                if kind === :chunk
                    pair = scheduled_pairs[idx]
                    from, to = pair
                    child_pairs = chunk_pairs_by_pair[idx]::Vector{Tuple{Int, Int}}
                    chunk_maps = chunk_maps_by_pair[idx]::Vector{
                        Union{Nothing, Axis1DPathConditionMap}
                    }
                    stats_vec = chunk_stats_by_pair[idx]::Vector{Axis1DPairSolveStats}
                    chunk_loads = chunk_loads_by_pair[idx]::Vector{Int}
                    chunk_indices = (chunk_indices_by_pair[idx]::Vector{Vector{Int}})[chunk_idx]
                    local_map = Axis1DPathConditionMap()
                    local_stats = Axis1DPairSolveStats()
                    chunk_start_ns = time_ns()
                    _merge_middle_join_chunk_indices_local!(
                        local_map, helper, from, child_pairs, chunk_indices, to, local_stats
                    )
                    chunk_seconds = (time_ns() - chunk_start_ns) / 1e9
                    lock(scheduler_lock) do
                        chunk_maps[chunk_idx] = local_map
                        stats_vec[chunk_idx] = local_stats
                        chunks_remaining[idx] -= 1
                        chunks_remaining[idx] == 0 && put!(ready, (:finalize, idx, 0))
                    end
                    lock(progress_lock) do
                        profile.queue_chunk_tasks += 1
                        profile.queue_max_chunk_seconds = max(
                            profile.queue_max_chunk_seconds, chunk_seconds
                        )
                        profile.queue_total_chunk_seconds += chunk_seconds
                        _update_queue_chunk_rate!(
                            profile, chunk_loads[chunk_idx], chunk_seconds
                        )
                    end
                    continue
                end

                if kind === :finalize
                    finalize_start_ns = time_ns()
                    conditions = pair_base_conditions[idx]
                    conditions === nothing && (conditions = Axis1DPathConditionMap())
                    local_maps = chunk_maps_by_pair[idx]::Vector{
                        Union{Nothing, Axis1DPathConditionMap}
                    }
                    local_stats = chunk_stats_by_pair[idx]::Vector{Axis1DPairSolveStats}
                    stats = pair_stats[idx]
                    for local_map in local_maps
                        local_map === nothing && error(
                            "Missing queued chunk result for pair $(scheduled_pairs[idx]).",
                        )
                        merge!(conditions, local_map)
                    end
                    for chunk_stats in local_stats
                        stats.middle_compute_ns += chunk_stats.middle_compute_ns
                        stats.middle_join_pairs += chunk_stats.middle_join_pairs
                    end
                    pair_seconds = (time_ns() - pair_start_ns[idx]) / 1e9
                    stats.pair_solve_ns += round(UInt64, pair_seconds * 1e9)
                    lock(progress_lock) do
                        profile.queue_finalize_tasks += 1
                        profile.queue_finalize_ns += time_ns() - finalize_start_ns
                    end
                    finish_queue_pair!(
                        idx, pair_seconds, length(conditions), stats; conditions
                    )
                    continue
                end

                pair = scheduled_pairs[idx]
                from, to = pair
                plan = plans[pair]
                lock(scheduler_lock) do
                    ready_pair_count[] = max(0, ready_pair_count[] - 1)
                end
                lock(progress_lock) do
                    profile.queue_pair_tasks += 1
                end
                pair_weights[idx] = lock(progress_lock) do
                    _begin_weighted_pair!(progress_state, profile, helper, pair, plan)
                end

                cached = _pair_conditions(helper, from, to)
                if !isnothing(cached)
                    finish_queue_pair!(
                        idx, 0.0, length(cached), Axis1DPairSolveStats(); was_cached=true
                    )
                    continue
                end

                target_entries, min_entries = chunk_estimator_params()
                if should_chunk_pair(idx, plan, target_entries, min_entries)
                    pair_start_ns[idx] = time_ns()
                    stats = pair_stats[idx]
                    base_conditions = Axis1DPathConditionMap()
                    _maybe_store_direct_path!(base_conditions, helper, from, to)
                    pair_base_conditions[idx] = base_conditions
                    collect_start_ns = time_ns()
                    child_pairs = _collect_middle_join_pairs(
                        helper, from, to, plan.successors, plan.predecessors
                    )
                    stats.middle_collect_ns += time_ns() - collect_start_ns
                    if length(child_pairs) >= AXIS1D_DAG_MIDDLE_PARALLEL_THRESHOLD
                        chunks = _middle_join_weighted_chunks(
                            helper,
                            child_pairs,
                            target_entries,
                            _axis1d_dag_inner_parallel_max_chunks(),
                        )
                        if length(chunks) > 1
                            stats.middle_parallel_nodes += 1
                            chunk_loads = _middle_join_chunk_entry_loads(
                                helper, child_pairs, chunks
                            )
                            chunk_pairs_by_pair[idx] = child_pairs
                            chunk_indices_by_pair[idx] = chunks
                            chunk_loads_by_pair[idx] = chunk_loads
                            chunk_maps_by_pair[idx] = Vector{
                                Union{Nothing, Axis1DPathConditionMap}
                            }(
                                nothing, length(chunks)
                            )
                            chunk_stats_by_pair[idx] = [
                                Axis1DPairSolveStats() for _ in eachindex(chunks)
                            ]
                            lock(scheduler_lock) do
                                chunks_remaining[idx] = length(chunks)
                            end
                            lock(progress_lock) do
                                profile.queue_chunked_pairs += 1
                                profile.queue_max_chunks_per_pair = max(
                                    profile.queue_max_chunks_per_pair, length(chunks)
                                )
                                if !isempty(chunk_loads)
                                    profile.queue_max_chunk_estimated_entries = max(
                                        profile.queue_max_chunk_estimated_entries,
                                        maximum(chunk_loads),
                                    )
                                    profile.queue_total_chunk_estimated_entries += sum(
                                        chunk_loads; init=0
                                    )
                                end
                            end
                            for chunk_task_idx in eachindex(chunks)
                                put!(ready, (:chunk, idx, chunk_task_idx))
                            end
                            continue
                        end
                    end
                end

                stats = Axis1DPairSolveStats()
                pair_start = time_ns()
                conditions = _compute_pair_plan_conditions!(
                    helper, from, to, plan, stats; use_inner_parallel=false
                )
                pair_seconds = (time_ns() - pair_start) / 1e9
                stats.pair_solve_ns += round(UInt64, pair_seconds * 1e9)
                finish_queue_pair!(idx, pair_seconds, length(conditions), stats; conditions)
            end
        end
    end
    return nothing
end

function _solve_pair_plan_layers!(
    helper::Axis1DPairMemoBackend,
    plans::Dict{Axis1DPairKey, Axis1DPairPlan},
    scheduled_pairs::AbstractVector{Axis1DPairKey},
    progress_state::Axis1DDAGProgressState,
    progress::ProgressMeter.Progress,
)::Nothing
    layers = _pair_plan_layers(scheduled_pairs, plans)
    profile = helper.dag_profile
    profile === nothing &&
        error("DAG profile must be initialized before layer-parallel solving.")

    @info "Solving DAG pair plans across $(length(layers)) dependency layers with staged layer-parallel commits."
    for (layer_idx, layer) in enumerate(layers)
        isempty(layer) && continue
        _prewarm_pair_plan_layer_prisms!(helper, layer, plans)

        n_layer_pairs = length(layer)
        conditions_by_pair = Vector{Union{Nothing, Axis1DPathConditionMap}}(
            nothing, n_layer_pairs
        )
        stats_by_pair = [Axis1DPairSolveStats() for _ in 1:n_layer_pairs]
        seconds_by_pair = zeros(Float64, n_layer_pairs)
        output_entries = zeros(Int, n_layer_pairs)
        pair_weights = zeros(Float64, n_layer_pairs)
        was_cached = falses(n_layer_pairs)

        for idx in eachindex(layer)
            pair = layer[idx]
            pair_weights[idx] = _begin_weighted_pair!(
                progress_state, profile, helper, pair, plans[pair]
            )
            cached = _pair_conditions(helper, pair[1], pair[2])
            if !isnothing(cached)
                was_cached[idx] = true
                output_entries[idx] = length(cached)
            end
        end

        inner_parallel_idxs = Set{Int}()
        if n_layer_pairs <= _axis1d_dag_layer_inner_parallel_width()
            union!(inner_parallel_idxs, eachindex(layer))
        else
            max_inner_pairs = _axis1d_dag_inner_parallel_max_pairs_per_layer()
            min_inner_weight = _axis1d_dag_inner_parallel_min_weight()
            if max_inner_pairs > 0
                candidates = Int[
                    idx for idx in eachindex(layer) if !was_cached[idx] &&
                    plans[layer[idx]].branch === :middle &&
                    pair_weights[idx] >= min_inner_weight
                ]
                sort!(candidates; by=idx -> pair_weights[idx], rev=true)
                for idx in Iterators.take(candidates, max_inner_pairs)
                    push!(inner_parallel_idxs, idx)
                end
            end
        end

        for idx in sort!(collect(inner_parallel_idxs))
            pair = layer[idx]
            from, to = pair
            if was_cached[idx]
                _finish_weighted_pair!(
                    progress_state,
                    profile,
                    progress,
                    pair,
                    pair_weights[idx],
                    seconds_by_pair[idx],
                    output_entries[idx],
                )
                continue
            end

            plan = plans[pair]
            stats = stats_by_pair[idx]
            pair_start_ns = time_ns()
            conditions = _compute_pair_plan_conditions!(
                helper, from, to, plan, stats; use_inner_parallel=true
            )
            pair_seconds = (time_ns() - pair_start_ns) / 1e9
            stats.pair_solve_ns += round(UInt64, pair_seconds * 1e9)
            conditions_by_pair[idx] = conditions
            seconds_by_pair[idx] = pair_seconds
            output_entries[idx] = length(conditions)
            _cache_pair_conditions!(helper, from, to, conditions)
            profile.pair_solve_calls += 1
            _add_pair_solve_stats!(profile, stats)
            _finish_weighted_pair!(
                progress_state,
                profile,
                progress,
                pair,
                pair_weights[idx],
                pair_seconds,
                output_entries[idx],
            )
        end

        outer_parallel_idxs = Int[
            idx for idx in eachindex(layer) if !(idx in inner_parallel_idxs)
        ]

        Threads.@threads :dynamic for idx_idx in eachindex(outer_parallel_idxs)
            idx = outer_parallel_idxs[idx_idx]
            was_cached[idx] && continue
            pair = layer[idx]
            from, to = pair
            plan = plans[pair]
            stats = stats_by_pair[idx]
            pair_start_ns = time_ns()
            conditions = _compute_pair_plan_conditions!(
                helper, from, to, plan, stats; use_inner_parallel=false
            )
            pair_seconds = (time_ns() - pair_start_ns) / 1e9
            stats.pair_solve_ns += round(UInt64, pair_seconds * 1e9)
            conditions_by_pair[idx] = conditions
            seconds_by_pair[idx] = pair_seconds
            output_entries[idx] = length(conditions)
        end

        for idx in outer_parallel_idxs
            pair = layer[idx]
            from, to = pair
            if !was_cached[idx]
                conditions = conditions_by_pair[idx]
                conditions === nothing && error(
                    "Missing computed conditions for pair ($(from), $(to)) in layer $(layer_idx).",
                )
                _cache_pair_conditions!(helper, from, to, conditions)
                profile.pair_solve_calls += 1
                _add_pair_solve_stats!(profile, stats_by_pair[idx])
            end
            _finish_weighted_pair!(
                progress_state,
                profile,
                progress,
                pair,
                pair_weights[idx],
                seconds_by_pair[idx],
                output_entries[idx],
            )
        end
    end
    return nothing
end

"""
    _find_all_path_conditions_dag!(helper) -> Axis1DPairMemoBackend

Selectively discover the same pair subproblems the recursive solver touches,
then solve them bottom-up. Heavy middle-join nodes are split into parallel
local join tasks while keeping pair ownership unique.
"""
function _find_all_path_conditions_dag!(
    helper::Axis1DPairMemoBackend,
    pair_queries::AbstractVector{<:Tuple{<:Integer, <:Integer}},
)::Axis1DPairMemoBackend
    helper.dag_profile = Axis1DDAGProfile()
    planning_start_ns = time_ns()
    plans, scheduled_pairs = _collect_pair_plan(helper, pair_queries)
    helper.dag_profile.planning_ns += time_ns() - planning_start_ns
    helper.dag_profile.planned_pairs = length(scheduled_pairs)
    isempty(scheduled_pairs) && return helper

    if length(scheduled_pairs) == 1
        pair = only(scheduled_pairs)
        from, to = pair
        plan = plans[pair]
        pair_weight = _adaptive_pair_weight(helper, plan)
        helper.dag_profile.weighted_work_total = pair_weight
        helper.dag_profile.current_pair_from = from
        helper.dag_profile.current_pair_to = to
        helper.dag_profile.current_pair_branch = plan.branch
        helper.dag_profile.current_pair_weight = pair_weight
        helper.dag_profile.current_pair_start_ns = time_ns()
        pair_start_ns = time_ns()
        conditions = _solve_pair_plan!(helper, from, to, plan)
        pair_seconds = (time_ns() - pair_start_ns) / 1e9
        helper.dag_profile.weighted_work_done = pair_weight
        helper.dag_profile.weighted_progress_units = AXIS1D_DAG_PROGRESS_UNITS
        helper.dag_profile.current_pair_start_ns = UInt64(0)
        helper.dag_profile.current_pair_elapsed_seconds = pair_seconds
        helper.dag_profile.current_pair_output_entries = length(conditions)
        helper.dag_profile.largest_pair_seconds = pair_seconds
        helper.dag_profile.largest_pair_from = from
        helper.dag_profile.largest_pair_to = to
        return helper
    end

    @info "Start finding all possible path conditions across $(length(scheduled_pairs)) selectively planned DAG pairs."
    progress_state = Axis1DDAGProgressState(scheduled_pairs, plans)
    helper.dag_profile.weighted_work_total = progress_state.weighted_total
    progress = ProgressMeter.Progress(
        AXIS1D_DAG_PROGRESS_UNITS;
        dt=1.0,
        desc="Finding path conditions (weighted)",
        showspeed=true,
    )
    ProgressMeter.update!(
        progress, 0; showvalues=_progress_showvalues(progress_state, helper.dag_profile)
    )
    if _axis1d_dag_pair_queue_enabled() || _axis1d_dag_use_queue_scheduler()
        _solve_pair_plan_queue!(helper, plans, scheduled_pairs, progress_state, progress)
    elseif _axis1d_dag_use_layer_scheduler()
        _solve_pair_plan_layers!(helper, plans, scheduled_pairs, progress_state, progress)
    else
        for (from, to) in scheduled_pairs
            pair = (from, to)
            plan = plans[pair]
            pair_weight = _begin_weighted_pair!(
                progress_state, helper.dag_profile, helper, pair, plan
            )
            pair_start_ns = time_ns()
            conditions = _solve_pair_plan!(helper, from, to, plan)
            pair_seconds = (time_ns() - pair_start_ns) / 1e9
            _finish_weighted_pair!(
                progress_state,
                helper.dag_profile,
                progress,
                pair,
                pair_weight,
                pair_seconds,
                length(conditions),
            )
        end
    end
    ProgressMeter.finish!(
        progress; showvalues=_progress_showvalues(progress_state, helper.dag_profile)
    )
    return helper
end

function _find_all_path_conditions_dag!(
    helper::Axis1DPairMemoBackend
)::Axis1DPairMemoBackend
    pair_queries = Tuple{Int, Int}[
        (source, sink) for source in helper.problem.dag.sources for
        sink in helper.problem.dag.sinks
    ]
    isempty(pair_queries) && return helper
    return _find_all_path_conditions_dag!(helper, pair_queries)
end

function get_path_conditions(helper::Axis1DPairMemoBackend, from::Integer, to::Integer)
    return _find_pair_path_conditions!(helper, Int(from), Int(to))
end
