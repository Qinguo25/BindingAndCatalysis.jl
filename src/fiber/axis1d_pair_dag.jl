struct Axis1DPairPlan
    branch::Symbol
    successors::Vector{Int}
    predecessors::Vector{Int}
    dependencies::Vector{Axis1DPairKey}
end

const AXIS1D_DAG_MIDDLE_PARALLEL_THRESHOLD = 8

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
        for successor in successors, predecessor in predecessors
            child_key = _pair_key(successor, predecessor)
            push!(dependencies, child_key)
            _build_pair_plan!(helper, plans, successor, predecessor)
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
    for dependency in plans[pair].dependencies
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
    dependencies = plans[pair].dependencies
    depth = if isempty(dependencies)
        1
    else
        1 + maximum(_pair_plan_depth!(depths, dep, plans) for dep in dependencies)
    end
    depths[pair] = depth
    return depth
end

function _pair_plan_layers(
    scheduled_pairs::AbstractVector{Axis1DPairKey},
    plans::Dict{Axis1DPairKey, Axis1DPairPlan},
)::Vector{Vector{Axis1DPairKey}}
    isempty(scheduled_pairs) && return Vector{Vector{Axis1DPairKey}}()
    depths = Dict{Axis1DPairKey, Int}()
    max_depth = maximum(_pair_plan_depth!(depths, pair, plans) for pair in scheduled_pairs)
    layers = [Axis1DPairKey[] for _ in 1:max_depth]
    for pair in scheduled_pairs
        push!(layers[depths[pair]], pair)
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

    _edge_exists(helper, from, to) && _axis1d_interface_condition!(helper, from, to)
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

@inline function _cached_pair_condition_count(
    helper::Axis1DPairMemoBackend, pair::Axis1DPairKey
)::Int
    conditions = _pair_conditions(helper, pair...)
    return isnothing(conditions) ? 0 : length(conditions)
end

function _merge_middle_join!(
    conditions::Axis1DPathConditionMap,
    helper::Axis1DPairMemoBackend,
    from::Int,
    successor::Int,
    predecessor::Int,
    to::Int,
)::Axis1DPathConditionMap
    left_condition = _axis1d_interface_condition!(helper, from, successor)
    isempty(left_condition) && return conditions
    right_condition = _axis1d_interface_condition!(helper, predecessor, to)
    isempty(right_condition) && return conditions
    middle_conditions = _pair_conditions(helper, successor, predecessor)
    middle_conditions === nothing &&
        error("Missing cached middle condition for pair ($(successor), $(predecessor)).")

    for (middle_path, middle_condition) in middle_conditions
        full_condition = _axis1d_intersect_nonempty(
            left_condition, middle_condition, right_condition
        )
        isnothing(full_condition) && continue
        conditions[_wrap_vertices(from, middle_path, to)] = full_condition
    end
    return conditions
end

function _collect_middle_join_pairs(
    helper::Axis1DPairMemoBackend,
    from::Int,
    to::Int,
    successors::AbstractVector{Int},
    predecessors::AbstractVector{Int},
)::Vector{Axis1DPairKey}
    child_pairs = Axis1DPairKey[]
    for successor in successors
        isempty(_axis1d_interface_condition!(helper, from, successor)) && continue
        for predecessor in predecessors
            isempty(_axis1d_interface_condition!(helper, predecessor, to)) && continue
            middle_conditions = _pair_conditions(helper, successor, predecessor)
            middle_conditions === nothing && error(
                "Missing cached middle condition for pair ($(successor), $(predecessor)).",
            )
            isempty(middle_conditions) || push!(child_pairs, (successor, predecessor))
        end
    end
    return child_pairs
end

function _middle_join_weighted_chunks(
    helper::Axis1DPairMemoBackend, child_pairs::AbstractVector{Axis1DPairKey}
)::Vector{Vector{Int}}
    n_chunks = min(length(child_pairs), max(1, Threads.nthreads()))
    n_chunks == 0 && return Vector{Int}[]

    weights = [max(1, _cached_pair_condition_count(helper, pair)) for pair in child_pairs]
    chunks = [Int[] for _ in 1:n_chunks]
    loads = zeros(Int, n_chunks)
    for index in sortperm(weights; rev=true)
        chunk = argmin(loads)
        push!(chunks[chunk], index)
        loads[chunk] += weights[index]
    end
    return chunks
end

function _compute_pair_plan_conditions!(
    helper::Axis1DPairMemoBackend,
    from::Int,
    to::Int,
    plan::Axis1DPairPlan;
    use_inner_parallel::Bool,
)::Axis1DPathConditionMap
    conditions = Axis1DPathConditionMap()
    if plan.branch === :diagonal
        condition = _axis1d_vertex_condition!(helper, from)
        isempty(condition) || (conditions[(from,)] = condition)
        return conditions
    end

    _maybe_store_direct_path!(conditions, helper, from, to)
    plan.branch === :no_bridge && return conditions

    if plan.branch === :middle
        child_pairs = _collect_middle_join_pairs(
            helper, from, to, plan.successors, plan.predecessors
        )
        if use_inner_parallel &&
            Threads.nthreads() > 1 &&
            length(child_pairs) >= AXIS1D_DAG_MIDDLE_PARALLEL_THRESHOLD
            chunks = _middle_join_weighted_chunks(helper, child_pairs)
            local_maps = [Axis1DPathConditionMap() for _ in chunks]
            Threads.@threads :dynamic for chunk_index in eachindex(chunks)
                local_map = local_maps[chunk_index]
                for pair_index in chunks[chunk_index]
                    successor, predecessor = child_pairs[pair_index]
                    _merge_middle_join!(local_map, helper, from, successor, predecessor, to)
                end
            end
            for local_map in local_maps
                merge!(conditions, local_map)
            end
        else
            for (successor, predecessor) in child_pairs
                _merge_middle_join!(conditions, helper, from, successor, predecessor, to)
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

function _compute_planned_pair(
    helper::Axis1DPairMemoBackend,
    pair::Axis1DPairKey,
    plan::Axis1DPairPlan;
    use_inner_parallel::Bool,
)::Axis1DPathConditionMap
    cached = _pair_conditions(helper, pair...)
    return if isnothing(cached)
        _compute_pair_plan_conditions!(
            helper, pair[1], pair[2], plan; use_inner_parallel=use_inner_parallel
        )
    else
        cached
    end
end

function _solve_pair_plan_layer!(
    helper::Axis1DPairMemoBackend,
    layer::AbstractVector{Axis1DPairKey},
    plans::Dict{Axis1DPairKey, Axis1DPairPlan},
    profile::Axis1DDAGProfile,
)::Nothing
    for pair in layer
        _prewarm_pair_plan_prisms!(helper, pair, plans[pair])
    end

    computed = Vector{Union{Nothing, Axis1DPathConditionMap}}(undef, length(layer))
    fill!(computed, nothing)
    if length(layer) == 1
        pair = only(layer)
        computed[1] = _compute_planned_pair(
            helper, pair, plans[pair]; use_inner_parallel=true
        )
    else
        Threads.@threads :dynamic for index in eachindex(layer)
            computed[index] = _compute_planned_pair(
                helper, layer[index], plans[layer[index]]; use_inner_parallel=false
            )
        end
    end

    for (index, pair) in enumerate(layer)
        if _pair_is_cached(helper, pair...)
            profile.cached_pairs += 1
        else
            _cache_pair_conditions!(
                helper, pair[1], pair[2], computed[index]::Axis1DPathConditionMap
            )
            profile.solved_pairs += 1
        end
    end
    return nothing
end

"""
    _find_all_path_conditions_dag!(helper, pair_queries) -> Axis1DPairMemoBackend

Plan the pair subproblems needed by `pair_queries`, then solve the DAG by dependency
layers. Independent pairs share the outer thread pool; a single heavy middle join may
instead split its child pairs across that pool.
"""
function _find_all_path_conditions_dag!(
    helper::Axis1DPairMemoBackend,
    pair_queries::AbstractVector{<:Tuple{<:Integer, <:Integer}},
)::Axis1DPairMemoBackend
    profile = Axis1DDAGProfile()
    helper.dag_profile = profile

    planning_start_ns = time_ns()
    plans, scheduled_pairs = _collect_pair_plan(helper, pair_queries)
    layers = _pair_plan_layers(scheduled_pairs, plans)
    profile.planning_ns = time_ns() - planning_start_ns
    profile.planned_pairs = length(scheduled_pairs)
    profile.layers = length(layers)

    solve_start_ns = time_ns()
    for layer in layers
        _solve_pair_plan_layer!(helper, layer, plans, profile)
    end
    profile.solve_ns = time_ns() - solve_start_ns
    return helper
end

function _find_all_path_conditions_dag!(
    helper::Axis1DPairMemoBackend
)::Axis1DPairMemoBackend
    pair_queries = Axis1DPairKey[
        (source, sink) for source in helper.problem.dag.sources for
        sink in helper.problem.dag.sinks
    ]
    return _find_all_path_conditions_dag!(helper, pair_queries)
end
