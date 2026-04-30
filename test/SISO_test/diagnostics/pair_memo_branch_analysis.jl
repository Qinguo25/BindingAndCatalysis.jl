using BindingAndCatalysis

const PairNode = Tuple{Int,Int}

mutable struct PairMemoBranchStats
    calls::Int
    cache_hits::Int
    diagonal_pairs::Int
    no_bridge_pairs::Int
    middle_overlap_pairs::Int
    suffix_pairs::Int
    prefix_pairs::Int
    middle_overlap_ns::UInt64
    suffix_ns::UInt64
    prefix_ns::UInt64
    middle_recursive_calls::Int
    suffix_recursive_calls::Int
    prefix_recursive_calls::Int
    middle_generated_paths::Int
    suffix_generated_paths::Int
    prefix_generated_paths::Int
    middle_left_empty_skips::Int
    middle_right_empty_skips::Int
    middle_empty_subproblem_skips::Int
    middle_intersection_empty_skips::Int
    suffix_left_empty_skips::Int
    suffix_empty_subproblem_skips::Int
    suffix_intersection_empty_skips::Int
    prefix_right_empty_skips::Int
    prefix_empty_subproblem_skips::Int
    prefix_intersection_empty_skips::Int
    middle_pair_calls::Dict{Tuple{Int,Int},Int}
    middle_pair_empty_hits::Dict{Tuple{Int,Int},Int}
    middle_pair_nonempty_hits::Dict{Tuple{Int,Int},Int}
    outer_middle_empty_hits::Dict{Tuple{Int,Int},Int}
    outer_middle_generated_paths::Dict{Tuple{Int,Int},Int}
    dependency_children::Dict{PairNode,Set{PairNode}}
    middle_children_by_outer::Dict{PairNode,Set{PairNode}}
    middle_nonempty_children_by_outer::Dict{PairNode,Set{PairNode}}
    pair_branch_kind::Dict{PairNode,Symbol}
    pair_total_ns::Dict{PairNode,UInt64}
    pair_self_ns::Dict{PairNode,UInt64}
end

PairMemoBranchStats() = PairMemoBranchStats(
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Dict{Tuple{Int,Int},Int}(),
    Dict{Tuple{Int,Int},Int}(),
    Dict{Tuple{Int,Int},Int}(),
    Dict{Tuple{Int,Int},Int}(),
    Dict{Tuple{Int,Int},Int}(),
    Dict{PairNode,Set{PairNode}}(),
    Dict{PairNode,Set{PairNode}}(),
    Dict{PairNode,Set{PairNode}}(),
    Dict{PairNode,Symbol}(),
    Dict{PairNode,UInt64}(),
    Dict{PairNode,UInt64}(),
)

function increment_pair_count!(counts::Dict{Tuple{Int,Int},Int}, key::Tuple{Int,Int})
    counts[key] = get(counts, key, 0) + 1
    return counts
end

function top_pair_counts(counts::Dict{Tuple{Int,Int},Int}; limit::Int=10)
    ordered = sort!(collect(counts); by=entry -> (-entry[2], entry[1][1], entry[1][2]))
    return [
        Dict("pair" => [pair[1], pair[2]], "count" => count)
        for (pair, count) in Iterators.take(ordered, limit)
    ]
end

function register_pair_dependency!(
    dependencies::Dict{PairNode,Set{PairNode}},
    parent::PairNode,
    child::PairNode,
)
    push!(get!(dependencies, parent, Set{PairNode}()), child)
    return dependencies
end

function register_middle_child!(
    children_by_outer::Dict{PairNode,Set{PairNode}},
    outer::PairNode,
    child::PairNode,
)
    push!(get!(children_by_outer, outer, Set{PairNode}()), child)
    return children_by_outer
end

function summarize_middle_pair_hits(stats::PairMemoBranchStats; limit::Int=10)
    pair_keys = union(
        keys(stats.middle_pair_calls),
        keys(stats.middle_pair_empty_hits),
        keys(stats.middle_pair_nonempty_hits),
    )
    rows = Dict{String,Any}[]
    for pair in pair_keys
        calls = get(stats.middle_pair_calls, pair, 0)
        empty_hits = get(stats.middle_pair_empty_hits, pair, 0)
        nonempty_hits = get(stats.middle_pair_nonempty_hits, pair, 0)
        push!(rows, Dict(
            "pair" => [pair[1], pair[2]],
            "calls" => calls,
            "empty_hits" => empty_hits,
            "nonempty_hits" => nonempty_hits,
            "empty_rate" => calls == 0 ? 0.0 : empty_hits / calls,
        ))
    end
    sort!(rows; by=row -> (-row["empty_hits"], -row["calls"], row["pair"][1], row["pair"][2]))
    return first(rows, min(limit, length(rows)))
end

function summarize_empty_middle_pair_distribution(stats::PairMemoBranchStats)
    empty_counts = collect(values(stats.middle_pair_empty_hits))
    isempty(empty_counts) && return Dict(
        "distinct_empty_middle_pairs" => 0,
        "max_empty_hits_per_pair" => 0,
        "pairs_with_multiple_empty_hits" => 0,
        "pairs_with_single_empty_hit" => 0,
        "top_10_empty_hit_share" => 0.0,
    )

    sorted_counts = sort(empty_counts; rev=true)
    return Dict(
        "distinct_empty_middle_pairs" => length(empty_counts),
        "max_empty_hits_per_pair" => first(sorted_counts),
        "pairs_with_multiple_empty_hits" => count(>(1), empty_counts),
        "pairs_with_single_empty_hit" => count(==(1), empty_counts),
        "top_10_empty_hit_share" => sum(Iterators.take(sorted_counts, 10)) / sum(sorted_counts),
    )
end

function pair_descendant_cone(
    pair::PairNode,
    dependencies::Dict{PairNode,Set{PairNode}},
    memo::Dict{PairNode,Set{PairNode}},
)::Set{PairNode}
    cached = get(memo, pair, nothing)
    !isnothing(cached) && return cached

    cone = Set([pair])
    for child in get(dependencies, pair, Set{PairNode}())
        union!(cone, pair_descendant_cone(child, dependencies, memo))
    end
    memo[pair] = cone
    return cone
end

function count_disjoint_child_pairs(
    children::AbstractVector{PairNode},
    cone_memo::Dict{PairNode,Set{PairNode}},
)::Int
    total = 0
    for i in 1:length(children)-1
        cone_i = cone_memo[children[i]]
        for j in (i + 1):length(children)
            cone_j = cone_memo[children[j]]
            isempty(intersect(cone_i, cone_j)) || continue
            total += 1
        end
    end
    return total
end

function summarize_parallel_middle_opportunities(stats::PairMemoBranchStats; limit::Int=10)
    cone_memo = Dict{PairNode,Set{PairNode}}()
    all_pairs = union(
        keys(stats.dependency_children),
        Iterators.flatten(values(stats.dependency_children)),
        keys(stats.middle_children_by_outer),
        keys(stats.middle_nonempty_children_by_outer),
    )
    for pair in all_pairs
        pair_descendant_cone(pair, stats.dependency_children, cone_memo)
    end

    total_middle_outer_pairs = length(stats.middle_children_by_outer)
    outer_pairs_with_2plus_children = 0
    outer_pairs_with_any_disjoint_child_pair = 0
    outer_pairs_with_any_disjoint_nonempty_child_pair = 0
    top_rows = Dict{String,Any}[]

    for outer in sort!(collect(keys(stats.middle_children_by_outer)))
        children = sort!(collect(stats.middle_children_by_outer[outer]))
        nonempty_children = sort!(collect(get(stats.middle_nonempty_children_by_outer, outer, Set{PairNode}())))
        if length(children) >= 2
            outer_pairs_with_2plus_children += 1
        end

        disjoint_child_pairs = length(children) < 2 ? 0 : count_disjoint_child_pairs(children, cone_memo)
        disjoint_nonempty_child_pairs = length(nonempty_children) < 2 ? 0 : count_disjoint_child_pairs(nonempty_children, cone_memo)
        disjoint_child_pairs > 0 && (outer_pairs_with_any_disjoint_child_pair += 1)
        disjoint_nonempty_child_pairs > 0 && (outer_pairs_with_any_disjoint_nonempty_child_pair += 1)

        push!(top_rows, Dict(
            "outer_pair" => [outer[1], outer[2]],
            "middle_children" => length(children),
            "nonempty_middle_children" => length(nonempty_children),
            "disjoint_child_pairs" => disjoint_child_pairs,
            "disjoint_nonempty_child_pairs" => disjoint_nonempty_child_pairs,
        ))
    end

    sort!(top_rows; by=row -> (-row["disjoint_nonempty_child_pairs"], -row["disjoint_child_pairs"], -row["nonempty_middle_children"], row["outer_pair"][1], row["outer_pair"][2]))
    return Dict(
        "total_middle_outer_pairs" => total_middle_outer_pairs,
        "outer_pairs_with_2plus_children" => outer_pairs_with_2plus_children,
        "outer_pairs_with_any_disjoint_child_pair" => outer_pairs_with_any_disjoint_child_pair,
        "outer_pairs_with_any_disjoint_nonempty_child_pair" => outer_pairs_with_any_disjoint_nonempty_child_pair,
        "top_outer_pairs_by_disjoint_middle_children" => first(top_rows, min(limit, length(top_rows))),
    )
end

function top_int_counts(counts::Dict{Int,Int}; limit::Int=10)
    ordered = sort!(collect(counts); by=entry -> (-entry[2], entry[1]))
    return [
        Dict("level" => level, "count" => count)
        for (level, count) in Iterators.take(ordered, limit)
    ]
end

function summarize_dependency_dag_parallelism(
    dependencies::Dict{PairNode,Set{PairNode}},
    roots::AbstractVector{PairNode};
    limit::Int=10,
)
    nodes = Set{PairNode}()
    for (parent, children) in dependencies
        push!(nodes, parent)
        union!(nodes, children)
    end
    isempty(nodes) && return Dict(
        "visited_pair_nodes" => 0,
        "dependency_edges" => 0,
        "critical_path_layers" => 0,
        "ideal_parallelism_work_over_span" => 0.0,
        "max_layer_width" => 0,
        "n_roots" => 0,
        "top_layer_widths" => Dict{String,Int}[],
    )

    indegree = Dict(node => 0 for node in nodes)
    for children in values(dependencies), child in children
        indegree[child] = get(indegree, child, 0) + 1
    end

    # Compute longest-distance-from-source levels on the visited dependency DAG.
    ready = sort!(collect(node for node in nodes if indegree[node] == 0))
    levels = Dict(node => 1 for node in ready)
    queue = copy(ready)
    head = 1
    while head <= length(queue)
        node = queue[head]
        head += 1
        node_level = levels[node]
        for child in get(dependencies, node, Set{PairNode}())
            levels[child] = max(get(levels, child, 1), node_level + 1)
            indegree[child] -= 1
            indegree[child] == 0 && push!(queue, child)
        end
    end

    layer_counts = Dict{Int,Int}()
    for level in values(levels)
        layer_counts[level] = get(layer_counts, level, 0) + 1
    end

    work = length(nodes)
    span = maximum(values(levels))
    max_width = maximum(values(layer_counts))
    dependency_edges = sum(length(children) for children in values(dependencies))
    root_set = Set(roots)

    return Dict(
        "visited_pair_nodes" => work,
        "dependency_edges" => dependency_edges,
        "critical_path_layers" => span,
        "ideal_parallelism_work_over_span" => work / span,
        "max_layer_width" => max_width,
        "n_roots" => length(intersect(root_set, nodes)),
        "top_layer_widths" => top_int_counts(layer_counts; limit=limit),
    )
end

function summarize_weighted_dependency_dag_parallelism(
    dependencies::Dict{PairNode,Set{PairNode}},
    pair_self_ns::Dict{PairNode,UInt64},
    pair_branch_kind::Dict{PairNode,Symbol},
)
    nodes = Set(keys(pair_self_ns))
    isempty(nodes) && return Dict(
        "weighted_work_seconds" => 0.0,
        "weighted_span_seconds" => 0.0,
        "ideal_weighted_parallelism_work_over_span" => 0.0,
        "branch_self_seconds" => Dict{String,Float64}(),
        "top_weighted_critical_path_pairs" => Dict{String,Any}[],
    )

    succ = Dict(node => collect(get(dependencies, node, Set{PairNode}())) for node in nodes)
    indegree = Dict(node => 0 for node in nodes)
    for parent in nodes, child in succ[parent]
        haskey(indegree, child) || continue
        indegree[child] += 1
    end

    topo = PairNode[]
    queue = sort!(collect(node for node in nodes if indegree[node] == 0))
    head = 1
    while head <= length(queue)
        node = queue[head]
        head += 1
        push!(topo, node)
        for child in succ[node]
            haskey(indegree, child) || continue
            indegree[child] -= 1
            indegree[child] == 0 && push!(queue, child)
        end
    end

    best_suffix_ns = Dict{PairNode,UInt64}()
    best_child = Dict{PairNode,Union{Nothing,PairNode}}()
    for node in Iterators.reverse(topo)
        node_self = pair_self_ns[node]
        if isempty(succ[node])
            best_suffix_ns[node] = node_self
            best_child[node] = nothing
            continue
        end
        best_next = nothing
        best_next_ns = UInt64(0)
        for child in succ[node]
            haskey(best_suffix_ns, child) || continue
            child_ns = best_suffix_ns[child]
            if child_ns > best_next_ns
                best_next_ns = child_ns
                best_next = child
            end
        end
        best_suffix_ns[node] = node_self + best_next_ns
        best_child[node] = best_next
    end

    start_node = nothing
    span_ns = UInt64(0)
    for (node, total_ns) in best_suffix_ns
        if total_ns > span_ns
            span_ns = total_ns
            start_node = node
        end
    end

    critical_pairs = Dict{String,Any}[]
    current = start_node
    while !isnothing(current)
        pair = current::PairNode
        push!(critical_pairs, Dict(
            "pair" => [pair[1], pair[2]],
            "branch" => String(pair_branch_kind[pair]),
            "self_seconds" => pair_self_ns[pair] / 1e9,
        ))
        current = best_child[pair]
    end

    branch_self_seconds = Dict{String,Float64}()
    for (pair, ns) in pair_self_ns
        branch = String(get(pair_branch_kind, pair, :unknown))
        branch_self_seconds[branch] = get(branch_self_seconds, branch, 0.0) + ns / 1e9
    end

    total_work_ns = sum(values(pair_self_ns))
    return Dict(
        "weighted_work_seconds" => total_work_ns / 1e9,
        "weighted_span_seconds" => span_ns / 1e9,
        "ideal_weighted_parallelism_work_over_span" => total_work_ns / max(1, span_ns),
        "branch_self_seconds" => branch_self_seconds,
        "top_weighted_critical_path_pairs" => critical_pairs,
    )
end

function complete_dimerization_matrix(n::Int)
    n >= 2 || error("CDN size must be at least 2 to define a binding edge.")
    pairs = [(i, j) for i in 1:n for j in (i + 1):n]
    m = length(pairs)
    N = zeros(Int, m, n + m)
    for (row, (i, j)) in enumerate(pairs)
        N[row, i] = 1
        N[row, j] = 1
        N[row, n + row] = -1
    end
    return N
end

cdn_model(n::Int) = Bnc(N = complete_dimerization_matrix(n))

function instrumented_find_pair_path_conditions!(
    helper::BindingAndCatalysis.SISOHelper,
    stats::PairMemoBranchStats,
    from::Int,
    to::Int,
)
    stats.calls += 1
    pair = (from, to)
    start_total_ns = time_ns()
    child_call_ns = UInt64(0)
    cached = BindingAndCatalysis._pair_conditions(helper, from, to)
    if !isnothing(cached)
        stats.cache_hits += 1
        return cached
    end

    conditions = BindingAndCatalysis.SISOPathConditionMap()
    if from == to
        stats.diagonal_pairs += 1
        stats.pair_branch_kind[pair] = :diagonal
        condition = BindingAndCatalysis._get_vertex_prism!(helper, from)
        isempty(condition) || (conditions[(from,)] = condition)
        out = BindingAndCatalysis._cache_pair_conditions!(helper, from, to, conditions)
        total_ns = time_ns() - start_total_ns
        stats.pair_total_ns[pair] = total_ns
        stats.pair_self_ns[pair] = total_ns
        return out
    end

    BindingAndCatalysis._maybe_store_direct_path!(conditions, helper, from, to)

    successors = BindingAndCatalysis._bridge_successors(helper, from, to)
    predecessors = BindingAndCatalysis._bridge_predecessors(helper, from, to)
    if isempty(successors) || isempty(predecessors)
        stats.no_bridge_pairs += 1
        stats.pair_branch_kind[pair] = :no_bridge
        out = BindingAndCatalysis._cache_pair_conditions!(helper, from, to, conditions)
        total_ns = time_ns() - start_total_ns
        stats.pair_total_ns[pair] = total_ns
        stats.pair_self_ns[pair] = total_ns
        return out
    end

    n_solved_successors = count(successor -> BindingAndCatalysis._pair_is_cached(helper, successor, to), successors)
    n_solved_predecessors = count(predecessor -> BindingAndCatalysis._pair_is_cached(helper, from, predecessor), predecessors)
    solved_successor_ratio = n_solved_successors / length(successors)
    solved_predecessor_ratio = n_solved_predecessors / length(predecessors)

    if n_solved_successors == 0 && n_solved_predecessors == 0
        stats.middle_overlap_pairs += 1
        stats.pair_branch_kind[pair] = :middle
        start_ns = time_ns()
        outer_pair = pair
        for successor in successors
            left_condition = BindingAndCatalysis._get_interface_prism!(helper, from, successor)
            if isempty(left_condition)
                stats.middle_left_empty_skips += 1
                continue
            end
            for predecessor in predecessors
                right_condition = BindingAndCatalysis._get_interface_prism!(helper, predecessor, to)
                if isempty(right_condition)
                    stats.middle_right_empty_skips += 1
                    continue
                end
                stats.middle_recursive_calls += 1
                child_pair = (successor, predecessor)
                increment_pair_count!(stats.middle_pair_calls, child_pair)
                register_pair_dependency!(stats.dependency_children, outer_pair, child_pair)
                register_middle_child!(stats.middle_children_by_outer, outer_pair, child_pair)
                child_start_ns = time_ns()
                middle_conditions = instrumented_find_pair_path_conditions!(helper, stats, successor, predecessor)
                child_call_ns += time_ns() - child_start_ns
                if isempty(middle_conditions)
                    stats.middle_empty_subproblem_skips += 1
                    increment_pair_count!(stats.middle_pair_empty_hits, child_pair)
                    increment_pair_count!(stats.outer_middle_empty_hits, outer_pair)
                    continue
                end
                increment_pair_count!(stats.middle_pair_nonempty_hits, child_pair)
                register_middle_child!(stats.middle_nonempty_children_by_outer, outer_pair, child_pair)
                for (middle_path, middle_condition) in middle_conditions
                    full_condition = BindingAndCatalysis._intersect_nonempty(left_condition, middle_condition, right_condition)
                    if isnothing(full_condition)
                        stats.middle_intersection_empty_skips += 1
                        continue
                    end
                    stats.middle_generated_paths += 1
                    increment_pair_count!(stats.outer_middle_generated_paths, outer_pair)
                    conditions[BindingAndCatalysis._wrap_vertices(from, middle_path, to)] = full_condition
                end
            end
        end
        stats.middle_overlap_ns += time_ns() - start_ns
        out = BindingAndCatalysis._cache_pair_conditions!(helper, from, to, conditions)
        total_ns = time_ns() - start_total_ns
        stats.pair_total_ns[pair] = total_ns
        stats.pair_self_ns[pair] = total_ns - child_call_ns
        return out
    end

    if solved_successor_ratio > solved_predecessor_ratio
        stats.suffix_pairs += 1
        stats.pair_branch_kind[pair] = :suffix
        start_ns = time_ns()
        for successor in successors
            stats.suffix_recursive_calls += 1
            register_pair_dependency!(stats.dependency_children, (from, to), (successor, to))
            child_start_ns = time_ns()
            suffix_conditions = instrumented_find_pair_path_conditions!(helper, stats, successor, to)
            child_call_ns += time_ns() - child_start_ns
            if isempty(suffix_conditions)
                stats.suffix_empty_subproblem_skips += 1
                continue
            end
            left_condition = BindingAndCatalysis._get_interface_prism!(helper, from, successor)
            if isempty(left_condition)
                stats.suffix_left_empty_skips += 1
                continue
            end
            for (suffix_path, suffix_condition) in suffix_conditions
                full_condition = BindingAndCatalysis._intersect_nonempty(left_condition, suffix_condition)
                if isnothing(full_condition)
                    stats.suffix_intersection_empty_skips += 1
                    continue
                end
                stats.suffix_generated_paths += 1
                conditions[BindingAndCatalysis._prepend_vertex(from, suffix_path)] = full_condition
            end
        end
        stats.suffix_ns += time_ns() - start_ns
        out = BindingAndCatalysis._cache_pair_conditions!(helper, from, to, conditions)
        total_ns = time_ns() - start_total_ns
        stats.pair_total_ns[pair] = total_ns
        stats.pair_self_ns[pair] = total_ns - child_call_ns
        return out
    end

    stats.prefix_pairs += 1
    stats.pair_branch_kind[pair] = :prefix
    start_ns = time_ns()
    for predecessor in predecessors
        stats.prefix_recursive_calls += 1
        register_pair_dependency!(stats.dependency_children, (from, to), (from, predecessor))
        child_start_ns = time_ns()
        prefix_conditions = instrumented_find_pair_path_conditions!(helper, stats, from, predecessor)
        child_call_ns += time_ns() - child_start_ns
        if isempty(prefix_conditions)
            stats.prefix_empty_subproblem_skips += 1
            continue
        end
        right_condition = BindingAndCatalysis._get_interface_prism!(helper, predecessor, to)
        if isempty(right_condition)
            stats.prefix_right_empty_skips += 1
            continue
        end
        for (prefix_path, prefix_condition) in prefix_conditions
            full_condition = BindingAndCatalysis._intersect_nonempty(prefix_condition, right_condition)
            if isnothing(full_condition)
                stats.prefix_intersection_empty_skips += 1
                continue
            end
            stats.prefix_generated_paths += 1
            conditions[BindingAndCatalysis._append_vertex(prefix_path, to)] = full_condition
        end
    end
    stats.prefix_ns += time_ns() - start_ns

    out = BindingAndCatalysis._cache_pair_conditions!(helper, from, to, conditions)
    total_ns = time_ns() - start_total_ns
    stats.pair_total_ns[pair] = total_ns
    stats.pair_self_ns[pair] = total_ns - child_call_ns
    return out
end

function analyze_pair_memo_branches(n::Int)
    model = cdn_model(n)
    t_find = @elapsed find_all_vertices!(model)
    helper = BindingAndCatalysis.SISOHelper(model, 1)
    stats = PairMemoBranchStats()
    pair_queries = [(source, sink) for source in get_sources(helper) for sink in get_sinks(helper)]
    t_pairs = @elapsed begin
        for (source, sink) in pair_queries
            instrumented_find_pair_path_conditions!(helper, stats, source, sink)
        end
    end
    cached_entries = sum(length(values) for values in values(helper.pair_conditions))
    return Dict(
        "cdn" => n,
        "find_all_vertices_seconds" => t_find,
        "pair_solver_seconds" => t_pairs,
        "source_sink_queries" => length(pair_queries),
        "cached_pairs" => length(helper.pair_conditions),
        "cached_path_condition_entries" => cached_entries,
        "calls" => stats.calls,
        "cache_hits" => stats.cache_hits,
        "diagonal_pairs" => stats.diagonal_pairs,
        "no_bridge_pairs" => stats.no_bridge_pairs,
        "middle_overlap_pairs" => stats.middle_overlap_pairs,
        "suffix_pairs" => stats.suffix_pairs,
        "prefix_pairs" => stats.prefix_pairs,
        "middle_overlap_seconds" => stats.middle_overlap_ns / 1e9,
        "suffix_seconds" => stats.suffix_ns / 1e9,
        "prefix_seconds" => stats.prefix_ns / 1e9,
        "middle_recursive_calls" => stats.middle_recursive_calls,
        "suffix_recursive_calls" => stats.suffix_recursive_calls,
        "prefix_recursive_calls" => stats.prefix_recursive_calls,
        "middle_generated_paths" => stats.middle_generated_paths,
        "suffix_generated_paths" => stats.suffix_generated_paths,
        "prefix_generated_paths" => stats.prefix_generated_paths,
        "middle_left_empty_skips" => stats.middle_left_empty_skips,
        "middle_right_empty_skips" => stats.middle_right_empty_skips,
        "middle_empty_subproblem_skips" => stats.middle_empty_subproblem_skips,
        "middle_intersection_empty_skips" => stats.middle_intersection_empty_skips,
        "empty_middle_pair_distribution" => summarize_empty_middle_pair_distribution(stats),
        "parallel_middle_opportunities" => summarize_parallel_middle_opportunities(stats),
        "dependency_dag_parallelism" => summarize_dependency_dag_parallelism(stats.dependency_children, pair_queries),
        "weighted_dependency_dag_parallelism" => summarize_weighted_dependency_dag_parallelism(stats.dependency_children, stats.pair_self_ns, stats.pair_branch_kind),
        "top_empty_middle_pairs" => summarize_middle_pair_hits(stats),
        "top_outer_pairs_by_empty_middle_hits" => top_pair_counts(stats.outer_middle_empty_hits),
        "top_outer_pairs_by_middle_generated_paths" => top_pair_counts(stats.outer_middle_generated_paths),
        "suffix_left_empty_skips" => stats.suffix_left_empty_skips,
        "suffix_empty_subproblem_skips" => stats.suffix_empty_subproblem_skips,
        "suffix_intersection_empty_skips" => stats.suffix_intersection_empty_skips,
        "prefix_right_empty_skips" => stats.prefix_right_empty_skips,
        "prefix_empty_subproblem_skips" => stats.prefix_empty_subproblem_skips,
        "prefix_intersection_empty_skips" => stats.prefix_intersection_empty_skips,
    )
end

function analyze_path_lengths(ns::AbstractVector{<:Integer})
    out = Dict{Int, Dict{String, Any}}()
    for n in ns
        model = cdn_model(n)
        t_find = @elapsed find_all_vertices!(model)
        graph = get_SISO_graph(model, 1)
        sources, sinks = get_sources_sinks(model, graph)
        t_paths = @elapsed paths = BindingAndCatalysis._enumerate_paths(graph; sources, sinks)
        max_vertices = isempty(paths) ? 0 : maximum(length, paths)
        out[n] = Dict(
            "cdn" => n,
            "find_all_vertices_seconds" => t_find,
            "enumerate_paths_seconds" => t_paths,
            "n_regimes" => n_regimes(model),
            "n_sources" => length(sources),
            "n_sinks" => length(sinks),
            "n_paths" => length(paths),
            "longest_path_vertices" => max_vertices,
            "longest_path_edges" => max(0, max_vertices - 1),
        )
    end
    return out
end

function main()
    mode = isempty(ARGS) ? "all" : ARGS[1]
    n = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
    if mode == "branch"
        println(repr(analyze_pair_memo_branches(n)))
        return
    end
    if mode == "scaling"
        println(repr(analyze_path_lengths(collect(2:5))))
        return
    end

    branch_result = analyze_pair_memo_branches(n)
    scaling_result = analyze_path_lengths(collect(2:5))
    println(repr(Dict("branch_analysis" => branch_result, "path_length_scaling" => scaling_result)))
end

main()
