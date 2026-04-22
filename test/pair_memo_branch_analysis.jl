using BindingAndCatalysis

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
end

PairMemoBranchStats() = PairMemoBranchStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

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
    cached = BindingAndCatalysis._pair_conditions(helper, from, to)
    if !isnothing(cached)
        stats.cache_hits += 1
        return cached
    end

    conditions = BindingAndCatalysis.SISOPathConditionMap()
    if from == to
        stats.diagonal_pairs += 1
        condition = BindingAndCatalysis._get_vertex_prism!(helper, from)
        isempty(condition) || (conditions[(from,)] = condition)
        return BindingAndCatalysis._cache_pair_conditions!(helper, from, to, conditions)
    end

    BindingAndCatalysis._maybe_store_direct_path!(conditions, helper, from, to)

    successors = BindingAndCatalysis._bridge_successors(helper, from, to)
    predecessors = BindingAndCatalysis._bridge_predecessors(helper, from, to)
    if isempty(successors) || isempty(predecessors)
        stats.no_bridge_pairs += 1
        return BindingAndCatalysis._cache_pair_conditions!(helper, from, to, conditions)
    end

    n_solved_successors = count(successor -> BindingAndCatalysis._pair_is_cached(helper, successor, to), successors)
    n_solved_predecessors = count(predecessor -> BindingAndCatalysis._pair_is_cached(helper, from, predecessor), predecessors)
    solved_successor_ratio = n_solved_successors / length(successors)
    solved_predecessor_ratio = n_solved_predecessors / length(predecessors)

    if n_solved_successors == 0 && n_solved_predecessors == 0
        stats.middle_overlap_pairs += 1
        start_ns = time_ns()
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
                middle_conditions = instrumented_find_pair_path_conditions!(helper, stats, successor, predecessor)
                if isempty(middle_conditions)
                    stats.middle_empty_subproblem_skips += 1
                    continue
                end
                for (middle_path, middle_condition) in middle_conditions
                    full_condition = BindingAndCatalysis._intersect_nonempty(left_condition, middle_condition, right_condition)
                    if isnothing(full_condition)
                        stats.middle_intersection_empty_skips += 1
                        continue
                    end
                    stats.middle_generated_paths += 1
                    conditions[BindingAndCatalysis._wrap_vertices(from, middle_path, to)] = full_condition
                end
            end
        end
        stats.middle_overlap_ns += time_ns() - start_ns
        return BindingAndCatalysis._cache_pair_conditions!(helper, from, to, conditions)
    end

    if solved_successor_ratio > solved_predecessor_ratio
        stats.suffix_pairs += 1
        start_ns = time_ns()
        for successor in successors
            stats.suffix_recursive_calls += 1
            suffix_conditions = instrumented_find_pair_path_conditions!(helper, stats, successor, to)
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
        return BindingAndCatalysis._cache_pair_conditions!(helper, from, to, conditions)
    end

    stats.prefix_pairs += 1
    start_ns = time_ns()
    for predecessor in predecessors
        stats.prefix_recursive_calls += 1
        prefix_conditions = instrumented_find_pair_path_conditions!(helper, stats, from, predecessor)
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

    return BindingAndCatalysis._cache_pair_conditions!(helper, from, to, conditions)
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
    if mode == "branch"
        println(repr(analyze_pair_memo_branches(4)))
        return
    end
    if mode == "scaling"
        println(repr(analyze_path_lengths(collect(2:5))))
        return
    end

    branch_result = analyze_pair_memo_branches(4)
    scaling_result = analyze_path_lengths(collect(2:5))
    println(repr(Dict("branch_analysis" => branch_result, "path_length_scaling" => scaling_result)))
end

main()
