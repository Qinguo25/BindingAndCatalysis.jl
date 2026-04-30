using BindingAndCatalysis
using JSON3
using Statistics

const CDN4_N = [
    1 1 0 0 -1 0 0 0 0 0
    1 0 1 0 0 -1 0 0 0 0
    1 0 0 1 0 0 -1 0 0 0
    0 1 1 0 0 0 0 -1 0 0
    0 1 0 1 0 0 0 0 -1 0
    0 0 1 1 0 0 0 0 0 -1
]

function complete_dimerization_matrix(n::Int)
    n >= 2 || error("CDN size must be at least 2.")
    pairs = [(i, j) for i in 1:n for j in (i + 1):n]
    N = zeros(Int, length(pairs), n + length(pairs))
    for (row, (i, j)) in enumerate(pairs)
        N[row, i] = 1
        N[row, j] = 1
        N[row, n + row] = -1
    end
    return N
end

cdn_model(n::Int) = n == 4 ? Bnc(N=CDN4_N) : Bnc(N=complete_dimerization_matrix(n))

function root_pairs(paths)
    roots = Set{Tuple{Int,Int}}()
    for path in getproperty(paths, :rgm_paths)
        push!(roots, (Int(first(path)), Int(last(path))))
    end
    return collect(roots)
end

function pair_depth!(
    depths::Dict{Tuple{Int,Int},Int},
    pair::Tuple{Int,Int},
    plans,
)::Int
    cached = get(depths, pair, nothing)
    cached === nothing || return cached
    plan = plans[pair]
    depth = isempty(plan.dependencies) ? 1 : 1 + maximum(pair_depth!(depths, dep, plans) for dep in plan.dependencies)
    depths[pair] = depth
    return depth
end

function branch_counts(pairs, plans)
    counts = Dict{String,Int}()
    for pair in pairs
        branch = string(plans[pair].branch)
        counts[branch] = get(counts, branch, 0) + 1
    end
    return counts
end

function greedy_makespan(times::Vector{Float64}, workers::Int)::Float64
    isempty(times) && return 0.0
    loads = zeros(Float64, max(1, workers))
    for t in sort(times; rev=true)
        idx = argmin(loads)
        loads[idx] += t
    end
    return maximum(loads)
end

function summarize_layers(pairs, plans, depths; pair_seconds=nothing)
    max_depth = maximum(values(depths))
    layers = Vector{Dict{String,Any}}()
    for depth in 1:max_depth
        layer_pairs = [pair for pair in pairs if depths[pair] == depth]
        dependency_counts = [length(plans[pair].dependencies) for pair in layer_pairs]
        static_weights = [BindingAndCatalysis._static_pair_weight(plans[pair]) for pair in layer_pairs]
        layer = Dict{String,Any}(
            "layer" => depth,
            "pairs" => length(layer_pairs),
            "branch_counts" => branch_counts(layer_pairs, plans),
            "dependency_count_sum" => sum(dependency_counts; init=0),
            "dependency_count_max" => isempty(dependency_counts) ? 0 : maximum(dependency_counts),
            "static_weight_sum" => sum(static_weights; init=0.0),
            "static_weight_max" => isempty(static_weights) ? 0.0 : maximum(static_weights),
            "static_8_worker_weight" => greedy_makespan(static_weights, 8),
            "static_50_worker_weight" => greedy_makespan(static_weights, 50),
        )
        if pair_seconds !== nothing
            times = [pair_seconds[pair] for pair in layer_pairs]
            layer["actual_seconds_sum"] = sum(times; init=0.0)
            layer["actual_seconds_max"] = isempty(times) ? 0.0 : maximum(times)
            layer["ideal_8_worker_seconds"] = greedy_makespan(times, 8)
            layer["ideal_50_worker_seconds"] = greedy_makespan(times, 50)
        end
        push!(layers, layer)
    end
    return layers
end

function main()
    n = parse(Int, get(ENV, "BNC_DIAG_CDN_N", "4"))
    solve_pairs = parse(Bool, get(ENV, "BNC_DIAG_SOLVE", "false"))
    model = cdn_model(n)
    find_time = @elapsed find_all_vertices!(model)
    paths_time = @elapsed paths = SISOPaths(model, 1; condition_solver=:dag)
    helper = BindingAndCatalysis._ensure_condition_helper!(paths)

    planning_time = @elapsed begin
        plans, scheduled_pairs = BindingAndCatalysis._collect_pair_plan(helper, root_pairs(paths))
    end

    depths = Dict{Tuple{Int,Int},Int}()
    for pair in scheduled_pairs
        pair_depth!(depths, pair, plans)
    end

    pair_seconds = solve_pairs ? Dict{Tuple{Int,Int},Float64}() : nothing
    pair_outputs = solve_pairs ? Dict{Tuple{Int,Int},Int}() : nothing
    solve_time = 0.0
    if solve_pairs
        helper.dag_profile = BindingAndCatalysis.SISODAGProfile()
        solve_time = @elapsed begin
            for pair in scheduled_pairs
                from, to = pair
                elapsed = @elapsed conditions = BindingAndCatalysis._solve_pair_plan!(helper, from, to, plans[pair])
                pair_seconds[pair] = elapsed
                pair_outputs[pair] = length(conditions)
            end
        end
    end

    layers = summarize_layers(scheduled_pairs, plans, depths; pair_seconds)
    layer_pair_counts = [layer["pairs"] for layer in layers]
    layer_weight_sums = [layer["static_weight_sum"] for layer in layers]
    result = Dict{String,Any}(
        "cdn" => n,
        "julia_threads" => Threads.nthreads(),
        "solve_pairs" => solve_pairs,
        "find_all_vertices_seconds" => find_time,
        "build_paths_seconds" => paths_time,
        "planning_seconds" => planning_time,
        "n_regimes" => n_regimes(model),
        "n_sources" => length(get_sources(paths)),
        "n_sinks" => length(get_sinks(paths)),
        "n_paths" => length(getproperty(paths, :rgm_paths)),
        "planned_pairs" => length(scheduled_pairs),
        "n_layers" => length(layers),
        "max_layer_pairs" => maximum(layer_pair_counts),
        "median_layer_pairs" => median(layer_pair_counts),
        "max_layer_static_weight" => maximum(layer_weight_sums),
        "total_static_weight" => sum(layer_weight_sums; init=0.0),
        "static_layer_infinite_worker_weight" =>
            sum([layer["static_weight_max"] for layer in layers]; init=0.0),
        "static_layer_8_worker_weight" =>
            sum([layer["static_8_worker_weight"] for layer in layers]; init=0.0),
        "static_layer_50_worker_weight" =>
            sum([layer["static_50_worker_weight"] for layer in layers]; init=0.0),
        "branch_counts" => branch_counts(scheduled_pairs, plans),
        "layers" => layers,
    )

    if solve_pairs
        times = [pair_seconds[pair] for pair in scheduled_pairs]
        result["solve_seconds"] = solve_time
        result["sum_pair_seconds"] = sum(times; init=0.0)
        result["max_pair_seconds"] = maximum(times)
        result["ideal_layer_serial_barrier_seconds"] = sum([layer["actual_seconds_sum"] for layer in layers]; init=0.0)
        result["ideal_layer_infinite_worker_seconds"] = sum([layer["actual_seconds_max"] for layer in layers]; init=0.0)
        result["ideal_layer_8_worker_seconds"] = sum([layer["ideal_8_worker_seconds"] for layer in layers]; init=0.0)
        result["ideal_layer_50_worker_seconds"] = sum([layer["ideal_50_worker_seconds"] for layer in layers]; init=0.0)
        result["cached_pairs"] = length(getproperty(helper, :pair_conditions))
        result["cached_path_condition_entries"] =
            sum((length(values) for values in values(getproperty(helper, :pair_conditions))); init=0)
    end

    println(JSON3.write(result))
end

main()
