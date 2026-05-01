using BindingAndCatalysis
using Dates
using Graphs
using JSON3

function complete_dimerization_matrix(n::Int)
    n >= 2 || error("CDN size must be at least 2.")
    pairs = [(i, j) for i in 1:n for j in (i + 1):n]
    N = zeros(Int, length(pairs), n + length(pairs))
    for (row, (i, j)) in enumerate(pairs)
        N[row, i] = 1
        N[row, j] = 1
        N[row, n + row] = -1
    end
    return N, pairs
end

function cdn_model(n::Int)
    N, pairs = complete_dimerization_matrix(n)
    monomer_names = Symbol.('A':'Z')[1:n]
    dimer_names = [Symbol(lowercase(string(monomer_names[i], monomer_names[j]))) for (i, j) in pairs]
    q_sym = [Symbol("t", monomer_names[i]) for i in 1:n]
    K_sym = [Symbol("K", i, j) for (i, j) in pairs]
    return Bnc(N = N, x_sym = [monomer_names; dimer_names], q_sym = q_sym, K_sym = K_sym), K_sym
end

function write_json(path, payload)
    open(path, "w") do io
        JSON3.write(io, payload)
        println(io)
    end
end

function relation_cases(model, K_sym)
    adjacent_order = Tuple{Any,Any,Any}[]
    adjacent_order_2x = Tuple{Any,Any,Any,Float64}[]
    margin = log10(2)
    for i in 1:(length(K_sym) - 1)
        push!(adjacent_order, (K_sym[i], :>, K_sym[i + 1]))
        push!(adjacent_order_2x, (K_sym[i], :>, K_sym[i + 1], margin))
    end

    return [
        ("baseline", nothing),
        ("single_K12_gt_K23", qK_preconstraint(model, :K12, :>, :K23)),
        ("ordered_equilibrium_constants", qK_preconstraints(model, adjacent_order)),
        ("ordered_equilibrium_constants_2x", qK_preconstraints(model, adjacent_order_2x)),
    ]
end

function summarize_graph_case(model, change_qK, name, preconstraints)
    result = Dict{String,Any}(
        "case" => name,
        "stage" => "graph_and_paths",
        "change_qK" => string(change_qK),
        "started_at" => string(now()),
    )

    graph_time = @elapsed begin
        graph, feasible_vertices, diagnostics =
            get_pruned_SISO_graph(model, change_qK; qK_preconstraints = preconstraints)
    end
    result["graph_seconds"] = graph_time
    result["graph_vertices"] = nv(graph)
    result["graph_edges"] = ne(graph)
    result["feasible_vertex_mask_count"] = count(feasible_vertices)
    result["diagnostics"] = Dict(
        "original_vertices" => diagnostics.original_vertices,
        "feasible_vertices" => diagnostics.feasible_vertices,
        "original_edges" => diagnostics.original_edges,
        "feasible_edges" => diagnostics.feasible_edges,
        "removed_vertices" => diagnostics.removed_vertices,
        "removed_edges" => diagnostics.removed_edges,
    )

    build_time = @elapsed paths = SISOPaths(model, change_qK; qK_preconstraints = preconstraints, condition_solver = :dag)
    result["siso_build_seconds"] = build_time
    result["n_sources"] = length(get_sources(paths))
    result["n_sinks"] = length(get_sinks(paths))
    result["n_paths"] = length(paths.rgm_paths)
    result["completed_at"] = string(now())
    return result, paths
end

function solve_path_conditions!(paths)
    solve_time = @elapsed polys = get_polyhedra(paths)
    return Dict{String,Any}(
        "stage" => "path_conditions",
        "get_polyhedra_seconds" => solve_time,
        "n_polyhedra" => length(polys),
    )
end

function main()
    output_dir = get(ENV, "BNC_RELATION_OUTPUT_DIR", joinpath(@__DIR__, "relation_pruning_artifacts"))
    mkpath(output_dir)
    status_path = joinpath(output_dir, "cdn5_relation_status.json")
    result_path = joinpath(output_dir, "cdn5_relation_result.json")

    change_qK = Symbol(get(ENV, "BNC_RELATION_CHANGE_QK", "tA"))
    run_full = parse(Bool, get(ENV, "BNC_RELATION_RUN_FULL", "false"))
    full_cases = Set(split(get(ENV, "BNC_RELATION_FULL_CASES", "ordered_equilibrium_constants,ordered_equilibrium_constants_2x"), ","))

    model, K_sym = cdn_model(5)
    payload = Dict{String,Any}(
        "benchmark" => "cdn5_relation_pruning",
        "threads" => Threads.nthreads(),
        "change_qK" => string(change_qK),
        "run_full" => run_full,
        "full_cases" => collect(full_cases),
        "started_at" => string(now()),
        "cases" => Any[],
    )
    write_json(status_path, payload)

    find_time = @elapsed find_all_regimes!(model)
    payload["find_all_regimes_seconds"] = find_time
    payload["n_regimes"] = n_regimes(model)
    write_json(status_path, payload)

    for (name, preconstraints) in relation_cases(model, K_sym)
        payload["current_case"] = name
        write_json(status_path, payload)

        case_result, paths = summarize_graph_case(model, change_qK, name, preconstraints)
        if run_full && name in full_cases
            merge!(case_result, solve_path_conditions!(paths))
        end
        push!(payload["cases"], case_result)
        write_json(status_path, payload)
    end

    payload["completed_at"] = string(now())
    delete!(payload, "current_case")
    write_json(status_path, payload)
    write_json(result_path, payload)
    println(JSON3.write(payload))
end

main()
