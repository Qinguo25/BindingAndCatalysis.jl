using BindingAndCatalysis
using JSON3
using Dates

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

function collect_helper_stats(paths)
    stats = Dict{String,Any}()
    if !hasproperty(paths, :condition_helper)
        return stats
    end

    helper = getproperty(paths, :condition_helper)
    isnothing(helper) && return stats

    pair_conditions = getproperty(helper, :pair_conditions)
    stats["cached_pairs"] = length(pair_conditions)
    stats["cached_path_condition_entries"] = sum(length(values) for values in values(pair_conditions))
    stats["cached_vertex_prisms"] = count(!isnothing, getproperty(helper, :vertex_prisms))
    stats["cached_interface_prisms"] = length(getproperty(helper, :interface_prisms))

    if hasproperty(helper, :dag_profile)
        profile = getproperty(helper, :dag_profile)
        if !isnothing(profile)
            stats["dag_planning_seconds"] = getproperty(profile, :planning_ns) / 1e9
            stats["dag_pair_solve_seconds"] = getproperty(profile, :pair_solve_ns) / 1e9
            stats["dag_middle_collect_seconds"] = getproperty(profile, :middle_collect_ns) / 1e9
            stats["dag_middle_compute_seconds"] = getproperty(profile, :middle_compute_ns) / 1e9
            stats["dag_middle_merge_seconds"] = getproperty(profile, :middle_merge_ns) / 1e9
            stats["dag_pair_solve_calls"] = getproperty(profile, :pair_solve_calls)
            stats["dag_planned_pairs"] = getproperty(profile, :planned_pairs)
            stats["dag_middle_parallel_nodes"] = getproperty(profile, :middle_parallel_nodes)
            stats["dag_middle_serial_nodes"] = getproperty(profile, :middle_serial_nodes)
            stats["dag_middle_join_pairs"] = getproperty(profile, :middle_join_pairs)
            if hasproperty(profile, :weighted_work_done)
                stats["dag_weighted_work_done"] = getproperty(profile, :weighted_work_done)
                stats["dag_weighted_work_total"] = getproperty(profile, :weighted_work_total)
                stats["dag_weighted_progress"] =
                    getproperty(profile, :weighted_work_done) / max(1.0, getproperty(profile, :weighted_work_total))
                stats["dag_weighted_progress_units"] = getproperty(profile, :weighted_progress_units)
                stats["dag_largest_pair_seconds"] = getproperty(profile, :largest_pair_seconds)
                stats["dag_largest_pair"] = [getproperty(profile, :largest_pair_from), getproperty(profile, :largest_pair_to)]
                stats["dag_current_pair"] = [getproperty(profile, :current_pair_from), getproperty(profile, :current_pair_to)]
                stats["dag_current_pair_branch"] = string(getproperty(profile, :current_pair_branch))
                stats["dag_current_pair_weight"] = getproperty(profile, :current_pair_weight)
                current_pair_start_ns = getproperty(profile, :current_pair_start_ns)
                current_pair_elapsed_seconds = getproperty(profile, :current_pair_elapsed_seconds)
                if current_pair_start_ns > 0
                    current_pair_elapsed_seconds = (time_ns() - current_pair_start_ns) / 1e9
                end
                stats["dag_current_pair_elapsed_seconds"] = current_pair_elapsed_seconds
                stats["dag_current_pair_running"] = current_pair_start_ns > 0
                stats["dag_current_pair_output_entries"] = getproperty(profile, :current_pair_output_entries)
            end
        end
    end

    if hasproperty(paths, :path_polys_is_calc)
        stats["computed_path_polyhedra"] = count(getproperty(paths, :path_polys_is_calc))
    end

    return stats
end

function write_status(path::AbstractString, payload::Dict{String,Any})
    open(path, "w") do io
        JSON3.write(io, payload)
        flush(io)
    end
    return nothing
end

function main()
    n = parse(Int, get(ENV, "BNC_CDN_N", "5"))
    solver = Symbol(get(ENV, "BNC_CDN_SOLVER", "dag"))
    heartbeat_seconds = parse(Float64, get(ENV, "BNC_HEARTBEAT_SECONDS", "60"))
    status_path = get(ENV, "BNC_STATUS_PATH", joinpath("test", "SISO_test", "cdn_status.json"))
    result_path = get(ENV, "BNC_RESULT_PATH", joinpath("test", "SISO_test", "cdn_result.json"))

    payload = Dict{String,Any}(
        "started_at" => string(now()),
        "stage" => "initializing",
        "cdn" => n,
        "condition_solver" => string(solver),
        "julia_threads" => Threads.nthreads(),
        "heartbeat_seconds" => heartbeat_seconds,
    )
    write_status(status_path, payload)

    model = cdn_model(n)
    t0 = time()

    payload["stage"] = "finding_regimes"
    write_status(status_path, payload)
    find_time = @elapsed find_all_vertices!(model)
    payload["find_all_vertices_seconds"] = find_time
    payload["n_regimes"] = n_regimes(model)
    write_status(status_path, payload)

    payload["stage"] = "building_paths"
    write_status(status_path, payload)
    paths = @timed SISOPaths(model, 1; condition_solver=solver)
    payload["build_paths_seconds"] = paths.time
    payload["n_sources"] = length(get_sources(paths.value))
    payload["n_sinks"] = length(get_sinks(paths.value))
    payload["n_paths"] = length(getproperty(paths.value, :rgm_paths))
    payload["stage"] = "solving_polyhedra"
    write_status(status_path, payload)

    done_ref = Ref(false)
    monitor_task = @async begin
        while !done_ref[]
            sleep(heartbeat_seconds)
            done_ref[] && break
            helper_stats = collect_helper_stats(paths.value)
            heartbeat = copy(payload)
            merge!(heartbeat, helper_stats)
            heartbeat["elapsed_seconds"] = time() - t0
            heartbeat["updated_at"] = string(now())
            heartbeat["stage"] = "solving_polyhedra"
            write_status(status_path, heartbeat)
        end
    end

    poly = @timed get_polyhedra(paths.value)
    done_ref[] = true
    wait(monitor_task)

    result = Dict{String,Any}(
        "finished_at" => string(now()),
        "stage" => "completed",
        "cdn" => n,
        "condition_solver" => string(solver),
        "julia_threads" => Threads.nthreads(),
        "find_all_vertices_seconds" => find_time,
        "build_paths_seconds" => paths.time,
        "get_polyhedra_seconds" => poly.time,
        "elapsed_seconds" => time() - t0,
        "n_regimes" => n_regimes(model),
        "n_sources" => length(get_sources(paths.value)),
        "n_sinks" => length(get_sinks(paths.value)),
        "n_paths" => length(getproperty(paths.value, :rgm_paths)),
        "n_polyhedra" => length(poly.value),
    )
    merge!(result, collect_helper_stats(paths.value))

    write_status(status_path, result)
    write_status(result_path, result)
    println(JSON3.write(result))
    flush(stdout)
end

main()
