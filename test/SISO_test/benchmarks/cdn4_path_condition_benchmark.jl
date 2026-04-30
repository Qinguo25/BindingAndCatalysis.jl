using BindingAndCatalysis

const CDN4_N = [
    1 1 0 0 -1 0 0 0 0 0
    1 0 1 0 0 -1 0 0 0 0
    1 0 0 1 0 0 -1 0 0 0
    0 1 1 0 0 0 0 -1 0 0
    0 1 0 1 0 0 0 0 -1 0
    0 0 1 1 0 0 0 0 0 -1
]

function build_cdn4_model()
    return Bnc(N = CDN4_N)
end

function build_paths(model, change_qK; condition_solver=:recursive)
    try
        return SISOPaths(model, change_qK; condition_solver=condition_solver)
    catch err
        if err isa UndefVarError && err.var === :paths && isdefined(BindingAndCatalysis, :SIMOPaths)
            qK_grh = get_SISO_graph(model, change_qK)
            sources, sinks = get_sources_sinks(model, qK_grh)
            rgm_paths = BindingAndCatalysis._enumerate_paths(qK_grh; sources, sinks)
            change_qK_idx = locate_sym_qK(model, change_qK)
            return BindingAndCatalysis.SIMOPaths(model, qK_grh, change_qK_idx, sources, sinks, rgm_paths)
        end
        rethrow()
    end
end

function time_call(f)
    GC.gc()
    elapsed = @elapsed value = f()
    return value, elapsed
end

function collect_backend_stats(paths)
    stats = Dict{String,Any}()

    if hasproperty(paths, :condition_helper)
        helper = getproperty(paths, :condition_helper)
        if !isnothing(helper)
            pair_conditions = getproperty(helper, :pair_conditions)
            stats["backend"] = "memoized_pair_solver"
            stats["condition_solver"] = getproperty(paths, :condition_solver)
            stats["cached_vertex_prisms"] = count(!isnothing, getproperty(helper, :vertex_prisms))
            stats["cached_interface_prisms"] = length(getproperty(helper, :interface_prisms))
            stats["cached_pairs"] = length(pair_conditions)
            stats["cached_path_condition_entries"] = sum(length(values) for values in values(pair_conditions))
            if hasproperty(helper, :dag_profile)
                profile = getproperty(helper, :dag_profile)
                if !isnothing(profile)
                    stats["dag_planning_seconds"] = getproperty(profile, :planning_ns) / 1e9
                    stats["dag_pair_solve_seconds"] = getproperty(profile, :pair_solve_ns) / 1e9
                    stats["dag_middle_collect_seconds"] = getproperty(profile, :middle_collect_ns) / 1e9
                    stats["dag_middle_compute_seconds"] = getproperty(profile, :middle_compute_ns) / 1e9
                    stats["dag_middle_merge_seconds"] = getproperty(profile, :middle_merge_ns) / 1e9
                    stats["dag_pair_solve_calls"] = getproperty(profile, :pair_solve_calls)
                    stats["dag_middle_parallel_nodes"] = getproperty(profile, :middle_parallel_nodes)
                    stats["dag_middle_serial_nodes"] = getproperty(profile, :middle_serial_nodes)
                    stats["dag_middle_join_pairs"] = getproperty(profile, :middle_join_pairs)
                    if hasproperty(profile, :queue_pair_tasks)
                        stats["dag_queue_pair_tasks"] = getproperty(profile, :queue_pair_tasks)
                        stats["dag_queue_chunk_tasks"] = getproperty(profile, :queue_chunk_tasks)
                        stats["dag_queue_chunked_pairs"] = getproperty(profile, :queue_chunked_pairs)
                        stats["dag_queue_finalize_tasks"] = getproperty(profile, :queue_finalize_tasks)
                        stats["dag_queue_max_chunks_per_pair"] = getproperty(profile, :queue_max_chunks_per_pair)
                        stats["dag_queue_max_chunk_estimated_entries"] =
                            getproperty(profile, :queue_max_chunk_estimated_entries)
                        stats["dag_queue_total_chunk_estimated_entries"] =
                            getproperty(profile, :queue_total_chunk_estimated_entries)
                        stats["dag_queue_max_chunk_seconds"] = getproperty(profile, :queue_max_chunk_seconds)
                        stats["dag_queue_total_chunk_seconds"] = getproperty(profile, :queue_total_chunk_seconds)
                        stats["dag_queue_finalize_seconds"] = getproperty(profile, :queue_finalize_ns) / 1e9
                        if hasproperty(profile, :queue_estimator_entries_per_second)
                            stats["dag_queue_chunk_candidate_pairs"] =
                                getproperty(profile, :queue_chunk_candidate_pairs)
                            stats["dag_queue_chunk_size_gate_skips"] =
                                getproperty(profile, :queue_chunk_size_gate_skips)
                            stats["dag_queue_chunk_width_gate_skips"] =
                                getproperty(profile, :queue_chunk_width_gate_skips)
                            stats["dag_queue_chunk_thread_gate_skips"] =
                                getproperty(profile, :queue_chunk_thread_gate_skips)
                            stats["dag_queue_estimator_entries_per_second"] =
                                getproperty(profile, :queue_estimator_entries_per_second)
                            stats["dag_queue_estimator_target_entries"] =
                                getproperty(profile, :queue_estimator_target_entries)
                            stats["dag_queue_estimator_min_parallel_entries"] =
                                getproperty(profile, :queue_estimator_min_parallel_entries)
                            stats["dag_queue_estimator_target_seconds"] =
                                getproperty(profile, :queue_estimator_target_seconds)
                        end
                    end
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
        end
    end

    if hasproperty(paths, :edge_keys)
        stats["backend"] = "node_edge_polyhedra"
        stats["unique_edges"] = length(getproperty(paths, :edge_keys))
        stats["path_edge_references"] = sum(length, getproperty(paths, :path_edge_idxs))
        stats["computed_node_polyhedra"] = count(getproperty(paths, :node_polys_is_calc))
        stats["computed_edge_polyhedra"] = count(getproperty(paths, :edge_polys_is_calc))
    end

    return stats
end

function benchmark_solver(condition_solver)
    model = build_cdn4_model()

    _, find_time = time_call(() -> find_all_vertices!(model))
    paths, build_time = time_call(() -> build_paths(model, 1; condition_solver=condition_solver))
    polys, poly_time = time_call(() -> get_polyhedra(paths))

    result = Dict{String,Any}(
        "branch" => try
            readchomp(`git rev-parse --abbrev-ref HEAD`)
        catch
            "unknown"
        end,
        "condition_solver" => string(condition_solver),
        "find_all_vertices_seconds" => find_time,
        "build_paths_seconds" => build_time,
        "get_polyhedra_seconds" => poly_time,
        "n_regimes" => n_regimes(model),
        "n_sources" => length(get_sources(paths)),
        "n_sinks" => length(get_sinks(paths)),
        "n_paths" => length(getproperty(paths, :rgm_paths)),
        "n_polyhedra" => length(polys),
    )

    merge!(result, collect_backend_stats(paths))
    return result
end

function main()
    solver = get(ENV, "BNC_CDN4_SOLVER", "both")
    result =
        solver == "recursive" ? benchmark_solver(:recursive) :
        solver == "dag" ? benchmark_solver(:dag) :
        Dict("recursive" => benchmark_solver(:recursive), "dag" => benchmark_solver(:dag))
    output_path = get(ENV, "BNC_CDN4_OUTPUT", "")
    if isempty(output_path)
        println(repr(result))
        flush(stdout)
    else
        open(output_path, "w") do io
            println(io, repr(result))
            flush(io)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
