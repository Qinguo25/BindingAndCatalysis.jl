include("cdn4_path_condition_benchmark.jl")

using Dates

const RESULT_DIR = joinpath(@__DIR__, "results")

function with_env(f, overrides::Dict{String,String})
    old_values = Dict{String,Union{Nothing,String}}()
    for key in keys(overrides)
        old_values[key] = get(ENV, key, nothing)
    end
    try
        for (key, value) in overrides
            ENV[key] = value
        end
        return f()
    finally
        for (key, value) in old_values
            if value === nothing
                delete!(ENV, key)
            else
                ENV[key] = value
            end
        end
    end
end

function benchmark_mode(name::String, env::Dict{String,String})
    println("Running CDN4 scheduler mode: ", name)
    result = with_env(env) do
        benchmark_solver(:dag)
    end
    result["scheduler_mode"] = name
    result["julia_threads"] = Threads.nthreads()
    return result
end

function write_results(results)
    mkpath(RESULT_DIR)
    timestamp = replace(string(Dates.now()), ':' => '-')
    output_path = joinpath(RESULT_DIR, "cdn4_scheduler_comparison_$(timestamp).txt")
    open(output_path, "w") do io
        for result in results
            println(io, repr(result))
        end
    end
    return output_path
end

function main()
    modes = [
        (
            "dag_default",
            Dict(
                "BNC_SISO_DAG_SCHEDULER" => "auto",
            ),
        ),
        (
            "dag_serial",
            Dict(
                "BNC_SISO_DAG_SCHEDULER" => "serial",
                "BNC_SISO_DAG_CHUNK_QUEUE" => "false",
            ),
        ),
        (
            "pair_chunk_queue_forced",
            Dict(
                "BNC_SISO_DAG_SCHEDULER" => "queue",
                "BNC_SISO_DAG_CHUNK_QUEUE" => "true",
                "BNC_SISO_DAG_INNER_PARALLEL_MIN_WEIGHT" => "1",
                "BNC_SISO_DAG_INNER_PARALLEL_TARGET_ENTRIES" => "50",
                "BNC_SISO_DAG_CHUNK_SIZE_GATE" => "false",
            ),
        ),
    ]

    results = [benchmark_mode(name, env) for (name, env) in modes]
    for result in results
        println(repr(result))
    end
    output_path = write_results(results)
    println("Wrote ", output_path)
end

main()
