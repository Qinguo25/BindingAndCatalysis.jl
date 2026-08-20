@testset "Nested Threaded Regime Propagation" begin
    if Threads.nthreads() > 1
        model = sparse_singular_model()
        @test begin
            find_all_regimes!(model)
            n_regimes(model) > 0
        end
    end
end

@testset "Concurrent Lazy Regime Cache Initialization" begin
    if Threads.nthreads() > 1
        n_tasks = 4

        binding_model = minimal_model()
        # Use byte-addressed Bool vectors: BitVector packs all four flags into
        # one word, so concurrent writes to distinct logical indices race.
        binding_ok = fill(false, n_tasks)
        Threads.@threads for i in 1:n_tasks
            ensure_binding_regimes!(binding_model)
            binding_ok[i] = n_bind_regimes(binding_model) > 0
        end
        @test all(binding_ok)

        graph_model = minimal_catalysis_model()
        graph_ok = fill(false, n_tasks)
        Threads.@threads for i in 1:n_tasks
            graph_ok[i] = !isnothing(get_catalysis_regimes_graph!(graph_model))
        end
        @test all(graph_ok)

        bnc_model = minimal_catalysis_model()
        bnc_ok = fill(false, n_tasks)
        Threads.@threads for i in 1:n_tasks
            ensure_bnc_regimes!(bnc_model)
            bnc_ok[i] = n_bnc_regimes(bnc_model) > 0
        end
        @test all(bnc_ok)
    end
end
