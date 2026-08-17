@testset "Seed Analysis Publishes Per-Vertex Entries" begin
    model = minimal_model()
    find_all_regimes!(model)
    regimes = get_binding_regimes(model)

    idx = findfirst(eachindex(regimes)) do i
        _, pdef = BindingAndCatalysis._get_Nρ_key_and_perm_nullity(
            regimes[i].perm, model.n
        )
        return pdef == 0
    end
    @test !isnothing(idx)

    state = BindingAndCatalysis.SeedAnalysisState(length(regimes), 1)
    first_result = BindingAndCatalysis._ensure_seed_analysis!(
        state, idx, regimes, model.N
    )
    second_result = BindingAndCatalysis._ensure_seed_analysis!(
        state, idx, regimes, model.N
    )

    @test first_result[1] == BindingAndCatalysis._SEED_STATUS_REGULAR
    @test first_result[1:3] == second_result[1:3]
    @test !isnothing(first_result[4])
    @test !isnothing(second_result[4])
    @test first_result[4].deficiency == second_result[4].deficiency == 0
    @test first_result[4].kind == second_result[4].kind
    @test first_result[4].inv === second_result[4].inv
    @test !haskey(state.cache, first_result[3])
end

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
