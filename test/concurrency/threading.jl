@testset "Nested Threaded Regime Propagation" begin
    if Threads.nthreads() > 1
        model = clique5_binding_model()
        @test begin
            find_all_regimes!(model)
            n_regimes(model) > 0
        end
    end
end
