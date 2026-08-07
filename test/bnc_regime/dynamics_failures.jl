@testset "Catalysis Binding Failures Terminate Integration" begin
    model = Bnc(
        ;
        L=[1 1 0; 1 0 0],
        N=[0 0 1],
        x_sym=[:x1, :x2, :x3],
        q_sym=[:qcat, :w],
        K_sym=[:K],
    )
    update_catalysis!(
        model;
        Γ=[1 -1],
        Π=[1 0 0; 0 1 0],
        q_picked=[:qcat],
        k_sym=[:k1, :k2],
    )

    removed_keyword_error = try
        qcat_traj_cat(
            model,
            [-1.0],
            zeros(4),
            (0.0, 0.01);
            fail_on_binding_error=false,
        )
        nothing
    catch error
        sprint(showerror, error)
    end
    @test removed_keyword_error ==
        "ArgumentError: keyword `fail_on_binding_error` is no longer supported; " *
        "binding solve failures now always terminate the integration."

    @test_throws ErrorException simulate_catalysis_trajectory(
        model;
        logqcat0=[-1.0],
        logwKk=zeros(4),
        tspan=(0.0, 0.01),
        method=:free_energy,
        qK2x_maxiters=20,
        homotopy_fallback=false,
        saveat=[0.0, 0.01],
    )
end
