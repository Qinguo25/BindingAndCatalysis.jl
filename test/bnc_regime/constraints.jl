@testset "BNC Analysis Constraints" begin
    model = minimal_catalysis_model()
    match_regimes!(model)

    constraints = parameter_constraints(
        model; equalities=[:k1 => :k2], inequalities=[(:K, :<, :tS)]
    )

    @test constraints.chart == :wKk
    @test constraints.compatible
    @test constraints.nullity == 1
    @test size(constraints.basis, 1) == length(wKk_symbol(model))
    @test size(constraints.basis, 2) == length(wKk_symbol(model)) - 1
    @test !isempty(constraints.notes)

    rgms = get_bnc_regimes(model)
    restricted = restrict_regimes(
        rgms, constraints; stable=nothing, singular=false, feasible=true, full_dim=nothing
    )
    @test all(rr.constraints === constraints for rr in restricted)
    @test all(rr.ambient_dim == size(constraints.basis, 2) for rr in restricted)

    pairs = stable_regime_intersections(rgms; constraints=constraints, full_dim=nothing)
    @test all(hasproperty(row, :regime_i) for row in pairs)
    @test all(hasproperty(row, :regime_j) for row in pairs)
    @test all(row.ambient_dim == size(constraints.basis, 2) for row in pairs)

    profile = multistability_profile(
        model;
        constraints=constraints,
        samples=50,
        max_draws=1_000,
        sampler=:uniform_box,
        log_lower=-2.0,
        log_upper=2.0,
        rng_seed=1,
    )
    @test profile.denominator == :constraint_region
    @test profile.accepted_samples == 50
    @test profile.R_atleast_1 >= profile.R_atleast_2 >= profile.R_atleast_3
    @test hasproperty(profile, :combination_counts)
    @test hasproperty(profile, :pair_table)
end

@testset "Binding Analysis Constraints Default to qK" begin
    model = minimal_model()
    constraints = parameter_constraints(model; equalities=[:tE => :tS])

    @test constraints.chart == :qK
    @test constraints.compatible
    @test size(constraints.basis, 1) == length(qK_symbol(model))
    @test size(constraints.basis, 2) == length(qK_symbol(model)) - 1
end
