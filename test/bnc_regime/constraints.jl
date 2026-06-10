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
    @test hasproperty(profile, :stable_count_histogram)
    @test hasproperty(profile, :R_exact_stable_count)
    @test hasproperty(profile, :R_atleast_stable_count)
    @test hasproperty(profile, :max_stable_count)
    @test get(profile.R_atleast_stable_count, 1, 0.0) >=
        get(profile.R_atleast_stable_count, 2, 0.0) >=
        get(profile.R_atleast_stable_count, 3, 0.0)
    @test sum(values(profile.R_exact_stable_count)) ≈ 1.0
    @test hasproperty(profile, :combination_counts)
    @test hasproperty(profile, :pair_table)

    syms = wKk_symbol(model)
    mapped_chart = parameter_chart(
        model; map=Dict(syms[1] => :shared_parameter, syms[2] => :shared_parameter)
    )
    @test mapped_chart.chart == :wKk
    @test mapped_chart.basis_kind == :identified_parameters
    @test :shared_parameter in mapped_chart.reduced_symbols
    @test size(mapped_chart.F, 1) == length(syms)
    @test size(mapped_chart.F, 2) == length(syms) - 1
    @test mapped_chart.basis == mapped_chart.F
    @test mapped_chart.offset == mapped_chart.F0

    empty_map_chart = parameter_chart(model; map=Dict{Symbol, Symbol}())
    @test empty_map_chart.basis_kind == :identity
    @test empty_map_chart.reduced_symbols == syms
    @test empty_map_chart.F ≈ Matrix{Float64}(I, length(syms), length(syms))

    original_constraints = parameter_constraints(
        model;
        map=Dict(syms[1] => :shared_parameter, syms[2] => :shared_parameter),
        inequalities=[(syms[1], :<, syms[end])],
    )
    reduced_constraints = parameter_constraints(
        mapped_chart; inequalities=[(:shared_parameter, :<, syms[end])]
    )
    @test original_constraints.basis_kind == :identified_parameters
    @test reduced_constraints.basis_kind == :identified_parameters
    @test original_constraints.reduced_symbols == reduced_constraints.reduced_symbols
    @test original_constraints.reduced_inequality_C ≈
        reduced_constraints.reduced_inequality_C

    identity_chart = parameter_chart(
        model;
        F=Matrix{Float64}(I, length(syms), length(syms)),
        F0=zeros(length(syms)),
        reduced_symbols=syms,
    )
    @test identity_chart.basis_kind == :provided
    @test identity_chart.original_symbols == syms

    asymptotic_profile = multistability_profile(
        model;
        constraints=original_constraints,
        samples=20,
        max_draws=1_000,
        sampler=:uniform_box,
        log_lower=-2.0,
        log_upper=2.0,
        rng_seed=2,
        mode=:asymptotic_R,
    )
    @test asymptotic_profile.mode == :asymptotic_R
    @test asymptotic_profile.denominator == :constraint_cone
    @test asymptotic_profile.basis_kind == :identified_parameters

    summary = multistability_R_index(
        model;
        constraints=original_constraints,
        samples=20,
        max_draws=1_000,
        sampler=:uniform_box,
        log_lower=-2.0,
        log_upper=2.0,
        rng_seed=3,
    )
    @test summary.mode == :asymptotic_R
    @test summary.denominator == :constraint_cone
    @test summary.R_atleast_stable_count === summary.profile.R_atleast_stable_count
    @test hasproperty(summary, :stderr_atleast_stable_count)
    @test hasproperty(summary, :full_dim_regimes)
    @test hasproperty(summary, :stable_full_dim_regimes)
    @test hasproperty(summary, :max_stable_count)
    @test summary.full_dim_regimes >= summary.stable_full_dim_regimes
    @test summary.stable_full_dim_regimes == length(summary.profile.restricted_regimes)
end

@testset "Binding Analysis Constraints Default to qK" begin
    model = minimal_model()
    constraints = parameter_constraints(model; equalities=[:tE => :tS])

    @test constraints.chart == :qK
    @test constraints.compatible
    @test size(constraints.basis, 1) == length(qK_symbol(model))
    @test size(constraints.basis, 2) == length(qK_symbol(model)) - 1
end
