function _strict_ring_model(n::Int)
    binding_matrix = zeros(Int, n, 3n)
    for i in 1:n
        predecessor = mod1(i - 1, n)
        binding_matrix[i, predecessor] = 1
        binding_matrix[i, n + i] = 1
        binding_matrix[i, 2n + i] = -1
    end

    model = Bnc(;
        N=binding_matrix,
        x_sym=vcat(
            [Symbol("P$i") for i in 1:n],
            [Symbol("D$i") for i in 1:n],
            [Symbol("C$i") for i in 1:n],
        ),
        q_sym=vcat([Symbol("tP$i") for i in 1:n], [Symbol("tD$i") for i in 1:n]),
        K_sym=[Symbol("K$i") for i in 1:n],
    )

    gamma_matrix = zeros(Int, n, 3n)
    for i in 1:n
        gamma_matrix[i, i] = -1
        gamma_matrix[i, n + i] = 1
        gamma_matrix[i, 2n + mod1(i + 1, n)] = -1
    end
    rate_map = zeros(Int, 3n, 2)
    rate_map[1:n, 2] .= 1
    rate_map[(n + 1):(2n), 1] .= 1
    rate_map[(2n + 1):(3n), 2] .= 1
    update_catalysis!(
        model;
        Π=Matrix{Int}(I, 3n, 3n),
        Γ=gamma_matrix,
        q_picked=[Symbol("tP$i") for i in 1:n],
        F=rate_map,
        F0=zeros(3n),
        k_sym=[:alpha, :gamma],
    )
    return model
end

function _strict_ring_chart(model)
    original = wKk_symbol(model)
    reduced = [:K, :qD, :alpha, :gamma]
    F = zeros(length(original), length(reduced))
    for (row, symbol) in enumerate(original)
        name = String(symbol)
        column = if startswith(name, "K")
            1
        elseif startswith(name, "tD")
            2
        elseif name == "alpha"
            3
        elseif name == "gamma"
            4
        else
            error("Unexpected two-node ring parameter $symbol.")
        end
        F[row, column] = 1
    end
    return parameter_chart(
        model; chart=:wKk, F=F, F0=zeros(length(original)), reduced_symbols=reduced
    )
end

function _strict_ring_binding_word(rgm, n::Int)
    perm = get_binding_perm(rgm)
    labels = Char[]
    for promoter in 1:n
        predecessor = mod1(promoter - 1, n)
        protein_is_free = perm[predecessor] == predecessor
        promoter_slot = n + promoter
        promoter_is_free = perm[promoter_slot] == promoter_slot
        push!(
            labels,
            if protein_is_free && promoter_is_free
                'F'
            elseif protein_is_free
                'B'
            elseif promoter_is_free
                'T'
            else
                'X'
            end,
        )
    end
    return String(labels)
end

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
    sample = first(restricted)
    legacy_constructed = RestrictedRegime(
        sample.regime,
        sample.constraints,
        sample.chart,
        sample.C,
        sample.C0,
        sample.nullity,
        sample.poly,
        sample.feasible,
        sample.dim,
        sample.ambient_dim,
        sample.full_dim,
        sample.reason,
    )
    @test (
        legacy_constructed.strict_feasible,
        legacy_constructed.strict_asymptotic,
        legacy_constructed.boundary_only,
    ) === (nothing, nothing, nothing)

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
    @test hasproperty(summary, :closure_full_dim_regimes)
    @test summary.closure_full_dim_regimes == summary.full_dim_regimes
    @test hasproperty(summary, :stable_full_dim_regimes)
    @test hasproperty(summary, :max_stable_count)
    @test summary.full_dim_regimes >= summary.stable_full_dim_regimes
    @test summary.stable_full_dim_regimes == length(summary.profile.restricted_regimes)
end

@testset "Strict Dominance Under Parameter Restriction" begin
    B = BindingAndCatalysis
    model = _strict_ring_model(2)
    match_regimes!(model; warn_singular_propagation=false)
    constraints = parameter_constraints(_strict_ring_chart(model))
    regimes = get_bnc_regimes(model; feasible=true)

    closures = restrict_regimes(
        regimes,
        constraints;
        stable=nothing,
        singular=nothing,
        feasible=true,
        full_dim=true,
        strict_feasible=nothing,
        strict_asymptotic=nothing,
    )
    @test length(closures) == 10
    by_word = Dict(_strict_ring_binding_word(rr.regime, 2) => rr for rr in closures)

    for word in ("BF", "FB", "BX", "XB")
        rr = by_word[word]
        @test rr.feasible
        @test rr.full_dim
        @test !rr.strict_feasible
        @test !rr.strict_asymptotic
        @test rr.boundary_only
    end
    for word in ("FF", "BB", "TT", "XX", "BT", "TB")
        rr = by_word[word]
        @test rr.feasible
        @test rr.full_dim
        @test rr.strict_feasible
        @test rr.strict_asymptotic
        @test !rr.boundary_only
    end

    expected_strict_words = Set(["FF", "BB", "TT", "XX", "BT", "TB"])
    strict_words = Set(
        _strict_ring_binding_word(rr.regime, 2) for rr in restrict_regimes(
            regimes,
            constraints;
            stable=nothing,
            singular=nothing,
            feasible=true,
            full_dim=true,
            strict_feasible=true,
            strict_asymptotic=true,
        )
    )
    @test strict_words == expected_strict_words

    for rgm in regimes
        rgm.is_stable = true
    end
    finite_profile = multistability_profile(
        model;
        constraints=constraints,
        regimes=regimes,
        samples=0,
        max_draws=0,
        singular=nothing,
        pair_intersections=false,
        mode=:finite_region,
    )
    asymptotic_profile = multistability_profile(
        model;
        constraints=constraints,
        regimes=regimes,
        samples=0,
        max_draws=0,
        singular=nothing,
        pair_intersections=false,
        mode=:asymptotic_R,
    )
    finite_profile_words = Set(
        _strict_ring_binding_word(rr.regime, 2) for rr in finite_profile.restricted_regimes
    )
    asymptotic_profile_words = Set(
        _strict_ring_binding_word(rr.regime, 2) for
        rr in asymptotic_profile.restricted_regimes
    )
    @test finite_profile_words == expected_strict_words
    @test asymptotic_profile_words == expected_strict_words

    summary = multistability_R_index(
        model;
        constraints=constraints,
        regimes=regimes,
        samples=0,
        max_draws=0,
        singular=nothing,
        mode=:asymptotic_R,
    )
    @test summary.closure_full_dim_regimes == 10
    @test summary.full_dim_regimes == summary.closure_full_dim_regimes
    summary_words = Set(
        _strict_ring_binding_word(rr.regime, 2) for rr in summary.profile.restricted_regimes
    )
    @test summary_words == expected_strict_words
    @test summary.stable_full_dim_regimes == 6

    bf = by_word["BF"].regime
    asymmetric = restrict_regime(bf, parameter_constraints(model; chart=:wKk); chart=:wKk)
    @test asymmetric.strict_feasible
    @test asymmetric.strict_asymptotic
    @test !asymmetric.boundary_only

    qKk_restriction = restrict_regime(
        bf, parameter_constraints(model; chart=:qKk); chart=:qKk
    )
    @test qKk_restriction.strict_feasible
    @test qKk_restriction.strict_asymptotic

    xb = by_word["XB"].regime
    @test is_singular(get_binding_regime(xb))
    singular_qKk = restrict_regime(xb, parameter_constraints(model; chart=:qKk); chart=:qKk)
    @test singular_qKk.strict_feasible isa Bool
    @test singular_qKk.strict_asymptotic isa Bool

    binding = get_binding_regime(bf)
    binding_restriction = restrict_regime(
        binding, parameter_constraints(model; chart=:qK); chart=:qK
    )
    @test binding_restriction.strict_feasible isa Bool
    @test binding_restriction.strict_asymptotic isa Bool

    raw = restrict_polyhedron(get_polyhedron(bf; chart=:wKk), constraints)
    @test isnothing(raw.strict_feasible)
    @test isnothing(raw.strict_asymptotic)
    @test isnothing(raw.boundary_only)

    base_finite_only = B._StrictJointSystem(
        zeros(0, 1),
        zeros(0, 1),
        Float64[],
        zeros(0, 1),
        Float64[],
        reshape([1.0, -1.0], 2, 1),
        [1.0, 1.0],
    )
    zero_fixed_point_equality = B._StrictJointSystem(
        zeros(1, 1),
        zeros(1, 1),
        [0.0],
        zeros(0, 1),
        Float64[],
        reshape([1.0, -1.0], 2, 1),
        [1.0, 1.0],
    )
    redundant_user_constraint = B._StrictJointSystem(
        zeros(0, 1),
        zeros(0, 1),
        Float64[],
        zeros(1, 1),
        [0.0],
        reshape([1.0, -1.0], 2, 1),
        [1.0, 1.0],
    )
    base_status = B._strict_joint_status(base_finite_only)
    zero_equality_status = B._strict_joint_status(zero_fixed_point_equality)
    redundant_user_status = B._strict_joint_status(redundant_user_constraint)
    @test base_status == (; strict_feasible=true, strict_asymptotic=false)
    @test zero_equality_status == base_status
    @test redundant_user_status == base_status

    forced_tie = B._StrictJointSystem(
        zeros(0, 1), zeros(0, 0), Float64[], zeros(0, 0), Float64[], zeros(1, 1), [0.0]
    )
    forced_tie_status = B._strict_joint_status(forced_tie)
    @test !forced_tie_status.strict_feasible
    @test !forced_tie_status.strict_asymptotic
end

@testset "Binding Analysis Constraints Default to qK" begin
    model = minimal_model()
    constraints = parameter_constraints(model; equalities=[:tE => :tS])

    @test constraints.chart == :qK
    @test constraints.compatible
    @test size(constraints.basis, 1) == length(qK_symbol(model))
    @test size(constraints.basis, 2) == length(qK_symbol(model)) - 1
end
