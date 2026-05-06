@testset "Exact Symbolics Without Decimal Suffix" begin
    model = minimal_model()
    find_all_regimes!(model)
    exprs = vcat(
        string.(show_condition_qK(model, 4; log_space = false)),
        string.(show_expression_x(model, 1; log_space = false)),
        string.(show_expression_qK(model, 1; log_space = false)),
    )
    @test all(s -> !occursin(".0", s), exprs)

    cat_model = minimal_catalysis_model()
    find_all_regimes!(cat_model)
    find_catalysis_regimes!(cat_model)
    match_regimes!(cat_model)
    regular = first(filter(r -> r.nlt == 0, get_bnc_regimes(cat_model)))
    cat_exprs = vcat(
        string.(show_expression_qcat(regular; log_space = false)),
        string.(show_condition_wKk(regular; log_space = false)),
        string.(show_condition_qKk(regular; log_space = false)),
    )
    @test all(s -> !occursin(".0", s), cat_exprs)
end

@testset "BncRegime Catalysis Dynamics Use Binding Chart" begin
    model = minimal_catalysis_model()
    find_all_regimes!(model)
    find_catalysis_regimes!(model)
    match_regimes!(model)
    rgm = first(filter(r -> r.nlt == 0 && !is_singular(get_binding_regime(r)), get_bnc_regimes(model)))

    dyn = string.(show_catalysis_dynamics(rgm))
    red = string.(show_catalysis_dynamics(rgm; reduced = true))

    @test any(occursin.(r"tE\*k1|k1\*tE", dyn))
    @test any(occursin.(r"tS\*k2|k2\*tS", dyn))
    @test !any(occursin.(r"(?<!t)E\*k1|k1\*(?<!t)E", dyn))
    @test !any(occursin.(r"(?<!t)S\*k2|k2\*(?<!t)S", dyn))
    @test any(occursin.(r"tE\*k1|k1\*tE", red))
    @test any(occursin.(r"tS\*k2|k2\*tS", red))
end

@testset "Catalysis Dynamics Dispatch Distinguishes Regime Types" begin
    model = offset_catalysis_model()
    find_all_regimes!(model)
    find_catalysis_regimes!(model)
    match_regimes!(model)

    bind = get_regime(model, 1)
    cat = get_catalysis_regime(model, 1; check = true)
    mixed = first(filter(r -> r.nlt == 0 && !is_singular(get_binding_regime(r)), get_bnc_regimes(model)))

    bind_dyn = string.(show_catalysis_dynamics(bind))
    cat_dyn = string.(show_catalysis_dynamics(cat))
    mixed_dyn = string.(show_catalysis_dynamics(mixed))

    @test any(occursin.(r"k2\*tS|tS\*k2", bind_dyn))
    @test any(occursin.("C*k3", cat_dyn))
    @test !any(occursin.(r"k2\*tS|tS\*k2", cat_dyn))
    @test any(occursin.(r"k1\*tE|tE\*k1|k2\*tS|tS\*k2", mixed_dyn))
    @test !any(occursin.("C*k3", mixed_dyn))
end

@testset "SIMO Path Symbolics Smoke" begin
    model = Bnc(N = [2 1 -1])
    find_all_regimes!(model)
    simo = SIMOPaths(model, 1)
    cond = string(only(show_condition(simo, 1; log_space = false)))
    @test occursin("K", cond)
    @test occursin("q", cond)
    @test occursin(">", cond) || occursin("<", cond) || occursin("~", cond)
end
