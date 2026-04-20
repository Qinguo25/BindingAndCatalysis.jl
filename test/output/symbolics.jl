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

@testset "SIMO Path Symbolics Smoke" begin
    model = Bnc(N = [2 1 -1])
    find_all_regimes!(model)
    simo = SIMOPaths(model, 1)
    cond = string(only(show_condition(simo, 1; log_space = false)))
    @test occursin("K", cond)
    @test occursin("q", cond)
    @test occursin(">", cond) || occursin("<", cond) || occursin("~", cond)
end
