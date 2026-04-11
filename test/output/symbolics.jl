@testset "Exact Symbolics Without Decimal Suffix" begin
    model = minimal_model()
    find_all_regimes!(model; mode = :exact)
    exprs = vcat(
        string.(show_condition_qK(model, 4; log_space = false)),
        string.(show_expression_x(model, 1; log_space = false)),
        string.(show_expression_qK(model, 1; log_space = false)),
    )
    @test all(s -> !occursin(".0", s), exprs)

    cat_model = minimal_catalysis_model()
    find_all_regimes!(cat_model; mode = :exact)
    find_catalysis_regimes!(cat_model)
    match_regimes!(cat_model)
    regular = first(filter(r -> r.nlt == 0, get_bnc_regimes(cat_model)))
    cat_exprs = vcat(
        string.(show_expression_qcat(regular; log_space = false)),
        string.(show_condition_qssKk(regular; log_space = false)),
        string.(show_condition_qKk(regular; log_space = false)),
    )
    @test all(s -> !occursin(".0", s), cat_exprs)
end
