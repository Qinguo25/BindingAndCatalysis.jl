@testset "D-Stability Certificates And Cache States" begin
    D = BindingAndCatalysis.DStable
    tol = 1.0e-8

    @test D._obvious_not_hurwitz(
        sparse(reshape([-1.0e-9], 1, 1)); spectral_tol=tol
    ) === missing
    @test D._obvious_not_hurwitz(
        sparse(reshape([1.0e-7], 1, 1)); spectral_tol=tol
    ) === true

    @test D.d_class(reshape([-1.0], 1, 1); tol=tol) === true
    @test D.d_class(reshape([1.0], 1, 1); tol=tol) === false
    @test D.d_class(reshape([-1.0e-9], 1, 1); tol=tol) === missing
    @test D.d_class(reshape([1.0e-9], 1, 1); tol=tol) === missing
    @test D.d_class(reshape([0.0], 1, 1); tol=tol) === missing

    @test D.d_class(Matrix(Diagonal([-1.0, -2.0])); tol=tol) === true
    @test D.d_class(Matrix(Diagonal([1.0, -1.0])); tol=tol) === false
    @test D.d_class(Matrix(Diagonal([-1.0, -1.0e-9])); tol=tol) === missing
    @test D.d_class([1.0e-9 1.0; -1.0 -1.0]; tol=tol) === missing

    @test D.d_class(-Matrix{Float64}(I, 3, 3); tol=tol) === true
    @test D.d_class(Matrix{Float64}(I, 3, 3); tol=tol) === false
    @test D.d_class(Matrix(Diagonal([-1.0, -1.0, -1.0e-9])); tol=tol) === missing
    @test D.d_class(Matrix(Diagonal([-1.0, -1.0, 1.0e-9])); tol=tol) === missing
    @test D.d_class([1.0e-9 -3.0 2.0; 0.0 -1.0 0.0; -1.0 -3.0 -1.0]; tol=tol) ===
        missing

    @test judge_dstable(
        reshape([-1.0e-9], 1, 1); spectral_tol=tol, margin_tol=tol
    ) === missing

    model = minimal_catalysis_model()
    match_regimes!(model; warn_singular_propagation=false)
    rgm = first(get_bnc_regimes(model; feasible=nothing))
    rgm.H_bd = spzeros(Float64, 1, 1)
    rgm.is_stable = nothing

    @test ismissing(is_stable(rgm; margin_tol=tol))
    @test ismissing(rgm.is_stable)
    @test stability_code(rgm) == 0

    rgm.H_bd = sparse(reshape([-1.0], 1, 1))
    @test ismissing(is_stable(rgm; margin_tol=tol))
    @test is_stable(rgm; recompute=true, margin_tol=tol) === true
    @test rgm.is_stable === true
    @test stability_code(rgm) == 1

    rgm.H_bd = spzeros(Float64, 1, 1)
    rgm.is_stable = nothing
    @test BindingAndCatalysis._get_filter(stable=true)(rgm) === false
    @test BindingAndCatalysis._get_filter(stable=false)(rgm) === false
end
