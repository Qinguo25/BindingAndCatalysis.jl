@testset "Two-Row N Singular H Sign Invariance" begin
    N = [
        1 1 -1 0
        0 1  1 -1
    ]
    singular_perm = [3, 3]

    model = Bnc(N = N)
    swapped_model = Bnc(N = N[[2, 1], :])

    find_all_regimes!(model)
    find_all_regimes!(swapped_model)

    @test have_perm(model, singular_perm)
    @test have_perm(swapped_model, singular_perm)
    @test get_nullity(model, singular_perm) == 1
    @test get_nullity(swapped_model, singular_perm) == 1

    H = Matrix(get_H(model, singular_perm))
    H_swapped = Matrix(get_H(swapped_model, singular_perm))

    @test H == H_swapped
    @test H != -H_swapped
end

@testset "Sparse L/N Singular Seed Fallback" begin
    model = sparse_singular_model()
    find_all_regimes!(model)

    @test n_regimes(model) == 24

    singular_idx = first(filter(i -> get_nullity(model, i) == 1, get_indices(model)))

    H, H0 = get_H_H0(model, singular_idx)
    CqK, C0qK, nullity = get_C_C0_nullity_qK(model, singular_idx)
    poly = get_polyhedron(model, singular_idx)

    @test size(H) == (model.n, model.n)
    @test length(H0) == model.n
    @test eltype(H) <: Rational
    @test eltype(H0) == ExactLogExpr
    @test nullity == 1
    @test eltype(CqK) == Float64
    @test eltype(C0qK) == Float64
    @test eltype(get_C(poly)) == Float64
end

@testset "Shared Hyperplane Assignment And Interface Orientation" begin
    model = notebook_model2()
    find_all_regimes!(model)

    Random.seed!(1234)
    samples = [rand(5) .* 12 .- 6 for _ in 1:100]
    assigned = [assign_regime(model, x; input_logspace = true, asymptotic_only = false, return_idx = true) for x in samples]
    fallback = [BindingAndCatalysis._assign_regime_qK_idx_fallback(model, x; asymptotic_only = false, eps = 0.0, warn_on_fallback = false) for x in samples]
    @test assigned == fallback

    simple = minimal_model()
    find_all_regimes!(simple)
    dir, ins = get_interface(simple, 2, 1)
    p_from = get_one_inner_point(simple, 2)
    p_to = get_one_inner_point(simple, 1)
    @test LinearAlgebra.dot(dir, p_from) + ins < 0
    @test LinearAlgebra.dot(dir, p_to) + ins > 0
end

@testset "High Nullity Exact Conditions" begin
    model = sparse_singular_model()
    find_all_regimes!(model)
    idx = first(filter(i -> get_nullity(model, i) > 1, get_indices(model)))
    cond_log = show_condition_qK(model, idx)
    cond_lin = show_condition_qK(model, idx; log_space = false)

    @test get_nullity(model, idx) == 2
    @test !isempty(cond_log)
    @test !isempty(cond_lin)
    @test all(c -> !occursin(".0", string(c)), cond_lin)
end
