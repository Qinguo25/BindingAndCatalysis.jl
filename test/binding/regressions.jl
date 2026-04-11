@testset "Two-Row N Singular H Sign Invariance" begin
    N = [
        1 1 -1 0
        0 1  1 -1
    ]
    singular_perm = [3, 3]

    for mode in (:float, :exact)
        model = Bnc(N = N)
        swapped_model = Bnc(N = N[[2, 1], :])

        find_all_regimes!(model; mode = mode)
        find_all_regimes!(swapped_model; mode = mode)

        @test have_perm(model, singular_perm)
        @test have_perm(swapped_model, singular_perm)
        @test get_nullity(model, singular_perm) == 1
        @test get_nullity(swapped_model, singular_perm) == 1

        H = Matrix(get_H(model, singular_perm))
        H_swapped = Matrix(get_H(swapped_model, singular_perm))

        @test H == H_swapped
        @test H != -H_swapped
    end
end

@testset "Sparse L/N Singular Seed Fallback" begin
    model_float = sparse_singular_model()
    model_rational = sparse_singular_model()

    find_all_regimes!(model_float; mode = :float)
    find_all_regimes!(model_rational; mode = :exact)

    @test n_regimes(model_float) == 24
    @test n_regimes(model_rational) == 24

    singular_idx = first(filter(i -> get_nullity(model_rational, i) == 1, get_indices(model_rational)))
    singular_perm = get_perm(model_rational, singular_idx)

    @test have_perm(model_float, singular_perm)

    Hf, H0f = get_H_H0(model_float, singular_perm)
    Hr, H0r = get_H_H0(model_rational, singular_perm)

    @test size(Hf) == (model_float.n, model_float.n)
    @test length(H0f) == model_float.n
    @test Matrix(Hf) ≈ Float64.(Matrix(Hr))
    @test H0f ≈ H0r
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
    find_all_regimes!(model; mode = :exact)
    idx = first(filter(i -> get_nullity(model, i) > 1, get_indices(model)))
    cond_log = show_condition_qK(model, idx)
    cond_lin = show_condition_qK(model, idx; log_space = false)

    @test get_nullity(model, idx) == 2
    @test !isempty(cond_log)
    @test !isempty(cond_lin)
    @test all(c -> !occursin(".0", string(c)), cond_lin)
end
