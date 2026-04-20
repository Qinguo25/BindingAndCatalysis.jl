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

    regular_ids = get_regimes(model; singular = false, return_idx = true)
    assigned = [
        assign_regime(model, get_one_inner_point(model, idx); input_logspace = true, asymptotic_only = false, return_idx = true)
        for idx in regular_ids
    ]
    @test assigned == regular_ids

    simple = minimal_model()
    find_all_regimes!(simple)
    dir, ins = get_interface(simple, 2, 1)
    p_from = get_one_inner_point(simple, 2)
    p_to = get_one_inner_point(simple, 1)
    @test LinearAlgebra.dot(dir, p_from) + ins < 0
    @test LinearAlgebra.dot(dir, p_to) + ins > 0
end

@testset "Strict qK Classifier Errors" begin
    model = minimal_model()
    find_all_regimes!(model)
    grh = get_regimes_graph!(model; full = true)

    info = BindingAndCatalysis._get_regime_qK_hyperplane_id_signs(grh, 1)
    edge_12 = get_edge(grh, 1, 2; full = true)
    @test haskey(info, edge_12.qK_interface_idx)
    @test info[edge_12.qK_interface_idx] == -edge_12.qK_interface_sign

    dir, ins = get_interface(model, 2, 1)
    p_from = get_one_inner_point(model, 2)
    p_to = get_one_inner_point(model, 1)
    step = p_to - p_from
    t = -(LinearAlgebra.dot(dir, p_from) + ins) / LinearAlgebra.dot(dir, step)
    p_boundary = p_from + t * step
    boundary_err = try
        assign_regime_qK(model, p_boundary; input_logspace = true, asymptotic_only = false, return_idx = true, eps = 1e-10)
        nothing
    catch err
        err
    end
    @test boundary_err isa ErrorException
    @test occursin("hit hyperplane boundary", sprint(showerror, boundary_err))
    @test occursin("logqK=", sprint(showerror, boundary_err))
    @test occursin("signature=", sprint(showerror, boundary_err))

    multi_classifier = BindingAndCatalysis.QKHyperplaneClassifier(
        [1, 2],
        SparseVector{Float64, Int}[],
        Float64[],
        BitVector[],
        BitVector[],
    )
    nonunique_err = try
        BindingAndCatalysis._resolve_unique_qK_candidate(multi_classifier, [0.0])
        nothing
    catch err
        err
    end
    @test nonunique_err isa ErrorException
    @test occursin("is not unique", sprint(showerror, nonunique_err))
    @test occursin("candidate_ids=[1, 2]", sprint(showerror, nonunique_err))

    no_candidate_classifier = BindingAndCatalysis.QKHyperplaneClassifier(
        [1],
        [SparseArrays.sparsevec([1], [1.0], 1)],
        [0.0],
        [falses(1)],
        [trues(1)],
    )
    no_candidate_err = try
        BindingAndCatalysis._resolve_unique_qK_candidate(no_candidate_classifier, [1.0])
        nothing
    catch err
        err
    end
    @test no_candidate_err isa ErrorException
    @test occursin("found no candidate regime", sprint(showerror, no_candidate_err))
    @test occursin("candidate_ids=Int[]", sprint(showerror, no_candidate_err))
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
