@testset "BindingAndCatalysis.jl" begin
    model = minimal_model()

    @test (model.r, model.n, model.d) == (1, 3, 2)
    @test length(show_conservation(model)) == model.d
    @test length(show_equilibrium(model; log_space = true)) == model.r
    @test length(show_equilibrium(model; log_space = false)) == model.r

    find_all_vertices!(model)
    @test ensure_binding_regimes!(model) === nothing
    @test !isempty(BindingAndCatalysis._bind_regimes_data(model))
    @test n_bind_regimes(BindingAndCatalysis._bind_regimes(model)) == n_regimes(model)

    first_perm = BindingAndCatalysis._bind_regimes_perms(model)[1]
    first_idx = get_idx(model, first_perm)
    @test have_perm(model, first_perm)
    @test have_perm(model, first_idx)
    @test get_perm(model, first_idx) == first_perm

    vertex = get_vertex(model, first_idx)
    @test ensure_regime_data!(vertex) === nothing
    C, C0, nullity = get_C_C0_nullity(vertex)
    @test size(C, 2) == model.n
    @test length(C0) == size(C, 1)
    @test nullity >= 0

    poly = get_polyhedron(model, first_idx)
    C_poly, C0_poly, nullity_poly = get_C_C0_nullity(poly)
    @test size(C_poly, 2) == model.n
    @test length(C0_poly) == size(C_poly, 1)
    @test nullity_poly == nullity
    @test zero(BindingAndCatalysis.Volume) == BindingAndCatalysis.Volume(0.0, 0.0)
    @test zeros(BindingAndCatalysis.Volume, 3) == fill(zero(BindingAndCatalysis.Volume), 3)

    Random.seed!(42)
    logqK = randomize(model, 1; log_lower = -2, log_upper = 2)[1]
    logx = qK2x(model, logqK; input_logspace = true, output_logspace = true)
    logqK_back = x2qK(model, logx; input_logspace = true, output_logspace = true)
    @test isapprox(logqK_back, logqK; atol = 1e-6, rtol = 1e-6)
end

@testset "Default qK2x Method Avoids Free Energy For Nonorthogonal L N" begin
    model = Bnc(N = [1 0], L = [1 1])

    @test BindingAndCatalysis._default_method(model) === :homotopy
    @test BindingAndCatalysis._resolve_qK2x_method(model, :free_energy) === :homotopy

    logqK = [0.0, 0.0]
    logx_default = qK2x(model, logqK; input_logspace = true, output_logspace = true)
    logx_explicit = qK2x(model, logqK; input_logspace = true, output_logspace = true, method = :free_energy)
    @test isapprox(logx_default, logx_explicit; atol = 1e-8, rtol = 1e-8)
end

@testset "Export Hygiene" begin
    undefined_exports = [s for s in names(BindingAndCatalysis, all=false) if !isdefined(BindingAndCatalysis, s)]
    @test isempty(undefined_exports)
    @test :benchmark_qK2x_methods ∉ names(BindingAndCatalysis, all=false)
    @test :lines ∉ names(BindingAndCatalysis, all=false)
end

@testset "Construction And Empty Catalysis Edge Cases" begin
    model = @test_logs (:warn, r"N has been reduced") Bnc(
        N = [1 1 -1; 2 2 -2],
        K_sym = [:K1, :K2],
    )
    @test model.r == 1
    @test string.(K_sym(model)) == ["K1"]

    no_cat = minimal_model()
    Lcat = @test_logs (:warn, r"does not contain a catalysis network") get_Lcat(no_cat)
    @test size(Lcat) == (0, no_cat.n)
end

@testset "Uniform Box Per-Dimension Sampling" begin
    sampling = BindingAndCatalysis._prepare_sampling_config(
        :uniform_box,
        2;
        log_lower = [-1, -2],
        log_upper = [1, 4],
    )

    @test sampling.log_lower == [-1.0, -2.0]
    @test sampling.box_width == [2.0, 6.0]
    @test sampling.sample_weight == 12.0

    rng = MersenneTwister(123)
    x = zeros(2)
    for _ in 1:100
        BindingAndCatalysis._draw_sample!(x, rng, sampling)
        @test -1.0 <= x[1] <= 1.0
        @test -2.0 <= x[2] <= 4.0
    end

    @test_throws AssertionError BindingAndCatalysis._prepare_sampling_config(
        :uniform_box,
        2;
        log_lower = [-1],
        log_upper = [1, 4],
    )
end

@testset "Minimal Notebook Workflow" begin
    model = minimal_model()

    find_all_regimes!(model)
    rgms = get_regimes(model)
    perms = get_perms(model)
    idxs = get_indices(model)
    perm_dict = get_bind_regimes_dict(model)

    @test length(rgms) == length(perms) == n_regimes(model) == length(idxs) == length(perm_dict)
    @test idxs == collect(1:n_regimes(model))
    @test all(r -> r isa BindingAndCatalysis.BindRegime, rgms)
    @test get_perms(rgms) == perms
    @test get_indices(rgms) == idxs
    @test model.vertices_graph !== nothing
    bind_data = BindingAndCatalysis._bind_regimes_data(model)
    @test all(i -> !isnothing(bind_data[i].H) && !isnothing(bind_data[i].H0),
        filter(i -> get_nullity(model, i) <= 1, idxs))

    r1_perm = perms[1]
    r2_perm = perms[2]
    r3_perm = perms[3]
    r1 = get_regime(model, r1_perm)
    @test r1 === get_regime(model, 1)

    C1, C01, nlt1 = get_C_C0_nullity(r1)
    C2, C02, nlt2 = get_C_C0_nullity(model, 1)
    C3, C03, nlt3 = get_C_C0_nullity(model, r1_perm)
    @test C1 == C2 == C3
    @test C01 == C02 == C03
    @test nlt1 == nlt2 == nlt3

    @test get_perm(r1) == r1_perm
    @test get_idx(model, r1_perm) == 1
    @test get_nullity(model, 1) == get_nullity(r1)
    @test is_singular(r1) == (get_nullity(r1) > 0)
    @test is_asymptotic(model, 1)
    @test occursin("dominant mode", sprint(show, MIME"text/plain"(), r1))
    @test occursin("nullity", sprint(show, MIME"text/plain"(), r1))
    @test occursin("asymptotic", sprint(show, MIME"text/plain"(), r1))

    P, P0 = get_P_P0(model, r1_perm)
    H, H0 = get_H_H0(model, 1)
    Cx, C0x = get_C_C0_x(model, 1)
    CqK, C0qK, nltqK = get_C_C0_nullity_qK(model, r1_perm)

    @test get_P(model, 1) == P
    @test get_P0(r1) == P0
    @test get_H(r1) == H
    @test get_H0(model, r1_perm) == H0
    @test get_C_x(model, r1_perm) == Cx
    @test get_C0_x(r1) == C0x
    @test get_C_C0(model, 1) == (CqK, C0qK)
    @test nltqK == get_nullity(r1)

    @test !isempty(show_condition_x(r1))
    @test !isempty(show_condition_qK(model, 1; log_space = false))
    @test !isempty(show_dominant_condition(r1; log_space = false))

    poly = get_polyhedron(model, r1_perm)
    Cpoly, C0poly, nltpoly = get_C_C0_nullity(poly)
    @test Cpoly == CqK
    @test C0poly == C0qK
    @test nltpoly == nltqK

    inner = get_one_inner_point(model, 2)
    @test assign_regime(model, inner; input_logspace = true, asymptotic_only = false, return_idx = true) == 2

    C_add = [1 -1 0]
    C0_add = [-log10(2)]
    feas = check_feasibility_with_constraint(model, 4; C = C_add, C0 = C0_add)
    feas_list = feasible_vertices_with_constraint(model; C = C_add, C0 = C0_add, return_idx = true)
    @test feas isa Bool
    @test all(i -> i in idxs, feas_list)

    vg = get_regimes_graph!(model; full = true)
    @test length(vg.neighbors) == n_regimes(model)
    @test size(get_regimes_neighbor_mat(model), 1) == n_regimes(model)
    @test BindingAndCatalysis.Graphs.nv(get_neighbor_graph_x(model)) == n_regimes(model)

    @test get_edge(vg, r2_perm, r1_perm) !== nothing
    @test get_edge(vg, r2_perm, r3_perm) === nothing
    edge_21 = get_edge(vg, r2_perm, r1_perm; full = true)
    edge_12 = get_edge(vg, r1_perm, r2_perm; full = true)
    qK_21 = BindingAndCatalysis._edge_idx_sign(edge_21, vg, :qK)
    qK_12 = BindingAndCatalysis._edge_idx_sign(edge_12, vg, :qK)
    @test qK_21[1] == qK_12[1] != 0
    @test qK_21[2] == -qK_12[2]

    Random.seed!(42)
    logqK_vec = randomize(model, 4; log_lower = -3, log_upper = 3)
    logx_vec = logqK_vec .|> qK -> qK2x(model, qK; input_logspace = true, output_logspace = true)
    logqK_vec_back = logx_vec .|> x -> x2qK(model, x; input_logspace = true, output_logspace = true)
    @test all(isapprox.(logqK_vec_back, logqK_vec; atol = 1e-6, rtol = 1e-6))

    assigned_qK = logqK_vec .|> qK -> assign_regime(model, qK; input_logspace = true, asymptotic_only = false, return_idx = true)
    assigned_from_x_qK = logx_vec .|> x -> assign_regime_qK(model; x = x, input_logspace = true, asymptotic_only = false, return_idx = true)
    assigned_from_x_x = logx_vec .|> x -> assign_regime_x(model, x; input_logspace = true, asymptotic_only = true, return_idx = true)
    @test assigned_qK == assigned_from_x_qK
    @test assigned_from_x_qK == assigned_from_x_x

    singular_bind_idx = only(filter(i -> get_nullity(model, i) == 1, get_regimes(model; return_idx = true)))
    Hs, H0s = get_H_H0(model, singular_bind_idx)
    @test size(Hs, 1) == model.n
    @test length(H0s) == model.n
    @test get_volume(model, singular_bind_idx; recalculate = true) == zero(BindingAndCatalysis.Volume)

    subset = [1, singular_bind_idx, 2]
    filtered_subset, subset_mask = BindingAndCatalysis.filter_regimes(
        model,
        subset;
        singular = false,
        return_mask = true,
    )
    @test filtered_subset == [1, 2]
    @test subset_mask == BitVector([true, false, true])
end

@testset "Small CDN3 Polyhedra And Volume Route" begin
    model = cdn3_small_model()
    find_all_regimes!(model)
    idxs = get_regimes(model; return_idx = true)

    polys_default = get_polyhedra(model)
    polys_unc = get_polyhedra(model; canonicalize = false)
    polys_can = get_polyhedra(model; canonicalize = true)

    @test length(polys_default) == length(polys_unc) == length(idxs) == length(polys_can)
    @test all(p -> p isa BindingAndCatalysis.Polyhedron, polys_default)
    @test all(p -> p isa BindingAndCatalysis.Polyhedron, polys_unc)
    @test all(p -> p isa BindingAndCatalysis.Polyhedron, polys_can)
    @test all(BindingAndCatalysis.same_polyhedron.(polys_default, polys_unc))
    @test all(BindingAndCatalysis.same_polyhedron.(polys_default, polys_can))

    regular_idx = first(filter(i -> get_nullity(model, i) == 0 && is_asymptotic(model, i), idxs))
    @test BindingAndCatalysis._bind_volume_route(model, [regular_idx]) == :classifier
    @test BindingAndCatalysis._bind_volume_route(model, [regular_idx]; contain_overlap = true) == :polyhedra

    classifier_vol = get_volume(
        model,
        regular_idx;
        recalculate = true,
        batch_size = 4_000,
        rel_tol = 0.2,
        abs_tol = 1e-3,
        time_limit = 1.0,
    )
    rgm = get_regime(model, regular_idx; inv_info = true)
    poly_vol = calc_volume(
        [rgm];
        contain_overlap = true,
        batch_size = 4_000,
        rel_tol = 0.2,
        abs_tol = 1e-3,
        time_limit = 1.0,
    )[1]

    @test classifier_vol.mean >= 0
    @test poly_vol.mean >= 0
    @test isapprox(classifier_vol.mean, poly_vol.mean; rtol = 0.35, atol = 0.02)

    qK_dim = model.d + model.r
    rebase_mat = Matrix(1.0I, qK_dim, qK_dim)
    rebase_mat[model.d + 1, model.d + 2] = 0.25
    rebase_mat[model.d + 2, model.d + 3] = -0.5
    classifier_rebased_vol = calc_volume(
        [rgm];
        rebase_mat = rebase_mat,
        batch_size = 4_000,
        rel_tol = 0.2,
        abs_tol = 1e-3,
        time_limit = 1.0,
    )[1]
    poly_rebased_vol = calc_volume(
        [rgm];
        contain_overlap = true,
        rebase_mat = rebase_mat,
        batch_size = 4_000,
        rel_tol = 0.2,
        abs_tol = 1e-3,
        time_limit = 1.0,
    )[1]

    @test classifier_rebased_vol.mean >= 0
    @test poly_rebased_vol.mean >= 0
    @test isapprox(classifier_rebased_vol.mean, poly_rebased_vol.mean; rtol = 0.35, atol = 0.02)
end
@testset "Exact Affine Data" begin
    model = minimal_model()
    find_all_regimes!(model)

    H = get_H(model, 1)
    H0 = get_H0(model, 1)
    CqK = get_C_qK(model, 1)
    poly = get_polyhedron(model, 1)
    vg = get_regimes_graph!(model; full = true)
    perms = get_perms(model)
    rational_singular_idx = only(filter(i -> get_nullity(model, i) == 1, get_indices(model)))
    Hs, H0s = get_H_H0(model, rational_singular_idx)

    @test eltype(H) <: Rational
    @test eltype(H0) == ExactLogExpr
    @test eltype(CqK) <: Rational
    @test get_nullity(poly) == get_nullity(model, 1)
    @test size(get_C(poly), 2) == model.d + model.r
    @test eltype(Hs) <: Rational
    @test eltype(H0s) == ExactLogExpr

    edge_21 = get_edge(vg, perms[2], perms[1]; full = true)
    edge_12 = get_edge(vg, perms[1], perms[2]; full = true)
    qK_21 = BindingAndCatalysis._edge_idx_sign(edge_21, vg, :qK)
    qK_12 = BindingAndCatalysis._edge_idx_sign(edge_12, vg, :qK)
    @test qK_21[1] == qK_12[1] != 0
    @test qK_21[2] == -qK_12[2]

    singular_C, singular_C0, singular_nullity = get_C_C0_nullity_qK(model, rational_singular_idx)
    singular_poly = get_polyhedron(model, rational_singular_idx)

    @test singular_nullity == 1
    @test eltype(singular_C) == Float64
    @test eltype(singular_C0) == Float64
    @test singular_poly isa BindingAndCatalysis.Polyhedron
    @test eltype(get_C(singular_poly)) == Float64
end
