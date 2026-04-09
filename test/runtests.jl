using BindingAndCatalysis
using Random
using SparseArrays
using Test

function minimal_model()
    N = [1 1 -1]
    x_sym = [:E, :S, :C]
    q_sym = [:tE, :tS]
    K_sym = [:K]
    return Bnc(N = N, x_sym = x_sym, q_sym = q_sym, K_sym = K_sym)
end

function minimal_catalysis_model()
    model = minimal_model()
    update_catalysis!(
        model;
        Γ = [1 -1],
        Π = [1 0 0; 0 1 0],
        q_picked = [:tE],
        k_sym = [:k1, :k2],
    )
    return model
end

function offset_catalysis_model()
    model = minimal_model()
    update_catalysis!(
        model;
        Γ = [2 1 -1],
        Π = [1 0 0; 0 1 0; 0 0 1],
        q_picked = [:tE],
        k_sym = [:k1, :k2, :k3],
    )
    return model
end

function notebook_model2()
    N = [
        1 1 -1 0 0
        1 0 1 -1 0
        0 1 0 1 -1
    ]
    return Bnc(N = N)
end

function clique5_binding_model()
    N = [
        1 1 0 0 0 -1 0 0 0 0 0 0 0 0 0
        1 0 1 0 0 0 -1 0 0 0 0 0 0 0 0
        1 0 0 1 0 0 0 -1 0 0 0 0 0 0 0
        1 0 0 0 1 0 0 0 -1 0 0 0 0 0 0
        0 1 1 0 0 0 0 0 0 -1 0 0 0 0 0
        0 1 0 1 0 0 0 0 0 0 -1 0 0 0 0
        0 1 0 0 1 0 0 0 0 0 0 -1 0 0 0
        0 0 1 1 0 0 0 0 0 0 0 0 -1 0 0
        0 0 1 0 1 0 0 0 0 0 0 0 0 -1 0
        0 0 0 1 1 0 0 0 0 0 0 0 0 0 -1
    ]
    x_sym = [:A, :B, :C, :D, :E, :ab, :ac, :ad, :ae, :bc, :bd, :be, :cd, :ce, :de]
    q_sym = [:tA, :tB, :tC, :tD, :tE]
    K_sym = [:K12, :K13, :K14, :K15, :K23, :K24, :K25, :K34, :K35, :K45]
    return Bnc(N = N, x_sym = x_sym, q_sym = q_sym, K_sym = K_sym)
end

@testset "BindingAndCatalysis.jl" begin
    model = minimal_model()

    @test (model.r, model.n, model.d) == (1, 3, 2)

    conservation = show_conservation(model)
    @test length(conservation) == model.d

    equilibrium_log = show_equilibrium(model; log_space = true)
    equilibrium_linear = show_equilibrium(model; log_space = false)
    @test length(equilibrium_log) == model.r
    @test length(equilibrium_linear) == model.r

    find_all_vertices!(model)
    @test !isempty(model.vertices_perm)

    first_perm = model.vertices_perm[1]
    first_idx = get_idx(model, first_perm)
    @test have_perm(model, first_perm)
    @test have_perm(model, first_idx)
    @test get_perm(model, first_idx) == first_perm

    vertex = get_vertex(model, first_idx)
    C, C0, nullity = get_C_C0_nullity(vertex)
    @test size(C, 2) == model.n
    @test length(C0) == size(C, 1)
    @test nullity >= 0

    poly = get_polyhedron(model, first_idx)
    C_poly, C0_poly, nullity_poly = get_C_C0_nullity(poly)
    @test size(C_poly, 2) == model.n
    @test length(C0_poly) == size(C_poly, 1)
    @test nullity_poly == nullity

    Random.seed!(42)
    logqK = randomize(model, 1; log_lower = -2, log_upper = 2)[1]
    logx = qK2x(model, logqK; input_logspace = true, output_logspace = true)
    logqK_back = x2qK(model, logx; input_logspace = true, output_logspace = true)
    @test isapprox(logqK_back, logqK; atol = 1e-6, rtol = 1e-6)
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
    @test all(i -> !isnothing(model.vertices_data[i].H) && !isnothing(model.vertices_data[i].H0),
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
    @test get_C(model, 1) == CqK
    @test get_C0(r1) == C0qK
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

    vol1 = get_volume(model, 1)
    vols = get_volumes(model)
    @test vol1.mean >= 0
    @test length(vols) == n_regimes(model)
    @test all(v -> v isa BindingAndCatalysis.Volume, vols)

    C_add = [1 -1 0]
    C0_add = [-log10(2)]
    feas = check_feasibility_with_constraint(model, 4; C = C_add, C0 = C0_add)
    feas_list = feasible_vertieces_with_constraint(model; C = C_add, C0 = C0_add, return_idx = true)
    @test feas isa Bool
    @test all(i -> i in idxs, feas_list)

    vg = get_regimes_graph!(model; full = true)
    @test length(vg.neighbors) == n_regimes(model)
    @test size(get_regimes_neighbor_mat(model), 1) == n_regimes(model)
    @test BindingAndCatalysis.Graphs.nv(get_neighbor_graph_x(model)) == n_regimes(model)

    @test get_edge(vg, r2_perm, r1_perm) !== nothing
    @test get_edge(vg, r2_perm, r3_perm) === nothing
    @test is_neighbor(model, r2_perm, r1_perm)

    edge_21 = get_edge(vg, r2_perm, r1_perm; full = true)
    edge_12 = get_edge(vg, r1_perm, r2_perm; full = true)
    @test edge_21.qK_interface_idx == edge_12.qK_interface_idx != 0
    @test edge_21.qK_interface_sign == -edge_12.qK_interface_sign

    inter = get_intersect(model, r2_perm, r1_perm)
    dir, ins = get_interface(model, r2_perm, r1_perm)
    @test get_nullity(inter) >= 0
    @test length(dir) == model.n
    @test ins isa Real
    @test sym_direction(model, dir) isa String
    @test show_interface(model, r2_perm, r1_perm) !== nothing
    @test get_interface(model, r2_perm, r3_perm) isa Tuple

    siso = SISOPaths(model, :tS)
    @test !isempty(siso.rgm_paths)

    p1_idx = 1
    p1 = get_path(siso, p1_idx)
    p1_idx_path = get_path(siso, p1_idx; return_idx = true)
    @test get_idx(siso, p1) == p1_idx
    @test get_idx(siso, p1_idx_path) == p1_idx
    @test get_path(siso, p1; return_idx = true) == p1_idx_path

    path_poly = get_polyhedron(siso, p1_idx)
    path_vol = get_volume(siso, p1_idx)
    @test get_nullity(path_poly) >= 0
    @test path_vol.mean >= 0
    @test !isempty(show_condition(siso, p1_idx; log_space = false))

    ro_path = get_RO_path(siso, p1_idx; observe_x = :E)
    @test !isempty(ro_path)
    @test format_arrow(ro_path) isa String

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
end

@testset "Rational H Mode" begin
    model = minimal_model()
    find_all_regimes!(model; H_mode = :rational)

    H = get_H(model, 1)
    H0 = get_H0(model, 1)
    CqK = get_C_qK(model, 1)
    poly = get_polyhedron(model, 1)
    vg = get_regimes_graph!(model; full = true)
    perms = get_perms(model)
    rational_singular_idx = only(filter(i -> get_nullity(model, i) == 1, get_indices(model)))
    Hs, H0s = get_H_H0(model, rational_singular_idx)

    @test model.affine_coeff_mode == :rational
    @test eltype(H) <: Rational
    @test eltype(H0) == Float64
    @test eltype(CqK) <: Rational
    @test get_nullity(poly) == get_nullity(model, 1)
    @test size(get_C(poly), 2) == model.d + model.r
    @test eltype(Hs) <: Rational
    @test eltype(H0s) == Float64

    edge_21 = get_edge(vg, perms[2], perms[1]; full = true)
    edge_12 = get_edge(vg, perms[1], perms[2]; full = true)
    @test edge_21.qK_interface_idx == edge_12.qK_interface_idx != 0
    @test edge_21.qK_interface_sign == -edge_12.qK_interface_sign

    find_all_regimes!(model; H_mode = :float)
    @test model.affine_coeff_mode == :float
    @test eltype(get_H(model, 1)) == Float64

    @test_throws ErrorException find_all_regimes!(minimal_model(); H_mode = :invalid_mode)
end

@testset "Two-Row N Singular H Sign Invariance" begin
    N = [
        1 1 -1 0
        0 1  1 -1
    ]
    singular_perm = [3, 3]

    for H_mode in (:float, :rational)
        model = Bnc(N = N)
        swapped_model = Bnc(N = N[[2, 1], :])

        find_all_regimes!(model; H_mode = H_mode)
        find_all_regimes!(swapped_model; H_mode = H_mode)

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
    L = SparseArrays.sparse(
        [1, 2, 3, 4, 4, 1, 3, 4, 2, 4],
        [1, 2, 3, 3, 4, 5, 5, 5, 6, 6],
        ones(Int, 10),
        4,
        6,
    )
    N = SparseArrays.sparse(
        [1, 2, 1, 2, 1, 2],
        [1, 2, 3, 4, 5, 6],
        [1, 1, 1, 1, -1, -1],
        2,
        6,
    )

    model_float = Bnc(L = L, N = N)
    model_rational = Bnc(L = L, N = N)

    find_all_regimes!(model_float; H_mode = :float)
    find_all_regimes!(model_rational; H_mode = :rational)

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

@testset "Larger RO Path Workflow" begin
    model = notebook_model2()
    pths = SISOPaths(model, 1)
    pths_single = SISOPaths(notebook_model2(), 1)

    grouped = group_sum([[1, 2], [1, 2], [2, 3]], fill(nothing, 3))
    @test length(grouped) == 2
    @test grouped[1][3] === nothing

    @test !isempty(pths.rgm_paths)
    @test get_path(pths, 1; return_idx = true) == pths.rgm_paths[1]
    @test get_idx(pths, get_path(pths, 1)) == 1
    @test pths.rgm_paths == pths_single.rgm_paths

    ro1 = get_RO_path(pths, 1; observe_x = 1)
    ro_paths_1 = get_RO_paths(pths; observe_x = 1)
    ro_paths_2 = get_RO_paths(pths; observe_x = 2, deduplicate = true)
    ro_paths_2_filtered = get_RO_paths(pths; observe_x = 2, deduplicate = true, keep_nonasymptotic = false, keep_singular = false)
    bulk_polys = get_polyhedra(pths)
    single_polys = [get_polyhedron(pths_single, i) for i in eachindex(pths_single.rgm_paths)]

    @test !isempty(ro1)
    @test length(ro_paths_1) == length(pths.rgm_paths)
    @test length(ro_paths_2) == length(pths.rgm_paths)
    @test length(ro_paths_2_filtered) == length(pths.rgm_paths)
    @test all(BindingAndCatalysis.same_polyhedron.(bulk_polys, single_polys))
end

@testset "SISO Helper Condition Alignment Example" begin
    model = notebook_model2()
    siso = SISOPaths(model, 1)
    path = [1, 4, 3]
    siso_poly = get_polyhedron(siso, path)

    helper = BindingAndCatalysis.SISOHelper(model, 1)
    BindingAndCatalysis._find_all_path_conditions!(helper)
    helper_path_polys = Dict{Tuple{Vararg{Int}},Any}()
    for source in get_sources(helper), sink in get_sinks(helper)
        for (path_key, poly) in BindingAndCatalysis.get_path_conditions(helper, source, sink)
            helper_path_polys[path_key] = poly
        end
    end

    @test haskey(helper_path_polys, (1, 3))
    @test haskey(helper_path_polys, Tuple(path))
    @test Set(keys(helper_path_polys)) == Set(Tuple.(siso.rgm_paths))

    @test BindingAndCatalysis.same_polyhedron(siso_poly, helper_path_polys[Tuple(path)])

    for (pth, poly) in helper_path_polys
        @test BindingAndCatalysis.same_polyhedron(get_polyhedron(siso, collect(pth)), poly)
    end
end

@testset "Nested Threaded Regime Propagation" begin
    if Threads.nthreads() > 1
        model = clique5_binding_model()
        @test begin
            find_all_regimes!(model)
            n_regimes(model) > 0
        end
    end
end

@testset "Catalysis And Mixed Regimes" begin
    model = minimal_catalysis_model()
    cn = get_catalysis_network(model)

    @test cn !== nothing
    @test all(isequal.(q_cat_sym(model), model.q_sym[1:cn.r_v]))
    @test all(isequal.(q_ss_sym(model), [w_sym(model); q_para_sym(model)]))
    @test all(isequal.(k_sym(model), cn.k_sym))

    find_catalysis_regimes!(model)
    @test n_regimes(cn) == 1

    cat_perm = first(get_catalysis_regimes(model))
    cat_rgm = get_catalysis_regime(model, cat_perm)
    @test get_idx(cn, cat_perm) == get_idx(cat_rgm)
    @test get_perm(cn, get_idx(cat_rgm)) == cat_perm
    @test size(get_P(cat_rgm), 1) == cn.r_v
    @test occursin("dominant mode", sprint(show, MIME"text/plain"(), cat_rgm))
    @test occursin("nullity", sprint(show, MIME"text/plain"(), cat_rgm))
    @test occursin("asymptotic", sprint(show, MIME"text/plain"(), cat_rgm))

    C_xk_cat, C0_xk_cat, nlt_xk_cat = get_C_C0_nullity_xk(cat_rgm)
    @test size(C_xk_cat, 2) == model.n + cn.n_v
    @test length(C0_xk_cat) == size(C_xk_cat, 1)
    @test nlt_xk_cat == cn.r_v

    dyn_full = show_catalysis_dynamics(model)
    dyn_reduced = show_reduced_catalysis_dynamics(model)
    @test length(dyn_full) == cn.r_v + cn.d_w + cn.d_para
    @test length(dyn_reduced) == cn.r_v + cn.d_w + cn.d_para
    @test length(show_condition_xk(cat_rgm)) == size(C_xk_cat, 1)

    match_regimes!(model)
    @test n_bnc_regimes(model) > 0

    bind_perm = first(get_perms(model))
    @test have_perm(model, bind_perm, cat_perm)

    mixed = get_bnc_regime(model, bind_perm, cat_perm)
    @test mixed !== nothing
    @test get_regime(model, bind_perm, cat_perm) === mixed
    @test get_binding_perm(mixed) == bind_perm
    @test get_catalysis_perm(mixed) == cat_perm
    @test get_idx(mixed) == CartesianIndex(get_idx(cat_rgm), get_idx(model, bind_perm))
    @test occursin("dominant mode", sprint(show, MIME"text/plain"(), mixed))
    @test occursin("nullity", sprint(show, MIME"text/plain"(), mixed))
    @test occursin("asymptotic", sprint(show, MIME"text/plain"(), mixed))

    C_qKk, C0_qKk, nlt_qKk = get_C_C0_nullity_qKk(mixed)
    @test size(C_qKk, 2) == model.d + model.r + cn.n_v
    @test length(C0_qKk) == size(C_qKk, 1)
    @test nlt_qKk >= 0

    C_qssKk, C0_qssKk, nlt_qssKk = get_C_C0_nullity_qssKk(mixed)
    @test size(C_qssKk, 2) == (cn.d_w + cn.d_para) + model.r + cn.n_v
    @test length(C0_qssKk) == size(C_qssKk, 1)
    @test nlt_qssKk >= 0

    @test !isempty(show_condition_xk(mixed; kind = :binding))
    @test !isempty(show_condition_xk(mixed; kind = :catalysis))
    @test !isempty(show_condition_qKk(mixed; kind = :binding))
    @test show_condition_qKk(mixed; kind = :catalysis) isa AbstractVector
    @test !isempty(show_condition_qKk(mixed))
    @test !isempty(show_condition_qssKk(mixed))
    @test !isempty(show_consistency_condition(mixed))

    regular = first(filter(r -> r.nlt == 0, get_bnc_regimes(model)))
    F_qcat, F0_qcat = get_qcat_F_F0(regular)
    @test size(F_qcat, 1) == cn.r_v
    @test length(F0_qcat) == cn.r_v
    @test length(show_expression_qcat(regular)) == cn.r_v
    @test length(show_expression_x(regular)) == model.n

    stable_flag = is_stable(regular)
    stable_code = is_stable(regular; return_code = true)
    @test stable_flag === true || stable_flag === false || ismissing(stable_flag)
    @test stable_code in (-1, 0, 1)

    singular_mixed = filter(r -> r.nlt == 1, get_bnc_regimes(model))
    if !isempty(singular_mixed)
        Hs, H0s = get_H_H0(first(singular_mixed))
        @test size(Hs, 1) == model.n
        @test length(H0s) == model.n
    end
end

@testset "Catalysis Offsets And Mixed Consistency" begin
    model = offset_catalysis_model()
    cn = get_catalysis_network(model)
    find_catalysis_regimes!(model)

    @test n_regimes(cn) == 2

    cat_rgms = [get_catalysis_regime(model, i) for i in 1:n_regimes(cn)]
    @test sort([only(get_P0(rgm)) for rgm in cat_rgms]) ≈ [0.0, log10(2.0)]
    @test sort([only(get_C0(rgm)) for rgm in cat_rgms]) ≈ [-log10(2.0), log10(2.0)]

    for cat_rgm in cat_rgms
        C_xk, C0_xk, nlt = get_C_C0_nullity_xk(cat_rgm)
        @test nlt == cn.r_v
        @test C0_xk[1:cn.r_v] == get_P0(cat_rgm)
        @test C0_xk[cn.r_v+1:end] == get_C0(cat_rgm)
        @test !isempty(show_condition_xk(cat_rgm; kind = :steady_state))
        @test !isempty(show_condition_xk(cat_rgm; kind = :dominance))
        @test size(C_xk, 2) == model.n + cn.n_v
    end

    match_regimes!(model)
    mixed_regular = first(filter(r -> r.nlt == 0 && !is_singular(get_binding_regime(r)), get_bnc_regimes(model)))
    bind_rgm = get_binding_regime(mixed_regular)
    cat_rgm = get_catalysis_regime(mixed_regular)
    r_v = size(get_P(cat_rgm), 1)

    P_ss = Matrix{Float64}(bind_rgm.P[r_v+1:end, :])
    P0_ss = Vector{Float64}(bind_rgm.P0[r_v+1:end])
    N = Matrix{Float64}(bind_rgm.network.N)
    PΠ = Matrix{Float64}(get_PΠ(cat_rgm))
    Pθ = Matrix{Float64}(get_P(cat_rgm))
    P0θ = Vector{Float64}(get_P0(cat_rgm))

    M_ss = vcat(P_ss, N, PΠ)
    M0_ss = vcat(P0_ss, zeros(Float64, size(N, 1) + r_v))
    H_ss = inv(M_ss)
    H0_ss = -(H_ss * M0_ss)
    split = size(H_ss, 2) - r_v
    H_right = H_ss[:, split+1:end]
    H_expected = hcat(H_ss[:, 1:split], -(H_right * Pθ))
    H0_expected = vec(H0_ss - H_right * P0θ)

    @test Matrix(get_H(mixed_regular)) ≈ H_expected
    @test get_H0(mixed_regular) ≈ H0_expected

    C_cat_qKk, C0_cat_qKk, nlt_cat_qKk = get_C_C0_nullity_qKk(mixed_regular, :catalysis)
    H_bind, H0_bind = get_H_H0(bind_rgm)
    C_expected = hcat(Matrix{Float64}(get_CΠ(cat_rgm) * H_bind), Matrix{Float64}(get_C_k(cat_rgm)))
    C0_expected = vec(get_CΠ(cat_rgm) * H0_bind + get_C0(cat_rgm))

    @test nlt_cat_qKk == 0
    @test Matrix(C_cat_qKk) ≈ C_expected
    @test C0_cat_qKk ≈ C0_expected
    @test !isempty(show_condition_qKk(mixed_regular; kind = :catalysis))
    @test !isempty(show_condition_qssKk(mixed_regular))
end
