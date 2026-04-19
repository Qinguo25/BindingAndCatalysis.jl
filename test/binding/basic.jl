@testset "BindingAndCatalysis.jl" begin
    model = minimal_model()

    @test (model.r, model.n, model.d) == (1, 3, 2)
    @test length(show_conservation(model)) == model.d
    @test length(show_equilibrium(model; log_space = true)) == model.r
    @test length(show_equilibrium(model; log_space = false)) == model.r

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

    simo = SIMOPaths(model, :tS)
    @test !isempty(simo.rgm_paths)

    p1_idx = 1
    p1 = get_path(simo, p1_idx)
    p1_idx_path = get_path(simo, p1_idx; return_idx = true)
    @test get_idx(simo, p1) == p1_idx
    @test get_idx(simo, p1_idx_path) == p1_idx
    @test get_path(simo, p1; return_idx = true) == p1_idx_path

    path_poly = get_polyhedron(simo, p1_idx)
    path_vol = get_volume(simo, p1_idx)
    @test get_nullity(path_poly) >= 0
    @test path_vol.mean >= 0
    @test !isempty(show_condition(simo, p1_idx; log_space = false))

    ro_path = get_RO_path(simo, p1_idx; observe_x = :E)
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

@testset "Nullity-1 Singular X Range" begin
    model = Bnc(N = [2 1 -1])
    find_all_regimes!(model; mode = :exact)
    rgm_idx = only(filter(i -> get_nullity(model, i) == 1, get_regimes(model; return_idx = true)))

    xr = get_singular_x_range(model, rgm_idx; observe_x = 1, log_space = false)
    @test xr isa SingularXRange
    @test isempty(xr.equalities)
    @test length(xr.lower_bounds) == 1
    @test length(xr.upper_bounds) == 1

    xr_dom = get_singular_x_range(model, rgm_idx; observe_x = 3, log_space = false)
    @test length(xr_dom.equalities) == 2
    qK_anchor = get_one_inner_point(get_polyhedron(model, rgm_idx))
    ev_x1 = BindingAndCatalysis._evaluate_singular_x_range(xr, qK_anchor, model; input_logspace = true)
    ev = BindingAndCatalysis._evaluate_singular_x_range(xr_dom, qK_anchor, model; input_logspace = true)
    @test ev_x1.consistent
    @test ev_x1.lower <= ev_x1.upper + 1e-8
    @test isapprox(ev.lower, ev.upper; atol = 1e-6)

    shown = show_expression_x_range(model, rgm_idx; observe_x = 1, log_space = false)
    @test shown == "max((K₁^(1//2))) < x₁ < min(q₁)"
    @test show_expression_x_range(model, rgm_idx; observe_x = 2, log_space = false) ==
        "max((((1//2)*K₁) / q₁)) < x₂ < min(((1//2)*q₁))"
    @test show_expression_x_range(model, rgm_idx; observe_x = 3, log_space = false) == "x₃ ~ q₂"
end

@testset "Higher-Nullity Singular X Range Projection" begin
    N = [
        1 2 1 -1 0 0 0
        1 1 1 0 -1 0 0
        0 0 1 1 0 -1 0
        2 1 0 0 1 0 -1
    ]
    L = BindingAndCatalysis.L_from_N(N)
    model = Bnc(N = N, L = L)
    find_all_regimes!(model; mode = :float)

    rgm_idx = first(filter(i -> get_nullity(model, i) == 2, get_regimes(model; return_idx = true)))
    xr = get_singular_x_range(model, rgm_idx; observe_x = 1, log_space = false)
    @test xr isa SingularXRange
    @test xr.projected_nullity >= 0

    qK_anchor = get_one_inner_point(get_polyhedron(model, rgm_idx))
    ev = BindingAndCatalysis._evaluate_singular_x_range(xr, qK_anchor, model; input_logspace = true)
    @test ev.consistent
    @test ev.lower <= ev.upper + 1e-8
end

@testset "Graph Plus q Bounds Are Not Equivalent" begin
    function _singular_component(model::Bnc, rgm_idx::Int)
        seen = Set([rgm_idx])
        queue = [rgm_idx]
        while !isempty(queue)
            cur = popfirst!(queue)
            for nb in get_neighbors(model, cur; singular = true, return_idx = true)
                get_nullity(model, nb) == 1 || continue
                if nb ∉ seen
                    push!(seen, nb)
                    push!(queue, nb)
                end
            end
        end
        return collect(seen)
    end

    function _graph_plus_q_numeric_bounds(model::Bnc, rgm_idx::Int, observe_x_idx::Int, logqK_anchor)
        xr = get_singular_x_range(model, rgm_idx; observe_x = observe_x_idx, log_space = false)
        direct = BindingAndCatalysis._evaluate_singular_x_range(xr, logqK_anchor, model; input_logspace = true)

        lowers = Float64[]
        uppers = Float64[]
        for srgm in _singular_component(model, rgm_idx)
            for nb in get_neighbors(model, srgm; singular = false, return_idx = true)
                H, H0 = get_H_H0(model, nb)
                val = LinearAlgebra.dot(Array(H[observe_x_idx, :]), logqK_anchor) + H0[observe_x_idx]
                val = Float64(val)
                val <= direct.lower + 1e-7 && push!(lowers, val)
                val >= direct.upper - 1e-7 && push!(uppers, val)
            end
        end
        for q_idx in BindingAndCatalysis._q_totals_containing_x(model, observe_x_idx)
            val = Float64(logqK_anchor[q_idx])
            val <= direct.lower + 1e-7 && push!(lowers, val)
            val >= direct.upper - 1e-7 && push!(uppers, val)
        end

        return (
            direct = direct,
            lower = isempty(lowers) ? -Inf : maximum(lowers),
            upper = isempty(uppers) ? Inf : minimum(uppers),
        )
    end

    N = [
        1 2 1 -1 0 0 0
        1 1 1 0 -1 0 0
        0 0 1 1 0 -1 0
        2 1 0 0 1 0 -1
    ]
    L = BindingAndCatalysis.L_from_N(N)
    Lnew = copy(L)
    Lnew[1, :] .= L[1, :] .+ L[2, :]
    model = Bnc(N = N, L = Lnew)
    find_all_regimes!(model; mode = :float)

    rgm_idx = only(filter(i -> get_nullity(model, i) == 1 && get_perm(model, i) == [2, 2, 7], get_regimes(model; return_idx = true)))
    logqK_anchor = get_one_inner_point(get_polyhedron(model, rgm_idx))
    rst = _graph_plus_q_numeric_bounds(model, rgm_idx, 1, logqK_anchor)

    @test isapprox(rst.upper, rst.direct.upper; atol = 1e-6, rtol = 1e-6)
    @test rst.lower < rst.direct.lower - 1e-3
end

@testset "Rational H Mode" begin
    model = minimal_model()
    find_all_regimes!(model; mode = :exact)

    H = get_H(model, 1)
    H0 = get_H0(model, 1)
    CqK = get_C_qK(model, 1)
    poly = get_polyhedron(model, 1)
    vg = get_regimes_graph!(model; full = true)
    perms = get_perms(model)
    rational_singular_idx = only(filter(i -> get_nullity(model, i) == 1, get_indices(model)))
    Hs, H0s = get_H_H0(model, rational_singular_idx)

    @test model.affine_coeff_mode == :exact
    @test eltype(H) <: Rational
    @test eltype(H0) == ExactLogExpr
    @test eltype(CqK) <: Rational
    @test get_nullity(poly) == get_nullity(model, 1)
    @test size(get_C(poly), 2) == model.d + model.r
    @test eltype(Hs) <: Rational
    @test eltype(H0s) == ExactLogExpr

    edge_21 = get_edge(vg, perms[2], perms[1]; full = true)
    edge_12 = get_edge(vg, perms[1], perms[2]; full = true)
    @test edge_21.qK_interface_idx == edge_12.qK_interface_idx != 0
    @test edge_21.qK_interface_sign == -edge_12.qK_interface_sign

    find_all_regimes!(model; mode = :float)
    @test model.affine_coeff_mode == :float
    @test eltype(get_H(model, 1)) == Float64

    @test_throws ErrorException find_all_regimes!(minimal_model(); mode = :invalid_mode)
end
