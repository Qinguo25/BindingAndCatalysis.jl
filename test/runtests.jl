using BindingAndCatalysis
using Random
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

    bind_perm = first(get_regimes(model))
    @test have_perm(model, bind_perm, cat_perm)

    mixed = get_bnc_regime(model, bind_perm, cat_perm)
    @test mixed !== nothing
    @test get_regime(model, bind_perm, cat_perm) === mixed
    @test get_binding_perm(mixed) == bind_perm
    @test get_catalysis_perm(mixed) == cat_perm
    @test get_idx(mixed) == CartesianIndex(get_idx(cat_rgm), get_idx(model, bind_perm))

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
end
