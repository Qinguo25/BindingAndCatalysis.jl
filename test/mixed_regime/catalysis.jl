@testset "Catalysis And Mixed Regimes" begin
    model = minimal_catalysis_model()
    cn = get_catalysis_network(model)

    @test cn !== nothing
    @test all(isequal.(q_cat_sym(model), model.q_sym[1:cn.r_v]))
    @test all(isequal.(w_sym(model), model.q_sym[cn.r_v + 1:model.d]))
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

    dyn_full = show_catalysis_dynamics(model; reduced = false)
    dyn_reduced = show_catalysis_dynamics(model; reduced = true)
    @test length(dyn_full) == cn.r_v + cn.d_w
    @test length(dyn_reduced) == cn.r_v + cn.d_w
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
    @test get_idx(mixed) == get_idx(model, bind_perm, cat_perm)
    @test occursin("dominant mode", sprint(show, MIME"text/plain"(), mixed))
    @test occursin("nullity", sprint(show, MIME"text/plain"(), mixed))
    @test occursin("asymptotic", sprint(show, MIME"text/plain"(), mixed))

    C_qKk, C0_qKk, nlt_qKk = get_C_C0_nullity_qKk(mixed)
    @test size(C_qKk, 2) == model.d + model.r + cn.n_v
    @test length(C0_qKk) == size(C_qKk, 1)
    @test nlt_qKk >= 0

    C_wKk, C0_wKk, nlt_wKk = get_C_C0_nullity_wKk(mixed)
    @test size(C_wKk, 2) == cn.d_w + model.r + cn.n_v
    @test length(C0_wKk) == size(C_wKk, 1)
    @test nlt_wKk >= 0

    @test !isempty(show_condition_xk(mixed; kind = :binding))
    @test !isempty(show_condition_xk(mixed; kind = :catalysis))
    @test !isempty(show_condition_qKk(mixed; kind = :binding))
    @test show_condition_qKk(mixed; kind = :catalysis) isa AbstractVector
    @test !isempty(show_condition_qKk(mixed))
    @test !isempty(show_condition_wKk(mixed))
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

@testset "Catalysis Symbol Overrides" begin
    model = minimal_model()
    update_catalysis!(
        model;
        Γ = [1 -1; -1 1],
        Π = [1 0 0; 0 1 0],
        q_picked = [:tE, :tS],
        k_sym = [:β, :γ],
        w_sym = [:wtot],
    )

    @test string.(k_sym(model)) == ["β", "γ"]
    @test string.(w_sym(model)) == ["wtot"]
    @test string.(q_sym(model)) == ["tE", "wtot"]
end

@testset "Minimal Binding-Catalysis wKk Example" begin
    L = [
        0 1 1
        1 0 1
    ]
    N = [1 1 -1]
    model = Bnc(L = L, N = N, x_sym = [:E, :S, :C], q_sym = [:tS, :tE], K_sym = [:K])
    update_catalysis!(
        model;
        Γ = [1 -1],
        Π = [0 0 1; 0 1 0],
        q_picked = [:tS],
        k_sym = [:β, :γ],
    )

    find_all_regimes!(model)
    find_catalysis_regimes!(model)
    match_regimes!(model)

    @test n_regimes(model) == 4
    @test n_regimes(get_catalysis_network(model)) == 1

    mixed = get_bnc_regime(model, 1, 1; check = true)
    @test get_binding_perm(mixed) == [2, 1]
    @test get_catalysis_perm(mixed) == [1, 2]
    @test string.(BindingAndCatalysis.wKk_sym(mixed)) == ["tE", "K", "β", "γ"]
    @test string.(show_condition_wKk(mixed; log_space = false)) == ["K*γ ~ tE*β", "K > tE"]

    C_wKk, C0_wKk, nlt_wKk = get_C_C0_nullity_wKk(mixed)
    @test Matrix(C_wKk) == Rational{Int}[-1 1 -1 1; -1 1 0 0]
    @test C0_wKk == ExactLogExpr[0, 0]
    @test nlt_wKk == 1
end

@testset "Catalysis Exact Mixed Mode" begin
    model = minimal_catalysis_model()
    find_all_regimes!(model)
    find_catalysis_regimes!(model)
    match_regimes!(model)

    mixed = first(get_bnc_regimes(model))
    regular = first(filter(r -> r.nlt == 0, get_bnc_regimes(model)))

    @test eltype(get_H0(mixed)) == ExactLogExpr
    @test eltype(get_C0_qKk(mixed)) == ExactLogExpr
    @test eltype(get_C0_wKk(mixed)) == ExactLogExpr
    @test !isempty(show_condition_qKk(mixed))
    @test !isempty(show_condition_wKk(mixed))
    @test !isempty(show_expression_qcat(regular))
end

@testset "Log-k Reparameterization Helpers" begin
    L = [
        0 1 1
        1 0 1
    ]
    N = [1 1 -1]
    model = Bnc(L = L, N = N, x_sym = [:E, :S, :C], q_sym = [:tS, :tE], K_sym = [:K])
    update_catalysis!(
        model;
        Γ = [1 -1],
        Π = [0 0 1; 0 1 0],
        q_picked = [:tS],
        k_sym = [:β, :γ],
    )

    match_regimes!(model)
    mixed = get_bnc_regime(model, 1, 1; check = true)
    R = reshape(Rational{Int}[1, 1], 2, 1)
    b = ExactLogExpr[exact_log10_ratio(1, 2), 0]

    C_wKθ, C0_wKθ, nlt_wKθ = get_C_C0_nullity_wKtheta(mixed; R = R, b = b)
    @test Matrix(C_wKθ) ≈ Float64.([-1 1 0; -1 1 0])
    @test Float64.(C0_wKθ) ≈ [log10(2), 0.0]
    @test nlt_wKθ == 1
    @test !is_feasible_under_logkmap(mixed; R = R, b = b)
    @test get_idx(mixed) ∉ feasible_bnc_regimes_under_logkmap(model; R = R, b = b, return_idx = true)
end

@testset "High-Nullity Mixed Regimes Keep Consistency" begin
    model = sparse_singular_model()
    update_catalysis!(
        model;
        Γ = [1 -1],
        Π = [1 0 0 0 0 0; 0 1 0 0 0 0],
        k_sym = [:k1, :k2],
    )
    match_regimes!(model)

    bind_high = filter(r -> get_binding_regime(r).nullity > 1 && r.nlt <= 1, get_bnc_regimes(model))
    @test !isempty(bind_high)

    low_mixed = first(bind_high)
    @test get_H_bd(low_mixed) isa AbstractMatrix
    @test is_stable(low_mixed) === true || is_stable(low_mixed) === false
    @test is_stable(low_mixed; return_code = true) in (-1, 1)
    @test !isempty(show_condition_qKk(low_mixed))
    @test !isempty(show_condition_wKk(low_mixed))

    H_low, H0_low = get_H_H0(low_mixed)
    @test size(H_low, 1) == model.n
    @test length(H0_low) == model.n

    high_model = sparse_singular_model()
    update_catalysis!(
        high_model;
        Γ = [1 -1],
        Π = [1 0 0 0 0 0; 1 0 0 0 0 0],
        k_sym = [:k1, :k2],
    )
    match_regimes!(high_model)

    consistency_only = filter(r -> r.nlt > 1, get_bnc_regimes(high_model))
    @test !isempty(consistency_only)

    high_mixed = first(consistency_only)
    @test get_H_bd(high_mixed) isa AbstractMatrix
    @test is_stable(high_mixed) === true || is_stable(high_mixed) === false
    @test is_stable(high_mixed; return_code = true) in (-1, 1)
    @test isnothing(high_mixed.H)
    @test isnothing(high_mixed.H0)
    @test !isempty(show_condition_qKk(high_mixed))
    @test !isempty(show_condition_wKk(high_mixed))
    @test_throws Exception get_H_H0(high_mixed)
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
    @test !isempty(show_condition_wKk(mixed_regular))
end
