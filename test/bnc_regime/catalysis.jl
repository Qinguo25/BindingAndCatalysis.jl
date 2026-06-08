@testset "Catalysis And Binding-Catalysis Regimes" begin
    model = minimal_catalysis_model()
    cn = get_catalysis_network(model)

    @test cn !== nothing
    @test all(isequal.(q_cat_sym(model), model.q_sym[1:(cn.r_v)]))
    @test all(isequal.(w_sym(model), model.q_sym[(cn.r_v + 1):(model.d)]))
    @test all(isequal.(k_sym(model), cn.k_sym))

    find_catalysis_regimes!(model)
    @test ensure_catalysis_regimes!(model) === nothing
    @test n_regimes(cn) == 1
    @test isdefined(Main, :get_catalysis_indices)
    @test isdefined(Main, :get_catalysis_perms)
    @test isdefined(Main, :get_bnc_indices)
    @test isdefined(Main, :get_bnc_perms)

    cat_perm = first(get_catalysis_regimes(model))
    cat_rgm = get_catalysis_regime(model, cat_perm)
    @test get_idx(cn, cat_perm) == get_idx(cat_rgm)
    @test get_perm(cn, get_idx(cat_rgm)) == cat_perm
    @test get_catalysis_indices(model) == get_idx.(get_catalysis_regimes(model))
    @test get_catalysis_perms(model) == get_perm.(get_catalysis_regimes(model))
    @test size(get_P(cat_rgm), 1) == cn.r_v
    @test occursin("dominant mode", sprint(show, MIME"text/plain"(), cat_rgm))
    @test occursin("nullity", sprint(show, MIME"text/plain"(), cat_rgm))
    @test occursin("asymptotic", sprint(show, MIME"text/plain"(), cat_rgm))

    C_xk_cat, C0_xk_cat, nlt_xk_cat = get_C_C0_nullity_xk(cat_rgm)
    @test size(C_xk_cat, 2) == model.n + cn.n_v
    @test length(C0_xk_cat) == size(C_xk_cat, 1)
    @test nlt_xk_cat == cn.r_v

    dyn_full = show_catalysis_dynamics(model; reduced=false)
    dyn_reduced = show_catalysis_dynamics(model; reduced=true)
    @test length(dyn_full) == cn.r_v + cn.d_w
    @test length(dyn_reduced) == cn.r_v + cn.d_w
    @test length(show_condition_xk(cat_rgm)) == size(C_xk_cat, 1)

    @test match_regimes!(model) === nothing
    @test ensure_bnc_regimes!(model) === nothing
    @test n_bnc_regimes(model) > 0
    @test get_bnc_indices(model) == get_idx.(get_bnc_regimes(model))
    @test get_bnc_perms(model) == get_perm.(get_bnc_regimes(model))
    @test get_binding_nullities(model) == get_nullity.(get_binding_regimes(model))
    @test get_bnc_nullities(model) == get_nullity.(get_bnc_regimes(model))

    bind_perm = first(get_perms(model))
    @test have_perm(model, bind_perm, cat_perm)

    bnc_rgm = get_bnc_regime(model, bind_perm, cat_perm)
    @test bnc_rgm !== nothing
    @test get_bnc_regime(model, bind_perm, cat_perm) === bnc_rgm
    @test get_binding_perm(bnc_rgm) == bind_perm
    @test get_catalysis_perm(bnc_rgm) == cat_perm
    @test get_idx(bnc_rgm) == get_idx(model, bind_perm, cat_perm)
    @test occursin("dominant mode", sprint(show, MIME"text/plain"(), bnc_rgm))
    @test occursin("nullity", sprint(show, MIME"text/plain"(), bnc_rgm))
    @test occursin("asymptotic", sprint(show, MIME"text/plain"(), bnc_rgm))

    C_qKk, C0_qKk, nlt_qKk = get_C_C0_nullity_qKk(bnc_rgm)
    @test size(C_qKk, 2) == model.d + model.r + cn.n_v
    @test length(C0_qKk) == size(C_qKk, 1)
    @test nlt_qKk >= 0

    C_wKk, C0_wKk, nlt_wKk = get_C_C0_nullity_wKk(bnc_rgm)
    @test size(C_wKk, 2) == cn.d_w + model.r + cn.n_v
    @test length(C0_wKk) == size(C_wKk, 1)
    @test nlt_wKk >= 0

    @test !isempty(show_condition_xk(bnc_rgm; kind=:binding))
    @test !isempty(show_condition_xk(bnc_rgm; kind=:catalysis))
    @test !isempty(show_condition_qKk(bnc_rgm; kind=:binding))
    @test show_condition_qKk(bnc_rgm; kind=:catalysis) isa AbstractVector
    @test !isempty(show_condition_qKk(bnc_rgm))
    @test !isempty(show_condition_wKk(bnc_rgm))
    @test !isempty(show_consistency_condition(bnc_rgm))

    regular = first(filter(r -> r.nlt == 0, get_bnc_regimes(model)))
    F_qcat, F0_qcat = get_qcat_F_F0(regular)
    @test size(F_qcat, 1) == cn.r_v
    @test length(F0_qcat) == cn.r_v
    @test length(show_expression_qcat(regular)) == cn.r_v
    @test length(show_expression_x(regular)) == model.n

    stable_flag = is_stable(regular)
    stable_code = stability_code(regular)
    @test stable_flag === true || stable_flag === false || ismissing(stable_flag)
    @test stable_code in (-1, 0, 1)

    singular_bnc_rgms = filter(r -> r.nlt == 1, get_bnc_regimes(model))
    if !isempty(singular_bnc_rgms)
        Hs, H0s = get_H_H0(first(singular_bnc_rgms))
        @test size(Hs, 1) == model.n
        @test length(H0s) == model.n
    end
end

@testset "Catalysis Symbol Overrides" begin
    model = minimal_model()
    update_catalysis!(
        model;
        Γ=[1 -1; -1 1],
        Π=[1 0 0; 0 1 0],
        q_picked=[:tE, :tS],
        k_sym=[:β, :γ],
        w_sym=[:wtot],
    )

    @test string.(k_sym(model)) == ["β", "γ"]
    @test string.(w_sym(model)) == ["wtot"]
    @test string.(q_sym(model)) == ["tE", "wtot"]
end

@testset "Catalysis Default Identity Pi From Picked Species" begin
    model = minimal_model()
    update_catalysis!(
        model; Γ=[1 -1; -1 1], x_picked=[:E, :S], q_picked=[:tE, :tS], k_sym=[:β, :γ]
    )

    @test Matrix(get_catalysis_network(model).Π) == [1 0 0; 0 1 0]
    @test string.(q_cat_sym(model)) == ["tE"]
end

@testset "Minimal Binding-Catalysis wKk Example" begin
    L = [
        0 1 1
        1 0 1
    ]
    N = [1 1 -1]
    model = Bnc(; L=L, N=N, x_sym=[:E, :S, :C], q_sym=[:tS, :tE], K_sym=[:K])
    update_catalysis!(model; Γ=[1 -1], Π=[0 0 1; 0 1 0], q_picked=[:tS], k_sym=[:β, :γ])

    find_all_regimes!(model)
    find_catalysis_regimes!(model)
    match_regimes!(model)

    @test n_regimes(model) == 4
    @test n_regimes(get_catalysis_network(model)) == 1

    bnc_rgm = get_bnc_regime(model, 1, 1; check=true)
    @test get_binding_perm(bnc_rgm) == [2, 1]
    @test get_catalysis_perm(bnc_rgm) == [1, 2]
    @test string.(BindingAndCatalysis.wKk_sym(bnc_rgm)) == ["tE", "K", "β", "γ"]
    @test string.(show_condition_wKk(bnc_rgm; log_space=false)) == ["K*γ ~ tE*β", "K > tE"]

    C_wKk, C0_wKk, nlt_wKk = get_C_C0_nullity_wKk(bnc_rgm)
    @test Matrix(C_wKk) == Rational{Int}[-1 1 -1 1; -1 1 0 0]
    @test C0_wKk == ExactLogExpr[0, 0]
    @test nlt_wKk == 1
end

@testset "Affine K Constraint Initialization" begin
    model = minimal_model()
    update_catalysis!(
        model;
        Γ=[1 -1],
        Π=[1 0 0; 0 1 0],
        q_picked=[:tE],
        F=reshape([1, 1], 2, 1),
        F0=zeros(2),
        k_sym=[:κ],
    )

    cn = get_catalysis_network(model)
    @test cn.n_v == 2
    @test cn.n_k == 1
    @test string.(k_sym(model)) == ["κ"]
    @test length(wKk_sym(model)) == cn.d_w + model.r + cn.n_k

    match_regimes!(model)
    bnc_rgm = first(get_bnc_regimes(model))
    @test all(is_feasible, get_bnc_regimes(model))
    @test size(get_C_C0_nullity_xk(bnc_rgm)[1], 2) == model.n + cn.n_k
    @test size(get_C_C0_nullity_wKk(bnc_rgm)[1], 2) == cn.d_w + model.r + cn.n_k
end

@testset "Affine K Constraint Bnc Graph Uses Reduced xk Facets" begin
    N = [
        1 0 0 1 -1 0
        0 1 1 0 0 -1
    ]
    model = Bnc(;
        N=N,
        x_sym=[:P1, :P2, :D1, :D2, :C1, :C2],
        q_sym=[:tP1, :tP2, :tD1, :tD2],
        K_sym=[:K1, :K2],
    )
    Pi = diagm(ones(Int, 6))
    Gamma = [
        -1 0 1 0 -1 0
        0 -1 0 1 0 -1
    ]
    F = [
        1 0
        1 0
        0 1
        0 1
        1 0
        1 0
    ]

    update_catalysis!(
        model; Π=Pi, Γ=Gamma, q_picked=[1, 2], F=F, F0=zeros(6), k_sym=[:gamma, :beta]
    )
    match_regimes!(model)

    @test n_bnc_regimes(model) == 16
    @test count(!is_feasible, model.BncRegimes) == 48

    grh = get_bnc_regimes_graph!(model)
    n_bind = n_bind_regimes(model)
    edges = [
        (i, e.to) for i in eachindex(grh.neighbors) for e in grh.neighbors[i] if i < e.to
    ]
    simultaneous = count(edges) do (i, j)
        bind_i, cat_i = BindingAndCatalysis._bnc_cart_index(n_bind, i)
        bind_j, cat_j = BindingAndCatalysis._bnc_cart_index(n_bind, j)
        bind_i != bind_j && cat_i != cat_j
    end

    @test length(edges) == 32
    @test simultaneous == 16
end

@testset "Catalysis Exact Binding-Catalysis Mode" begin
    model = minimal_catalysis_model()
    find_all_regimes!(model)
    find_catalysis_regimes!(model)
    match_regimes!(model)

    bnc_rgm = first(get_bnc_regimes(model))
    regular = first(filter(r -> r.nlt == 0, get_bnc_regimes(model)))

    @test eltype(get_H0(bnc_rgm)) == ExactLogExpr
    @test eltype(get_C0_qKk(bnc_rgm)) == ExactLogExpr
    @test eltype(get_C0_wKk(bnc_rgm)) == ExactLogExpr
    @test !isempty(show_condition_qKk(bnc_rgm))
    @test !isempty(show_condition_wKk(bnc_rgm))
    @test !isempty(show_expression_qcat(regular))
end

@testset "High-Nullity Binding-Catalysis Regimes Keep Consistency" begin
    model = sparse_singular_model()
    update_catalysis!(model; Γ=[1 -1], Π=[1 0 0 0 0 0; 0 1 0 0 0 0], k_sym=[:k1, :k2])
    match_regimes!(model)

    bind_high = filter(
        r -> get_binding_regime(r).nullity > 1 && r.nlt <= 1, get_bnc_regimes(model)
    )
    @test !isempty(bind_high)

    low_bnc_rgm = first(bind_high)
    @test get_H_bd(low_bnc_rgm) isa AbstractMatrix
    @test is_stable(low_bnc_rgm) === true || is_stable(low_bnc_rgm) === false
    @test stability_code(low_bnc_rgm) in (-1, 1)
    @test !isempty(show_condition_qKk(low_bnc_rgm))
    @test !isempty(show_condition_wKk(low_bnc_rgm))

    H_low, H0_low = get_H_H0(low_bnc_rgm)
    @test size(H_low, 1) == model.n
    @test length(H0_low) == model.n

    high_model = sparse_singular_model()
    update_catalysis!(high_model; Γ=[1 -1], Π=[1 0 0 0 0 0; 1 0 0 0 0 0], k_sym=[:k1, :k2])
    match_regimes!(high_model)

    consistency_only = filter(r -> r.nlt > 1, get_bnc_regimes(high_model))
    @test !isempty(consistency_only)

    high_bnc_rgm = first(consistency_only)
    @test get_H_bd(high_bnc_rgm) isa AbstractMatrix
    @test is_stable(high_bnc_rgm) === true || is_stable(high_bnc_rgm) === false
    @test stability_code(high_bnc_rgm) in (-1, 1)
    @test isnothing(high_bnc_rgm.H)
    @test isnothing(high_bnc_rgm.H0)
    @test !isempty(show_condition_qKk(high_bnc_rgm))
    @test !isempty(show_condition_wKk(high_bnc_rgm))
    @test_throws Exception get_H_H0(high_bnc_rgm)
end

@testset "Catalysis Offsets And Binding-Catalysis Consistency" begin
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
        @test C0_xk[1:(cn.r_v)] == get_P0(cat_rgm)
        @test C0_xk[(cn.r_v + 1):end] == get_C0(cat_rgm)
        @test !isempty(show_condition_xk(cat_rgm; kind=:steady_state))
        @test !isempty(show_condition_xk(cat_rgm; kind=:dominance))
        @test size(C_xk, 2) == model.n + cn.n_v
    end

    match_regimes!(model)
    regular_bnc_rgm = first(
        filter(
            r -> r.nlt == 0 && !is_singular(get_binding_regime(r)), get_bnc_regimes(model)
        ),
    )
    bind_rgm = get_binding_regime(regular_bnc_rgm)
    cat_rgm = get_catalysis_regime(regular_bnc_rgm)
    r_v = size(get_P(cat_rgm), 1)

    P_ss = Matrix{Float64}(bind_rgm.P[(r_v + 1):end, :])
    P0_ss = Vector{Float64}(bind_rgm.P0[(r_v + 1):end])
    N = Matrix{Float64}(bind_rgm.network.N)
    PΠ = Matrix{Float64}(get_PΠ(cat_rgm))
    Pθ = Matrix{Float64}(get_P(cat_rgm))
    P0θ = Vector{Float64}(get_P0(cat_rgm))

    M_ss = vcat(P_ss, N, PΠ)
    M0_ss = vcat(P0_ss, zeros(Float64, size(N, 1) + r_v))
    H_ss = inv(M_ss)
    H0_ss = -(H_ss * M0_ss)
    split = size(H_ss, 2) - r_v
    H_right = H_ss[:, (split + 1):end]
    H_expected = hcat(H_ss[:, 1:split], -(H_right * Pθ))
    H0_expected = vec(H0_ss - H_right * P0θ)

    @test Matrix(get_H(regular_bnc_rgm)) ≈ H_expected
    @test get_H0(regular_bnc_rgm) ≈ H0_expected

    C_cat_qKk, C0_cat_qKk, nlt_cat_qKk = get_C_C0_nullity_qKk(regular_bnc_rgm, :catalysis)
    H_bind, H0_bind = get_H_H0(bind_rgm)
    C_expected = hcat(
        Matrix{Float64}(get_CΠ(cat_rgm) * H_bind), Matrix{Float64}(get_C_k(cat_rgm))
    )
    C0_expected = vec(get_CΠ(cat_rgm) * H0_bind + get_C0(cat_rgm))

    @test nlt_cat_qKk == 0
    @test Matrix(C_cat_qKk) ≈ C_expected
    @test C0_cat_qKk ≈ C0_expected
    @test !isempty(show_condition_qKk(regular_bnc_rgm; kind=:catalysis))
    @test !isempty(show_condition_wKk(regular_bnc_rgm))
end
