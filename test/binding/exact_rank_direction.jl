@testset "Exact Rank-One Denominator Classification" begin
    B = BindingAndCatalysis
    Q = Rational{BigInt}
    tiny = Q(1, big(10)^30)
    hp = B.Hyperplane_perm(1, 2, big(1), big(1), zero(Q))
    H0 = Q[0, 0]

    H_tiny = sparse(Q[tiny 1; 1 0])
    _, _, tiny_nullity, _, _ = B._rank1_step_update_from_regular(H_tiny, H0, 1, hp, Int8(1))
    @test tiny_nullity == 0

    H_zero = sparse(Q[0 1; 1 0])
    _, _, zero_nullity, _, _ = B._rank1_step_update_from_regular(H_zero, H0, 1, hp, Int8(1))
    @test zero_nullity == 1

    hp_float = B.Hyperplane_perm(1, 2, 1, 1, 0.0)
    H_float = sparse([1.0e-13 1.0; 1.0 0.0])
    _, _, float_nullity, _, _ = B._rank1_step_update_from_regular(
        H_float, zeros(2), 1, hp_float, Int8(1)
    )
    @test float_nullity == 1
end

@testset "Exact Binding Direction And Singular Summary" begin
    B = BindingAndCatalysis
    b = big(10)^30
    positive = [b b-1; b+1 b]

    @test B._det_sign_exact(positive) === 1
    @test B._det_sign_exact(positive[[2, 1], :]) === -1
    @test B._det_sign_exact([b b; b b]) === 0

    a = 10^9
    L = [a a-1 0 0; a+1 a 0 0]
    N = [0 0 1 0; 0 0 0 1]
    model = Bnc(; L=L, N=N)

    @test model.direction == Int8(1)

    model.direction = Int8(0)
    shown = sprint(show, MIME("text/plain"), model)
    @test occursin("Direction of binding reactions: singular/undefined", shown)
    @test !occursin("Direction of binding reactions: backward", shown)

    model.direction = Int8(-1)
    B._rebuild_helper!(model)
    @test model.direction == Int8(1)
end
