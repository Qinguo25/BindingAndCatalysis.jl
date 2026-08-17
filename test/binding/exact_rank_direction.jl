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

@testset "Immutable Exact Hyperplane Identity" begin
    B = BindingAndCatalysis
    input_coeffs = Dict(3 => 1//2, 5 => 0//1, 2 => 2//3)
    x1 = ExactLogExpr(1//7, input_coeffs)

    input_coeffs[3] = 9//1
    input_coeffs[7] = 1//1
    @test collect(x1.coeffs) == [2 => 2//3, 3 => 1//2]
    @test Dict(x1.coeffs) == Dict(2 => 2//3, 3 => 1//2)
    @test_throws MethodError setindex!(x1.coeffs, 1//1, 2)
    @test_throws MethodError delete!(x1.coeffs, 2)

    x2 = ExactLogExpr(1//7, Dict(2 => 2//3, 3 => 1//2))
    @test x1 == x2
    @test isequal(x1, x2)
    @test hash(x1) == hash(x2)
    @test Dict(x1 => :present)[x2] === :present

    c1 = sparsevec([1, 2], Rational{Int}[1, -1], 2)
    c2 = sparsevec([1, 2], Rational{Int}[1, -1], 2)
    key1 = B.get_hp_key(c1, x1)
    key2 = B.get_hp_key(c2, x2)
    @test key1 == key2
    @test isequal(key1, key2)
    @test hash(key1) == hash(key2)
    @test Dict(key1 => :present)[key2] === :present

    pool = B.RegimeToHyperplanePool(2)
    hid1 = B.add_hyperplane!(pool, c1, x1)
    hid2 = B.add_hyperplane!(pool, c2, x2)
    @test hid1 == hid2 == 1
    @test length(pool.hyperplanes) == 1
    @test length(pool.hp_dict) == 1

    hid3 = B.add_hyperplane!(pool, c2, x2 + 1)
    @test hid3 == 2
    @test length(pool.hyperplanes) == 2
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
