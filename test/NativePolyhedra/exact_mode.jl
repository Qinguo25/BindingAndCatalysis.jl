@testset "Native Polyhedra Exact Mode" begin
    exact_val = exact_log10(100)
    exact_ratio = exact_log10_ratio(2, 5)
    @test exact_val isa ExactLogExpr
    @test Float64(exact_val) ≈ 2.0
    @test Float64(exact_ratio) ≈ log10(2 / 5)

    exact_rep = NP.hrep(
        Rational{Int}[1 0; -1 0; 0 1; 0 -1],
        ExactLogExpr[exact_log10(2), ExactLogExpr(0), exact_log10(3), ExactLogExpr(0)],
    )
    exact_poly = NP.polyhedron(exact_rep)
    exact_vr = NP.vrep(exact_poly)
    @test length(NP.points(exact_vr)) == 4
    @test first(first(NP.points(exact_vr))) isa ExactLogExpr
    @test same_polyhedron(exact_poly, NP.polyhedron(exact_vr))
end
