@testset "Native Polyhedra API" begin
    rep_raw = NP.hrep(
        [1 0; -1 0; 0 1; 0 1],
        [1.0, -1.0, 1.0, 1.0],
    )
    @test rep_raw isa NP.HRep
    @test fieldnames(typeof(rep_raw)) == (:halfspaces, :ambient_dim)
    @test fieldnames(NP.HalfSpace) == (:p, :sign)
    @test fieldnames(NP.HyperPlane) == (:a, :β)

    poly_raw = NP.Polyhedron(copy(rep_raw.A), copy(rep_raw.b), copy(rep_raw.linset), false, false)
    @test fieldnames(typeof(poly_raw)) == (:halfspaces, :ambient_dim, :empty, :normalized)
    NP.detecthlinearity!(poly_raw)
    NP.removehredundancy!(poly_raw)
    @test NP.fulldim(poly_raw) == 2
    @test NP.hashyperplanes(poly_raw)
    @test length(NP.hyperplanes(poly_raw)) == 1
    @test length(NP.allhalfspaces(poly_raw)) == 1
    @test NP.dim(poly_raw) == 1
    @test NP.feasible_point(poly_raw) !== nothing

    box = NP.polyhedron(NP.hrep(
        [1 0; -1 0; 0 1; 0 -1],
        [1.0, 0.0, 1.0, 0.0],
    ))
    @test NP.hrep(box) isa NP.HRep
    @test NP.interior_point(box) !== nothing

    cut = NP.intersect(
        box,
        NP.HyperPlane([1.0, 0.0], 1.0),
        NP.HalfSpace([0.0, 1.0], 1.0),
    )
    proj = NP.eliminate(cut, 1)
    @test NP.issubset(cut, NP.HyperPlane([1.0, 0.0], 1.0))
    @test NP.fulldim(proj) == 1
    @test NP.feasible_point(proj) !== nothing

    vr_box = NP.vrep(box)
    @test length(NP.points(vr_box)) == 4
    @test isempty(NP.rays(vr_box))
    @test isempty(NP.lines(vr_box))
    @test get_one_inner_point(box) ≈ [0.5, 0.5]
    @test get_C_C0_nullity(NP.polyhedron(vr_box)) == get_C_C0_nullity(box)

    halfline = NP.polyhedron(NP.hrep(reshape([-1], 1, 1), [0]))
    vr_halfline = NP.vrep(halfline)
    @test length(NP.points(vr_halfline)) == 1
    @test length(NP.rays(vr_halfline)) == 1
    @test isempty(NP.lines(vr_halfline))
    @test Float64.(first(NP.points(vr_halfline))) == [0.0]
    @test Float64.(first(NP.rays(vr_halfline))) == [1.0]
end
