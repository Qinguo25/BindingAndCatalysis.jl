@testset "cddlib And lrslib Reference Roundtrip" begin
    cube_h, _ = _polyhedron_from_polyhedra_file("src/lrslib-main/lrslib-main/cube.ine")
    cube_vrep, _ = _vrep_from_polyhedra_file("src/lrslib-main/lrslib-main/cube.ext")
    cube_from_v = NP.polyhedron(cube_vrep)
    cube_v = NP.vrep(cube_h)
    @test same_polyhedron(cube_h, cube_from_v)
    @test length(NP.points(cube_v)) == 8
    @test isempty(NP.rays(cube_v))
    @test isempty(NP.lines(cube_v))

    samplev_poly, _ = _polyhedron_from_polyhedra_file(_poly_example_path("samplev1.ext"))
    samplev_round = NP.vrep(samplev_poly)
    @test NP.dim(samplev_poly) == 1
    @test length(NP.points(samplev_round)) == 1
    @test length(NP.rays(samplev_round)) == 1

    samplev2_poly, _ = _polyhedron_from_polyhedra_file(_poly_example_path("samplev2.ext"))
    @test same_polyhedron(samplev2_poly, NP.polyhedron(NP.vrep(samplev2_poly)))

    for name in ("sampleh1", "sampleh2", "sampleh3", "sampleh4", "sampleh5")
        poly, _ = _polyhedron_from_polyhedra_file(_poly_example_path("$name.ine"))
        @testset "$name roundtrip" begin
            @test same_polyhedron(poly, NP.polyhedron(NP.vrep(poly)))
        end
    end

    bug_poly, _ = _polyhedron_from_polyhedra_file(_poly_example_path("bug45.ine"))
    bug_expected, _ = _polyhedron_from_polyhedra_file(_poly_example_path("bug45res.ine"))
    @test same_polyhedron(bug_poly, bug_expected)
end
