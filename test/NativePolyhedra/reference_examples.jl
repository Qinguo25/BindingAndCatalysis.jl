@testset "cddlib Reference Roundtrip" begin
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
