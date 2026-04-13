@testset "Native Polyhedra Projection And Redundancy" begin
    proj_src = NP.polyhedron(NP.hrep(
        [1 0 1; -1 0 0; 0 1 0; 0 -1 0],
        [1, 0, 1, 0],
    ))
    proj_fourier = NP.eliminate(proj_src, BitSet((2, 3)); method = :fourier)
    proj_block = NP.eliminate(proj_src, BitSet((2, 3)); method = :block)
    @test last(get_C_C0_nullity(proj_fourier)) == last(get_C_C0_nullity(proj_block))
    @test NP.fulldim(proj_fourier) == 1
    @test NP.fulldim(proj_block) == 1
    @test NP.feasible_point(proj_fourier) !== nothing
    @test NP.feasible_point(proj_block) !== nothing

    for name in ("sampleh6", "sampleh7")
        poly_orig, _ = _polyhedron_from_polyhedra_file(_poly_example_path("$name.ine"))
        poly_reduced, _ = _polyhedron_from_polyhedra_file(_poly_example_path("$name.ine"))
        NP.removehredundancy!(poly_reduced)
        @testset "$name redundancy" begin
            @test same_polyhedron(poly_orig, poly_reduced)
            @test size(NP.hrep(poly_reduced).A, 1) <= size(NP.hrep(poly_orig).A, 1)
        end
    end

    project_poly, project_parsed = _polyhedron_from_polyhedra_file(_poly_example_path("project1.ine"))
    project_expected, _ = _polyhedron_from_polyhedra_file(_poly_example_path("project1res.ine"))
    project_axes = _project_axes_from_option(project_parsed, NP.fulldim(project_poly))
    project_fourier = NP.eliminate(project_poly, project_axes; method = :fourier)
    project_block = NP.eliminate(project_poly, project_axes; method = :block)
    @test same_polyhedron(project_expected, project_fourier)
    @test same_polyhedron(project_expected, project_block)

    if get(ENV, "BNC_RUN_HEAVY_POLY_TESTS", "0") == "1"
        project2_poly, project2_parsed = _polyhedron_from_polyhedra_file(_poly_example_path("project2.ine"))
        project2_expected, _ = _polyhedron_from_polyhedra_file(_poly_example_path("project2res.ine"))
        project2_axes = _project_axes_from_option(project2_parsed, NP.fulldim(project2_poly))
        project2_block = NP.eliminate(project2_poly, project2_axes; method = :block)
        @test same_polyhedron(project2_expected, project2_block)
    end
end
