@testset "Larger RO Path Workflow" begin
    model = notebook_model2()
    pths = SISOPaths(model, 1)
    pths_single = SISOPaths(notebook_model2(), 1)

    grouped = group_sum([[1, 2], [1, 2], [2, 3]], fill(nothing, 3))
    @test length(grouped) == 2
    @test grouped[1][3] === nothing

    @test !isempty(pths.rgm_paths)
    @test get_path(pths, 1; return_idx = true) == pths.rgm_paths[1]
    @test get_idx(pths, get_path(pths, 1)) == 1
    @test pths.rgm_paths == pths_single.rgm_paths

    ro1 = get_RO_path(pths, 1; observe_x = 1)
    ro_paths_1 = get_RO_paths(pths; observe_x = 1)
    ro_paths_2 = get_RO_paths(pths; observe_x = 2, deduplicate = true)
    bulk_polys = get_polyhedra(pths)
    single_polys = [get_polyhedron(pths_single, i) for i in eachindex(pths_single.rgm_paths)]

    @test !isempty(ro1)
    @test length(ro_paths_1) == length(pths.rgm_paths)
    @test length(ro_paths_2) == length(pths.rgm_paths)
    @test all(BindingAndCatalysis.same_polyhedron.(bulk_polys, single_polys))
end

@testset "SISO Export Smoke" begin
    model = notebook_model2()
    pths = SISOPaths(model, 1)
    @test get_SISO_graph(model, 1) == get_SISO_graph(pths)
    @test get_neighbor_graph(pths) == get_neighbor_graph_qK(pths)
    sources = get_sources(get_SISO_graph(pths))
    sinks = get_sinks(get_SISO_graph(pths))
    @test sources == first(get_sources_sinks(get_SISO_graph(pths)))
    @test sinks == last(get_sources_sinks(get_SISO_graph(pths)))
    @test summary_RO_path(pths; observe_x = 1, show_volume = false) === nothing
    @test first(show_expression_path(pths, 1, 1; log_space = false)) isa AbstractVector
end

@testset "SISO Exact Smoke" begin
    N = [1 1 0 -1 0 0;
         0 1 1 0 -1 0;
         1 0 1 0 0 -1]
    model = Bnc(N = N)
    find_all_regimes!(model; mode = :exact)
    pths = SISOPaths(model, 1)
    polys = get_polyhedra(pths, 1:3)
    @test length(polys) == 3
    @test all(p -> p isa BindingAndCatalysis.Polyhedron, polys)
    @test eltype(get_C0_qK(get_regime(model, 1))) == ExactLogExpr
end
