@testset "Relation Pruning" begin
    model = minimal_model()
    find_all_regimes!(model)

    unconstrained = SISOPaths(model, :tS)
    unconstrained_kw = SISOPaths(minimal_model(), :tS; qK_preconstraints = nothing)
    @test unconstrained.rgm_paths == unconstrained_kw.rgm_paths
    @test get_qK_preconstraints(unconstrained_kw) === nothing
    @test get_qK_constraints(unconstrained_kw) === nothing
    @test get_pruning_diagnostics(unconstrained_kw) === nothing

    g = get_SISO_graph(model, :tS)
    pruned_g, feasible_vertices, diagnostics =
        get_pruned_SISO_graph(model, :tS; qK_preconstraints = nothing)
    @test BindingAndCatalysis.Graphs.nv(pruned_g) == n_regimes(model)
    @test feasible_vertices == trues(n_regimes(model))
    @test diagnostics.original_vertices == n_regimes(model)
    @test diagnostics.feasible_vertices == n_regimes(model)
    @test diagnostics.original_edges == BindingAndCatalysis.Graphs.ne(g)
    @test diagnostics.feasible_edges == BindingAndCatalysis.Graphs.ne(g)
    @test diagnostics.removed_vertices == 0
    @test diagnostics.removed_edges == 0
    for edge in BindingAndCatalysis.Graphs.edges(g)
        @test BindingAndCatalysis.Graphs.has_edge(pruned_g, BindingAndCatalysis.Graphs.src(edge), BindingAndCatalysis.Graphs.dst(edge))
    end

    impossible_C = [
        1.0 0.0 0.0
       -1.0 0.0 0.0
    ]
    impossible_C0 = [-1.0, -1.0]
    empty_g, empty_feasible, empty_diagnostics =
        get_pruned_SISO_graph(model, :tS; qK_preconstraints = (impossible_C, impossible_C0))
    @test BindingAndCatalysis.Graphs.nv(empty_g) == n_regimes(model)
    @test BindingAndCatalysis.Graphs.ne(empty_g) == 0
    @test count(empty_feasible) == 0
    @test empty_diagnostics.feasible_vertices == 0
    @test empty_diagnostics.feasible_edges == 0

    empty_paths = SISOPaths(model, :tS; qK_preconstraints = (impossible_C, impossible_C0))
    @test isempty(empty_paths.rgm_paths)
    @test isempty(get_sources(empty_paths))
    @test isempty(get_sinks(empty_paths))
    @test isempty(get_polyhedra(empty_paths))
    @test get_pruning_diagnostics(empty_paths).feasible_edges == 0

    @test_throws ArgumentError SISOPaths(model, :tS; qK_preconstraints = ([1.0 0.0], [0.0]))
    @test_throws ArgumentError SISOPaths(model, :tS; qK_preconstraints = :not_a_constraint)
    @test_throws ArgumentError SISOPaths(
        model,
        :tS;
        qK_preconstraints = (impossible_C, impossible_C0),
        qK_constraints = (impossible_C, impossible_C0),
    )

    regular_vertex = first(filter(i -> get_nullity(model, i) == 0, get_indices(model)))
    point = get_one_inner_point(model, regular_vertex; rand_line = false, rand_ray = false)
    point_C = zeros(Float64, model.n, model.n)
    for i in 1:model.n
        point_C[i, i] = 1.0
    end
    point_C0 = -Float64.(point)
    point_g, point_feasible, point_diagnostics =
        get_pruned_SISO_graph(model, :tS; qK_preconstraints = (point_C, point_C0, model.n))
    @test BindingAndCatalysis.Graphs.nv(point_g) == n_regimes(model)
    @test point_feasible[regular_vertex]
    @test point_diagnostics.feasible_vertices >= 1
    @test point_diagnostics.feasible_vertices < n_regimes(model)
    @test point_diagnostics.feasible_edges == 0
    @test point_diagnostics.removed_vertices > 0
    @test point_diagnostics.removed_edges == point_diagnostics.original_edges

    constrained_paths = SISOPaths(model, :tS; qK_preconstraints = (point_C, point_C0, model.n))
    @test isempty(constrained_paths.rgm_paths)
    @test get_qK_preconstraints(constrained_paths) !== nothing
    @test get_pruning_diagnostics(constrained_paths).feasible_edges == 0

    change_idx = locate_sym_qK(model, :tS)
    symbolic_R = qK_preconstraint(model, :tS, :>, :tE, 0.25)
    manual_symbolic_C = zeros(Float64, 1, model.n)
    manual_symbolic_C[1, locate_sym_qK(model, :tS)] = 1.0
    manual_symbolic_C[1, locate_sym_qK(model, :tE)] = -1.0
    manual_symbolic_R = get_polyhedron(manual_symbolic_C, [-0.25])
    @test BindingAndCatalysis.same_polyhedron(symbolic_R, manual_symbolic_R)

    combined_R = qK_preconstraints(model, (:tS, :>, :tE, 0.25), (:K, :(==), :K))
    combined_C, combined_C0, combined_nullity = get_C_C0_nullity(combined_R)
    @test combined_nullity == 1
    @test size(combined_C, 2) == model.n
    @test length(combined_C0) == size(combined_C, 1)

    slice_C = zeros(Float64, 1, model.n)
    slice_C[1, change_idx] = 1.0
    slice_C0 = [-Float64(point[change_idx])]
    slice_R = get_polyhedron(slice_C, slice_C0, 1)
    helper = BindingAndCatalysis.SISOHelper(model, :tS; qK_preconstraints = slice_R)
    cached_prism = BindingAndCatalysis._get_vertex_prism!(helper, regular_vertex)
    manual_prism = BindingAndCatalysis._project_polyhedron(
        BindingAndCatalysis.Polyhedra.intersect(get_polyhedron(model, regular_vertex), slice_R),
        change_idx,
    )
    @test BindingAndCatalysis.same_polyhedron(cached_prism, manual_prism)
end
