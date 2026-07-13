@testset "Fiber/Chamber Architecture" begin
    coordinate_space = VariationSubspace(4, [2, 4])
    @test ambient_dimension(coordinate_space) == 4
    @test fiber_dimension(coordinate_space) == 2
    @test base_dimension(coordinate_space) == 2

    chart = FiberChart(4, [2, 4])
    @test chart.quotient_map * coordinate_space.basis ≈ zeros(2, 2)
    @test chart.quotient_map * chart.section ≈ Matrix{Float64}(I, 2, 2)
    @test chart.quotient_map == [1.0 0.0 0.0 0.0; 0.0 0.0 1.0 0.0]

    toy_problem = FiberProblem(:toy, coordinate_space; parameter_chart=:qK)
    fiber = fiber_at(toy_problem, [3.0, -2.0])
    @test fiber.problem === toy_problem
    @test toy_problem.chart.quotient_map * fiber.offset ≈ fiber.base_point
    @test_throws DimensionMismatch fiber_at(toy_problem, [1.0])
    @test_throws ArgumentError VariationSubspace([1.0 2.0; 2.0 4.0])

    pair_paths = SIMOPaths(minimal_model(), 1; condition_method=:pair_memo_dag)
    suffix_paths = SIMOPaths(minimal_model(), 1; condition_method=:suffix_dag)

    @test pair_paths.rgm_paths == suffix_paths.rgm_paths
    @test all(isnothing, pair_paths.path_feasible)
    @test get_fiber_problem(pair_paths).model === pair_paths.bn
    @test fiber_dimension(get_fiber_problem(pair_paths)) == 1
    @test base_dimension(get_fiber_problem(pair_paths)) == pair_paths.bn.n - 1
    @test get_slice_types(pair_paths) == OrderedRegimePath.(pair_paths.rgm_paths)
    @test get_sources(pair_paths) == sort(pair_paths.sources)
    @test get_sinks(pair_paths) == sort(pair_paths.sinks)
    @test BindingAndCatalysis.get_change_qK_idx(pair_paths) == 1

    axis2_pair = SIMOPaths(minimal_model(), 2; condition_method=:pair_memo_dag)
    axis2_suffix = SIMOPaths(minimal_model(), 2; condition_method=:suffix_dag)
    @test axis2_pair.fiber_problem.chart.quotient_map == [1.0 0.0 0.0; 0.0 0.0 1.0]
    @test axis2_pair.rgm_paths == axis2_suffix.rgm_paths
    @test all(
        BindingAndCatalysis.same_polyhedron.(
            get_polyhedra(axis2_pair), get_polyhedra(axis2_suffix)
        ),
    )

    pair_polys = get_polyhedra(pair_paths)
    suffix_polys = get_polyhedra(suffix_paths)
    @test all(BindingAndCatalysis.same_polyhedron.(pair_polys, suffix_polys))
    @test pair_paths.path_feasible == .!isempty.(pair_polys)
    @test all(is_feasible(pair_paths, idx) for idx in eachindex(pair_paths.rgm_paths))

    recursive_oracle = BindingAndCatalysis.Axis1DPairMemoBackend(
        BindingAndCatalysis._build_axis1d_problem(
            pair_paths.bn,
            pair_paths.change_qK_idx,
            pair_paths.qK_grh,
            pair_paths.sources,
            pair_paths.sinks,
        ),
    )
    endpoint_pairs = unique((first(path), last(path)) for path in pair_paths.rgm_paths)
    for (source, sink) in endpoint_pairs
        BindingAndCatalysis._find_pair_path_conditions!(recursive_oracle, source, sink)
    end
    for (idx, path) in enumerate(pair_paths.rgm_paths)
        oracle_map = BindingAndCatalysis._pair_conditions(
            recursive_oracle, first(path), last(path)
        )
        oracle_condition = get(oracle_map, BindingAndCatalysis._path_key(path), nothing)
        @test !isnothing(oracle_condition)
        @test BindingAndCatalysis.same_polyhedron(pair_polys[idx], oracle_condition)
    end

    conditional_types = get_conditional_slice_types(pair_paths)
    @test length(conditional_types) == length(pair_paths.rgm_paths)
    @test getfield.(conditional_types, :feasible) == pair_paths.path_feasible
    @test all(is_feasible, conditional_types)
    @test all(
        t -> t.condition_dimension <= base_dimension(pair_paths.fiber_problem),
        conditional_types,
    )

    # Requesting one endpoint pair must not make a later endpoint fall back to
    # the recursive oracle. Each uncached request gets a fresh DAG profile.
    incremental = SIMOPaths(minimal_model(), 1; condition_method=:pair_memo_dag)
    @test length(incremental.rgm_paths) >= 2
    get_polyhedron(incremental, 1)
    first_cache_size = length(incremental.condition_backend.pair_conditions)
    get_polyhedron(incremental, 2)
    @test length(incremental.condition_backend.pair_conditions) > first_cache_size
    @test incremental.condition_backend.dag_profile.planned_pairs > 0

    # Lock backend parity to a nontrivial fixture rather than relying only on
    # the two-path minimal model.
    medium_model = notebook_model2()
    medium_pair = SIMOPaths(medium_model, 1; condition_method=:pair_memo_dag)
    medium_suffix = SIMOPaths(medium_model, 1; condition_method=:suffix_dag)
    @test length(medium_pair.rgm_paths) == 15
    @test medium_pair.rgm_paths == medium_suffix.rgm_paths
    @test all(
        BindingAndCatalysis.same_polyhedron.(
            get_polyhedra(medium_pair), get_polyhedra(medium_suffix)
        ),
    )

    # Custom paths must use real, correctly oriented SIMO edges. Fabricating
    # an interface between non-neighbor regimes is not a candidate path.
    @test_throws ArgumentError SIMOPaths(
        medium_model, 1; rgm_paths=[[1, 3, 5]], condition_method=:pair_memo_dag
    )

    # A solved endpoint pair may legitimately omit a candidate tuple. Lock the
    # public representation of that result: an empty condition and false.
    missing_tuple = SIMOPaths(minimal_model(), 1; condition_method=:pair_memo_dag)
    missing_path = first(missing_tuple.rgm_paths)
    BindingAndCatalysis._cache_pair_conditions!(
        missing_tuple.condition_backend,
        first(missing_path),
        last(missing_path),
        BindingAndCatalysis.Axis1DPathConditionMap(),
    )
    @test all(isnothing, missing_tuple.path_feasible)
    @test isempty(get_polyhedron(missing_tuple, 1))
    @test !is_feasible(missing_tuple, 1)
    @test missing_tuple.path_feasible[1] === false
    @test all(isnothing, @view(missing_tuple.path_feasible[2:end]))

    singleton_pair = SIMOPaths(
        minimal_model(), 1; rgm_paths=[[1]], condition_method=:pair_memo_dag
    )
    singleton_suffix = SIMOPaths(
        minimal_model(), 1; rgm_paths=[[1]], condition_method=:suffix_dag
    )
    @test BindingAndCatalysis.same_polyhedron(
        get_polyhedron(singleton_pair, 1), get_polyhedron(singleton_suffix, 1)
    )

    @test_throws ArgumentError SIMOPaths(minimal_model(), 1; rgm_paths=[Int[]])
    @test_throws ArgumentError SIMOPaths(minimal_model(), 1; rgm_paths=[[1, 1]])
    @test_throws ArgumentError SIMOPaths(minimal_model(), 1; rgm_paths=[[1, 2], [2, 1]])
end
