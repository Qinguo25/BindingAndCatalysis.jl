@testset "Legacy And Numeric API Smoke" begin
    model = minimal_model()
    find_all_vertices!(model)

    @test n_vertices(model) == n_regimes(model)
    @test get_vertices_perm_dict(model) == get_bind_regimes_dict(model)
    @test get_vertices(model) == get_regimes(model)
    @test get_vertex(model, 1) === get_regime(model, 1)
    @test get_vertices_graph!(model) === get_regimes_graph!(model)
    @test get_vertices_neighbor_mat(model) == get_regimes_neighbor_mat(model)

    inner = get_one_inner_point(model, 2)
    @test assign_vertex(
        model, inner; input_logspace=true, asymptotic_only=false, return_idx=true
    ) == 2
    @test assign_vertex_qK(
        model, inner; input_logspace=true, asymptotic_only=false, return_idx=true
    ) == 2
    @test assign_vertex_x(
        model,
        qK2x(model, inner; input_logspace=true, output_logspace=true);
        input_logspace=true,
        return_idx=true,
    ) == 2

    jac_qK_x = ∂logqK_∂logx(model; qK=inner, input_logspace=true)
    jac_x_qK = ∂logx_∂logqK(model; qK=inner, input_logspace=true)
    @test logder_qK_x(model; qK=inner, input_logspace=true) == jac_qK_x
    @test logder_x_qK(model; qK=inner, input_logspace=true) ≈ jac_x_qK

    @test locate_sym_x(model, :E) == 1
    @test locate_sym_qK(model, :K) == 3
    @test size(N_generator(2, 4)) == (2, 4)
    @test size(L_generator(2, 4)) == (2, 4)
    @test length(randomize(model, 2)) == 2
    @test_nowarn pythonprint([1, 2, 3])

    @test get_nullities(model) == get_nullity.(get_regimes(model))
    @test length(get_neighbors(model, 1)) == 2
    @test get_function(get_regime(model, 1))(
        inner; input_logspace=true, output_logspace=true
    ) isa AbstractVector
    singular_idx = only(filter(i -> get_nullity(model, i) > 0, get_indices(model)))
    get_regime(model, singular_idx).volume = nothing
    @test summary_vertex(model, singular_idx) === nothing
    @test get_regime(model, singular_idx).volume === nothing
end
