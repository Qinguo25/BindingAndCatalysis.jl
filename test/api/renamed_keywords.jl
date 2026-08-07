function captured_error_message(f)
    try
        f()
    catch error
        return sprint(showerror, error)
    end
    return nothing
end

@testset "Renamed Keyword Errors" begin
    model = minimal_model()

    removed_full_message =
        "ArgumentError: keyword `full` is no longer supported by " *
        "`get_regimes_graph!`; remove the keyword."
    @test captured_error_message(() -> get_regimes_graph!(model; full=true)) ==
        removed_full_message
    @test captured_error_message(() -> get_regimes_graph!(model; full=false)) ==
        removed_full_message
    @test captured_error_message(() -> get_vertices_graph!(model; full=true)) ==
        removed_full_message

    graph = get_regimes_graph!(model)
    removed_edge_full_message =
        "ArgumentError: keyword `full` is no longer supported by " *
        "`get_edge`; remove the keyword."
    @test captured_error_message(() -> get_edge(graph, 1, 2; full=true)) ==
        removed_edge_full_message
    @test captured_error_message(() -> get_edge(model, 1, 2; full=false)) ==
        removed_edge_full_message
    @test !isdefined(BindingAndCatalysis, :x_traj_with_q_change)
    @test !isdefined(BindingAndCatalysis, :get_binding_network_grh)

    @test captured_error_message(() -> SIMOPaths(model, 1; condition_solver=:dag)) ==
        "ArgumentError: keyword `condition_solver` is no longer supported; use " *
          "`condition_method` instead. For `condition_solver=:dag`, use " *
          "`condition_method=:pair_memo_dag`."

    @test captured_error_message(() -> SIMOPaths(model, 1; condition_solver=:recursive)) ==
        "ArgumentError: keyword `condition_solver` is no longer supported; use " *
          "`condition_method` instead. For `condition_solver=:recursive`, the recursive " *
          "solver was removed; use `condition_method=:pair_memo_dag`."

    @test captured_error_message(() -> SIMOPaths(model, 1; condition_method=:dag)) ==
        "ArgumentError: unsupported condition_method=:dag; only `:pair_memo_dag` is supported."
    @test captured_error_message(() -> SIMOPaths(model, 1; condition_method=:suffix_dag)) ==
        "ArgumentError: unsupported condition_method=:suffix_dag; only " *
          "`:pair_memo_dag` is supported."

    paths = SIMOPaths(model, 1)
    @test captured_error_message(() -> get_volumes(paths; recalculate=false)) ==
        "ArgumentError: keyword `recalculate` is no longer supported; use `recompute` instead."
    @test captured_error_message(
        () -> get_volumes(paths; recompute=true, recalculate=false)
    ) ==
        "ArgumentError: keyword `recalculate` is no longer supported; use `recompute` instead."
    @test captured_error_message(() -> get_volumes(model; recalculate=false)) ==
        "ArgumentError: keyword `recalculate` is no longer supported; use `recompute` instead."

    C = [1.0;;]
    C0 = [0.0]
    @test captured_error_message(() -> calc_volume(C, C0; abs_tol=1.0e-3)) ==
        "ArgumentError: keyword `abs_tol` is no longer supported; use `abstol` instead."
    @test captured_error_message(() -> calc_volume(C, C0; rel_tol=1.0e-3)) ==
        "ArgumentError: keyword `rel_tol` is no longer supported; use `reltol` instead."
    @test captured_error_message(() -> calc_volume(C, C0; abstol=1.0e-3, abs_tol=1.0e-3)) ==
        "ArgumentError: keyword `abs_tol` is no longer supported; use `abstol` instead."

    empty_polys = Polyhedron[]
    @test captured_error_message(() -> calc_volume(empty_polys; rel_tol=1.0e-3)) ==
        "ArgumentError: keyword `rel_tol` is no longer supported; use `reltol` instead."
end
