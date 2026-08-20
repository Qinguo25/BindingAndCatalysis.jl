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

    @test captured_error_message(() -> SIMOPaths(model, 1; condition_solver=:dag)) ==
        "ArgumentError: keyword `condition_solver` is no longer supported; use " *
          "`condition_method` instead. For `condition_solver=:dag`, use " *
          "`condition_method=:pair_memo_dag`."

    @test captured_error_message(() -> SIMOPaths(model, 1; condition_solver=:recursive)) ==
        "ArgumentError: keyword `condition_solver` is no longer supported; use " *
          "`condition_method` instead. For `condition_solver=:recursive`, the recursive " *
          "solver is now an internal test oracle; use `condition_method=:pair_memo_dag`."

    @test captured_error_message(() -> SIMOPaths(model, 1; condition_method=:dag)) ==
        "ArgumentError: unsupported condition_method=:dag; supported values are " *
          "`:pair_memo_dag` and `:suffix_dag`."

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
