@testset "BNC Polyhedral Condition Correctness" begin
    B = BindingAndCatalysis

    @testset "condition blocks keep equalities first" begin
        C_bind = sparse([1.0 0.0; -1.0 0.0])
        C0_bind = [0.0, 1.0]
        C_cat = sparse([0.0 1.0; 0.0 -1.0])
        C0_cat = [0.0, 1.0]

        C, C0, nlt = B._stack_conditions(
            (C_bind, C0_bind, 0), (C_cat, C0_cat, 1)
        )

        @test nlt == 1
        @test Matrix(C) == [0.0 1.0; 1.0 0.0; -1.0 0.0; 0.0 -1.0]
        @test C0 == [0.0, 0.0, 1.0, 1.0]

        C_redundant, C0_redundant, nlt_redundant = B._stack_conditions(
            (sparse([1.0 0.0]), [2.0], 1),
            (sparse([2.0 0.0; 0.0 1.0]), [4.0, 3.0], 2),
        )
        @test nlt_redundant == 2
        @test rank(Matrix(C_redundant[1:nlt_redundant, :])) == nlt_redundant
        @test C0_redundant == [2.0, 3.0]

        C_empty, C0_empty, nlt_empty = B._stack_conditions(
            (sparse([1.0 0.0]), [0.0], 1),
            (sparse([1.0 0.0]), [1.0], 1),
        )
        @test nlt_empty == 0
        @test isempty(get_polyhedron(C_empty, C0_empty, nlt_empty))

        C_exact_empty, C0_exact_empty, nlt_exact_empty = B._stack_conditions(
            (sparse(reshape([1//1], 1, 1)), [0//1], 1),
            (sparse(reshape([1//1], 1, 1)), [1//10^12], 1),
        )
        @test nlt_exact_empty == 0
        @test isempty(get_polyhedron(C_exact_empty, C0_exact_empty, nlt_exact_empty))

        C_canonical, C0_canonical, nlt_canonical = B._maybe_remove_h_redundancy(
            C_exact_empty,
            C0_exact_empty,
            nlt_exact_empty;
            remove_h_redundancy=true,
        )
        @test nlt_canonical == 0
        @test size(C_canonical, 1) == 1
        @test C0_canonical == [-1//1]
    end

    @testset "zero coefficient rows respect equality semantics" begin
        C_zero = spzeros(Float64, 1, 2)

        C, C0, nlt = B._drop_trivial_true_rows(C_zero, [0.0], 1)
        @test size(C, 1) == 0
        @test isempty(C0)
        @test nlt == 0

        C, C0, nlt = B._drop_trivial_true_rows(C_zero, [1.0], 1)
        @test nlt == 0
        @test isempty(get_polyhedron(C, C0, nlt))

        C, C0, nlt = B._drop_trivial_true_rows(
            spzeros(Rational{Int}, 1, 1), [1//10^12], 1
        )
        @test nlt == 0
        @test isempty(get_polyhedron(C, C0, nlt))

        C, C0, nlt = B._drop_trivial_true_rows(C_zero, [1.0], 0)
        @test size(C, 1) == 0
        @test isempty(C0)
        @test nlt == 0

        C, C0, nlt = B._drop_trivial_true_rows(C_zero, [-1.0], 0)
        @test nlt == 0
        @test isempty(get_polyhedron(C, C0, nlt))
    end

    @testset "combined public condition preserves catalysis balance rows" begin
        model = minimal_catalysis_model()
        match_regimes!(model; warn_singular_propagation=false)
        rgm = first(get_bnc_regimes(model; feasible=nothing))

        C_bind, C0_bind = get_C_C0_xk(get_binding_regime(rgm))
        C_cat, C0_cat, nlt_cat = get_C_C0_nullity_xk(get_catalysis_regime(rgm))
        C, C0, nlt = get_C_C0_nullity_xk(rgm, :combined)

        @test nlt == nlt_cat
        @test Matrix(C[1:nlt, :]) == Matrix(C_cat[1:nlt_cat, :])
        @test C0[1:nlt] == C0_cat[1:nlt_cat]
        @test Matrix(C[(nlt + 1):(nlt + size(C_bind, 1)), :]) == Matrix(C_bind)
        @test C0[(nlt + 1):(nlt + length(C0_bind))] == C0_bind
    end

    @testset "empty affine-k projections remain ordinary infeasible regimes" begin
        model = minimal_model()
        update_catalysis!(
            model;
            Γ=[2 1 -1],
            Π=Matrix{Int}(I, 3, 3),
            F=reshape([1, 1, 1], 3, 1),
            F0=[-1, 0, 0],
            q_picked=[:tE],
            k_sym=[:k],
        )

        match_regimes!(model; warn_singular_propagation=false)
        diagnostics = bnc_regime_diagnostics(model)
        @test diagnostics.n_regimes == 8
        @test diagnostics.n_feasible == 7
        @test diagnostics.n_infeasible == 1
        @test length(get_bnc_regimes(model; feasible=nothing)) == 8
        @test length(get_bnc_regimes(model; feasible=false)) == 1
        infeasible = only(get_bnc_regimes(model; feasible=false))
        @test isempty(get_polyhedron(infeasible; chart=:wKk))
    end

    @testset "empty polyhedra have an explicit infeasible status" begin
        empty_poly = get_polyhedron(spzeros(Float64, 1, 2), [-1.0], 0)
        status = B._poly_dim_status(empty_poly)
        @test status.feasible === false
        @test status.dim == -1
        @test status.full_dim === false
    end
end
