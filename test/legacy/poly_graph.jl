@testset "Polyhedra Regime Graph" begin
    left_A = [
        1.0 0.0
        -1.0 0.0
        0.0 1.0
        0.0 -1.0
    ]
    left_b = [0.0, 1.0, 1.0, 0.0]
    right_A = copy(left_A)
    right_b = [1.0, 0.0, 1.0, 0.0]

    left = polyhedron(hrep(left_A, left_b), BindingAndCatalysis.POLY_BACK_END)
    right = polyhedron(hrep(right_A, right_b), BindingAndCatalysis.POLY_BACK_END)

    grh = polyhedra_regime_graph([left, right])
    g = get_neighbor_graph(grh)

    @test length(grh.neighbors) == 2
    @test Graphs.ne(g) == 1
    @test size(grh.hp_data[1].hp_to_poly.M, 1) == 2
    @test size(grh.hp_data[1].hp_to_poly.M, 2) == length(grh.hp_data[1].hyperplanes)
end
