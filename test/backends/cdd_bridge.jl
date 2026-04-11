@testset "cddlog Exact Projection Regression" begin
    if isnothing(BindingAndCatalysis.CddBridge._cddlog_bindir())
        @test true
    else
        C = sparse(Rational{Int}[
            -1 1 0 0 0 0
            -1 0 1 0 0 0
            -1 0 0 1 0 0
            -2 0 1 1 -1 1
            -1 0 0 0 0 1
            -2 1 0 1 -1 1
        ])
        C0 = fill(ExactLogExpr(0), 6)

        Cproj, C0proj, nullity_proj = BindingAndCatalysis.CddBridge.cddlog_project_hrep(C, C0, 0, BitSet([1]))
        @test size(Cproj) == (0, 5)
        @test isempty(C0proj)
        @test nullity_proj == 0

        poly = BindingAndCatalysis.CddBridge._polyhedron_from_C_C0_nullity(C, C0, 0)
        proj_poly = BindingAndCatalysis.CddBridge.maybe_cddlog_eliminate(poly, BitSet([1]); canonicalize = false)
        @test proj_poly !== nothing
        @test get_C_C0_nullity(proj_poly) == (spzeros(Rational{Int}, 0, 5), ExactLogExpr[], 0)
    end
end

@testset "CddBridge Conversion Is Read Only" begin
    poly = BindingAndCatalysis.polyhedron(BindingAndCatalysis.hrep(
        [1 0; 1 0; -1 0; 0 1; 0 -1],
        [1.0, 1.0, 0.0, 1.0, 0.0],
    ); strong = false)
    poly.normalized = false
    before_len = length(poly.halfspaces)
    before_norm = poly.normalized

    cdd_poly = BindingAndCatalysis.CddBridge._native_to_cdd(poly)
    roundtrip = BindingAndCatalysis.CddBridge._cdd_to_native(cdd_poly)

    @test BindingAndCatalysis.NativePolyhedra.fulldim(roundtrip) == 2
    @test length(poly.halfspaces) == before_len
    @test poly.normalized == before_norm
end
