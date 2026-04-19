using SparseArrays

isdefined(@__MODULE__, :_polyhedron_from_polyhedra_file) || include(joinpath(@__DIR__, "..", "NativePolyhedra", "helpers.jl"))

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

@testset "local cdd Float Projection Regression" begin
    if isnothing(BindingAndCatalysis.CddBridge._cdd_bindir())
        @test true
    else
        C = sparse([1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0])
        C0 = [1.0, 0.0, 1.0, 0.0]
        Cproj, C0proj, nullity_proj = BindingAndCatalysis.CddBridge.cdd_project_hrep(C, C0, 0, BitSet([1]); canonicalize=true)
        expected_proj = get_polyhedron(sparse(reshape([1.0, -1.0], 2, 1)), [1.0, 0.0], 0)
        @test same_polyhedron(get_polyhedron(Cproj, C0proj, nullity_proj), expected_proj)
        @test nullity_proj == 0

        poly, parsed = _polyhedron_from_polyhedra_file(_poly_example_path("project1.ine"))
        expected, _ = _polyhedron_from_polyhedra_file(_poly_example_path("project1res.ine"))
        axes = _project_axes_from_option(parsed, BindingAndCatalysis.fulldim(poly))
        out = BindingAndCatalysis.CddBridge.cdd_eliminate(poly, axes; canonicalize=true)
        @test same_polyhedron(out, expected)
    end
end

@testset "CddBridge Serialization Is Read Only" begin
    poly = BindingAndCatalysis.polyhedron(BindingAndCatalysis.hrep(
        [1 0; 1 0; -1 0; 0 1; 0 -1],
        [1.0, 1.0, 0.0, 1.0, 0.0],
    ); strong = false)
    poly.normalized = false
    before_len = length(poly.halfspaces)
    before_norm = poly.normalized

    C, C0, nullity = BindingAndCatalysis.CddBridge._polyhedron_to_C_C0_nullity(poly)
    roundtrip = BindingAndCatalysis.CddBridge._polyhedron_from_C_C0_nullity(C, C0, nullity)

    @test BindingAndCatalysis.NativePolyhedra.fulldim(roundtrip) == 2
    @test length(poly.halfspaces) == before_len
    @test poly.normalized == before_norm
end
