const NP = BindingAndCatalysis.NativePolyhedra

function _parse_polyhedra_number(tok::AbstractString, numbertype::Symbol)
    s = strip(tok)
    if occursin("/", s)
        a, b = split(s, "/"; limit = 2)
        return parse(Int, strip(a)) // parse(Int, strip(b))
    elseif numbertype === :real || occursin(".", s) || occursin("e", lowercase(s))
        return parse(Float64, s)
    else
        return parse(Int, s)
    end
end

function _read_polyhedra_file(path::AbstractString)
    raw_lines = readlines(path)
    lines = strip.(raw_lines)
    rep = nothing
    linset = Int[]
    begin_idx = 0

    for (i, line) in enumerate(lines)
        isempty(line) && continue
        startswith(line, "*") && continue
        if occursin("H-representation", line)
            rep = :H
        elseif occursin("V-representation", line)
            rep = :V
        elseif startswith(lowercase(line), "linearity")
            toks = split(line)
            count = parse(Int, toks[2])
            linset = parse.(Int, toks[3:(2 + count)])
        elseif lowercase(line) == "begin"
            begin_idx = i
            break
        end
    end

    rep === nothing && error("Could not detect representation in $path")
    begin_idx > 0 || error("Could not find begin in $path")

    dims = split(lines[begin_idx + 1])
    m = parse(Int, dims[1])
    n = parse(Int, dims[2])
    numbertype = Symbol(lowercase(dims[3]))
    data = Vector{Vector{Any}}(undef, m)
    for i in 1:m
        toks = split(lines[begin_idx + 1 + i])
        data[i] = [_parse_polyhedra_number(tok, numbertype) for tok in toks]
        length(data[i]) == n || error("Bad row length in $path at row $i")
    end

    end_idx = begin_idx + 2 + m
    lowercase(lines[end_idx]) == "end" || error("Could not find end in $path")
    options = String[]
    for line in lines[(end_idx + 1):end]
        isempty(line) && continue
        startswith(line, "*") && continue
        push!(options, line)
    end

    return (representation = rep, linset = BitSet(linset), numbertype = numbertype, matrix = data, options = options)
end

function _polyhedron_from_polyhedra_file(path::AbstractString)
    parsed = _read_polyhedra_file(path)
    T = foldl(promote_type, (typeof(x) for row in parsed.matrix for x in row); init = Int)
    M = Matrix{T}(undef, length(parsed.matrix), length(first(parsed.matrix)))
    for i in axes(M, 1), j in axes(M, 2)
        M[i, j] = parsed.matrix[i][j]
    end
    if parsed.representation === :H
        b = vec(M[:, 1])
        A = -M[:, 2:end]
        return NP.polyhedron(NP.hrep(A, b, parsed.linset)), parsed
    else
        gens = M
        flags = vec(gens[:, 1])
        coords = gens[:, 2:end]
        pts = Vector{Vector}()
        rays = Vector{Vector}()
        lines = Vector{Vector}()
        for i in 1:size(coords, 1)
            coord = collect(coords[i, :])
            if i in parsed.linset
                push!(lines, coord)
            elseif flags[i] == 1
                push!(pts, coord)
            else
                push!(rays, coord)
            end
        end
        return NP.polyhedron(NP.VRep(pts, rays, lines)), parsed
    end
end

function _vrep_from_polyhedra_file(path::AbstractString)
    parsed = _read_polyhedra_file(path)
    parsed.representation === :V || error("Expected V-representation file: $path")
    T = foldl(promote_type, (typeof(x) for row in parsed.matrix for x in row); init = Int)
    M = Matrix{T}(undef, length(parsed.matrix), length(first(parsed.matrix)))
    for i in axes(M, 1), j in axes(M, 2)
        M[i, j] = parsed.matrix[i][j]
    end
    flags = vec(M[:, 1])
    coords = M[:, 2:end]
    pts = Vector{Vector}()
    rays = Vector{Vector}()
    lines = Vector{Vector}()
    for i in 1:size(coords, 1)
        coord = collect(coords[i, :])
        if i in parsed.linset
            push!(lines, coord)
        elseif flags[i] == 1
            push!(pts, coord)
        else
            push!(rays, coord)
        end
    end
    return NP.VRep(pts, rays, lines), parsed
end

function _project_axes_from_option(parsed, d::Int)
    line = only(filter(opt -> startswith(lowercase(opt), "project"), parsed.options))
    toks = split(line)
    k = parse(Int, toks[2])
    keep = parse.(Int, toks[3:(2 + k)])
    return BitSet(setdiff(1:d, keep))
end

@testset "Native Polyhedra API" begin
    exact_val = exact_log10(100)
    exact_ratio = exact_log10_ratio(2, 5)
    @test exact_val isa ExactLogExpr
    @test Float64(exact_val) ≈ 2.0
    @test Float64(exact_ratio) ≈ log10(2 / 5)

    rep_raw = NP.hrep(
        [1 0; -1 0; 0 1; 0 1],
        [1.0, -1.0, 1.0, 1.0],
    )
    @test rep_raw isa NP.HRep
    @test rep_raw isa NP.MixedMatHRep

    poly_raw = NP.Polyhedron(copy(rep_raw.A), copy(rep_raw.b), copy(rep_raw.linset), false, false)
    NP.detecthlinearity!(poly_raw)
    NP.removehredundancy!(poly_raw)
    @test poly_raw isa NP.Polyhedron
    @test NP.fulldim(poly_raw) == 2
    @test NP.hashyperplanes(poly_raw)
    @test length(NP.hyperplanes(poly_raw)) == 1
    @test length(NP.allhalfspaces(poly_raw)) == 1
    @test NP.dim(poly_raw) == 1
    @test NP.feasible_point(poly_raw) !== nothing

    box = NP.polyhedron(NP.hrep(
        [1 0; -1 0; 0 1; 0 -1],
        [1.0, 0.0, 1.0, 0.0],
    ))
    @test NP.hrep(box) isa NP.HRep
    @test NP.interior_point(box) !== nothing

    cut = NP.intersect(
        box,
        NP.HyperPlane([1.0, 0.0], 1.0),
        NP.HalfSpace([0.0, 1.0], 1.0),
    )
    proj = NP.eliminate(cut, 1)
    @test NP.issubset(cut, NP.HyperPlane([1.0, 0.0], 1.0))
    @test NP.fulldim(proj) == 1
    @test NP.feasible_point(proj) !== nothing

    vr_box = NP.vrep(box)
    @test length(NP.points(vr_box)) == 4
    @test isempty(NP.rays(vr_box))
    @test isempty(NP.lines(vr_box))
    @test get_one_inner_point(box) ≈ [0.5, 0.5]
    @test get_C_C0_nullity(NP.polyhedron(vr_box)) == get_C_C0_nullity(box)

    halfline = NP.polyhedron(NP.hrep(reshape([-1], 1, 1), [0]))
    vr_halfline = NP.vrep(halfline)
    @test length(NP.points(vr_halfline)) == 1
    @test length(NP.rays(vr_halfline)) == 1
    @test isempty(NP.lines(vr_halfline))
    @test Float64.(first(NP.points(vr_halfline))) == [0.0]
    @test Float64.(first(NP.rays(vr_halfline))) == [1.0]

    exact_rep = NP.hrep(
        Rational{Int}[1 0; -1 0; 0 1; 0 -1],
        ExactLogExpr[exact_log10(2), ExactLogExpr(0), exact_log10(3), ExactLogExpr(0)],
    )
    exact_poly = NP.polyhedron(exact_rep)
    exact_vr = NP.vrep(exact_poly)
    @test length(NP.points(exact_vr)) == 4
    @test first(first(NP.points(exact_vr))) isa ExactLogExpr
    @test same_polyhedron(exact_poly, NP.polyhedron(exact_vr))

    proj_src = NP.polyhedron(NP.hrep(
        [1 0 1; -1 0 0; 0 1 0; 0 -1 0],
        [1, 0, 1, 0],
    ))
    proj_fourier = NP.eliminate(proj_src, BitSet((2, 3)); method = :fourier)
    proj_block = NP.eliminate(proj_src, BitSet((2, 3)); method = :block)
    @test last(get_C_C0_nullity(proj_fourier)) == last(get_C_C0_nullity(proj_block))
    @test NP.fulldim(proj_fourier) == 1
    @test NP.fulldim(proj_block) == 1
    @test NP.feasible_point(proj_fourier) !== nothing
    @test NP.feasible_point(proj_block) !== nothing
end

@testset "cddlib And lrslib Reference Examples" begin
    cube_h, _ = _polyhedron_from_polyhedra_file("src/lrslib-main/lrslib-main/cube.ine")
    cube_vrep, _ = _vrep_from_polyhedra_file("src/lrslib-main/lrslib-main/cube.ext")
    cube_from_v = NP.polyhedron(cube_vrep)
    cube_v = NP.vrep(cube_h)
    @test same_polyhedron(cube_h, cube_from_v)
    @test length(NP.points(cube_v)) == 8
    @test isempty(NP.rays(cube_v))
    @test isempty(NP.lines(cube_v))

    samplev_poly, _ = _polyhedron_from_polyhedra_file("src/cddlib-master/examples/samplev1.ext")
    samplev_round = NP.vrep(samplev_poly)
    @test NP.dim(samplev_poly) == 1
    @test length(NP.points(samplev_round)) == 1
    @test length(NP.rays(samplev_round)) == 1

    samplev2_poly, _ = _polyhedron_from_polyhedra_file("src/cddlib-master/examples/samplev2.ext")
    samplev2_round = NP.vrep(samplev2_poly)
    @test same_polyhedron(samplev2_poly, NP.polyhedron(samplev2_round))

    sampleh1_poly, _ = _polyhedron_from_polyhedra_file("src/cddlib-master/examples/sampleh1.ine")
    sampleh1_v = NP.vrep(sampleh1_poly)
    @test same_polyhedron(sampleh1_poly, NP.polyhedron(sampleh1_v))

    sampleh2_poly, _ = _polyhedron_from_polyhedra_file("src/cddlib-master/examples/sampleh2.ine")
    sampleh2_v = NP.vrep(sampleh2_poly)
    @test same_polyhedron(sampleh2_poly, NP.polyhedron(sampleh2_v))

    sampleh3_poly, _ = _polyhedron_from_polyhedra_file("src/cddlib-master/examples/sampleh3.ine")
    sampleh3_v = NP.vrep(sampleh3_poly)
    @test same_polyhedron(sampleh3_poly, NP.polyhedron(sampleh3_v))

    sampleh4_poly, _ = _polyhedron_from_polyhedra_file("src/cddlib-master/examples/sampleh4.ine")
    sampleh4_v = NP.vrep(sampleh4_poly)
    @test same_polyhedron(sampleh4_poly, NP.polyhedron(sampleh4_v))

    sampleh5_poly, _ = _polyhedron_from_polyhedra_file("src/cddlib-master/examples/sampleh5.ine")
    sampleh5_v = NP.vrep(sampleh5_poly)
    @test same_polyhedron(sampleh5_poly, NP.polyhedron(sampleh5_v))

    sampleh6_poly_orig, _ = _polyhedron_from_polyhedra_file("src/cddlib-master/examples/sampleh6.ine")
    sampleh6_poly, _ = _polyhedron_from_polyhedra_file("src/cddlib-master/examples/sampleh6.ine")
    NP.removehredundancy!(sampleh6_poly)
    @test same_polyhedron(sampleh6_poly_orig, sampleh6_poly)
    @test size(NP.hrep(sampleh6_poly).A, 1) <= size(NP.hrep(sampleh6_poly_orig).A, 1)
    @test last(get_C_C0_nullity(sampleh6_poly)) == last(get_C_C0_nullity(sampleh6_poly_orig))

    sampleh7_poly_orig, _ = _polyhedron_from_polyhedra_file("src/cddlib-master/examples/sampleh7.ine")
    sampleh7_poly, _ = _polyhedron_from_polyhedra_file("src/cddlib-master/examples/sampleh7.ine")
    NP.removehredundancy!(sampleh7_poly)
    @test same_polyhedron(sampleh7_poly_orig, sampleh7_poly)
    @test size(NP.hrep(sampleh7_poly).A, 1) <= size(NP.hrep(sampleh7_poly_orig).A, 1)
    @test last(get_C_C0_nullity(sampleh7_poly)) == last(get_C_C0_nullity(sampleh7_poly_orig))

    project_poly, project_parsed = _polyhedron_from_polyhedra_file("src/cddlib-master/examples/project1.ine")
    project_expected, _ = _polyhedron_from_polyhedra_file("src/cddlib-master/examples/project1res.ine")
    project_axes = _project_axes_from_option(project_parsed, NP.fulldim(project_poly))
    project_fourier = NP.eliminate(project_poly, project_axes; method = :fourier)
    project_block = NP.eliminate(project_poly, project_axes; method = :block)
    @test same_polyhedron(project_expected, project_fourier)
    @test same_polyhedron(project_expected, project_block)

    if get(ENV, "BNC_RUN_HEAVY_POLY_TESTS", "0") == "1"
        project2_poly, project2_parsed = _polyhedron_from_polyhedra_file("src/cddlib-master/examples/project2.ine")
        project2_expected, _ = _polyhedron_from_polyhedra_file("src/cddlib-master/examples/project2res.ine")
        project2_axes = _project_axes_from_option(project2_parsed, NP.fulldim(project2_poly))
        project2_block = NP.eliminate(project2_poly, project2_axes; method = :block)
        @test same_polyhedron(project2_expected, project2_block)
    end

    bug_poly, _ = _polyhedron_from_polyhedra_file("src/cddlib-master/examples/bug45.ine")
    bug_expected, _ = _polyhedron_from_polyhedra_file("src/cddlib-master/examples/bug45res.ine")
    @test same_polyhedron(bug_poly, bug_expected)
    @test last(get_C_C0_nullity(bug_poly)) == last(get_C_C0_nullity(bug_expected))
end
