const NP = BindingAndCatalysis.NativePolyhedra
using Pkg.Artifacts: artifact_hash, artifact_exists, artifact_path
const _REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const _ARTIFACTS_TOML = joinpath(_REPO_ROOT, "Artifacts.toml")

function _normalize_cddlog_source_root(root::AbstractString)
    isdir(root) || return nothing
    if isfile(joinpath(root, "lib-src", "cddcore.c"))
        return root
    end
    subdirs = filter(name -> isdir(joinpath(root, name)), readdir(root))
    if length(subdirs) == 1
        nested = joinpath(root, only(subdirs))
        if isfile(joinpath(nested, "lib-src", "cddcore.c"))
            return nested
        end
    end
    return nothing
end

function _cddlog_source_roots()
    roots = String[]

    if haskey(ENV, "BNC_CDDLOG_SOURCE_DIR")
        src_root = _normalize_cddlog_source_root(ENV["BNC_CDDLOG_SOURCE_DIR"])
        src_root !== nothing && push!(roots, src_root)
    end

    if isfile(_ARTIFACTS_TOML)
        try
            hash = artifact_hash("cddlog_source", _ARTIFACTS_TOML)
            if hash !== nothing && artifact_exists(hash)
                src_root = _normalize_cddlog_source_root(artifact_path(hash))
                src_root !== nothing && push!(roots, src_root)
            end
        catch
        end
    end

    append!(roots, [
        joinpath(_REPO_ROOT, "src", "cddlib-master"),
    ])

    return unique(filter(isdir, roots))
end

function _poly_example_path(rel::AbstractString)
    candidates = String[]
    for root in _cddlog_source_roots()
        push!(candidates, joinpath(root, "examples", rel))
    end
    for path in candidates
        isfile(path) && return path
    end
    error("Could not locate polyhedra example file: $rel")
end

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
    end

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
