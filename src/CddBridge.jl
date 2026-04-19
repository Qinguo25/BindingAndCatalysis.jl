module CddBridge

using ..NativePolyhedra
using ..ExactTypes: ExactLogExpr, exact_log10_ratio
using SparseArrays

const _DEFAULT_LOCAL_CDD_BINDIR = joinpath(dirname(@__DIR__), ".build", "cddlog", "src")

const _LOCAL_CDD_BUILD_HINT = "Run `Pkg.build()` after installing gcc/cc/clang and libgmp-dev, or point `BNC_CDDLOG_SOURCE_DIR` at a local cddlib-logarithmic source tree."

@inline _local_cdd_disabled() = get(ENV, "BNC_DISABLE_LOCAL_CDD", "0") == "1"

function _local_cdd_bindir(; require_log::Bool=false)
    _local_cdd_disabled() && return nothing
    require_log && get(ENV, "BNC_DISABLE_CDDLOG", "0") == "1" && return nothing

    candidates = String[]
    if require_log
        haskey(ENV, "BNC_CDDLOG_BINDIR") && push!(candidates, ENV["BNC_CDDLOG_BINDIR"])
        haskey(ENV, "BNC_CDDLOG_BUILD_DIR") && push!(candidates, joinpath(ENV["BNC_CDDLOG_BUILD_DIR"], "src"))
    end
    haskey(ENV, "BNC_CDD_BINDIR") && push!(candidates, ENV["BNC_CDD_BINDIR"])
    haskey(ENV, "BNC_CDD_BUILD_DIR") && push!(candidates, joinpath(ENV["BNC_CDD_BUILD_DIR"], "src"))
    push!(candidates, _DEFAULT_LOCAL_CDD_BINDIR)

    required = require_log ? ("projection_log", "redcheck_log") : ("projection", "redcheck")
    for dir in candidates
        isempty(dir) && continue
        all(isfile(joinpath(dir, tool)) for tool in required) || continue
        return dir
    end
    return nothing
end

_cdd_bindir() = _local_cdd_bindir(require_log=false)
_cddlog_bindir() = _local_cdd_bindir(require_log=true)

_cdd_available() = !isnothing(_cdd_bindir())
_cddlog_available() = !isnothing(_cddlog_bindir())

function _require_local_cdd!()
    bindir = _cdd_bindir()
    isnothing(bindir) && error("Local cdd backend is required but not available. $(_LOCAL_CDD_BUILD_HINT)")
    return bindir
end

function _require_local_cddlog!()
    bindir = _cddlog_bindir()
    isnothing(bindir) && error("Local cddlog backend is required but not available. $(_LOCAL_CDD_BUILD_HINT)")
    return bindir
end

@inline _can_use_cdd_fastpath(is_exact::Bool) = !is_exact && _cdd_available()

@inline _supports_cddlog(poly::NativePolyhedra.Polyhedron) = any(h -> h.p.β isa ExactLogExpr, poly.halfspaces) &&
    all(h -> all(x -> x isa Rational{Int} || x isa Integer, h.p.a), poly.halfspaces)

function _require_cddlog_support!(poly::NativePolyhedra.Polyhedron)
    _supports_cddlog(poly) && return nothing
    error("Local cddlog backend only supports exact polyhedra with rational coefficients and `ExactLogExpr` right-hand sides.")
end

@inline _is_exact_rhs(C0::AbstractVector) = any(x -> x isa ExactLogExpr, C0)

function _has_float_data(C::AbstractMatrix, C0::AbstractVector)
    eltype(C) <: AbstractFloat && return true
    eltype(C0) <: AbstractFloat && return true
    for x in C0
        x isa AbstractFloat && return true
    end
    for x in C
        iszero(x) && continue
        x isa AbstractFloat && return true
    end
    return false
end

function _cdd_numbertype(C::AbstractMatrix, C0::AbstractVector)
    if _is_exact_rhs(C0)
        return :logarithmic
    elseif _has_float_data(C, C0)
        return :real
    else
        return :rational
    end
end

function _empty_polyhedron(dim::Integer)
    return NativePolyhedra.Polyhedron(NativePolyhedra.HalfSpace[], Int(dim), true, false)
end

function _polyhedron_to_C_C0_nullity(poly::NativePolyhedra.Polyhedron)
    poly.empty && throw(ArgumentError("Cannot serialize an empty polyhedron to an H-representation."))
    n = NativePolyhedra.fulldim(poly)
    eq_rows = Vector{Vector{Any}}()
    eq_rhs = Any[]
    ineq_rows = Vector{Vector{Any}}()
    ineq_rhs = Any[]

    for hs in poly.halfspaces
        # Project-level H-representation uses C * x + C0 >= 0.
        # NativePolyhedra stores halfspaces as a * x <= β, so C = -a and C0 = β.
        a = [-x for x in NativePolyhedra._constraint_vector(hs)]
        β = NativePolyhedra._constraint_rhs(hs)
        if NativePolyhedra._isequality(hs)
            push!(eq_rows, Any[x for x in a])
            push!(eq_rhs, β)
        else
            push!(ineq_rows, Any[x for x in a])
            push!(ineq_rhs, β)
        end
    end

    rows = vcat(eq_rows, ineq_rows)
    rhs = vcat(eq_rhs, ineq_rhs)
    TA = isempty(rows) ? Rational{Int} : foldl(promote_type, (typeof(x) for row in rows for x in row); init=Rational{Int})
    TB = isempty(rhs) ? ExactLogExpr : foldl(promote_type, map(typeof, rhs); init=ExactLogExpr)
    A = spzeros(TA, length(rows), n)
    for i in eachindex(rows), j in 1:n
        v = rows[i][j]
        iszero(v) && continue
        A[i, j] = convert(TA, v)
    end
    C0 = TB[convert(TB, v) for v in rhs]
    return A, C0, length(eq_rows)
end

function _polyhedron_from_C_C0_nullity(C::AbstractMatrix{<:Real}, C0::AbstractVector, nullity::Integer)
    halfspaces = NativePolyhedra.HalfSpace[]
    sizehint!(halfspaces, size(C, 1))
    for i in 1:size(C, 1)
        sign = i <= nullity ? Int8(0) : Int8(1)
        row = vec(Array(C[i, :]))
        push!(halfspaces, NativePolyhedra.HalfSpace([-x for x in row], C0[i], sign))
    end
    return NativePolyhedra.Polyhedron(halfspaces, size(C, 2), false, false)
end

function _combine_polyhedra(polys::AbstractVector{<:NativePolyhedra.Polyhedron})
    isempty(polys) && throw(ArgumentError("Need at least one polyhedron."))
    n = NativePolyhedra.fulldim(polys[1])
    any(p -> p.empty, polys) && return _empty_polyhedron(n)
    halfspaces = NativePolyhedra.HalfSpace[]
    for poly in polys
        NativePolyhedra.fulldim(poly) == n || throw(DimensionMismatch("All polyhedra must have the same ambient dimension."))
        append!(halfspaces, copy(poly.halfspaces))
    end
    return NativePolyhedra.Polyhedron(halfspaces, n, false, false)
end

function _rational_str(x::Rational{<:Integer})
    return denominator(x) == 1 ? string(numerator(x)) : string(numerator(x), "/", denominator(x))
end

_rational_str(x::Integer) = string(x)

function _exactlogexpr_to_cddlog(x::ExactLogExpr)
    iszero(x) && return "0"

    parts = String[]
    if !iszero(x.constant)
        push!(parts, _rational_str(x.constant))
    end
    for (p, c) in sort!(collect(x.coeffs); by=first)
        ratio = "log($(p)/1)/log(10/1)"
        coeff = _rational_str(c)
        term = c == 1//1 ? ratio : c == -1//1 ? "-" * ratio : coeff * "*" * ratio
        push!(parts, term)
    end

    out = first(parts)
    for part in Iterators.drop(parts, 1)
        if startswith(part, "-") || startswith(part, "+")
            out *= part
        else
            out *= "+" * part
        end
    end
    return out
end

function _scalar_to_cdd_token(x, numbertype::Symbol)
    if numbertype === :logarithmic
        return x isa ExactLogExpr ? _exactlogexpr_to_cddlog(x) : x isa Rational ? _rational_str(x) : string(x)
    elseif numbertype === :real
        return string(Float64(x))
    elseif numbertype === :rational
        return x isa Rational ? _rational_str(x) : x isa Integer ? string(x) : _rational_str(rationalize(x))
    else
        error("Unsupported cdd number type: $numbertype")
    end
end

function _write_cdd_hrep(path::AbstractString, C::AbstractMatrix{<:Real}, C0::AbstractVector, nullity::Integer)
    size(C, 1) == length(C0) || throw(DimensionMismatch("size(C,1) must match length(C0)."))
    numbertype = _cdd_numbertype(C, C0)
    open(path, "w") do io
        println(io, "H-representation")
        nullity > 0 && println(io, "linearity ", nullity, " ", join(1:nullity, " "))
        println(io, "begin")
        println(io, " ", size(C, 1), " ", size(C, 2) + 1, " ", String(numbertype))
        for i in 1:size(C, 1)
            parts = String[_scalar_to_cdd_token(C0[i], numbertype)]
            for j in 1:size(C, 2)
                push!(parts, _scalar_to_cdd_token(C[i, j], numbertype))
            end
            println(io, join(parts, " "))
        end
        println(io, "end")
    end
    return path
end

function _eval_exact_expr_ast(ex)
    if ex isa Integer
        return ex
    elseif ex isa Rational
        return ex
    elseif ex isa Expr && ex.head === :call
        fn = ex.args[1]
        if fn === :exact_log10_ratio
            return exact_log10_ratio(Int(ex.args[2]), Int(ex.args[3]))
        elseif fn === :+
            return _eval_exact_expr_ast(ex.args[2]) + _eval_exact_expr_ast(ex.args[3])
        elseif fn === :-
            if length(ex.args) == 2
                return -_eval_exact_expr_ast(ex.args[2])
            end
            return _eval_exact_expr_ast(ex.args[2]) - _eval_exact_expr_ast(ex.args[3])
        elseif fn === :*
            return _eval_exact_expr_ast(ex.args[2]) * _eval_exact_expr_ast(ex.args[3])
        elseif fn === :/
            return _eval_exact_expr_ast(ex.args[2]) / _eval_exact_expr_ast(ex.args[3])
        end
    end
    throw(ArgumentError("Unsupported cddlog expression: $ex"))
end

function _parse_cddlog_constant(token::AbstractString)
    stripped = replace(strip(token), " " => "")
    lowered = replace(stripped, r"log\(([0-9]+)/([0-9]+)\)/log\(10/1\)" => s"exact_log10_ratio(\1,\2)")
    value = _eval_exact_expr_ast(Meta.parse(lowered))
    return value isa ExactLogExpr ? value : ExactLogExpr(value)
end

function _parse_rational_token(token::AbstractString)
    s = strip(token)
    occursin("/", s) || return parse(Int, s) // 1
    a, b = split(s, "/"; limit=2)
    return parse(Int, a) // parse(Int, b)
end

_parse_float_token(token::AbstractString) = parse(Float64, strip(token))

function _parse_constant_token(token::AbstractString, numbertype::Symbol)
    if numbertype === :logarithmic
        return _parse_cddlog_constant(token)
    elseif numbertype === :real
        return _parse_float_token(token)
    elseif numbertype === :rational
        return _parse_rational_token(token)
    else
        error("Unsupported cdd number type: $numbertype")
    end
end

function _parse_coeff_token(token::AbstractString, numbertype::Symbol)
    if numbertype === :real
        return _parse_float_token(token)
    elseif numbertype === :rational || numbertype === :logarithmic
        return _parse_rational_token(token)
    else
        error("Unsupported cdd number type: $numbertype")
    end
end

function _extract_last_hrep_block(text::AbstractString)
    lines = split(replace(text, "\r\n" => "\n"), '\n')
    starts = findall(i -> strip(lines[i]) == "H-representation", eachindex(lines))
    isempty(starts) && error("No H-representation block found in cdd output.")
    return @view lines[starts[end]:end]
end

function _parse_cdd_hrep(text::AbstractString)
    lines = _extract_last_hrep_block(text)
    idx = 2
    linset = BitSet()
    while idx <= length(lines)
        line = strip(lines[idx])
        isempty(line) && (idx += 1; continue)
        if startswith(line, "linearity")
            toks = split(line)
            nlin = parse(Int, toks[2])
            for tok in toks[3:min(end, 2 + nlin)]
                push!(linset, parse(Int, tok))
            end
            idx += 1
            continue
        end
        line == "begin" && break
        idx += 1
    end
    idx <= length(lines) || error("Malformed cdd output: missing begin.")

    dims = split(strip(lines[idx + 1]))
    m = parse(Int, dims[1])
    n = parse(Int, dims[2]) - 1
    numbertype = Symbol(lowercase(dims[3]))
    numbertype === :integer && (numbertype = :rational)
    row_lines = String[]
    for line in @view lines[(idx + 2):end]
        s = strip(line)
        s == "end" && break
        isempty(s) && continue
        push!(row_lines, s)
    end
    length(row_lines) == m || error("Malformed cdd output: expected $m rows, got $(length(row_lines)).")

    if numbertype === :real
        I = Int[]
        J = Int[]
        V = Float64[]
        C0 = Vector{Float64}(undef, m)
        for (i, line) in enumerate(row_lines)
            toks = split(line)
            length(toks) == n + 1 || error("Unsupported cdd row format: $line")
            C0[i] = _parse_constant_token(toks[1], numbertype)
            for j in 1:n
                cij = _parse_coeff_token(toks[j + 1], numbertype)
                iszero(cij) && continue
                push!(I, i)
                push!(J, j)
                push!(V, cij)
            end
        end
        C = sparse(I, J, V, m, n)
        eq_rows = sort!(collect(linset))
        ineq_rows = [i for i in 1:m if !(i in linset)]
        perm = vcat(eq_rows, ineq_rows)
        return C[perm, :], C0[perm], length(eq_rows)
    else
        I = Int[]
        J = Int[]
        V = Rational{Int}[]
        C0 = Vector{numbertype === :logarithmic ? ExactLogExpr : Rational{Int}}(undef, m)
        for (i, line) in enumerate(row_lines)
            toks = split(line)
            length(toks) == n + 1 || error("Unsupported cdd row format: $line")
            C0[i] = _parse_constant_token(toks[1], numbertype)
            for j in 1:n
                cij = _parse_coeff_token(toks[j + 1], numbertype)
                iszero(cij) && continue
                push!(I, i)
                push!(J, j)
                push!(V, cij)
            end
        end
        C = sparse(I, J, V, m, n)
        eq_rows = sort!(collect(linset))
        ineq_rows = [i for i in 1:m if !(i in linset)]
        perm = vcat(eq_rows, ineq_rows)
        return C[perm, :], C0[perm], length(eq_rows)
    end
end

function _tool_bindir(toolname::AbstractString)
    endswith(toolname, "_log") ? _cddlog_bindir() : _cdd_bindir()
end

function _run_cdd_hrep_tool(toolname::AbstractString, C::AbstractMatrix{<:Real}, C0::AbstractVector, nullity::Integer; stdin_text::AbstractString="")
    bindir = endswith(toolname, "_log") ? _require_local_cddlog!() : _require_local_cdd!()
    tool = joinpath(bindir, toolname)
    isfile(tool) || error("Missing local cdd tool: $tool")

    mktempdir() do tmp
        input_path = joinpath(tmp, "poly.ine")
        _write_cdd_hrep(input_path, C, C0, nullity)
        cmd = Cmd([tool, input_path])
        if isempty(stdin_text)
            return read(pipeline(cmd; stderr=devnull), String)
        end
        stdin_path = joinpath(tmp, "stdin.txt")
        write(stdin_path, stdin_text)
        open(stdin_path, "r") do io
            return read(pipeline(cmd; stdin=io, stderr=devnull), String)
        end
    end
end

function _canonicalize_hrep(C::AbstractMatrix{<:Real}, C0::AbstractVector, nullity::Integer)
    toolname = _is_exact_rhs(C0) ? "redcheck_log" : "redcheck"
    stdout_text = _run_cdd_hrep_tool(toolname, C, C0, nullity)
    return _parse_cdd_hrep(stdout_text)
end

function cdd_project_hrep(
    C::AbstractMatrix{<:Real},
    C0::AbstractVector,
    nullity::Integer,
    delset;
    canonicalize::Bool=false,
)
    axes = sort!(collect(Int.(delset)))
    if isempty(axes)
        Ccopy = sparse(C)
        if canonicalize
            return _canonicalize_hrep(Ccopy, C0, nullity)
        end
        return Ccopy, collect(C0), Int(nullity)
    end

    _is_exact_rhs(C0) ? _require_local_cddlog!() : _require_local_cdd!()
    toolname = _is_exact_rhs(C0) ? "projection_log" : "projection"
    stdin_text = string(length(axes), "\n", join(axes, "\n"), "\n")
    stdout_text = _run_cdd_hrep_tool(toolname, C, C0, nullity; stdin_text=stdin_text)
    Cproj, C0proj, nullity_proj = _parse_cdd_hrep(stdout_text)
    return canonicalize ? _canonicalize_hrep(Cproj, C0proj, nullity_proj) : (Cproj, C0proj, nullity_proj)
end

cddlog_project_hrep(C::AbstractMatrix{<:Real}, C0::AbstractVector, nullity::Integer, delset) =
    cdd_project_hrep(C, C0, nullity, delset)

function _cdd_canonicalize_poly(poly::NativePolyhedra.Polyhedron)
    poly.empty && return poly
    C, C0, nullity = _polyhedron_to_C_C0_nullity(poly)
    Ccan, C0can, ncan = _canonicalize_hrep(C, C0, nullity)
    return _polyhedron_from_C_C0_nullity(Ccan, C0can, ncan)
end

function cdd_eliminate(
    poly::NativePolyhedra.Polyhedron,
    delset::BitSet;
    canonicalize::Bool=false,
)
    out_dim = NativePolyhedra.fulldim(poly) - length(delset)
    poly.empty && return _empty_polyhedron(out_dim)
    if any(h -> h.p.β isa ExactLogExpr, poly.halfspaces)
        _require_local_cddlog!()
        _require_cddlog_support!(poly)
    else
        _require_local_cdd!()
    end
    C, C0, nullity = _polyhedron_to_C_C0_nullity(poly)
    Cproj, C0proj, nproj = cdd_project_hrep(C, C0, nullity, delset; canonicalize=canonicalize)
    return _polyhedron_from_C_C0_nullity(Cproj, C0proj, nproj)
end

function cdd_intersect_eliminate(
    poly1::NativePolyhedra.Polyhedron,
    poly2::NativePolyhedra.Polyhedron,
    delset::BitSet;
    canonicalize::Bool=false,
)
    combined = _combine_polyhedra([poly1, poly2])
    return cdd_eliminate(combined, delset; canonicalize=canonicalize)
end

function cdd_intersect_many(
    polys::AbstractVector{<:NativePolyhedra.Polyhedron};
    canonicalize::Bool=false,
)
    combined = _combine_polyhedra(polys)
    return canonicalize ? _cdd_canonicalize_poly(combined) : combined
end

end
