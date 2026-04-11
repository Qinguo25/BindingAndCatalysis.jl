module CddBridge

using ..NativePolyhedra
using ..ExactTypes: ExactLogExpr, exact_log10_ratio
using CDDLib
using Polyhedra
using SparseArrays

const _CDD_FLOAT_LIB = CDDLib.Library(:float)
const _DEFAULT_CDDLOG_BINDIR = joinpath(dirname(@__DIR__), ".build", "cddlog", "src")

@inline _can_use_cdd_fastpath(is_exact::Bool) = !is_exact

function _native_to_cdd(poly::NativePolyhedra.Polyhedron)
    n = NativePolyhedra.fulldim(poly)
    if poly.empty
        A = zeros(Float64, 1, n)
        b = Float64[-1.0]
        return Polyhedra.polyhedron(Polyhedra.hrep(A, b), _CDD_FLOAT_LIB)
    end
    rep = NativePolyhedra.hrep(poly)
    A = Matrix{Float64}(rep.A)
    b = isempty(rep.b) ? Float64[] : Float64[x for x in rep.b]
    return Polyhedra.polyhedron(Polyhedra.hrep(A, b, rep.linset), _CDD_FLOAT_LIB)
end

function _cdd_to_native(poly; canonicalize::Bool=false)
    h = Polyhedra.hrep(poly)
    n = Polyhedra.fulldim(poly)
    halfspaces = NativePolyhedra.HalfSpace{Float64,Float64}[]
    sizehint!(halfspaces, length(collect(Polyhedra.hyperplanes(h))) + length(collect(Polyhedra.halfspaces(h))))

    for hp in Polyhedra.hyperplanes(h)
        push!(halfspaces, NativePolyhedra.HalfSpace(NativePolyhedra.HyperPlane(Float64.(hp.a), Float64(hp.β)), 0))
    end
    for hs in Polyhedra.halfspaces(h)
        push!(halfspaces, NativePolyhedra.HalfSpace(NativePolyhedra.HyperPlane(Float64.(hs.a), Float64(hs.β)), 1))
    end

    out = NativePolyhedra.Polyhedron(halfspaces, n, Base.isempty(poly), false)
    canonicalize && !Base.isempty(poly) && NativePolyhedra.removehredundancy!(out; strong=false)
    return out
end

function cdd_intersect_eliminate(
    poly1::NativePolyhedra.Polyhedron,
    poly2::NativePolyhedra.Polyhedron,
    delset::BitSet,
    ;
    canonicalize::Bool=false,
)
    cdd_poly = Base.intersect(_native_to_cdd(poly1), _native_to_cdd(poly2))
    proj = Polyhedra.eliminate(cdd_poly, sort!(collect(delset)))
    return _cdd_to_native(proj; canonicalize=canonicalize)
end

function cdd_intersect_many(
    polys::AbstractVector{<:NativePolyhedra.Polyhedron};
    canonicalize::Bool=false,
)
    Base.isempty(polys) && throw(ArgumentError("Need at least one polyhedron."))
    cdd_poly = _native_to_cdd(polys[1])
    for p in @view polys[2:end]
        cdd_poly = Base.intersect(cdd_poly, _native_to_cdd(p))
    end
    return _cdd_to_native(cdd_poly; canonicalize=canonicalize)
end

function cdd_eliminate(
    poly::NativePolyhedra.Polyhedron,
    delset::BitSet;
    canonicalize::Bool=false,
)
    proj = Polyhedra.eliminate(_native_to_cdd(poly), sort!(collect(delset)))
    return _cdd_to_native(proj; canonicalize=canonicalize)
end

function cdd_project_hrep(
    C::AbstractMatrix{<:Real},
    C0::AbstractVector{<:Real},
    nullity::Integer,
    delset,
)
    A = -Matrix{Float64}(C)
    b = Float64[x for x in C0]
    rep = nullity == 0 ? Polyhedra.hrep(A, b) : Polyhedra.hrep(A, b, BitSet(1:nullity))
    poly = Polyhedra.polyhedron(rep, _CDD_FLOAT_LIB)
    proj = Polyhedra.eliminate(poly, sort!(collect(delset)))
    h = Polyhedra.hrep(proj)
    n = Polyhedra.fulldim(proj)
    hps = collect(Polyhedra.hyperplanes(h))
    hss = collect(Polyhedra.halfspaces(h))
    m = length(hps) + length(hss)
    Cproj = Matrix{Float64}(undef, m, n)
    C0proj = Vector{Float64}(undef, m)

    row = 1
    for hp in hps
        @inbounds begin
            Cproj[row, :] = -Float64.(hp.a)
            C0proj[row] = Float64(hp.β)
        end
        row += 1
    end
    for hs in hss
        @inbounds begin
            Cproj[row, :] = -Float64.(hs.a)
            C0proj[row] = Float64(hs.β)
        end
        row += 1
    end
    return sparse(Cproj), C0proj, length(hps)
end

function _cddlog_bindir()
    get(ENV, "BNC_DISABLE_CDDLOG", "0") == "1" && return nothing
    candidates = String[]
    haskey(ENV, "BNC_CDDLOG_BINDIR") && push!(candidates, ENV["BNC_CDDLOG_BINDIR"])
    haskey(ENV, "BNC_CDDLOG_BUILD_DIR") && push!(candidates, joinpath(ENV["BNC_CDDLOG_BUILD_DIR"], "src"))
    push!(candidates, _DEFAULT_CDDLOG_BINDIR)

    for dir in candidates
        isempty(dir) && continue
        isfile(joinpath(dir, "projection_log")) || continue
        return dir
    end
    return nothing
end

_cddlog_available() = !isnothing(_cddlog_bindir())

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
        if startswith(part, "-")
            out *= part
        elseif startswith(part, "+")
            out *= part
        else
            out *= "+" * part
        end
    end
    return out
end

_scalar_to_cddlog(x::ExactLogExpr) = _exactlogexpr_to_cddlog(x)
_scalar_to_cddlog(x::Integer) = string(x)
_scalar_to_cddlog(x::Rational{<:Integer}) = _rational_str(x)
_scalar_to_cddlog(x::AbstractFloat) = string(x)

function _polyhedron_to_C_C0_nullity(poly::NativePolyhedra.Polyhedron)
    n = NativePolyhedra.fulldim(poly)
    eq_rows = Vector{Vector{Any}}()
    eq_rhs = Any[]
    ineq_rows = Vector{Vector{Any}}()
    ineq_rhs = Any[]

    for hs in poly.halfspaces
        a = NativePolyhedra._constraint_vector(hs)
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

function _write_cddlog_hrep(path::AbstractString, C::AbstractMatrix{<:Real}, C0::AbstractVector, nullity::Integer)
    size(C, 1) == length(C0) || throw(DimensionMismatch("size(C,1) must match length(C0)."))
    open(path, "w") do io
        println(io, "H-representation")
        nullity > 0 && println(io, "linearity ", nullity, " ", join(1:nullity, " "))
        println(io, "begin")
        println(io, " ", size(C, 1), " ", size(C, 2) + 1, " logarithmic")
        for i in 1:size(C, 1)
            parts = String[_scalar_to_cddlog(C0[i])]
            for j in 1:size(C, 2)
                push!(parts, _scalar_to_cddlog(C[i, j]))
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

function _extract_last_hrep_block(text::AbstractString)
    lines = split(replace(text, "\r\n" => "\n"), '\n')
    starts = findall(i -> strip(lines[i]) == "H-representation", eachindex(lines))
    isempty(starts) && error("No H-representation block found in cddlog output.")
    return @view lines[starts[end]:end]
end

function _parse_cddlog_hrep(text::AbstractString)
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
    idx <= length(lines) || error("Malformed cddlog output: missing begin.")
    dims = split(strip(lines[idx + 1]))
    m = parse(Int, dims[1])
    n = parse(Int, dims[2]) - 1
    row_lines = String[]
    for line in @view lines[(idx + 2):end]
        s = strip(line)
        s == "end" && break
        isempty(s) && continue
        push!(row_lines, s)
    end
    length(row_lines) == m || error("Malformed cddlog output: expected $m rows, got $(length(row_lines)).")

    I = Int[]
    J = Int[]
    V = Rational{Int}[]
    C0 = Vector{ExactLogExpr}(undef, m)
    for (i, line) in enumerate(row_lines)
        toks = split(line)
        length(toks) == n + 1 || error("Unsupported cddlog row format: $line")
        C0[i] = _parse_cddlog_constant(toks[1])
        for j in 1:n
            cij = _parse_rational_token(toks[j + 1])
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

function _polyhedron_from_C_C0_nullity(C::SparseMatrixCSC{<:Real,Int}, C0::AbstractVector, nullity::Integer)
    halfspaces = NativePolyhedra.HalfSpace[]
    sizehint!(halfspaces, size(C, 1))
    for i in 1:size(C, 1)
        sign = i <= nullity ? Int8(0) : Int8(1)
        push!(halfspaces, NativePolyhedra.HalfSpace(collect(Array(C[i, :])), C0[i], sign))
    end
    return NativePolyhedra.Polyhedron(halfspaces, size(C, 2), false, false)
end

function _run_cddlog_hrep_tool(toolname::AbstractString, C::AbstractMatrix{<:Real}, C0::AbstractVector, nullity::Integer; stdin_text::AbstractString="")
    bindir = _cddlog_bindir()
    isnothing(bindir) && error("cddlog backend is not available.")
    tool = joinpath(bindir, toolname)
    isfile(tool) || error("Missing cddlog tool: $tool")

    mktempdir() do tmp
        input_path = joinpath(tmp, "poly.ine")
        _write_cddlog_hrep(input_path, C, C0, nullity)
        if isempty(stdin_text)
            stdout_text = read(`/usr/sbin/bash -lc "$tool $input_path 2>/dev/null"`, String)
            return stdout_text
        end
        stdin_path = joinpath(tmp, "stdin.txt")
        write(stdin_path, stdin_text)
        return read(`/usr/sbin/bash -lc "$tool $input_path < $stdin_path 2>/dev/null"`, String)
    end
end

function cddlog_project_hrep(
    C::AbstractMatrix{<:Real},
    C0::AbstractVector,
    nullity::Integer,
    delset,
)
    axes = sort!(collect(Int.(delset)))
    isempty(axes) && return sparse(C), ExactLogExpr[c isa ExactLogExpr ? c : ExactLogExpr(c) for c in C0], Int(nullity)
    stdin_text = string(length(axes), "\n", join(axes, "\n"), "\n")
    stdout_text = _run_cddlog_hrep_tool("projection_log", C, C0, nullity; stdin_text=stdin_text)
    return _parse_cddlog_hrep(stdout_text)
end

function maybe_cddlog_eliminate(poly::NativePolyhedra.Polyhedron, axes::BitSet; canonicalize::Bool=true, method::Symbol=:auto)
    _cddlog_available() || return nothing
    any(h -> h.p.β isa ExactLogExpr, poly.halfspaces) || return nothing
    all(h -> all(x -> x isa Rational{Int} || x isa Integer, h.p.a), poly.halfspaces) || return nothing

    try
        C, C0, nullity = _polyhedron_to_C_C0_nullity(poly)
        Cproj, C0proj, nullity_proj = cddlog_project_hrep(C, C0, nullity, axes)
        out = _polyhedron_from_C_C0_nullity(Cproj, C0proj, nullity_proj)
        canonicalize && NativePolyhedra.removehredundancy!(out; strong=false)
        return out
    catch
        return nothing
    end
end

end
