module NativePolyhedra

using LinearAlgebra
using SparseArrays
using JuMP
import Clarabel
import MathOptInterface as MOI

import Base: +, -, *, /, ==, hash, show, zero, iszero, convert, promote_rule, Float64, BigFloat, isless, ^, isempty, in, intersect, issubset, float, abs, abs2, real, conj, <, <=, >, >=

export ExactLogExpr, exact_log10, exact_log10_ratio
export Polyhedron, HRep, MixedMatHRep, hrep, polyhedron
export HalfSpace, HyperPlane, intersect, eliminate, detecthlinearity!, removehredundancy!
export dim, fulldim, hashyperplanes, hyperplanes, allhalfspaces, issubset
export feasible_point, interior_point

struct ExactLogExpr <: Real
    constant::Rational{Int}
    coeffs::Dict{Int,Rational{Int}}

    function ExactLogExpr(
        constant::Rational{Int}=0//1,
        coeffs::AbstractDict{<:Integer,<:Rational}=Dict{Int,Rational{Int}}(),
    )
        cleaned = Dict{Int,Rational{Int}}()
        for (p, c) in coeffs
            ci = Int(numerator(c)) // Int(denominator(c))
            iszero(ci) && continue
            cleaned[Int(p)] = get(cleaned, Int(p), 0//1) + ci
            iszero(cleaned[Int(p)]) && delete!(cleaned, Int(p))
        end
        return new(constant, cleaned)
    end
end

ExactLogExpr(x::Integer) = ExactLogExpr(Int(x)//1)
ExactLogExpr(x::Rational{<:Integer}) = ExactLogExpr(Int(numerator(x)) // Int(denominator(x)))

function _factor_positive_integer(n::Int)
    n > 0 || throw(ArgumentError("Only positive integers can be factorized, got $n."))
    out = Dict{Int,Int}()
    m = n
    p = 2
    while p * p <= m
        while m % p == 0
            out[p] = get(out, p, 0) + 1
            m = div(m, p)
        end
        p = p == 2 ? 3 : p + 2
    end
    m > 1 && (out[m] = get(out, m, 0) + 1)
    return out
end

function exact_log10(n::Integer)
    n == 0 && throw(ArgumentError("log10(0) is undefined."))
    n < 0 && throw(ArgumentError("log10 is only supported for positive integers."))
    n == 1 && return zero(ExactLogExpr)
    coeffs = Dict{Int,Rational{Int}}()
    for (p, e) in _factor_positive_integer(Int(n))
        coeffs[p] = e // 1
    end
    return ExactLogExpr(0//1, coeffs)
end

function exact_log10_ratio(num::Integer, den::Integer=1)
    num == 0 && throw(ArgumentError("log10(0) is undefined."))
    den == 0 && throw(ArgumentError("Division by zero in log10(num/den)."))
    sign(num) == sign(den) || throw(ArgumentError("log10 is only supported for positive rational ratios."))
    num = abs(Int(num))
    den = abs(Int(den))
    num == den && return zero(ExactLogExpr)
    return exact_log10(num) - exact_log10(den)
end

zero(::Type{ExactLogExpr}) = ExactLogExpr()
zero(::ExactLogExpr) = zero(ExactLogExpr)
iszero(x::ExactLogExpr) = iszero(x.constant) && isempty(x.coeffs)

function +(a::ExactLogExpr, b::ExactLogExpr)
    coeffs = Dict{Int,Rational{Int}}(a.coeffs)
    for (p, c) in b.coeffs
        coeffs[p] = get(coeffs, p, 0//1) + c
        iszero(coeffs[p]) && delete!(coeffs, p)
    end
    return ExactLogExpr(a.constant + b.constant, coeffs)
end
-(a::ExactLogExpr) = ExactLogExpr(-a.constant, Dict(p => -c for (p, c) in a.coeffs))
-(a::ExactLogExpr, b::ExactLogExpr) = a + (-b)
+(a::ExactLogExpr, b::Integer) = a + ExactLogExpr(b)
+(a::Integer, b::ExactLogExpr) = ExactLogExpr(a) + b
-(a::ExactLogExpr, b::Integer) = a - ExactLogExpr(b)
-(a::Integer, b::ExactLogExpr) = ExactLogExpr(a) - b
+(a::ExactLogExpr, b::Rational{<:Integer}) = a + ExactLogExpr(b)
+(a::Rational{<:Integer}, b::ExactLogExpr) = ExactLogExpr(a) + b
-(a::ExactLogExpr, b::Rational{<:Integer}) = a - ExactLogExpr(b)
-(a::Rational{<:Integer}, b::ExactLogExpr) = ExactLogExpr(a) - b

function *(c::Rational{<:Integer}, x::ExactLogExpr)
    coeffs = Dict{Int,Rational{Int}}()
    cc = Int(numerator(c)) // Int(denominator(c))
    for (p, v) in x.coeffs
        coeffs[p] = cc * v
    end
    return ExactLogExpr(cc * x.constant, coeffs)
end
*(c::Integer, x::ExactLogExpr) = (Int(c)//1) * x
*(x::ExactLogExpr, c::Rational{<:Integer}) = c * x
*(x::ExactLogExpr, c::Integer) = c * x
/(x::ExactLogExpr, c::Integer) = x * (1 // Int(c))
/(x::ExactLogExpr, c::Rational{<:Integer}) = x * inv(Int(numerator(c)) // Int(denominator(c)))

convert(::Type{ExactLogExpr}, x::Integer) = ExactLogExpr(x)
convert(::Type{ExactLogExpr}, x::Rational{<:Integer}) = ExactLogExpr(x)
promote_rule(::Type{ExactLogExpr}, ::Type{<:Integer}) = ExactLogExpr
promote_rule(::Type{ExactLogExpr}, ::Type{<:Rational}) = ExactLogExpr
promote_rule(::Type{ExactLogExpr}, ::Type{<:AbstractFloat}) = Float64

==(a::ExactLogExpr, b::ExactLogExpr) = a.constant == b.constant && a.coeffs == b.coeffs
==(a::ExactLogExpr, b::Integer) = a == ExactLogExpr(b)
==(a::Integer, b::ExactLogExpr) = ExactLogExpr(a) == b
hash(x::ExactLogExpr, h::UInt) = hash((x.constant, sort!(collect(x.coeffs); by=first)), h)

function Float64(x::ExactLogExpr)
    val = float(x.constant)
    for (p, c) in x.coeffs
        val += Float64(c) * log10(Float64(p))
    end
    return val
end

function BigFloat(x::ExactLogExpr)
    val = BigFloat(numerator(x.constant)) / BigFloat(denominator(x.constant))
    for (p, c) in x.coeffs
        val += (BigFloat(numerator(c)) / BigFloat(denominator(c))) * log10(BigFloat(p))
    end
    return val
end

float(x::ExactLogExpr) = Float64(x)
abs(x::ExactLogExpr) = abs(Float64(x))
abs2(x::ExactLogExpr) = abs2(Float64(x))
real(x::ExactLogExpr) = x
conj(x::ExactLogExpr) = x

isless(a::ExactLogExpr, b::ExactLogExpr) = Float64(a) < Float64(b)
isless(a::ExactLogExpr, b::Real) = Float64(a) < Float64(b)
isless(a::Real, b::ExactLogExpr) = Float64(a) < Float64(b)
<(a::ExactLogExpr, b::ExactLogExpr) = Float64(a) < Float64(b)
<(a::ExactLogExpr, b::Real) = Float64(a) < Float64(b)
<(a::Real, b::ExactLogExpr) = Float64(a) < Float64(b)
<=(a::ExactLogExpr, b::ExactLogExpr) = Float64(a) <= Float64(b)
<=(a::ExactLogExpr, b::Real) = Float64(a) <= Float64(b)
<=(a::Real, b::ExactLogExpr) = Float64(a) <= Float64(b)
>(a::ExactLogExpr, b::ExactLogExpr) = Float64(a) > Float64(b)
>(a::ExactLogExpr, b::Real) = Float64(a) > Float64(b)
>(a::Real, b::ExactLogExpr) = Float64(a) > Float64(b)
>=(a::ExactLogExpr, b::ExactLogExpr) = Float64(a) >= Float64(b)
>=(a::ExactLogExpr, b::Real) = Float64(a) >= Float64(b)
>=(a::Real, b::ExactLogExpr) = Float64(a) >= Float64(b)
^(x::Number, y::ExactLogExpr) = x ^ Float64(y)

function show(io::IO, x::ExactLogExpr)
    if iszero(x)
        print(io, "0")
        return
    end

    parts = String[]
    !iszero(x.constant) && push!(parts, string(x.constant))
    for (p, c) in sort!(collect(x.coeffs); by=first)
        term = c == 1//1 ? "log10($p)" : c == -1//1 ? "-log10($p)" : "$(c)*log10($p)"
        push!(parts, term)
    end

    out = first(parts)
    for part in Iterators.drop(parts, 1)
        if startswith(part, "-")
            out *= " - " * part[2:end]
        else
            out *= " + " * part
        end
    end
    print(io, out)
end

struct HalfSpace{TA<:Real,TB<:Real}
    a::Vector{TA}
    β::TB
end

struct HyperPlane{TA<:Real,TB<:Real}
    a::Vector{TA}
    β::TB
end

struct HRep{TA<:Real,TB<:Real}
    A::SparseMatrixCSC{TA,Int}
    b::Vector{TB}
    linset::BitSet
end

const MixedMatHRep = HRep

mutable struct Polyhedron{TA<:Real,TB<:Real}
    A::SparseMatrixCSC{TA,Int}
    b::Vector{TB}
    linset::BitSet
    empty::Bool
end

function hrep(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real}, linset::BitSet=BitSet())
    size(A, 1) == length(b) || throw(DimensionMismatch("size(A,1) must match length(b)."))
    TA = promote_type(eltype(A), Int)
    TB = eltype(b)
    return HRep(sparse(TA.(A)), TB.(collect(b)), copy(linset))
end

hrep(poly::Polyhedron) = HRep(copy(poly.A), copy(poly.b), copy(poly.linset))

function polyhedron(rep::HRep, _backend=nothing)
    poly = Polyhedron(copy(rep.A), copy(rep.b), copy(rep.linset), false)
    removehredundancy!(poly)
    return poly
end

fulldim(poly::Polyhedron) = size(poly.A, 2)

function _row_sparse(A::SparseMatrixCSC, i::Int)
    row = vec(Array(A[i:i, :]))
    idxs = findall(x -> !iszero(x), row)
    return sparsevec(idxs, row[idxs], length(row))
end

function _row_entries(A::SparseMatrixCSC, i::Int)
    row = A[i, :]
    I, V = findnz(row)
    return Int.(I), collect(V)
end

function _is_rational_row(vals)
    return all(v -> v isa Rational || v isa Integer, vals)
end

function _normalize_signed_row(vals::AbstractVector, β)
    if isempty(vals)
        return Any[], β
    end
    if _is_rational_row(vals)
        rats = Rational{Int}[Int(numerator(v)) // Int(denominator(v)) for v in vals]
        lcm_den = foldl(lcm, (denominator(v) for v in rats); init=1)
        ints = [Int(numerator(v * lcm_den)) for v in rats]
        g = foldl(gcd, abs.(ints); init=0)
        g == 0 && (g = 1)
        scale = (lcm_den // g)
        ints = Int.(ints ./ g)
        return ints, scale * β
    else
        nz = findfirst(!iszero, vals)
        nz === nothing && return Float64[], Float64(β)
        scale = maximum(abs, Float64.(vals))
        norm_vals = round.(Float64.(vals) ./ scale; digits=12)
        norm_β = round(Float64(β) / scale; digits=12)
        return collect(norm_vals), norm_β
    end
end

function _row_signature(A::SparseMatrixCSC, i::Int)
    I, V = _row_entries(A, i)
    vals, β = _normalize_signed_row(V, nothing)
    return I, vals
end

function _signed_signature(A::SparseMatrixCSC, b::AbstractVector, i::Int)
    I, V = _row_entries(A, i)
    vals, β = _normalize_signed_row(V, b[i])
    return (Tuple(I), Tuple(vals), β)
end

function _negate_signature(sig)
    I, vals, β = sig
    vals_neg = map(v -> -v, vals)
    return (I, Tuple(vals_neg), -β)
end

function _unsigned_signature(sig)
    I, vals, β = sig
    if isempty(vals)
        return sig
    end
    first_nz = findfirst(v -> !iszero(v), vals)
    first_nz === nothing && return sig
    if vals[first_nz] isa Real && vals[first_nz] < 0
        return _negate_signature(sig)
    end
    return sig
end

function _rebuild_polyhedron(
    rows::AbstractVector{<:SparseVector},
    bs::AbstractVector,
    linrows::AbstractVector{Bool},
    nvars::Int,
) 
    TA = isempty(rows) ? Int : foldl(promote_type, map(eltype, rows); init=Int)
    TB = isempty(bs) ? Int : foldl(promote_type, map(typeof, bs); init=Int)
    nnz_total = isempty(rows) ? 0 : sum(nnz, rows)
    I = Vector{Int}(undef, nnz_total)
    J = Vector{Int}(undef, nnz_total)
    V = Vector{TA}(undef, nnz_total)
    ptr = 1
    for (i, row) in enumerate(rows)
        idxs, vals = findnz(row)
        for k in eachindex(idxs)
            I[ptr] = i
            J[ptr] = idxs[k]
            V[ptr] = vals[k]
            ptr += 1
        end
    end
    A = sparse(I, J, V, length(rows), nvars)
    linset = BitSet(findall(identity, linrows))
    return Polyhedron(A, TB.(collect(bs)), linset, false)
end

function detecthlinearity!(poly::Polyhedron)
    poly.empty && return poly
    nrows = size(poly.A, 1)
    nvars = fulldim(poly)
    paired = falses(nrows)
    eq_rows = SparseVector[]
    eq_bs = eltype(poly.b)[]
    ineq_rows = SparseVector[]
    ineq_bs = eltype(poly.b)[]

    eq_seen = Dict{Any,Int}()
    signed_seen = Dict{Any,Int}()

    for i in 1:nrows
        sig = _signed_signature(poly.A, poly.b, i)
        if i in poly.linset
            usig = _unsigned_signature(sig)
            haskey(eq_seen, usig) && continue
            eq_seen[usig] = i
            push!(eq_rows, poly.A[i, :])
            push!(eq_bs, poly.b[i])
            continue
        end

        negsig = _negate_signature(sig)
        if haskey(signed_seen, negsig)
            j = signed_seen[negsig]
            paired[j] = true
            paired[i] = true
            usig = _unsigned_signature(sig)
            if !haskey(eq_seen, usig)
                eq_seen[usig] = i
                push!(eq_rows, poly.A[i, :])
                push!(eq_bs, poly.b[i])
            end
        else
            signed_seen[sig] = i
        end
    end

    for i in 1:nrows
        (i in poly.linset || paired[i]) && continue
        push!(ineq_rows, poly.A[i, :])
        push!(ineq_bs, poly.b[i])
    end

    rows = vcat(eq_rows, ineq_rows)
    bs = vcat(eq_bs, ineq_bs)
    linrows = vcat(fill(true, length(eq_rows)), fill(false, length(ineq_rows)))
    rebuilt = _rebuild_polyhedron(rows, bs, linrows, nvars)
    poly.A = rebuilt.A
    poly.b = rebuilt.b
    poly.linset = rebuilt.linset
    return poly
end

function _violates_zero_row(is_eq::Bool, β)
    rhs = Float64(β)
    return is_eq ? abs(rhs) > 1e-9 : rhs < -1e-9
end

function removehredundancy!(poly::Polyhedron)
    poly.empty && return poly
    detecthlinearity!(poly)

    nrows = size(poly.A, 1)
    nvars = fulldim(poly)
    eq_rows = SparseVector[]
    eq_bs = eltype(poly.b)[]
    ineq_rows = SparseVector[]
    ineq_bs = eltype(poly.b)[]
    eq_seen = Set{Any}()
    ineq_seen = Set{Any}()

    for i in 1:nrows
        row = poly.A[i, :]
        idxs, vals = findnz(row)
        is_eq = i in poly.linset
        if isempty(idxs)
            if _violates_zero_row(is_eq, poly.b[i])
                poly.empty = true
                return poly
            end
            continue
        end

        sig = _signed_signature(poly.A, poly.b, i)
        key = is_eq ? _unsigned_signature(sig) : sig
        target = is_eq ? eq_seen : ineq_seen
        key in target && continue
        push!(target, key)
        if is_eq
            push!(eq_rows, row)
            push!(eq_bs, poly.b[i])
        else
            push!(ineq_rows, row)
            push!(ineq_bs, poly.b[i])
        end
    end

    rows = vcat(eq_rows, ineq_rows)
    bs = vcat(eq_bs, ineq_bs)
    linrows = vcat(fill(true, length(eq_rows)), fill(false, length(ineq_rows)))
    rebuilt = _rebuild_polyhedron(rows, bs, linrows, nvars)
    poly.A = rebuilt.A
    poly.b = rebuilt.b
    poly.linset = rebuilt.linset
    poly.empty = false
    return poly
end

function _constraint_vectors(poly::Polyhedron)
    rows = Vector{Vector{Float64}}(undef, size(poly.A, 1))
    for i in 1:size(poly.A, 1)
        rows[i] = vec(Float64.(Array(poly.A[i:i, :])))
    end
    return rows, Float64.(poly.b)
end

function _new_optimizer()
    model = JuMP.Model(Clarabel.Optimizer)
    set_silent(model)
    return model
end

function feasible_point(poly::Polyhedron; tol::Float64=1e-8)
    poly.empty && return nothing
    n = fulldim(poly)
    rows, rhs = _constraint_vectors(poly)
    model = _new_optimizer()
    @variable(model, x[1:n])
    @variable(model, t >= 0)
    @objective(model, Min, t)

    for i in 1:length(rows)
        expr = sum(rows[i][j] * x[j] for j in 1:n)
        if i in poly.linset
            @constraint(model, expr <= rhs[i] + t)
            @constraint(model, expr >= rhs[i] - t)
        else
            @constraint(model, expr <= rhs[i] + t)
        end
    end

    optimize!(model)
    term = termination_status(model)
    if !(term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL))
        return nothing
    end
    value(t) <= tol || return nothing
    return value.(x)
end

function interior_point(poly::Polyhedron; tol::Float64=1e-8)
    pt = feasible_point(poly; tol=tol)
    isnothing(pt) && return nothing

    n = fulldim(poly)
    rows, rhs = _constraint_vectors(poly)
    model = _new_optimizer()
    @variable(model, x[1:n])
    @variable(model, t)
    @objective(model, Max, t)
    @constraint(model, t >= -1.0)

    for i in 1:length(rows)
        expr = sum(rows[i][j] * x[j] for j in 1:n)
        if i in poly.linset
            @constraint(model, expr == rhs[i])
        else
            @constraint(model, expr <= rhs[i] - t)
        end
    end

    for j in 1:n
        set_start_value(x[j], pt[j])
    end
    set_start_value(t, 0.0)

    optimize!(model)
    term = termination_status(model)
    if term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL)
        return value.(x)
    end
    return pt
end

function isempty(poly::Polyhedron)
    poly.empty && return true
    pt = feasible_point(poly)
    if isnothing(pt)
        poly.empty = true
        return true
    end
    return false
end

dim(poly::Polyhedron) = isempty(poly) ? -1 : fulldim(poly) - length(poly.linset)
hashyperplanes(poly::Polyhedron) = !isempty(poly.linset)

function hyperplanes(poly::Polyhedron)
    out = HyperPlane[]
    for i in sort!(collect(poly.linset))
        push!(out, HyperPlane(vec(Array(poly.A[i:i, :])), poly.b[i]))
    end
    return out
end

function allhalfspaces(rep::Union{HRep,Polyhedron})
    out = HalfSpace[]
    for i in 1:size(rep.A, 1)
        i in rep.linset && continue
        push!(out, HalfSpace(vec(Array(rep.A[i:i, :])), rep.b[i]))
    end
    return out
end

function _append_constraint!(
    rows::Vector,
    bs::Vector,
    linrows::Vector{Bool},
    c::HalfSpace,
)
    idxs = findall(x -> !iszero(x), c.a)
    push!(rows, sparsevec(idxs, c.a[idxs], length(c.a)))
    push!(bs, c.β)
    push!(linrows, false)
    return nothing
end

function _append_constraint!(
    rows::Vector,
    bs::Vector,
    linrows::Vector{Bool},
    c::HyperPlane,
)
    idxs = findall(x -> !iszero(x), c.a)
    push!(rows, sparsevec(idxs, c.a[idxs], length(c.a)))
    push!(bs, c.β)
    push!(linrows, true)
    return nothing
end

function intersect(poly::Polyhedron)
    return poly
end

function intersect(poly::Polyhedron, others...)
    nvars = fulldim(poly)
    rows = SparseVector[]
    bs = Any[]
    linrows = Bool[]

    for i in 1:size(poly.A, 1)
        push!(rows, poly.A[i, :])
        push!(bs, poly.b[i])
        push!(linrows, i in poly.linset)
    end

    for other in others
        if other isa Polyhedron
            fulldim(other) == nvars || throw(DimensionMismatch("Polyhedra must live in the same dimension."))
            for i in 1:size(other.A, 1)
                push!(rows, other.A[i, :])
                push!(bs, other.b[i])
                push!(linrows, i in other.linset)
            end
        elseif other isa Union{HalfSpace,HyperPlane}
            length(other.a) == nvars || throw(DimensionMismatch("Constraint dimension mismatch."))
            _append_constraint!(rows, bs, linrows, other)
        else
            throw(ArgumentError("Unsupported constraint type $(typeof(other))."))
        end
    end

    Atype = foldl(promote_type, map(eltype, rows); init=Int)
    Btype = foldl(promote_type, map(typeof, bs); init=Int)
    rows_typed = SparseVector{Atype,Int}[SparseVector{Atype,Int}(r) for r in rows]
    bs_typed = Btype[b for b in bs]
    poly_new = _rebuild_polyhedron(rows_typed, bs_typed, linrows, nvars)
    removehredundancy!(poly_new)
    return poly_new
end

function _scale_bound(c, β)
    return c * β
end

function _scaled_row(row::SparseVector{T,Int}, c) where {T<:Real}
    idxs, vals = findnz(row)
    new_vals = similar(vals, promote_type(T, typeof(c)))
    for i in eachindex(vals)
        new_vals[i] = vals[i] * c
    end
    return sparsevec(idxs, new_vals, length(row))
end

function _expand_rows(poly::Polyhedron)
    rows = SparseVector[]
    bs = Any[]
    for i in 1:size(poly.A, 1)
        row = poly.A[i, :]
        if i in poly.linset
            push!(rows, row)
            push!(bs, poly.b[i])
            push!(rows, -row)
            push!(bs, -poly.b[i])
        else
            push!(rows, row)
            push!(bs, poly.b[i])
        end
    end
    return rows, bs
end

function _drop_axis(row::SparseVector, axis::Int, nvars_new::Int)
    idxs, vals = findnz(row)
    kept_idxs = Int[]
    kept_vals_any = Any[]
    for k in eachindex(idxs)
        idxs[k] == axis && continue
        push!(kept_idxs, idxs[k] > axis ? idxs[k] - 1 : idxs[k])
        push!(kept_vals_any, vals[k])
    end
    T = isempty(kept_vals_any) ? Int : foldl(promote_type, map(typeof, kept_vals_any); init=Int)
    kept_vals = T[v for v in kept_vals_any]
    return sparsevec(kept_idxs, kept_vals, nvars_new)
end

function _eliminate_one(poly::Polyhedron, axis::Int)
    axis in 1:fulldim(poly) || throw(ArgumentError("Invalid elimination axis $axis."))
    isempty(poly) && return polyhedron(hrep(spzeros(Int, 0, fulldim(poly) - 1), Int[]))

    rows, bs = _expand_rows(poly)
    pos_rows = Int[]
    neg_rows = Int[]
    zero_rows = Int[]

    coeffs_axis = Any[]
    for i in eachindex(rows)
        coeff = rows[i][axis]
        push!(coeffs_axis, coeff)
        if Float64(coeff) > 1e-12
            push!(pos_rows, i)
        elseif Float64(coeff) < -1e-12
            push!(neg_rows, i)
        else
            push!(zero_rows, i)
        end
    end

    out_rows = SparseVector[]
    out_bs = Any[]

    for i in zero_rows
        push!(out_rows, _drop_axis(rows[i], axis, fulldim(poly) - 1))
        push!(out_bs, bs[i])
    end

    for ip in pos_rows, ineg in neg_rows
        ap = coeffs_axis[ip]
        an = coeffs_axis[ineg]
        row = _scaled_row(rows[ip], -an) + _scaled_row(rows[ineg], ap)
        push!(out_rows, _drop_axis(row, axis, fulldim(poly) - 1))
        push!(out_bs, _scale_bound(-an, bs[ip]) + _scale_bound(ap, bs[ineg]))
    end

    Atype = isempty(out_rows) ? Int : foldl(promote_type, map(eltype, out_rows); init=Int)
    Btype = isempty(out_bs) ? Int : foldl(promote_type, map(typeof, out_bs); init=Int)
    rows_typed = SparseVector{Atype,Int}[SparseVector{Atype,Int}(r) for r in out_rows]
    bs_typed = Btype[b for b in out_bs]
    poly_new = _rebuild_polyhedron(rows_typed, bs_typed, falses(length(rows_typed)), fulldim(poly) - 1)
    detecthlinearity!(poly_new)
    removehredundancy!(poly_new)
    return poly_new
end

function eliminate(poly::Polyhedron, axis::Integer)
    return _eliminate_one(poly, Int(axis))
end

function eliminate(poly::Polyhedron, axes::BitSet)
    out = poly
    for axis in sort!(collect(axes); rev=true)
        out = _eliminate_one(out, axis)
    end
    return out
end

function in(x::AbstractVector{<:Real}, poly::Polyhedron)
    length(x) == fulldim(poly) || return false
    for i in 1:size(poly.A, 1)
        lhs = dot(vec(Float64.(Array(poly.A[i:i, :]))), Float64.(x))
        rhs = Float64(poly.b[i])
        if i in poly.linset
            abs(lhs - rhs) <= 1e-7 || return false
        else
            lhs <= rhs + 1e-7 || return false
        end
    end
    return true
end

function _optimize_linear(poly::Polyhedron, a::AbstractVector{<:Real}, β::Real)
    isempty(poly) && return :empty, -Inf
    n = fulldim(poly)
    rows, rhs = _constraint_vectors(poly)
    model = _new_optimizer()
    @variable(model, x[1:n])
    @objective(model, Max, sum(Float64(a[j]) * x[j] for j in 1:n) - Float64(β))

    for i in 1:length(rows)
        expr = sum(rows[i][j] * x[j] for j in 1:n)
        if i in poly.linset
            @constraint(model, expr == rhs[i])
        else
            @constraint(model, expr <= rhs[i])
        end
    end

    optimize!(model)
    term = termination_status(model)
    if term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL)
        return :ok, objective_value(model)
    elseif term in (MOI.DUAL_INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
        return :unbounded, Inf
    else
        return :failed, Inf
    end
end

function issubset(poly::Polyhedron, h::HalfSpace; tol::Float64=1e-7)
    status, val = _optimize_linear(poly, h.a, h.β)
    return status == :empty || (status == :ok && val <= tol)
end

function issubset(poly::Polyhedron, h::HyperPlane; tol::Float64=1e-7)
    return issubset(poly, HalfSpace(h.a, h.β); tol=tol) &&
           issubset(poly, HalfSpace(-h.a, -h.β); tol=tol)
end

end
