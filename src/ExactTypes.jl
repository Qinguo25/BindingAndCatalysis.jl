module ExactTypes

import Base:
    +,
    -,
    *,
    /,
    ==,
    hash,
    show,
    zero,
    iszero,
    convert,
    promote_rule,
    Float64,
    BigFloat,
    isless,
    ^,
    float,
    abs,
    abs2,
    real,
    conj,
    <,
    <=,
    >,
    >=

export ExactLogExpr, exact_log10, exact_log10_ratio

struct ExactLogExpr <: Real
    constant::Rational{Int}
    coeffs::Dict{Int, Rational{Int}}

    function ExactLogExpr(
        constant::Rational{Int}=0//1,
        coeffs::AbstractDict{<:Integer, <:Rational}=Dict{Int, Rational{Int}}(),
    )
        cleaned = Dict{Int, Rational{Int}}()
        for (p, c) in coeffs
            ci = Int(numerator(c))//Int(denominator(c))
            iszero(ci) && continue
            cleaned[Int(p)] = get(cleaned, Int(p), 0//1) + ci
            iszero(cleaned[Int(p)]) && delete!(cleaned, Int(p))
        end
        return new(constant, cleaned)
    end
end

ExactLogExpr(x::Integer) = ExactLogExpr(Int(x)//1)
ExactLogExpr(x::Rational{<:Integer}) = ExactLogExpr(Int(numerator(x))//Int(denominator(x)))

function _factor_positive_integer(n::Int)
    n > 0 || throw(ArgumentError("Only positive integers can be factorized, got $n."))
    out = Dict{Int, Int}()
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
    coeffs = Dict{Int, Rational{Int}}()
    for (p, e) in _factor_positive_integer(Int(n))
        coeffs[p] = e//1
    end
    return ExactLogExpr(0//1, coeffs)
end

function exact_log10_ratio(num::Integer, den::Integer=1)
    num == 0 && throw(ArgumentError("log10(0) is undefined."))
    den == 0 && throw(ArgumentError("Division by zero in log10(num/den)."))
    sign(num) == sign(den) ||
        throw(ArgumentError("log10 is only supported for positive rational ratios."))
    num = abs(Int(num))
    den = abs(Int(den))
    num == den && return zero(ExactLogExpr)
    return exact_log10(num) - exact_log10(den)
end

zero(::Type{ExactLogExpr}) = ExactLogExpr()
zero(::ExactLogExpr) = zero(ExactLogExpr)
iszero(x::ExactLogExpr) = iszero(x.constant) && isempty(x.coeffs)

function +(a::ExactLogExpr, b::ExactLogExpr)
    coeffs = Dict{Int, Rational{Int}}(a.coeffs)
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
    coeffs = Dict{Int, Rational{Int}}()
    cc = Int(numerator(c))//Int(denominator(c))
    for (p, v) in x.coeffs
        coeffs[p] = cc * v
    end
    return ExactLogExpr(cc * x.constant, coeffs)
end
*(c::Integer, x::ExactLogExpr) = (Int(c)//1) * x
*(x::ExactLogExpr, c::Rational{<:Integer}) = c * x
*(x::ExactLogExpr, c::Integer) = c * x
/(x::ExactLogExpr, c::Integer) = x * (1//Int(c))
/(x::ExactLogExpr, c::Rational{<:Integer}) = x * inv(Int(numerator(c))//Int(denominator(c)))

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
^(x::Number, y::ExactLogExpr) = x^Float64(y)

function show(io::IO, x::ExactLogExpr)
    if iszero(x)
        print(io, "0")
        return nothing
    end

    parts = String[]
    !iszero(x.constant) && push!(parts, string(x.constant))
    for (p, c) in sort!(collect(x.coeffs); by=first)
        term = if c == 1//1
            "log10($p)"
        elseif c == -1//1
            "-log10($p)"
        else
            "$(c)*log10($p)"
        end
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
    return print(io, out)
end

end
