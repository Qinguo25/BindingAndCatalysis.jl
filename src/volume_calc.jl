"""
    Volume

Estimated volume with an uncertainty encoded as variance.
"""
struct Volume
    mean::Float64
    var::Float64
end
"""
    fetch_mean_re(V::Volume) -> (Float64, Float64)

Return `(mean, relative_error)` for a `Volume`.
"""
@inline function fetch_mean_re(V::Volume)
    relerr = iszero(V.mean) ? (iszero(V.var) ? 0.0 : Inf) : (sqrt(V.var) / V.mean)
    return V.mean, relerr
end

function _volume_summary_string(V::Volume)
    mean, relerr = fetch_mean_re(V)
    relpct = isfinite(relerr) ? Printf.@sprintf("%.2f%%", relerr * 100) : string(relerr)
    return Printf.@sprintf(
        "Volume(Mean=%.3e, STD=%.3e, RelError=%s)", mean, sqrt(V.var), relpct,
    )
end

Base.display(V::Volume) = _volume_summary_string(V)
Base.show(io::IO, V::Volume) = print(io, _volume_summary_string(V))

Base.:+(v1::Volume, v2::Volume) = Volume(v1.mean + v2.mean, v1.var + v2.var)
Base.:-(v1::Volume, v2::Volume) = Volume(v1.mean - v2.mean, v1.var + v2.var)
Base.isless(a::Volume, b::Volume) = isless((a.mean, a.var), (b.mean, b.var))
Base.:(==)(a::Volume, b::Volume) = a.mean == b.mean && a.var == b.var
Base.hash(v::Volume, h::UInt) = hash((v.mean, v.var), h)
Base.zero(::Type{Volume}) = Volume(0.0, 0.0)
Base.zero(::Volume) = zero(Volume)
Base.iszero(v::Volume) = iszero(v.mean) && iszero(v.var)
Base.:*(c::Real, v::Volume) = Volume(c * v.mean, abs2(c) * v.var)
Base.:*(v::Volume, c::Real) = c * v
Base.:/(v::Volume, c::Real) = Volume(v.mean / c, v.var / abs2(c))
