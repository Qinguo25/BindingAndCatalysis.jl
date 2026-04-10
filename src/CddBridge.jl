module CddBridge

using ..NativePolyhedra
using CDDLib
using Polyhedra
using SparseArrays

const _CDD_FLOAT_LIB = CDDLib.Library(:float)

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

end
