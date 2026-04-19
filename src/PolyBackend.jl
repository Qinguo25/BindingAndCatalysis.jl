module PolyBackend

using ..NativePolyhedra
using ..CddBridge
using ..ExactTypes: ExactLogExpr

@inline _poly_is_exact(poly::NativePolyhedra.Polyhedron) = any(h -> h.p.β isa ExactLogExpr, poly.halfspaces)

function backend_prefers_fastpath(is_exact::Bool)
    is_exact && return false
    CddBridge._require_local_cdd!()
    return true
end

@inline backend_prefers_fastpath(poly::NativePolyhedra.Polyhedron) = backend_prefers_fastpath(any(h -> h.p.β isa ExactLogExpr, poly.halfspaces))

function backend_intersect_eliminate(
    poly1::NativePolyhedra.Polyhedron,
    poly2::NativePolyhedra.Polyhedron,
    delset::BitSet;
    canonicalize::Bool=false,
    prefer_fastpath::Bool=false,
)
    return CddBridge.cdd_intersect_eliminate(poly1, poly2, delset; canonicalize=canonicalize)
end

function backend_intersect_many(
    polys::AbstractVector{<:NativePolyhedra.Polyhedron};
    canonicalize::Bool=false,
    prefer_fastpath::Bool=false,
)
    return CddBridge.cdd_intersect_many(polys; canonicalize=canonicalize)
end

function backend_eliminate(
    poly::NativePolyhedra.Polyhedron,
    delset::BitSet;
    canonicalize::Bool=true,
    prefer_fastpath::Bool=false,
    method::Symbol=:auto,
)
    method === :auto || nothing
    return CddBridge.cdd_eliminate(poly, delset; canonicalize=canonicalize)
end

function backend_project_hrep(
    C::AbstractMatrix{<:Real},
    C0::AbstractVector,
    nullity::Integer,
    delset,
)
    return CddBridge.cdd_project_hrep(C, C0, nullity, delset; canonicalize=true)
end

function backend_prepare_fastpath(
    poly::NativePolyhedra.Polyhedron;
    prefer_fastpath::Bool=false,
)
    return poly
end

function backend_fast_eliminate(
    poly,
    delset::BitSet;
    prefer_fastpath::Bool=false,
)
    return backend_eliminate(poly, delset; canonicalize=false, prefer_fastpath=prefer_fastpath)
end

function backend_fast_intersect(poly1, poly2; prefer_fastpath::Bool=false)
    return CddBridge.cdd_intersect_many([poly1, poly2]; canonicalize=false)
end

function backend_from_fastpath(poly; canonicalize::Bool=false, prefer_fastpath::Bool=false)
    if canonicalize
        return CddBridge._cdd_canonicalize_poly(poly)
    end
    return poly
end

end
