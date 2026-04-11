module PolyBackend

using ..NativePolyhedra
using ..CddBridge
using ..ExactTypes: ExactLogExpr
using SparseArrays: sparse
import Polyhedra as PolyhedraExt

@inline backend_prefers_fastpath(is_exact::Bool) = CddBridge._can_use_cdd_fastpath(is_exact)
@inline backend_prefers_fastpath(poly::NativePolyhedra.Polyhedron) = backend_prefers_fastpath(any(h -> h.p.β isa ExactLogExpr, poly.halfspaces))

function backend_intersect_eliminate(
    poly1::NativePolyhedra.Polyhedron,
    poly2::NativePolyhedra.Polyhedron,
    delset::BitSet;
    canonicalize::Bool=false,
    prefer_fastpath::Bool=false,
)
    if prefer_fastpath
        return CddBridge.cdd_intersect_eliminate(poly1, poly2, delset; canonicalize=canonicalize)
    end
    p = NativePolyhedra.intersect(poly1, poly2; canonicalize=false)
    return backend_eliminate(p, delset; canonicalize=canonicalize, prefer_fastpath=false)
end

function backend_intersect_many(
    polys::AbstractVector{<:NativePolyhedra.Polyhedron};
    canonicalize::Bool=false,
    prefer_fastpath::Bool=false,
)
    if prefer_fastpath
        return CddBridge.cdd_intersect_many(polys; canonicalize=canonicalize)
    end
    return NativePolyhedra.intersect(polys...; canonicalize=canonicalize)
end

function backend_eliminate(
    poly::NativePolyhedra.Polyhedron,
    delset::BitSet;
    canonicalize::Bool=true,
    prefer_fastpath::Bool=false,
    method::Symbol=:auto,
)
    if prefer_fastpath
        return CddBridge.cdd_eliminate(poly, delset; canonicalize=canonicalize)
    end

    exact_fast = CddBridge.maybe_cddlog_eliminate(poly, delset; canonicalize=canonicalize, method=method)
    exact_fast === nothing || return exact_fast
    return NativePolyhedra.eliminate(poly, delset; canonicalize=canonicalize, method=method)
end

function backend_project_hrep(
    C::AbstractMatrix{<:Real},
    C0::AbstractVector,
    nullity::Integer,
    delset,
)
    if !any(x -> x isa ExactLogExpr, C0)
        return CddBridge.cdd_project_hrep(C, C0, nullity, delset)
    end

    try
        return CddBridge.cddlog_project_hrep(C, C0, nullity, delset)
    catch
        poly = CddBridge._polyhedron_from_C_C0_nullity(sparse(C), C0, nullity)
        poly_elim = backend_eliminate(poly, BitSet(delset); canonicalize=true, prefer_fastpath=false)
        return CddBridge._polyhedron_to_C_C0_nullity(poly_elim)
    end
end

function backend_prepare_fastpath(
    poly::NativePolyhedra.Polyhedron;
    prefer_fastpath::Bool=false,
)
    return prefer_fastpath ? CddBridge._native_to_cdd(poly) : poly
end

function backend_fast_eliminate(
    poly,
    delset::BitSet;
    prefer_fastpath::Bool=false,
)
    if prefer_fastpath
        return PolyhedraExt.eliminate(poly, sort!(collect(delset)))
    end
    return backend_eliminate(poly, delset; canonicalize=false, prefer_fastpath=false)
end

function backend_fast_intersect(poly1, poly2; prefer_fastpath::Bool=false)
    if prefer_fastpath
        return Base.intersect(poly1, poly2)
    end
    return NativePolyhedra.intersect(poly1, poly2; canonicalize=false)
end

function backend_from_fastpath(poly; canonicalize::Bool=false, prefer_fastpath::Bool=false)
    if prefer_fastpath
        return CddBridge._cdd_to_native(poly; canonicalize=canonicalize)
    end
    if canonicalize
        NativePolyhedra.removehredundancy!(poly; strong=false)
    end
    return poly
end

end
