module PolyBackend

using ..NativePolyhedra
using ..CddBridge
using ..ExactTypes: ExactLogExpr
using SparseArrays: sparse

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
    if prefer_fastpath
        try
            return CddBridge.cdd_intersect_eliminate(poly1, poly2, delset; canonicalize=canonicalize)
        catch
        end
    end
    poly = NativePolyhedra.intersect(poly1, poly2; canonicalize=false)
    return backend_eliminate(poly, delset; canonicalize=canonicalize, prefer_fastpath=prefer_fastpath)
end

function backend_intersect_many(
    polys::AbstractVector{<:NativePolyhedra.Polyhedron};
    canonicalize::Bool=false,
    prefer_fastpath::Bool=false,
)
    if prefer_fastpath
        try
            return CddBridge.cdd_intersect_many(polys; canonicalize=canonicalize)
        catch
        end
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
    method === :auto || nothing

    if prefer_fastpath
        try
            return CddBridge.cdd_eliminate(poly, delset; canonicalize=canonicalize)
        catch
        end
    end

    if _poly_is_exact(poly)
        try
            return CddBridge.cdd_eliminate(poly, delset; canonicalize=canonicalize)
        catch
        end
    end

    return NativePolyhedra.eliminate(poly, delset; canonicalize=canonicalize, method=method)
end

function backend_project_hrep(
    C::AbstractMatrix{<:Real},
    C0::AbstractVector,
    nullity::Integer,
    delset,
)
    try
        return CddBridge.cdd_project_hrep(C, C0, nullity, delset; canonicalize=true)
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
    return NativePolyhedra.intersect(poly1, poly2; canonicalize=false)
end

function backend_from_fastpath(poly; canonicalize::Bool=false, prefer_fastpath::Bool=false)
    if canonicalize
        NativePolyhedra.removehredundancy!(poly; strong=false)
    end
    return poly
end

end
