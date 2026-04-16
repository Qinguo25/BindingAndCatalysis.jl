module PolyBackend

using ..NativePolyhedra
using ..CddBridge
using ..ExactTypes: ExactLogExpr
using Logging: @warn
using SparseArrays: sparse, spzeros

const _backend_warn_lock = ReentrantLock()
const _warned_missing_local_cdd = Ref(false)
const _warned_missing_local_cddlog = Ref(false)

function _warn_missing_local_backend!(which::Symbol)
    warned = which === :exact ? _warned_missing_local_cddlog : _warned_missing_local_cdd
    lock(_backend_warn_lock) do
        warned[] && return
        warned[] = true
        backend_name = which === :exact ? "cddlog" : "cdd"
        @warn "Local $backend_name backend is not available or disabled; falling back to NativePolyhedra. Run `Pkg.build()` after installing gcc/cc/clang to enable the local backend." build_script="deps/build.jl"
    end
    return nothing
end

@inline _poly_is_exact(poly::NativePolyhedra.Polyhedron) = any(h -> h.p.β isa ExactLogExpr, poly.halfspaces)

function _empty_hrep_result(C::AbstractMatrix, C0::AbstractVector, dim::Integer)
    coeffs = spzeros(eltype(sparse(C)), 1, dim)
    rhs = if CddBridge._is_exact_rhs(C0)
        ExactLogExpr[-1]
    elseif CddBridge._has_float_data(C, C0)
        Float64[-1.0]
    else
        Rational{Int}[-1//1]
    end
    return coeffs, rhs, 0
end

function backend_prefers_fastpath(is_exact::Bool)
    is_exact && return false
    CddBridge._cdd_available() && return true
    _warn_missing_local_backend!(:float)
    return false
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
    p = NativePolyhedra.intersect(poly1, poly2; canonicalize=false)
    return backend_eliminate(p, delset; canonicalize=canonicalize, prefer_fastpath=prefer_fastpath)
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
    if prefer_fastpath
        try
            return CddBridge.cdd_eliminate(poly, delset; canonicalize=canonicalize)
        catch
        end
    end

    _poly_is_exact(poly) && !CddBridge._cddlog_available() && _warn_missing_local_backend!(:exact)
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
    if CddBridge._is_exact_rhs(C0)
        !CddBridge._cddlog_available() && _warn_missing_local_backend!(:exact)
    else
        !CddBridge._cdd_available() && _warn_missing_local_backend!(:float)
    end
    try
        return CddBridge.cdd_project_hrep(C, C0, nullity, delset; canonicalize=true)
    catch
        poly = CddBridge._polyhedron_from_C_C0_nullity(sparse(C), C0, nullity)
        poly_elim = backend_eliminate(poly, BitSet(delset); canonicalize=true, prefer_fastpath=false)
        poly_elim.empty && return _empty_hrep_result(C, C0, NativePolyhedra.fulldim(poly_elim))
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
