const POLY_BACK_END = CDDLib.Library(:float)



function _build_polyhedron_from_C_C0(
    C::AbstractMatrix{<:Real},
    C0::AbstractVector,
    nullity::Integer=0;
    canonicalize::Bool=false,
)::Polyhedron
    A = Matrix(-Float64.(C))
    b = collect(Float64.(C0))
    rep = nullity == 0 ? hrep(A, b) : hrep(A, b, BitSet(1:nullity))
    poly = polyhedron(rep, POLY_BACK_END)
    canonicalize && removehredundancy!(poly)
    return poly
end

function _polyhedron_to_C_C0_nullity(poly::Polyhedron)
    detecthlinearity!(poly)
    rep = MixedMatHRep(hrep(poly))
    eq_rows = sort!(collect(rep.linset))
    ineq_rows = [i for i in axes(rep.A, 1) if !(i in rep.linset)]
    order = vcat(eq_rows, ineq_rows)
    C = sparse(-rep.A[order, :])
    C0 = collect(rep.b[order])
    return C, C0, length(eq_rows)
end

function _poly_eliminate(
    poly::Polyhedron,
    delset;
    canonicalize::Bool=false,
)::Polyhedron
    axes = BitSet(Int.(collect(delset)))
    out = isempty(axes) ? poly : eliminate(poly, axes)
    canonicalize && removehredundancy!(out)
    return out
end

function _poly_intersect_many(
    polys::AbstractVector{<:Polyhedron};
    canonicalize::Bool=false,
)::Polyhedron
    isempty(polys) && throw(ArgumentError("Need at least one polyhedron."))
    poly = if length(polys) == 1
        polyhedron(hrep(first(polys)), POLY_BACK_END)
    else
        reduce(intersect, polys)
    end
    canonicalize && removehredundancy!(poly)
    return poly
end

function _poly_intersect_eliminate(
    poly1::Polyhedron,
    poly2::Polyhedron,
    delset;
    canonicalize::Bool=false,
)::Polyhedron
    poly = intersect(poly1, poly2)
    return _poly_eliminate(poly, delset; canonicalize=canonicalize)
end

function _poly_project_hrep(
    C::AbstractMatrix{<:Real},
    C0::AbstractVector,
    nullity::Integer,
    delset;
    canonicalize::Bool=true,
)
    poly = _build_polyhedron_from_C_C0(C, C0, nullity; canonicalize=false)
    projected = _poly_eliminate(poly, delset; canonicalize=canonicalize)
    return _polyhedron_to_C_C0_nullity(projected)
end

function _clean_polyhedron!(poly::Polyhedron)
    removehredundancy!(poly)
    return poly
end
