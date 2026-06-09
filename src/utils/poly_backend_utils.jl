const POLY_BACK_END = CDDLib.Library(:float)

function _poly_normalize!(
    poly::Polyhedron; detect_linearities::Bool=true, remove_redundancy::Bool=true
)
    detect_linearities && detecthlinearity!(poly)
    remove_redundancy && removehredundancy!(poly)
    return poly
end

function _poly_normalized_copy(
    poly::Polyhedron; canonicalize::Bool=true, detect_linearities::Bool=true
)
    out = polyhedron(hrep(poly), POLY_BACK_END)
    return _poly_normalize!(
        out; detect_linearities=detect_linearities, remove_redundancy=canonicalize
    )
end

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
    canonicalize && _poly_normalize!(poly)
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

function _poly_eliminate(poly::Polyhedron, delset; canonicalize::Bool=false)::Polyhedron
    axes = BitSet(Int.(collect(delset)))
    out = isempty(axes) ? poly : eliminate(poly, axes)
    canonicalize && _poly_normalize!(out)
    return out
end

function _poly_intersect_many(
    polys::AbstractVector{<:Polyhedron}; canonicalize::Bool=false
)::Polyhedron
    isempty(polys) && throw(ArgumentError("Need at least one polyhedron."))
    poly = if length(polys) == 1
        polyhedron(hrep(first(polys)), POLY_BACK_END)
    else
        reduce(intersect, polys)
    end
    canonicalize && _poly_normalize!(poly)
    return poly
end

function _poly_intersect_eliminate(
    poly1::Polyhedron, poly2::Polyhedron, delset; canonicalize::Bool=false
)::Polyhedron
    poly = intersect(poly1, poly2)
    return _poly_eliminate(poly, delset; canonicalize=canonicalize)
end

# Seems this function is redundent to _calc_C_C0_nullily
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
    return _poly_normalize!(poly)
end

function _poly_dim_status(
    poly::Polyhedron;
    ambient_dim=nothing,
    canonicalize::Bool=true,
    detect_linearities::Bool=canonicalize,
)
    p = if canonicalize
        _poly_normalized_copy(
            poly; canonicalize=canonicalize, detect_linearities=detect_linearities
        )
    else
        detect_linearities && detecthlinearity!(poly)
        poly
    end

    resolved_ambient_dim = isnothing(ambient_dim) ? fulldim(p) : Int(ambient_dim)
    try
        feasible = !isempty(p)
        dim_val = feasible ? dim(p) : -1
        return (;
            poly=p,
            feasible=feasible,
            dim=dim_val,
            ambient_dim=resolved_ambient_dim,
            full_dim=feasible && dim_val == resolved_ambient_dim,
        )
    catch
        return (;
            poly=p,
            feasible=false,
            dim=typemin(Int),
            ambient_dim=resolved_ambient_dim,
            full_dim=false,
        )
    end
end

function _poly_is_full_dimensional(poly::Polyhedron; kwargs...)
    return _poly_dim_status(poly; kwargs...).full_dim
end

function _poly_intersection_status(
    poly1::Polyhedron,
    poly2::Polyhedron;
    ambient_dim=nothing,
    canonicalize::Bool=true,
    detect_linearities::Bool=true,
)
    ins = intersect(poly1, poly2)
    return _poly_dim_status(
        ins;
        ambient_dim=ambient_dim,
        canonicalize=canonicalize,
        detect_linearities=detect_linearities,
    )
end

function _poly_pullback_hrep(
    C,
    C0,
    nullity::Integer,
    basis::AbstractMatrix{<:Real},
    offset::AbstractVector{<:Real};
    inequality_C=nothing,
    inequality_C0=nothing,
)
    C_mat = Matrix{Float64}(C)
    C0_vec = Float64.(vec(C0))
    nlt = Int(nullity)

    eq_C = C_mat[1:nlt, :] * basis
    eq_C0 = C_mat[1:nlt, :] * offset + C0_vec[1:nlt]
    ineq_C = C_mat[(nlt + 1):end, :] * basis
    ineq_C0 = C_mat[(nlt + 1):end, :] * offset + C0_vec[(nlt + 1):end]

    if isnothing(inequality_C)
        return vcat(eq_C, ineq_C), vcat(eq_C0, ineq_C0), nlt
    end

    return vcat(eq_C, ineq_C, inequality_C), vcat(eq_C0, ineq_C0, inequality_C0), nlt
end
