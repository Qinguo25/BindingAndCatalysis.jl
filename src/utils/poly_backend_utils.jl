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

@inline _condition_scalar_iszero(x, atol::Real) =
    x isa AbstractFloat ? abs(x) <= atol : iszero(x)

@inline _condition_scalar_isnonnegative(x, atol::Real) =
    x isa AbstractFloat ? x >= -atol : x >= zero(x)

@inline _condition_coefficient_is_exact(x) = x isa Integer || x isa Rational
@inline _condition_bias_is_exact(x) =
    x isa Integer || x isa Rational || x isa ExactLogExpr

@inline _as_exact_fraction(x::Integer) = x // one(x)
@inline _as_exact_fraction(x::Rational) = x

function _canonical_empty_condition(C, C0; ncols::Integer=size(C, 2))
    TC = eltype(C)
    TC0 = eltype(C0)
    impossible_C = spzeros(TC, 1, Int(ncols))
    impossible_C0 = TC0[convert(TC0, -1)]
    return impossible_C, impossible_C0, 0
end

function _independent_compatible_equalities(C, C0; atol::Real=1.0e-10)
    n_rows = size(C, 1)
    n_rows == 0 && return C, C0, true

    exact_data = all(_condition_coefficient_is_exact, C) &&
                 all(_condition_bias_is_exact, C0)
    if exact_data
        C_exact = [
            _as_exact_fraction(C[i, j]) for i in axes(C, 1), j in axes(C, 2)
        ]
        basis_rows = Vector{Vector{eltype(C_exact)}}()
        basis_offsets = Any[]
        pivot_columns = Int[]
        kept = Int[]

        for row in axes(C_exact, 1)
            candidate = collect(@view C_exact[row, :])
            offset = C0[row]
            for basis_idx in eachindex(basis_rows)
                pivot = pivot_columns[basis_idx]
                factor = candidate[pivot]
                iszero(factor) && continue
                candidate .-= factor .* basis_rows[basis_idx]
                offset -= factor * basis_offsets[basis_idx]
            end

            pivot = findfirst(x -> !iszero(x), candidate)
            if isnothing(pivot)
                iszero(offset) || return C[1:0, :], C0[1:0], false
                continue
            end

            scale = candidate[pivot]
            candidate ./= scale
            offset /= scale
            push!(basis_rows, candidate)
            push!(basis_offsets, offset)
            push!(pivot_columns, pivot)
            push!(kept, row)
        end
        return C[kept, :], C0[kept], true
    end

    C_float = Matrix{Float64}(C)
    C0_float = Float64.(C0)
    rank_C = rank(C_float; atol=Float64(atol), rtol=0)
    rank_augmented = rank(hcat(C_float, C0_float); atol=Float64(atol), rtol=0)
    rank_augmented == rank_C || return C[1:0, :], C0[1:0], false

    kept = Int[]
    current_rank = 0
    for row in axes(C, 1)
        candidate = @view C_float[vcat(kept, row), :]
        candidate_rank = rank(candidate; atol=Float64(atol), rtol=0)
        if candidate_rank > current_rank
            push!(kept, row)
            current_rank = candidate_rank
        end
    end
    return C[kept, :], C0[kept], true
end

"""
    _stack_conditions(parts...; atol=1e-10)

Combine `(C, C0, equality_count)` condition triples while preserving the
package contract that all independent equality rows precede all inequalities.
An incompatible equality system is returned as a canonical empty condition.
"""
function _stack_conditions(parts...; atol::Real=1.0e-10)
    isempty(parts) && throw(ArgumentError("at least one condition block is required."))

    ambient_dim = size(first(parts)[1], 2)
    all(size(part[1], 2) == ambient_dim for part in parts) ||
        throw(DimensionMismatch("all condition blocks must use the same ambient dimension."))

    equality_C = [part[1][1:Int(part[3]), :] for part in parts]
    equality_C0 = [part[2][1:Int(part[3])] for part in parts]
    inequality_C = [part[1][(Int(part[3]) + 1):end, :] for part in parts]
    inequality_C0 = [part[2][(Int(part[3]) + 1):end] for part in parts]

    C_eq = reduce(vcat, equality_C)
    C0_eq = reduce(vcat, equality_C0)
    C_ineq = reduce(vcat, inequality_C)
    C0_ineq = reduce(vcat, inequality_C0)
    C_eq, C0_eq, compatible = _independent_compatible_equalities(
        C_eq, C0_eq; atol=atol
    )

    if !compatible
        C_all = vcat(C_eq, C_ineq)
        C0_all = vcat(C0_eq, C0_ineq)
        return _canonical_empty_condition(C_all, C0_all; ncols=ambient_dim)
    end

    return vcat(C_eq, C_ineq), vcat(C0_eq, C0_ineq), size(C_eq, 1)
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
    feasible = !isempty(p)
    dim_val = feasible ? dim(p) : -1
    return (;
        poly=p,
        feasible=feasible,
        dim=dim_val,
        ambient_dim=resolved_ambient_dim,
        full_dim=feasible && dim_val == resolved_ambient_dim,
    )
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
