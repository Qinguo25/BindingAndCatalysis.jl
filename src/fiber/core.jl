export VariationSubspace,
    FiberChart,
    FiberProblem,
    AffineFiber,
    AbstractSliceType,
    OrderedRegimePath,
    ConditionalSliceType,
    FiberChamber,
    ChamberComplex,
    ambient_dimension,
    fiber_dimension,
    base_dimension,
    fiber_at,
    get_fiber_problem,
    get_slice_types,
    get_conditional_slice_types

"""
A full-column-rank basis for the parameter directions allowed to vary.
"""
struct VariationSubspace
    basis::Matrix{Float64}

    function VariationSubspace(basis::AbstractMatrix{<:Real})
        basis_float = Matrix{Float64}(basis)
        ambient_dim, variation_dim = size(basis_float)
        ambient_dim > 0 || throw(ArgumentError("ambient dimension must be positive."))
        variation_dim > 0 || throw(ArgumentError("variation dimension must be positive."))
        variation_dim <= ambient_dim || throw(
            ArgumentError(
                "variation dimension $(variation_dim) exceeds ambient dimension $(ambient_dim).",
            ),
        )
        rank(basis_float) == variation_dim ||
            throw(ArgumentError("variation basis must have full column rank."))
        return new(basis_float)
    end
end

"""
Construct a coordinate-aligned variation subspace from one or more axes.
"""
function VariationSubspace(ambient_dim::Integer, varying_indices::AbstractVector{<:Integer})
    m = Int(ambient_dim)
    m > 0 || throw(ArgumentError("ambient dimension must be positive."))
    indices = Int.(collect(varying_indices))
    isempty(indices) && throw(ArgumentError("at least one varying index is required."))
    length(unique(indices)) == length(indices) ||
        throw(ArgumentError("varying indices must be unique."))
    all(i -> 1 <= i <= m, indices) ||
        throw(ArgumentError("varying indices must lie in 1:$(m)."))

    basis = zeros(Float64, m, length(indices))
    for (column, index) in enumerate(indices)
        basis[index, column] = 1.0
    end
    return VariationSubspace(basis)
end

function VariationSubspace(ambient_dim::Integer, varying_index::Integer)
    return VariationSubspace(ambient_dim, [Int(varying_index)])
end

ambient_dimension(space::VariationSubspace) = size(space.basis, 1)
fiber_dimension(space::VariationSubspace) = size(space.basis, 2)
base_dimension(space::VariationSubspace) = ambient_dimension(space) - fiber_dimension(space)

"""
Coordinate data for the quotient `Q -> Q/U`.

`quotient_map` has kernel `U`, while `section` maps base coordinates back to a
chosen zero-fiber representative. The default constructor uses an orthonormal
complement of the variation basis.
"""
struct FiberChart
    variation::VariationSubspace
    quotient_map::Matrix{Float64}
    section::Matrix{Float64}

    function FiberChart(
        variation::VariationSubspace,
        quotient_map::AbstractMatrix{<:Real},
        section::AbstractMatrix{<:Real};
        atol::Real=1.0e-10,
    )
        quotient = Matrix{Float64}(quotient_map)
        section_float = Matrix{Float64}(section)
        m = ambient_dimension(variation)
        b = base_dimension(variation)
        size(quotient) == (b, m) ||
            throw(DimensionMismatch("quotient_map must have size ($(b), $(m))."))
        size(section_float) == (m, b) ||
            throw(DimensionMismatch("section must have size ($(m), $(b))."))
        isapprox(
            quotient * variation.basis,
            zeros(b, fiber_dimension(variation));
            atol=atol,
            rtol=0,
        ) ||
            throw(ArgumentError("the quotient map must annihilate the variation subspace."))
        isapprox(quotient * section_float, Matrix{Float64}(I, b, b); atol=atol, rtol=0) ||
            throw(ArgumentError("section must be a right inverse of the quotient map."))
        return new(variation, quotient, section_float)
    end
end

function FiberChart(variation::VariationSubspace)
    complement = nullspace(transpose(variation.basis))
    return FiberChart(variation, transpose(complement), complement)
end

function FiberChart(ambient_dim::Integer, varying_indices::AbstractVector{<:Integer})
    variation = VariationSubspace(ambient_dim, varying_indices)
    fixed_indices = setdiff(1:ambient_dimension(variation), Int.(varying_indices))
    identity_map = Matrix{Float64}(
        I, ambient_dimension(variation), ambient_dimension(variation)
    )
    quotient_map = identity_map[fixed_indices, :]
    section = identity_map[:, fixed_indices]
    return FiberChart(variation, quotient_map, section)
end

function FiberChart(ambient_dim::Integer, varying_index::Integer)
    return FiberChart(ambient_dim, [Int(varying_index)])
end

ambient_dimension(chart::FiberChart) = ambient_dimension(chart.variation)
fiber_dimension(chart::FiberChart) = fiber_dimension(chart.variation)
base_dimension(chart::FiberChart) = base_dimension(chart.variation)

"""
A model together with its parameter chart and allowed variation fiber.
"""
struct FiberProblem{M}
    model::M
    chart::FiberChart
    parameter_chart::Symbol
end

function FiberProblem(model, chart::FiberChart; parameter_chart::Symbol=:qK)
    return FiberProblem(model, chart, parameter_chart)
end

function FiberProblem(
    model,
    variation::VariationSubspace;
    parameter_chart::Symbol=:qK,
    quotient_map=nothing,
    section=nothing,
)
    xor(isnothing(quotient_map), isnothing(section)) &&
        throw(ArgumentError("quotient_map and section must be provided together."))
    chart = if isnothing(quotient_map)
        FiberChart(variation)
    else
        FiberChart(variation, quotient_map, section)
    end
    return FiberProblem(model, chart, parameter_chart)
end

ambient_dimension(problem::FiberProblem) = ambient_dimension(problem.chart)
fiber_dimension(problem::FiberProblem) = fiber_dimension(problem.chart)
base_dimension(problem::FiberProblem) = base_dimension(problem.chart)

"""
One affine fiber selected by a point in quotient/base coordinates.
"""
struct AffineFiber
    problem::FiberProblem
    base_point::Vector{Float64}
    offset::Vector{Float64}
end

function fiber_at(problem::FiberProblem, base_point::AbstractVector{<:Real})
    base = Float64.(collect(base_point))
    length(base) == base_dimension(problem) || throw(
        DimensionMismatch(
            "base point has length $(length(base)); expected $(base_dimension(problem)).",
        ),
    )
    return AffineFiber(problem, base, problem.chart.section * base)
end

abstract type AbstractSliceType end

"""
The canonical slice-type label for a one-dimensional ordered regime path.
"""
struct OrderedRegimePath{P <: Tuple} <: AbstractSliceType
    regimes::P
end

OrderedRegimePath(path::AbstractVector{<:Integer}) = OrderedRegimePath(Tuple(Int.(path)))

"""
A slice type together with its closed existence condition in base space.

This record deliberately does not call the condition a chamber: closed
existence conditions can overlap on discriminant strata and may need further
connected-stratum refinement.
"""
struct ConditionalSliceType{S <: AbstractSliceType, C}
    slice_type::S
    condition::C
    feasible::Bool
    condition_dimension::Int
    full_dimensional::Bool
end

is_feasible(slice::ConditionalSliceType) = slice.feasible

"""
One connected stratum of an exact chamber decomposition.
"""
struct FiberChamber{S <: AbstractSliceType, C, W}
    id::Int
    slice_type::S
    condition::C
    witness::W
    dimension::Int
end

"""
Exact chamber nodes and their verified codimension-one adjacency graph.
"""
struct ChamberComplex{C <: FiberChamber, G <: AbstractGraph}
    chambers::Vector{C}
    adjacency::G

    function ChamberComplex(
        chambers::Vector{C}, adjacency::G
    ) where {C <: FiberChamber, G <: AbstractGraph}
        nv(adjacency) == length(chambers) ||
            throw(DimensionMismatch("the chamber graph must have one vertex per chamber."))
        return new{C, G}(chambers, adjacency)
    end
end
