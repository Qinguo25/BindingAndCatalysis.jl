# __precompile__(false)
module BindingAndCatalysis

#========================================================================================#
# Dependencies
#========================================================================================#

using LinearAlgebra
using Symbolics

import DiffEqCallbacks as CB
import OrdinaryDiffEq as ODE
using OrdinaryDiffEq: AutoForwardDiff, ODESolution

using Distributions: Normal
using Graphs
using Polyhedra
using ProgressMeter
using SparseArrays
using Statistics: mean, median!, quantile

import Base: summary, show
using CDDLib: CDDLib
using JSON3: JSON3
using Printf: Printf
using Random: Random

# Latex rendering is used by symbolic display helpers.
using Latexify

#========================================================================================#
# Internal paths and exact numeric types
#========================================================================================#

const _SRC_DIR = @__DIR__
const _MATHCORE_DIR = joinpath(_SRC_DIR, "Mathcore")

_include_src(path...) = include(joinpath(_SRC_DIR, path...))
_include_mathcore(path...) = include(joinpath(_MATHCORE_DIR, path...))

# Exact log rational types
_include_src("ExactTypes.jl")
using .ExactTypes: ExactLogExpr, exact_log10, exact_log10_ratio

#========================================================================================#
# Public API exports
#========================================================================================#

export Bnc, update_catalysis!
export ExactLogExpr, exact_log10, exact_log10_ratio

# Polyhedra backend exports
export Polyhedron, HRep, MixedMatHRep, hrep, polyhedron
export VRep, vrep, points, rays
export HalfSpace, HyperPlane, intersect, eliminate, detecthlinearity!, removehredundancy!
export dim, fulldim, hashyperplanes, hyperplanes, allhalfspaces, issubset
export get_Lcat

#========================================================================================#
# Core shared types
#========================================================================================#

_include_src("volume_calc.jl")

#========================================================================================#
# Numerical integration cache
#========================================================================================#

"""
    IntegrationHelper

Container for cached integration starting points and sparse matrix index helpers.
Used for Integration during homotopy continuation.
"""
mutable struct IntegrationHelper
    _anchor_log_x::Vector{<:Real}
    _anchor_log_qK::Vector{<:Real}

    _LN_top_idx::Vector{Int} # first d row index of _LN_sparse
    _LN_top_rows::Vector{Int} # the corresponding row number in L for _LN_top_idx
    _LN_top_cols::Vector{Int} # the corresponding column number in L for _LN_top_idx

    _LN_bottom_idx::Vector{Int} # last r row index of _LN_sparse
    _LN_bottom_rows::Vector{Int} # the corresponding row number in N for _LN_bottom_idx
    _LN_bottom_cols::Vector{Int} # the corresponding column number in N for _LN_bottom_idx
    _LN_top_diag_idx::Vector{Int} # one perturbation nzval index per top row, chosen to preserve nonsingularity

    _LN_sparse::SparseMatrixCSC{Float64, Int} # cached Float64.(sparse([L; N])) for numerical integration
    _LN_lu::Union{SparseArrays.UMFPACK.UmfpackLU{Float64, Int}, Nothing} # LU decomposition of _LNt_sparse, used for fast calculation
end

@inline function calc_integration_helper(L, N)
    n = size(L, 2)
    d = size(L, 1)
    r = size(N, 1)
    _anchor_log_x = zeros(n)
    _anchor_log_qK = vcat(vec(log10.(sum(L; dims=2))), zeros(r))

    _LN_sparse = Float64.(sparse([L; N]))
    (_LN_top_rows, _LN_top_cols, _LN_top_idx) = rowmask_indices(_LN_sparse, 1, d)
    (_LN_bottom_rows, _LN_bottom_cols, _LN_bottom_idx) = rowmask_indices(
        _LN_sparse, d + 1, n
    )
    _LN_top_diag_idx = diag_indices(_LN_sparse, d)

    _LN_lu = rank(_LN_sparse) == n ? lu(_LN_sparse) : nothing

    return IntegrationHelper(
        _anchor_log_x,
        _anchor_log_qK,
        _LN_top_idx,
        _LN_top_rows,
        _LN_top_cols,
        _LN_bottom_idx,
        _LN_bottom_rows,
        _LN_bottom_cols,
        _LN_top_diag_idx,
        _LN_sparse,
        _LN_lu,
    )
end

#========================================================================================#
# Abstract interfaces
#========================================================================================#

abstract type AbstractBnc end
abstract type AbstractRegime end
abstract type AbstractHyperPlane end
abstract type AbstractHelper end

#========================================================================================#
# f(L) -> {P, P0, C, C0} helpers
#========================================================================================#

_include_src("utils", "HyperPlanes.jl")

#========================================================================================#
# Regime containers
#========================================================================================#

struct Regimes{T, R <: AbstractRegime, A <: AbstractArray{R}}
    regimes_perm_dict::Dict{Vector{T}, Int}
    regimes_data::A
end

function Base.getproperty(rgms::Regimes, name::Symbol)
    if name === :vertices_perm_dict
        return getfield(rgms, :regimes_perm_dict)
    elseif name === :vertices_data
        return getfield(rgms, :regimes_data)
    end
    return getfield(rgms, name)
end

function Base.propertynames(::Regimes, private::Bool=false)
    names = (:regimes_perm_dict, :regimes_data)
    return private ? (names..., :vertices_perm_dict, :vertices_data) : names
end

const BindAffineMatrix = Union{
    SparseMatrixCSC{Float64, Int},  # Keep this for now as singular regimes's C
    SparseMatrixCSC{Rational{Int}, Int},
}
const BindConditionBiasVector = Union{Vector{Float64}, Vector{ExactLogExpr}}

#========================================================================================#
# Binding regimes
#========================================================================================#

"""
    BindRegime

Representation of a binding regime in a binding network, including cached
linear maps and polyhedral conditions.
"""
mutable struct BindRegime{F, T} <: AbstractRegime
    #--- Parent Bnc model reference ---
    network::Union{AbstractBnc, Nothing} # Reference to the parent Bnc model

    # --- Initial / Identifying Properties ---
    perm::Vector{T} # The regime vector
    idx::Int # Index of the regime in the parent Regimes container
    is_asymptotic::Bool # Whether the regime is asymptotic or not.

    # --- Basic Properties ---
    P::Union{SparseMatrixCSC{Int, Int}, Nothing}
    P0::Union{Vector{F}, Nothing}
    M::Union{SparseMatrixCSC{Int, Int}, Nothing}
    M0::Union{Vector{F}, Nothing}
    C_x::Union{SparseMatrixCSC{Int, Int}, Nothing}
    C0_x::Union{Vector{F}, Nothing}

    # --- Expensive Calculated Properties ---
    nullity::T
    H::Union{BindAffineMatrix, Nothing}
    H0::Union{Vector{F}, Nothing}
    C_qK::Union{BindAffineMatrix, Nothing}
    C0_qK::Union{BindConditionBiasVector, Nothing}

    volume::Union{Volume, Nothing}

    function BindRegime(;
        network=nothing, perm, idx, is_asymptotic, nullity::T
    ) where {T <: Integer}
        return new{ExactLogExpr, T}(
            network,
            perm,
            idx,
            is_asymptotic,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing, # P, P0, M, M0, C_x, C0_x
            nullity,
            nothing,
            nothing, # H, H0
            nothing,
            nothing, # C_qK,C0_qK
            nothing,
        )
    end
end

#========================================================================================#
# Catalysis regimes
#========================================================================================#

mutable struct CatalysisRegime{F <: Real} <: AbstractRegime
    network::Union{AbstractBnc, Nothing} # Reference to the parent Bnc model
    perm::Vector{Int} # The regime vector
    idx::Int # Index of the regime in the parent Regimes container
    is_asymptotic::Bool # Whether this catalysis regime is asymptotic or not.

    #--- Basic Properties ---
    P_pos_neg::Union{SparseMatrixCSC{Int, Int}, Nothing} # the vcat of P_pos and P_neg
    P0_pos_neg::Union{Vector{F}, Nothing} # the vcat of P0_pos and P0_neg

    P::Union{SparseMatrixCSC{Int, Int}, Nothing} # P_pos - P_neg
    P0::Union{Vector{F}, Nothing} # P0_pos - P0_neg
    C::Union{SparseMatrixCSC{Int, Int}, Nothing} # the vcat of C_pos and C_neg
    C0::Union{Vector{F}, Nothing} # the vcat of C0_pos and C0_neg

    CΠ::Union{SparseMatrixCSC{Int, Int}, Nothing} # the vcat of C_pos*Π and C_neg*Π
    # [CΠH C ] \log( (q_{cat},w,K), k) + C_0 + CΠH_0 >0 is the condition for catalysis regime
    # [CΠH*_{w,K} -CΠH*_{̃k}P +C] \log((w,K),k) + C_0 + CΠH*_0 >0 is the consistency condition for fixed point.

    PΠ::Union{SparseMatrixCSC{Int, Int}, Nothing} # the vcat of (P_pos - P_neg)*Π
    # Act as N for catalysis regime.

    function CatalysisRegime(; network=nothing, perm, idx, is_asymptotic)
        return new{ExactLogExpr}(
            network,
            perm,
            idx,
            is_asymptotic,
            nothing, # P_pos_neg
            nothing, # P0_pos_neg
            nothing, # P
            nothing, # P0
            nothing, # C
            nothing, # C0
            nothing, # CΠ
            nothing,  # PΠ
        )
    end
end

#========================================================================================#
# Matched binding-catalysis regimes
#========================================================================================#
#
# x/xk conditions live on the binding and catalysis regime objects. BncRegime
# caches only the reduced steady-state maps and conditions in qKk/wKk bases.

mutable struct BncRegime <: AbstractRegime
    bind_rgm::BindRegime
    catalysis_rgm::CatalysisRegime

    # Fixed point information:
    H_bd::Union{SparseMatrixCSC{Float64, Int}, Nothing}
    is_stable::Union{Nothing, Missing, Bool} # nothing=uncomputed, missing=undetermined
    is_feasible::Bool

    # wKk̃2x mapping, the core matrix
    H_inner::Union{AbstractMatrix{<:Real}, Nothing}
    H0_inner::Union{AbstractVector{<:Real}, Nothing}

    nlt::Int
    H::Union{AbstractMatrix{<:Real}, Nothing} # x's reaction order to w,K,k
    H0::Union{AbstractVector{<:Real}, Nothing} # Intersection

    # Conditions
    ## x, k base
    # Directly extract from bind_rgm and catalysis_rgm, no need to calculate separately.

    # bind_conds: bind_rgm.C_x, bind_rgm.C0_x,
    # Catalysis_conds: catalysis_rgm.

    ## q_cat, K, k base
    # Binding could directly extract from bind_rgm, catalysis needs to calculate seperately
    # If binding is singular, we need to Combine with M,M0 to do the elimination again

    C_qKk_cat::Union{AbstractMatrix{<:Real}, Nothing}
    C0_qKk_cat::Union{AbstractVector{<:Real}, Nothing}
    nlt_qKk_cat::Int

    ## w, K, k base
    C_wKk::Union{AbstractMatrix{<:Real}, Nothing}
    C0_wKk::Union{AbstractVector{<:Real}, Nothing}
    nlt_wKk::Int

    volume::Union{Volume, Nothing}

    function BncRegime(bind_rgm, catalysis_rgm)
        PΠ = get_PΠ(catalysis_rgm)
        H_bind = if bind_rgm.nullity == 0
            get_affine_qK2x(bind_rgm)[1]
        else
            get_H_numerically(bind_rgm)
        end
        r_v = size(PΠ, 1)
        H_bd = sparse(Float64.(PΠ * H_bind[:, 1:r_v]))

        return new(
            bind_rgm,
            catalysis_rgm,
            H_bd,
            nothing, # is_stable: not computed
            true, # is_feasible
            nothing, # H_inner
            nothing, # H0_inner
            -1, # nlt
            nothing,
            nothing,
            nothing,
            nothing,
            -1,
            nothing,
            nothing,
            -1,
            nothing,
        )
    end
end

#========================================================================================#
# Catalysis network data
#========================================================================================#

"""
    CatalysisData

Container for catalysis network metadata, including stoichiometric changes,
reaction orders, and rate constants.
"""
mutable struct CatalysisData <: AbstractBnc
    # Parameters for the catalysis networks
    bn::AbstractBnc # reference to the parent Bnc model, used for validation and consistency checks

    # Catalysis determining Matrix
    Γ::SparseMatrixCSC{Int, Int} # catalysis change in qK space, each column is a reaction
    Π::SparseMatrixCSC{Int, Int} # catalysis index and coefficients, rate will be vⱼ=k_oldⱼ∏xᵢ^Π_{j,i}, denote what species catalysis the reaction.
    F::SparseMatrixCSC{Rational{Int}, Int} # affine map from independent log k to old flux log k: log k_old = F log k + F0
    F0::Vector{ExactLogExpr}

    # Derived matrices 
    S::SparseMatrixCSC{Int, Int} # the full row rank version of Γ
    L_Γ::SparseMatrixCSC{Int, Int} # the left null space of Γ such that L_Γ^⊤ * Γ = 0

    # Derived parameters
    r_v::Int # number of independent catalysis reactions, L_w = L[r_v+1:end, :]
    n_v::Int # number of fluxes / old rate constants
    n_k::Int # number of independent rate constants after affine k constraints
    d_w::Int # total number of reduced conserved quantities collected into w, (r_v+d_w = d)
    a_w::Int # split row: L_w[1:a_w, :] is old dependent-part, L_w[a_w+1:end, :] is former parameter part

    # symbols of independent k and flux v, with log v = Π log x + F log k + F0
    k_sym::Vector{Num}
    v_sym::Vector{Num}

    # helper parameters for fast calculation, used for fast calculation of H and C_qK
    _S_sparse::SparseMatrixCSC{Float64, Int} # sparse version of Γ, used for fast calculation
    _Π_sparse::SparseMatrixCSC{Float64, Int}  # sparse version of Π, used for fast calculation

    #Catalysis regimes
    S_pos_neg::SparseMatrixCSC{Int, Int} # the vcat of positive and negative parts of S
    _S_helper::AbstractHelper

    CatalysisRegimes::Union{Regimes, Nothing} # Using Any for placeholder for CatalysisRegimes
    vertices_graph::Union{Any, Nothing} # legacy field name for the regime graph

    function CatalysisData(
        bn, Γ, Π, k_sym, w_sym=nothing, v_sym=nothing, F=nothing, F0=nothing
    )
        Γ = sparse(Γ)
        Π = sparse(Π)
        d_wv, nv = size(Γ)
        n = size(Π, 2)
        F = if isnothing(F)
            Matrix{Rational{Int}}(I, nv, nv)
        else
            rationalize.(Int, Float64.(F); tol=1e-10)
        end
        size(F, 1) == nv || throw(
            ArgumentError(
                "F must have one row per catalysis flux/rate constant: expected $nv, got $(size(F, 1)).",
            ),
        )
        nk = size(F, 2)
        F0 = if isnothing(F0)
            fill(zero(ExactLogExpr), nv)
        else
            [
                if x isa ExactLogExpr
                    x
                else
                    ExactLogExpr(rationalize(Int, Float64(x); tol=1e-10))
                end for x in vec(F0)
            ]
        end
        length(F0) == nv || throw(
            ArgumentError(
                "F0 length must match the number of catalysis fluxes/rate constants: expected $nv, got $(length(F0)).",
            ),
        )
        F = sparse(F)
        v_sym = isnothing(v_sym) ? Symbolics.variables(:v, 1:nv) : name_converter(v_sym)
        # Validation
        @assert size(Π, 1) == nv "Γ's column number must match Π's row number."
        @assert length(k_sym) == nk "k_sym length must match the number of independent rate constants (size(F, 2))."
        @assert length(v_sym) == nv "v_sym length must match the number of fluxes"
        @assert n == bn.n "Π's column number have to meet with the number of species n in the binding network"
        L_Γ, pivits = left_nullspace_integer(Γ)

        r_v = length(pivits) # Maximum of non-redundant flux, also the number of independent catalysis reactions.
        a_w = size(L_Γ, 2)
        d_w = bn.d - r_v

        # reorder and fix the binding network
        no_pivits = setdiff(1:d_wv, pivits)
        S = Γ[pivits, :]
        new_ord = vcat(pivits, no_pivits)
        Γ = Γ[new_ord, :]
        L_Γ = L_Γ[new_ord, :]
        fix_bn_catalysis!(bn, new_ord, L_Γ, w_sym)

        # Create sparse matrices
        _S_sparse = sparse(Float64.(S))
        _Π_sparse = sparse(Float64.(Π))

        S_pos_neg = S_to_S_pos_neg(S)
        _S_helper = _build_matrix_helper(S_pos_neg)

        return new(
            bn,
            Γ,
            Π,
            F,
            F0,
            S,
            L_Γ,
            r_v,
            nv,
            nk,
            d_w,
            a_w,
            k_sym,
            v_sym,
            _S_sparse,
            _Π_sparse,
            S_pos_neg,
            _S_helper,
            nothing,
            nothing,
        )
    end
end

#========================================================================================#
# Binding network model
#========================================================================================#

"""
    Bnc

Binding network model with stoichiometry, conservation laws, and derived
structures for regime analysis.
"""
mutable struct Bnc{T} <: AbstractBnc # T is the int type used for regime indices.
    # Binding network matrices
    N::SparseMatrixCSC{Int, Int} # binding reaction matrix
    L::SparseMatrixCSC{Int, Int} # conservation law matrix

    r::Int # number of reactions
    n::Int # number of variables
    d::Int # number of conserved quantities
    # lcm::Int # least common multiple of [L;N]^{-1}

    # Symbol metadata
    x_sym::Vector{Num} # species symbols, each column is a species
    q_sym::Vector{Num}
    K_sym::Vector{Num}

    # Catalysis network data
    catalysis::Union{Any, Nothing} # Using Any for placeholder for CatalysisData

    # Cached regime data
    BindRegimes::Union{Regimes, Nothing}
    BncRegimes::Union{Any, Nothing}

    # Graph and affine propagation caches
    vertices_graph::Union{Any, Nothing} # legacy field name for the regime graph
    _vertices_Nρ_inv_dict::Union{Any, Nothing} # legacy field name for regime affine caches
    _regimes_affine_ready::Bool
    _regimes_affine_lock::ReentrantLock
    _integration_helper_lock::ReentrantLock
    _diagnostics::Dict{Symbol, Any}

    # Numeric helpers
    direction::Int8 # direction of the binding reactions, determine the ray direction for invertible regime, calculated by sign of det[L;N]
    IntegrationHelper::Union{IntegrationHelper, Nothing}
    _L_helper::AbstractHelper # MatrixHelper

    # Inner constructor
    function Bnc{T}(N, L, x_sym, q_sym, K_sym, catalysis) where {T <: Integer}
        N_sparse = sparse(N)
        L_sparse = sparse(L)
        N_dense = Matrix{Int}(N)
        L_dense = Matrix{Int}(L)

        # get desired values
        r, n = size(N_dense)
        d, n_L = size(L_dense)
        # Validate dimensions for binding network, check if its legal.
        let
            @assert n == d + r "d+r is not equal to n"
            @assert n_L == n "L must have the same number of columns as N"
            @assert length(x_sym) == n "x_sym length must equal number of species (n)"
            @assert length(q_sym) == d "q_sym length must equal number of conserved quantities (d)"
            @assert length(K_sym) == r "K_sym length must equal number of reactions (r)"
        end

        # Direction of the binding reactions.
        M = vcat(L_dense, N_dense)
        direction = sign(det(M))

        _L_helper = _build_matrix_helper(L)
        return new(
            N_sparse,
            L_sparse,
            r,
            n,
            d,
            x_sym,
            q_sym,
            K_sym,
            catalysis,
            nothing,                         # BindRegimes
            nothing,                         # BncRegimes
            nothing,                         # vertices_graph, legacy field name
            nothing,                         # _vertices_Nρ_inv_dict, legacy field name
            false,                           # _regimes_affine_ready
            ReentrantLock(),                 # _regimes_affine_lock
            ReentrantLock(),                 # _integration_helper_lock
            Dict{Symbol, Any}(),             # _diagnostics
            direction,
            nothing,
            _L_helper,
        )
    end
end
#========================================================================================#
# Source loading
#========================================================================================#

_include_src("initialize.jl")
_include_mathcore("find_matrix_vertex.jl") # before regime files
_include_mathcore("d_stable.jl")
using .DStable: judge_dstable
export judge_dstable

_include_mathcore("perm_graph_core.jl")
_include_mathcore("SparseSparse_modified.jl")
_include_mathcore("matrix_inverse.jl")
_include_mathcore("graph_propagate.jl")

_include_src("helperfunctions.jl")
_include_src("qK_x_mapping.jl")
_include_src("catalysis_dynamics.jl")
_include_src("regime_assign.jl")
_include_src("volume_calc_impl.jl")
_include_src("numeric.jl")

# Regime models and APIs
_include_src("RegimeCore.jl")
_include_src("BindingRegime.jl")
_include_src("CatalysisRegime.jl")
_include_src("BncRegime.jl")
_include_src("BncControl.jl")

# Regime graph APIs
_include_src("BindingRegimeGraph.jl")
_include_src("CatalysisRegimeGraph.jl")
_include_src("BncRegimeGraph.jl")

# Higher-level workflows, rendering, and compatibility
_include_src("FiberChamber.jl")
_include_src("SIMO.jl")
_include_src("symbolics.jl")
_include_src("RegimeConstraints.jl")
_include_src("visualize.jl")
_include_src("old_api.jl")

end # module
