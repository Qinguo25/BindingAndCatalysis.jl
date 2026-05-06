# __precompile__(false)
module BindingAndCatalysis

# using GLMakie
# using Plots
using Symbolics
using Parameters
using LinearAlgebra
# using DifferentialEquations
import OrdinaryDiffEq as ODE
import DiffEqCallbacks as CB
using StatsBase
using SparseArrays
# using IntegerSmithNormalForm # to get the maximum of denum 
# using JuMP
# using CUDA # Speedup calculation for distance matrix
using DataStructures:Queue,enqueue!,dequeue!,isempty
# using Interpolations
using NonlinearSolve
using Statistics:quantile
using Distributions:Uniform, Normal

# Exact log rational types
include(joinpath(@__DIR__, "ExactTypes.jl"))
using .ExactTypes: ExactLogExpr, exact_log10, exact_log10_ratio

using Polyhedra
import CDDLib

using Graphs

import Printf
import JSON3
import ImageFiltering: imfilter, Kernel

import Random
import Base: summary,show

#---------------------------plot dependency-----------------------------
using Makie
using GraphMakie
using GraphMakie.NetworkLayout
using Latexify

using ProgressMeter





export Bnc, update_catalysis!
export ExactLogExpr, exact_log10, exact_log10_ratio

#Polyhedra export
export Polyhedron, HRep, MixedMatHRep, hrep, polyhedron
export VRep, vrep, points, rays, lines
export HalfSpace, HyperPlane, intersect, eliminate, detecthlinearity!, removehredundancy!
export dim, fulldim, hashyperplanes, hyperplanes, allhalfspaces, issubset
export get_Lcat




# ---------------------Define the struct of binding and catalysis networks----------------------------------

include(joinpath(@__DIR__,"volume_calc.jl"))


#===============================================================================================#
# Integration Helper struct
#===============================================================================================#

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
    _LN_top_diag_idx::Vector{Int} # the diagonal index of the top d rows of _LN_sparse, used for fast calculation

    _LN_sparse::SparseMatrixCSC{Float64,Int} # cached Float64.(sparse([L; N])) for numerical integration
    _LN_lu::Union{SparseArrays.UMFPACK.UmfpackLU{Float64,Int}, Nothing} # LU decomposition of _LNt_sparse, used for fast calculation
end


@inline function calc_integration_helper(L,N)
    n = size(L,2)
    d = size(L,1)
    r = size(N,1)
    _anchor_log_x = zeros(n)
    _anchor_log_qK = vcat(vec(log10.(sum(L; dims=2))), zeros(r))
    
    _LN_sparse = Float64.(sparse([L; N]))
    (_LN_top_rows, _LN_top_cols, _LN_top_idx) = rowmask_indices(_LN_sparse, 1,d) # record the position of non-zero elements in L within _LN_sparse
    (_LN_bottom_rows, _LN_bottom_cols, _LN_bottom_idx) = rowmask_indices(_LN_sparse, d+1,n) # record the position of non-zero elements in N within _LN_sparse
    _LN_top_diag_idx = diag_indices(_LN_sparse, d)
    
    _LN_lu = rank(_LN_sparse)== n ? lu(_LN_sparse) : nothing # LU decomposition of _LNt_sparse, used for fast calculation

    IntegrationHelper(
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




abstract type AbstractBnc end
abstract type AbstractRegime end
abstract type AbstractHyperPlane end
abstract type AbstractHelper end

#=================================================================================#
# f(L) -> {P,P0,C,C0} associated structs and helpers
#=================================================================================#

include(joinpath(@__DIR__, "utils/HyperPlanes.jl"))



#=================================================================================#
# Regimes associated structs, including regimes for binding, catalysis and the combined Bnc regimes, 
#=================================================================================#


struct Regimes{T,R<:AbstractRegime,A<:AbstractArray{R}}
    vertices_perm_dict::Dict{Vector{T},Int}
    vertices_data::A
end


const BindAffineMatrix = Union{
    SparseMatrixCSC{Float64,Int},  # Keep this for now as singular regimes's C
    SparseMatrixCSC{Rational{Int},Int},
}
const BindConditionBiasVector = Union{Vector{Float64}, Vector{ExactLogExpr}}


"""
    BindRegime

Representation of a regime/vertex in a binding network, including cached
linear maps and polyhedral conditions.
"""
mutable struct BindRegime{F,T} <: AbstractRegime
    #--- Parent Bnc model reference ---
    network::Union{AbstractBnc,Nothing} # Reference to the parent Bnc model

    # --- Initial / Identifying Properties ---
    perm::Vector{T} # The regime vector
    idx::Int # Index of the vertex in the Bnc.vertices list
    is_asymptotic::Bool # Whether the vertex is asymptotic or not.

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

    function BindRegime(; network=nothing, perm, idx, is_asymptotic, nullity::T) where {T<:Integer}
        return new{ExactLogExpr,T}(network, perm, idx, is_asymptotic,
            nothing,nothing, nothing, nothing, nothing, nothing, # P, P0, M, M0, C_x, C0_x
            nullity,
            nothing, nothing, # H, H0
            nothing, nothing, # C_qK,C0_qK
            nothing
        )
    end
end




mutable struct CatalysisRegime{F<:Real} <:AbstractRegime
    network::Union{AbstractBnc,Nothing} # Reference to the parent Bnc model
    perm::Vector{Int} # The regime vector
    idx::Int # Index of the vertex in the Catalysis.vertices list
    is_asymptotic::Bool # Whether this catalysis regime is asymptotic or not.

    #--- Basic Properties ---
    P_pos_neg::Union{SparseMatrixCSC{Int, Int}, Nothing} # the vcat of P_pos and P_neg
    P0_pos_neg::Union{Vector{F}, Nothing} # the vcat of P0_pos and P0_neg
    
    P:: Union{SparseMatrixCSC{Int, Int}, Nothing} # P_pos - P_neg
    P0::Union{Vector{F}, Nothing} # P0_pos - P0_neg
    C::Union{SparseMatrixCSC{Int, Int}, Nothing} # the vcat of C_pos and C_neg
    C0::Union{Vector{F}, Nothing} # the vcat of C0_pos and C0_neg

    CΠ:: Union{SparseMatrixCSC{Int, Int}, Nothing} # the vcat of C_pos*Π and C_neg*Π
    # [CΠH C ] \log( (q_{cat},w,K), k) + C_0 + CΠH_0 >0 is the condition for catalysis regime
    # [CΠH*_{w,K} -CΠH*_{̃k}P +C] \log((w,K),k) + C_0 + CΠH*_0 >0 is the consistency condition for fixed point.

    PΠ:: Union{SparseMatrixCSC{Int, Int}, Nothing} # the vcat of (P_pos - P_neg)*Π
    # Act as N for catalysis regime.

    function CatalysisRegime(; network=nothing, perm, idx, is_asymptotic) 
        return new{ExactLogExpr}(network, perm, idx, is_asymptotic,
            nothing, # P_pos_neg
            nothing, # P0_pos_neg
            nothing, # P
            nothing, # P0
            nothing, # C
            nothing, # C0
            nothing, # CΠ
            nothing  # PΠ
        )
    end
end

# for BncRegime, the x /xk conditions are already within bind_rgm or catalysis_rgm, 
# H_w, H0_w, C_wKk, C0_wKk
# C_qKk_cat, C_0qKk_cat, 
# C_xk_ss


mutable struct BncRegime <:AbstractRegime
    bind_rgm::BindRegime
    catalysis_rgm::CatalysisRegime


    # Fixed point information:
    H_bd::Union{SparseMatrixCSC{Float64, Int}, Nothing}
    is_stable::Int8 # 1 for stable, -1 for unstable, 0 for unknown # mapped from d_stable

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
            Int8(0), #is_stable
            nothing, # H_inner
            nothing, # H0_inner

            -1, # nlt
            
            nothing, 
            nothing,
            nothing,
            nothing, 
            -1,  
            
            nothing, 
            nothing)
    end
end



"""
    CatalysisData

Container for catalysis network metadata, including stoichiometric changes,
reaction orders, and rate constants.
"""
mutable struct CatalysisData <:AbstractBnc
    # Parameters for the catalysis networks
    bn::AbstractBnc # reference to the parent Bnc model, used for validation and consistency checks

    # Catalysis determining Matrix
    Γ::SparseMatrixCSC{Int,Int} # catalysis change in qK space, each column is a reaction
    Π::SparseMatrixCSC{Int,Int} # catalysis index and coefficients, rate will be vⱼ=kⱼ∏xᵢ^Π_{j,i}, denote what species catalysis the reaction.

    # Derived matrices 
    S::SparseMatrixCSC{Int,Int} # the full row rank version of Γ
    L_Γ::SparseMatrixCSC{Int,Int} # the left null space of Γ such that L_Γ^⊤ * Γ = 0

    # Derived parameters
    r_v::Int # number of independent catalysis reactions, L_w = L[r_v+1:end, :]
    n_v::Int # number of flux, typically equal to the number of k 
    d_w::Int # total number of reduced conserved quantities collected into w, (r_v+d_w = d)
    a_w::Int # split row: L_w[1:a_w, :] is old dependent-part, L_w[a_w+1:end, :] is former parameter part

    # symbols of k
    k_sym::Vector{Num}


    # helper parameters for fast calculation, used for fast calculation of H and C_qK
    _S_sparse::SparseMatrixCSC{Float64,Int} # sparse version of Γ, used for fast calculation
    _Π_sparse::SparseMatrixCSC{Float64,Int}  # sparse version of Π, used for fast calculation

    #Catalysis regimes
    S_pos_neg::SparseMatrixCSC{Int,Int} # the vcat of positive and negative parts of S
    _S_helper::AbstractHelper

    CatalysisRegimes::Union{Regimes,Nothing} # Using Any for placeholder for CatalysisRegimes
    vertices_graph::Union{Any,Nothing}

    function CatalysisData(bn,Γ, Π, k_sym, w_sym=nothing)
        Γ = sparse(Γ)
        Π = sparse(Π)
        d_wv, nv = size(Γ)
        n = size(Π,2)
        # Validation
        @assert size(Π,1) == length(k_sym) == nv "Γ's column number have to meet with total flux number and k_sym"
        @assert n == bn.n "Π's column number have to meet with the number of species n in the binding network"
        L_Γ, pivits = left_nullspace_integer(Γ)

        r_v = length(pivits) # Maximum of non-redundant flux, also the number of independent catalysis reactions.
        a_w = size(L_Γ,2)
        d_w = bn.d - r_v

        # reorder and fix the binding network
        no_pivits = setdiff(1:d_wv, pivits)
        S = Γ[pivits, :]
        new_ord = vcat(pivits,no_pivits)
        Γ = Γ[new_ord, :]
        L_Γ = L_Γ[new_ord, :]
        fix_bn_catalysis!(bn, new_ord, L_Γ, w_sym)

        # Create sparse matrices
        _S_sparse = sparse(Float64.(S))
        _Π_sparse = sparse(Float64.(Π))

        S_pos_neg = S_to_S_pos_neg(S)
        _S_helper = _build_matrix_helper(S_pos_neg)

        new(bn, Γ, Π, S, L_Γ,
            r_v, nv, d_w, a_w,
            k_sym, _S_sparse, _Π_sparse,
            S_pos_neg, _S_helper, nothing,nothing)
    end
end






"""
    Bnc

Binding network model with stoichiometry, conservation laws, and derived
structures for regime analysis.
"""
mutable struct Bnc{T} <: AbstractBnc # T is the int type to save all the indices
    # ----Parameters of the binding networks------
    N::SparseMatrixCSC{Int,Int} # binding reaction matrix
    L::SparseMatrixCSC{Int,Int} # conservation law matrix

    r::Int # number of reactions
    n::Int # number of variables
    d::Int # number of conserved quantities
    # lcm::Int # least common multiple of [L;N]^{-1}

    #-------symbols of species -----------
    x_sym::Vector{Num} # species symbols, each column is a species
    q_sym::Vector{Num}
    K_sym::Vector{Num}

    #-------Parameters of the catalysis networks------
    catalysis::Union{Any,Nothing} # Using Any for placeholder for CatalysisData

    #--------Binding regimes data--------
    BindRegimes::Union{Regimes, Nothing}

    #-------Mixed regimes data--------
    BncRegimes::Union{Any, Nothing}

    #The following are computed when building graphs.
    vertices_graph::Union{Any,Nothing} # Using Any for placeholder for RegimeGraph
    # _vertices_Nρ_inv_dict::Dict{Vector{T}, Tuple{SparseMatrixCSC{Float64, Int},T}} # cache the N_inv for each vertex permutation
    _vertices_Nρ_inv_dict :: Union{Any,Nothing}
    _regimes_affine_ready::Bool
    _regimes_affine_lock::ReentrantLock
    _integration_helper_lock::ReentrantLock

    #------other helper parameters------
    direction::Int8 # direction of the binding reactions, determine the ray direction for invertible regime, calculated by sign of det[L;N]
    IntegrationHelper::Union{IntegrationHelper,Nothing}
    _L_helper::AbstractHelper # MatrixHelper


    # Inner constructor 
    function Bnc{T}(N, L, x_sym, q_sym, K_sym, catalysis) where {T<:Integer}
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

        #The direction and lcm
        M = vcat(L_dense, N_dense)
        direction = sign(det(M)) # Ensure matrix is Float64 for det
        # lcm = get_max_denom(M)
        #-------helper parameters-------------
        # paramters for default homotopcontinuous starting point.
        _L_helper = _build_matrix_helper(L)
        new(
            # Fields 1-5
            N_sparse, L_sparse, r, n, d,# lcm,
            # Fields 6-9
            x_sym, q_sym, K_sym, catalysis,
            # Fields 10-12 (Initialized empty)
            nothing,                         # BindRegimes
            nothing,                         # BncRegimes
            nothing,                         # vertices_graph
            nothing,                         # _vertices_perm_Ninv_dict
            false,                           # _regimes_affine_ready
            ReentrantLock(),                 # _regimes_affine_lock
            ReentrantLock(),                 # _integration_helper_lock
            # Fields 13-28 (Calculated values)
            direction,
            nothing,
            _L_helper,
        )
    end
end

    


pth1 = joinpath(@__DIR__,"Mathcore/")

include(joinpath(@__DIR__, "initialize.jl"))
include(joinpath(pth1,"find_matrix_vertex.jl")) # before regimes.jl
include(joinpath(pth1,"d_stable.jl"))
using .DStable: judge_dstable
export judge_dstable
include(joinpath(pth1,"perm_graph_core.jl"))
include(joinpath(pth1,"SparseSparse_modified.jl"))

include(joinpath(@__DIR__,"helperfunctions.jl"))
include(joinpath(pth1,"matrix_inverse.jl"))
include(joinpath(pth1,"graph_propagate.jl"))
include(joinpath(@__DIR__,"qK_x_mapping.jl"))
include(joinpath(@__DIR__,"regime_assign.jl"))
include(joinpath(@__DIR__,"volume_calc_impl.jl"))
include(joinpath(@__DIR__,"numeric.jl"))

# three different level of regime
include(joinpath(@__DIR__,"RegimeCore.jl"))
include(joinpath(@__DIR__,"BindingRegime.jl"))
include(joinpath(@__DIR__,"CatalysisRegime.jl"))
include(joinpath(@__DIR__,"BncRegime.jl"))

# three different level of regime graph
include(joinpath(@__DIR__,"BindingRegimeGraph.jl"))
include(joinpath(@__DIR__,"CatalysisRegimeGraph.jl"))
include(joinpath(@__DIR__,"BncRegimeGraph.jl"))


include(joinpath(@__DIR__,"SIMO.jl"))
include(joinpath(@__DIR__,"symbolics.jl"))
include(joinpath(@__DIR__,"additional_constrain.jl"))
include(joinpath(@__DIR__,"visualize.jl"))
include(joinpath(@__DIR__,"old_api.jl"))

end # module
