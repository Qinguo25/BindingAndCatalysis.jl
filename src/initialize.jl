
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
# using JuMP
# using CUDA # Speedup calculation for distance matrix
using DataStructures:Queue,enqueue!,dequeue!,isempty
# using Interpolations
using NonlinearSolve
using Statistics:quantile
using Distributions:Uniform, Normal

using Polyhedra#:vrep,hrep,eliminate,MixedMatHRep,MixedMatVRep,polyhedron,Polyhedron
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



# ---------------------Define the struct of binding and catalysis networks----------------------------------


abstract type AbstractBnc end

abstract type AbstractRegime end


"""
    CatalysisData

Container for catalysis network metadata, including stoichiometric changes,
reaction orders, and rate constants.
"""
struct CatalysisData <:AbstractBnc
    # Parameters for the catalysis networks
    bn::AbstractBnc # reference to the parent Bnc model, used for validation and consistency checks
    Γ::SparseMatrixCSC{Int,Int} # catalysis change in qK space, each column is a reaction
    S::SparseMatrixCSC{Int,Int} # the full row rank version of Γ
    L_Γ::SparseMatrixCSC{Int,Int} # the left null space of Γ such that L_Γ^⊤ * Γ = 0
    Π::SparseMatrixCSC{Int,Int} # catalysis index and coefficients, rate will be vⱼ=kⱼ∏xᵢ^Π_{j,i}, denote what species catalysis the reaction.

    r_v::Int # number of independent catalysis reactions
    n_v::Int # number of flux
    d_w::Int # number of dependent conserved quantities.
    d_para::Int # number of parameter total concentrations



    k_sym::Vector{Num}
    # cat_x_idx::Vector{Int} # index of the species that catalysis the reaction, if not provided, will be inferred from Γ

    _S_sparse::SparseMatrixCSC{Float64,Int} # sparse version of Γ, used for fast calculation
    _Π_sparse::SparseMatrixCSC{Float64,Int}  # sparse version of Π, used for fast calculation

    #Catalysis regimes
    S_pos_neg::SparseMatrixCSC{Int,Int} # the vcat of positive and negative parts of S
    _S_helper::MatrixHelper




    function CatalysisData(bn,Γ, Π, k_sym)
        d_wv, nv = size(Γ)
        n = size(Π,2)
        # Validation
        @assert size(Π,1) == length(k_sym) == nv "Γ's column number have to meet with total flux number and k_sym"
        @assert n == bn.n "Π's column number have to meet with the number of species n in the binding network"
        L_Γ, pivits = left_nullspace_integer(Γ)

        r_v = length(pivits)
        d_w = size(L_Γ,1)
        d_para = bn.d - r_v

        # reorder and fix the binding network
        no_pivits = setdiff(1:d_wv, pivits)
        S = Γ[pivits, :]
        new_ord = vcat(pivits,no_pivits)
        Γ = Γ[new_ord, :]
        L_Γ = L_Γ[new_ord, :]
        fix_bn_catalysis!(bn, new_ord, L_Γ)

        # Create sparse matrices
        _S_sparse = sparse(Float64.(S))
        _Π_sparse = sparse(Float64.(Π))

        S_pos_neg = S_to_S_pos_neg(S)
        _S_helper = _build_matrix_helper(S)

        new(bn, Γ,S, L_Γ, Π, 
            r_v, nv, d_w, d_para,    
            k_sym, _S_sparse, _Π_sparse, S_pos_neg)
    end
end



struct Volume
    mean::Float64
    var::Float64
end
"""
    fetch_mean_re(V::Volume) -> (Float64, Float64)

Return the mean and relative error (standard deviation / mean) for a `Volume`.
"""
fetch_mean_re(V::Volume) = (V.mean, sqrt(V.var)/V.mean)
"""
    Base.display(V::Volume)

Display a compact summary of a `Volume`.
"""
Base.display(V::Volume) = Printf.@sprintf("Volume(Mean=%.3e, STD=%.3e, RelError=%.2f%%)", V.mean, sqrt(V.var), (sqrt(V.var)/V.mean)*100)
Base.show(io::IO, V::Volume) = print(io, Printf.@sprintf("Volume(Mean=%.3e, STD=%.3e, RelError=%.2f%%)", V.mean, sqrt(V.var), (sqrt(V.var)/V.mean)*100))
"""
    Base.:+(v1::Volume, v2::Volume) -> Volume

Add two `Volume` values by summing means and variances.
"""
Base.:+(v1::Volume, v2::Volume) = Volume(v1.mean + v2.mean, v1.var + v2.var)
"""
    Base.:-(v1::Volume, v2::Volume) -> Volume

Add two `Volume` values by summing means and variances.
"""
Base.:-(v1::Volume, v2::Volume) = Volume(v1.mean - v2.mean, v1.var + v2.var)
"""
    Base.isless(a::Volume, b::Volume) -> Bool

Compare `Volume` objects by mean value.
"""
Base.isless(a::Volume, b::Volume) = a.mean < b.mean
"""
    Base.:(==)(a::Volume, b::Volume) -> Bool

Return `true` when two `Volume` objects have identical means.
"""
Base.:(==)(a::Volume, b::Volume) = a.mean == b.mean 
"""
    Base.zero(::Volume) -> Volume

Return a zero `Volume` with zero mean and variance.
"""
Base.zero(::Volume) = Volume(0.0, 0.0)

Base.:*(c::Real, v::Volume) = Volume(c * v.mean, c^2 * v.var)
Base.:*(v::Volume, c::Real) = c * v
Base.:/(v::Volume, c::Real) = Volume(v.mean / c, v.var / c^2)


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
    
    # --- Basic Calculated Properties ---
    P::SparseMatrixCSC{Int, Int}
    P0::Vector{F} 
    M::SparseMatrixCSC{Int, Int}
    M0::Vector{F} #
    C_x::SparseMatrixCSC{Int, Int}
    C0_x::Vector{F} 

    # --- Expensive Calculated Properties ---
    nullity::T
    H::SparseMatrixCSC{Float64, Int} # Taking inverse, can have Float.
    H0::Vector{F} 
    C_qK::SparseMatrixCSC{Float64, Int}
    C0_qK::Vector{F} 
    
    #---Realizibility Index
    volume::Volume

    # The inner constructor also needs to be updated for the parametric type
    function BindRegime(;network::Union{AbstractBnc,Nothing}=nothing, perm, P, P0::Vector{F}, M, M0, C_x, C0_x, idx,is_asymptotic,nullity::T) where {T<:Integer,F<:Real}
        # _M_lu = lu(M, check=false) # It's good practice to ensure M is Float64 for LU
        # Use new{T} to construct an instance of BindRegime{T}
        return new{F,T}(network, perm, idx,is_asymptotic, P, P0, M, M0, C_x, C0_x,
            nullity,
            SparseMatrixCSC{Float64, Int}(undef, 0, 0), # H
            Vector{F}(undef, 0),          # H0
            SparseMatrixCSC{Float64, Int}(undef, 0, 0), # C_qK
            Vector{F}(undef, 0),          # C0_qK
            Volume(0.0, 0.0) # volume
        )
    end
end

"""
    VertexEdge

Edge metadata connecting neighboring vertices in a regime graph.
"""
mutable struct VertexEdge{T}
    to::Int
    diff_r::Int
    change_dir_x::SparseVector{Int8, T}
    intersect_x::Float64
    change_dir_qK::Union{Nothing, SparseVector{Float64, T}}
    intersect_qK::Union{Nothing, Float64}
    function VertexEdge(to::Int, diff_r::Int, change_dir_x::SparseVector{Int8, T}, intersect_x::Float64) where {T}
        return new{T}(to, diff_r, change_dir_x, intersect_x,nothing,nothing)
    end
end

# Adjacency list + optional caches
"""
    VertexGraph

Adjacency structure for vertices with optional caches for change directions.
"""
mutable struct VertexGraph{T}
    bn::AbstractBnc
    x_grh::SimpleGraph 
    neighbors::Vector{Vector{VertexEdge{T}}}
    change_dir_qK_computed::Bool
    edge_pos::Vector{Dict{Int, Int}}  # (u,v) -> (u,edge_pos[u][v]) to locate the VertexEdge.
    function VertexGraph(bn::AbstractBnc, neighbors::Vector{Vector{VertexEdge{T}}}) where {T}
        edge_pos = [Dict{Int, Int}() for _ in 1:length(neighbors)]
        g = SimpleGraph(length(neighbors))
        for i in 1:length(neighbors)
            edges = neighbors[i]
            for (k, e) in enumerate(edges)
                edge_pos[i][e.to] = k
                add_edge!(g, i, e.to)
            end
        end
        return new{T}(bn, g, neighbors, false, edge_pos)
    end
end





mutable struct CatalysisRegime <:AbstractRegime
    network::Union{AbstractBnc,Nothing} # Reference to the parent Bnc model
    perm::Vector{Int} # The regime vector
    idx::Int # Index of the vertex in the Catalysis.vertices list
end
























"""
Canonical hyperplane

Stored in canonical form with `u < v`:

    z_u - z_v + log10(num/den) = 0

where `(num, den)` is the reduced integer ratio.
"""
struct Hyperplane_perm{Tv<:Integer} 
    u::Int # fast access 
    v::Int # fast access
    num::Tv # reduced positive integer
    den::Tv # reduced positive integer
    c0::Float64 # pre-logarithm log10(num/den)
    crow::SparseVector{Int8,Int}      # +1 at u, -1 at v
    crow_neg::SparseVector{Int8,Int}  # +1 at v, -1 at u
end

"""
One oriented inequality induced by choosing p in row i.
If `sign == +1`, use the canonical side:
    crow * z + c0 > 0
If `sign == -1`, use the opposite side:
    crow_neg * z - c0 > 0
`competitor` is the losing column k compared against the perm dominant p.
`oriented_c0 = log10(L[i,p] / L[i,k])`
so the actual inequality is:
    z_p - z_k + oriented_c0 > 0
"""
struct ChoiceIneq
    hid::Int  # index into global hyperplane pool
    sign::Int8 # +1 for canonical side, -1 for opposite side

    #Fast access
    _competitor::Int # fast access
    _oriented_c0::Float64 # fast access.
end

"""
Helper struct for managing matrix operations.
- `J[i]`: positive columns in row i
- `choice_slot[i][p]`: local slot of column p inside J[i], or 0 if p ∉ J[i]
- `choice_map[i][t]`: all oriented inequalities for choosing p = J[i][t]
- `hyperplanes`: global deduplicated hyperplane pool
- `asymptotic`: all asymptotic regimes
- `feasible`: all regimes feasible under the weighted constraints
"""
struct MatrixHelper{Tv<:Integer}
    n::Int # number of columns
    J::Vector{Vector{Int}} # positive columns idx for each row

    # Fast access from column index to "local slot" in J[i]/ choice_logcoeff[i]
    choice_slot::Vector{Vector{Int}} # k = choice_slot[i][p] denotes p is the k th positive column in row i, or 0 if p ∉ J[i]
    choice_logcoeff::Vector{Vector{Float64}} # choice_logcoeff[i] = [log10(L[i, j]) for j in Ji]

    rowptr::Vector{Int} # rowptr[i] gives the starting index of constraints for row i in the global constraint list

    total_constraints::Int # total number of constraints across all rows
    choice_map::Vector{Vector{Vector{ChoiceIneq}}} # choice_map[i][t] gives the list of oriented inequalities for choosing p = J[i][t]
    hyperplanes::Vector{Hyperplane_perm{Tv}} # global deduplicated hyperplane pool
end

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
        _LN_lu,
    )
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

    #-------symbols of species -----------
    x_sym::Vector{Num} # species symbols, each column is a species
    q_sym::Vector{Num}
    K_sym::Vector{Num}

    #-------Parameters of the catalysis networks------
    catalysis::Union{Any,Nothing} # Using Any for placeholder for CatalysisData

    #--------Vertex data--------

    #The following four are computed when finding regimes.
    vertices_perm::Vector{Vector{T}} # all feasible regimes.
    vertices_perm_dict::Dict{Vector{T},Int} # map from permutation vector to its idx in the vertices list
    vertices_asymptotic_flag::Vector{Bool} # While this vertice is real
    vertices_nullity::Vector{T} # nullity of one vertex.
    
    #The following are computed when building graphs.
    vertices_graph::Union{Any,Nothing} # Using Any for placeholder for VertexGraph
    vertices_data::Vector{BindRegime} # Using Any for placeholder for BindRegime
    _vertices_is_initialized::BitVector
    _vertices_volume_is_calced::BitVector
    _vertices_Nρ_inv_dict::Dict{Vector{T}, Tuple{SparseMatrixCSC{Float64, Int},T}} # cache the N_inv for each vertex permutation

    #------other helper parameters------
    direction::Int8 # direction of the binding reactions, determine the ray direction for invertible regime, calculated by sign of det[L;N]
    IntegrationHelper::IntegrationHelper
    _L_helper::MatrixHelper

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

        #The direction
        direction = sign(det([L_dense;N_dense])) # Ensure matrix is Float64 for det
        #-------helper parameters-------------
        # paramters for default homotopcontinuous starting point.
        integration_helper = calc_integration_helper(L, N)
        _L_helper = _build_matrix_helper(L)
        new(
            # Fields 1-5
            N_sparse, L_sparse, r, n, d,
            # Fields 6-9
            x_sym, q_sym, K_sym, catalysis,
            # Fields 10-12 (Initialized empty)
            Vector{T}[],                # vertices_perm
            Dict{Vector{T},Int}(),            # vertices_perm_dict
            Bool[],                          # vertices_asymptotic_flag
            T[],                          # vertices_nullity
            nothing,                         # vertices_graph
            Vector{BindRegime}(),              # vertices_data
            BitVector(),                     # _vertices_is_initialized
            BitVector(),                     # _R_idx_is_calced
            Dict{Vector{T}, Tuple{SparseMatrixCSC{Float64, Int},T}}(), # _vertices_perm_Ninv_dict
            # Fields 13-28 (Calculated values)
            direction,
            integration_helper,
            _L_helper,
        )
    end
end





struct SISOPaths{T} 
    bn::Bnc{T}   # binding Newtork
    qK_grh::SimpleDiGraph # SimpleDiGraph in qK space
    change_qK_idx::T  # which qK is changing in this SISO graph

    sources::Vector{Int}  # source vertices in the graph
    sinks::Vector{Int}    # sink vertices in the graph
    paths_dict::Dict{Vector{Int},Int} # map from path (vector of vertex idx) to its idx in rgm_paths
    rgm_paths::Vector{Vector{Int}} #All paths from sources to sinks, each path is represented as a vector of vertex idx. Grows exponentially
    path_polys::Vector{Polyhedron} # the polyhedron for each path, lazily calculated when needed, stored in the same order as rgm_paths
    path_volume::Vector{Volume}# the volume for each path, lazily calculated when needed, stored in the same order as rgm_paths

    path_volume_is_calc::BitVector # whether the volume for each path is calculated, stored in the same order as rgm_paths
    path_polys_is_calc::BitVector # whether the polyhedron for each path is calculated, stored in the same order as rgm_paths
    
     function SISOPaths(model::Bnc{T}, qK_grh, change_qK_idx, sources, sinks, rgm_paths) where T
        path_polys = Vector{Polyhedron}(undef, length(rgm_paths))
        path_volume = Vector{Volume}(undef, length(rgm_paths))
        path_volume_is_calc = falses(length(rgm_paths))
        path_polys_is_calc = falses(length(rgm_paths))
        paths_dict = Dict{Vector{Int},Int}()
        for (i, p) in enumerate(rgm_paths)
            paths_dict[p] = i
        end  
        new{T}(model, qK_grh, change_qK_idx, 
            sources, sinks, 
            paths_dict,
            rgm_paths, path_polys, path_volume,
            path_volume_is_calc, path_polys_is_calc)
    end
end





"""
    Bnc(; N=nothing, L=nothing, x_sym=nothing, q_sym=nothing, K_sym=nothing,
        Γ=nothing, Π=nothing, k=nothing, cat_x_idx=nothing) -> Bnc

Construct a binding network model from stoichiometry (`N`) or conservation (`L`)
matrices and optional symbol metadata. Catalysis data can be attached through
`Γ`, `Π`, and `k`.

# Keyword Arguments
- `N`: Stoichiometry matrix (reactions × species).
- `L`: Conservation matrix (totals × species).
- `x_sym`: Symbols for species concentrations.
- `q_sym`: Symbols for total concentrations.
- `K_sym`: Symbols for binding constants.
- `Γ`: Catalysis change matrix in qK space.
- `Π`: Catalysis index and coefficient matrix.
- `k`: Catalysis rate constants.
- `cat_x_idx`: Index of catalytic species.

# Returns
- A `Bnc` model with derived matrices and caches initialized.
"""
function Bnc(;N=nothing,L=nothing,
    x_sym=nothing,q_sym=nothing,K_sym=nothing,
    kwargs...
)::Bnc
    # if N is not provided, derive it from L, if provided, check its linear indenpendency
    
    N = isnothing(N) ? N_from_L(L) : N
    row_idx = independent_row_idx(N)
    r = length(row_idx)

    if isnothing(L)
        if r != size(N,1) @warn("N has been reduced from $r to $r_new rows, for linear dependent.") : nothing
            N = N[row_idx, :] # reduce N to independent rows
            if !isnothing(K_sym) && length(K_sym) == r
                K_sym = K_sym[row_idx] # reduce K_sym to independent rows 
            end
        end
        L = L_from_N(N)
    else # L is provided
        if r!= size(N,1) && size(N,1) +size(L,1) ==size(N,2)
            @warn "N is not full row rank and can't be reduced, numerical issures could happen"
        end
    end

    r,n = size(N)
    d = size(L,1)
    

    # Call the inner constructor
    # Number of variables in the binding network
    x_sym = isnothing(x_sym) ? Symbolics.variables(:x, 1:n) : name_converter(x_sym) # convert x_sym to a vector of symbols
    q_sym = isnothing(q_sym) ? Symbolics.variables(:q, 1:d) : name_converter(q_sym) # convert q_sym to a vector of symbols
    K_sym = isnothing(K_sym) ? Symbolics.variables(:K, 1:r) : name_converter(K_sym) # convert K_sym to a vector of symbols

    model = Bnc{Int}(N, L, x_sym, q_sym, K_sym, nothing)
    update_catalysis!(model; kwargs...)
    return model
end



"""
    update_catalysis!(bnc::Bnc; Γ=nothing, Π=nothing, k=nothing, cat_x_idx=nothing) -> Bnc

Attach or update catalysis data on a `Bnc` model in-place.

# Arguments
- `bnc`: Binding network model to update.

# Keyword Arguments
- `Γ`: Catalysis change matrix in qK space.
- `Π`: Catalysis index and coefficient matrix.
- `k`: Rate constants.
- `cat_x_idx`: Index of catalytic species.

# Returns
- The updated `bnc`.
"""
function update_catalysis!(model::Bnc;
    Γ::Union{<:AbstractMatrix{Int},Nothing}=nothing,
    Π::Union{<:AbstractMatrix{Int},Nothing}=nothing,
    k_sym::Union{<:AbstractVector,Nothing}=nothing,
    x_picked::Union{<:AbstractVector,Nothing}=nothing,
    q_picked::Union{<:AbstractVector,Nothing}=nothing,
    )
    if isnothing(Γ) && isnothing(Π)
        return nothing
    else 
        @assert !isnothing(Γ) && !isnothing(Π) "You shall provide both Γ and Π"
    end

    Π = if isnothing(x_picked)
            Π
        else
            x_idx = locate_sym_x.(Ref(model), x_picked)
            Π2 = zeros(Int, (size(Π,1),model.n))
            for (i,x) in enumerate(x_idx)
                Π2[:,x] .= Π[:, i]
            end
            Π2
        end

    if !isnothing(q_picked)
        q_idx = locate_sym_qK.(Ref(model), q_picked)
        new_order = vcat(q_idx, setdiff(1:model.d, q_idx))
        _change_q_L_order!(model, new_order) # reorder the q and L in the model to make the picked q first, since the catalysis will involve the first r_v q.
        _remove_regime_data!(model) # remove the cached regime data, since the regimes will be changed after reordering q and L.
    end

    k_sym = isnothing(k_sym) ? Symbolics.variables(:k, 1:size(Π,1)) : name_converter(k_sym)
    model.catalysis = CatalysisData(model,Γ,Π, k_sym)
    return nothing
end

function fix_bn_catalysis!(bn::Bnc, new_ord::Vector{Int},L_Γ::AbstractMatrix{Int})
    if new_ord !== collect(1:length(new_ord)) # no reording should be made
        _change_q_L_order!(bn, new_ord)

        @info "q is reordered to make catalysis-involving species first"
        d_dep = size(L_Γ,2)
        d_cat_full = length(new_ord)
        d_cat = d_cat_full - d_dep

        if d_dep >0
            @info "New conservation forms as catalysis involves"
            #update the name of q_sym to make the first d_cat are q_cat, and the rest are q_dep
            bn.q_sym[(d_cat+1):d_cat_full] = Symbolics.variables(:w, 1:d_dep)
            # Calculate the L_w
            L_w = L_Γ' * bn.L[1:d_cat_full,:]
            @assert all(L_w .>=0) "L_w should be non-negative"
            #update L_w to replace L_dep
            bn.L[(d_cat+1):d_cat_full,:] = L_w
        end

        dropzeros!(bn.L)
        _remove_regime_data!(bn) # remove the cached regime data, since the regimes will be changed.
        #other initializing
    end

    _rebuild_helper!(bn) # rebuild the helper parameters since L has been changed.
    return nothing
end


@inline function _change_q_L_order!(bn::Bnc, new_ord::Vector{Int})
    bn.q_sym[1:length(new_ord)] = bn.q_sym[new_ord]
    bn.L[1:length(new_ord),:] = bn.L[new_ord, :]
end

@inline function _rebuild_helper!(bn::Bnc)
    bn.direction = sign(det([bn.L;bn.N])) # recalculate the direction, since L has been changed.
    bn.IntegrationHelper = calc_integration_helper(bn.L, bn.N) # recalculate the integration helper, since L has been changed.
    bn._L_helper = _build_matrix_helper(bn.L)
    return nothing
end

@inline function _remove_regime_data!(bn::Bnc{T}) where T 
    bn.vertices_perm = T[]
    bn.vertices_perm_dict = Dict{Vector{T},Int}() # reset the vertices_perm_dict since the vertices_perm will be reset.
    bn.vertices_asymptotic_flag = Bool[]
    bn.vertices_nullity = T[]
    bn.vertices_graph = nothing
    bn.vertices_data = BindRegime[]
    bn._vertices_is_initialized = BitVector()
    bn._vertices_volume_is_calced = BitVector()
    # questionable whether to delete the following cache.
    bn._vertices_Nρ_inv_dict = Dict{Vector{T}, Tuple{SparseMatrixCSC{Float64, Int},T}}() # reset the Nρ_inv_dict since the vertices will be reset.
    return nothing
end


include(joinpath(@__DIR__,"helperfunctions.jl"))
include(joinpath(@__DIR__,"qK_x_mapping.jl"))
include(joinpath(@__DIR__,"volume_calc.jl"))
include(joinpath(@__DIR__,"numeric.jl"))
include(joinpath(@__DIR__,"find_matrix_vertex.jl")) # before regimes.jl
include(joinpath(@__DIR__,"regimes.jl"))
include(joinpath(@__DIR__,"regime_assign.jl"))
include(joinpath(@__DIR__,"symbolics.jl"))
include(joinpath(@__DIR__,"regime_graphs.jl"))
include(joinpath(@__DIR__,"visualize.jl"))
include(joinpath(@__DIR__,"old_api.jl"))


"""
    summary(bnc::Bnc) -> String

Print a summary of a binding network model to standard output.
"""
function summary(Bnc::Bnc)
    println("----------Binding Network Summary:-------------")
    println("Number of species (n): ", Bnc.n)
    println("Number of conserved quantities (d): ", Bnc.d)
    println("Number of reactions (r): ", Bnc.r)
    println("L matrix: ", Bnc.L)
    println("N matrix: ", Bnc.N)
    println("Direction of binding reactions: ", Bnc.direction > 0 ? "forward" : "backward")
    catalysis_str = isnothing(Bnc.catalysis) ? "No" : "Yes"
    println("Catalysis involved: ", catalysis_str)
    is_regimes_built = isempty(Bnc.vertices_perm) ? "No" : "Yes"
    println("Regimes constructed: ", is_regimes_built)
    if !isempty(Bnc.vertices_perm)
        map = zip(Bnc.vertices_asymptotic_flag, Bnc.vertices_nullity .> 0) |> countmap
        println("Number of regimes: ", length(Bnc.vertices_perm))
        println("  - Invertible + Asymptotic: ", get(map, (true, false), 0))
        println("  - Singular +  Asymptotic: ", get(map, (true, true), 0))
        println("  - Invertible +  Non-Asymptotic: ", get(map, (false, false), 0))
        println("  - Singular +  Non-Asymptotic: ", get(map, (false, true), 0))
    end
    println("-----------------------------------------------")
end

"""
    show(io::IO, ::MIME"text/plain", bnc::Bnc)

Pretty-print a `Bnc` model in plain text contexts.
"""
function show(io::IO, ::MIME"text/plain", bnc::Bnc)
    println(io, "----------Binding Network Summary:-------------")
    println(io, "Number of species (n): ", bnc.n)
    println(io, "Number of conserved quantities (d): ", bnc.d)
    println(io, "Number of reactions (r): ", bnc.r)
    println(io, "L matrix: ", bnc.L)
    println(io, "N matrix: ", bnc.N)
    println(io, "Direction of binding reactions: ", bnc.direction > 0 ? "forward" : "backward")
    catalysis_str = isnothing(bnc.catalysis) ? "No" : "Yes"
    println(io, "Catalysis involved: ", catalysis_str)
    is_regimes_built = isempty(bnc.vertices_perm) ? "No" : "Yes"
    println(io, "Regimes constructed: ", is_regimes_built)
    if !isempty(bnc.vertices_perm)
        map = zip(bnc.vertices_asymptotic_flag, bnc.vertices_nullity .> 0) |> countmap
        println(io, "Number of regimes: ", length(bnc.vertices_perm))
        println(io, "  - Invertible + Asymptotic: ", get(map, (true, false), 0))
        println(io, "  - Singular +  Asymptotic: ", get(map, (true, true), 0))
        println(io, "  - Invertible +  Non-Asymptotic: ", get(map, (false, false), 0))
        println(io, "  - Singular +  Non-Asymptotic: ", get(map, (false, true), 0))
    end
    print(io, "-----------------------------------------------") # 最后一行可用 print 避免额外空行
end