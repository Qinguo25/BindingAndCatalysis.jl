
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

import JSON3
import ImageFiltering: imfilter, Kernel
# ---------------------Define the struct of binding and catalysis networks----------------------------------


"""
Catalysis data structure
"""
struct CatalysisData
    # Parameters for the catalysis networks
    S::Matrix{Int} # catalysis change in qK space, each column is a reaction
    aT::Matrix{Int} # catalysis index and coefficients, rate will be vⱼ=kⱼ∏xᵢ^aT_{j,i}, denote what species catalysis the reaction.

    k::Vector{<:Real} # rate constants for catalysis reactions
    cat_x_idx::Vector{Int} # index of the species that catalysis the reaction, if not provided, will be inferred from S

    r_cat::Int # number of catalysis reactions/species
    _S_sparse::SparseMatrixCSC{Float64,Int} # sparse version of S, used for fast calculation
    _aT_sparse::SparseMatrixCSC{Float64,Int}  # sparse version of aT, used for fast calculation

    function CatalysisData(n::Int, S, aT, k, cat_x_idx)
        
        # fufill aT and cat_x_idx, either derive one from another
        if isnothing(aT) && !isnothing(cat_x_idx)
            println("catalysis coefficients is set to 1 for all catalysis species as aT is not provided.")
            aT = _idx_val2Mtx(cat_x_idx, 1, n)
        elseif isnothing(cat_x_idx) && !isnothing(aT)
            cat_x_idx, _ = _Mtx2idx_val(aT)
        elseif !isnothing(aT) && !isnothing(cat_x_idx)
            tmp,_ = _Mtx2idx_val(aT)
            @assert tmp == cat_x_idx "cat_x_idx must be the same as the index of aT"
        end 

        if isnothing(S)
            @error "S must be provided"
        end
        if isnothing(aT)
            @error "aT or cat_x_idx must be provided"
        end

        #fufill k if not provided.
        if isnothing(k) 
            k = ones(size(aT, 1))
            @warn("k is not provided, initialized to ones")
        end

        #fufill S to fulllength to contain K, could be nothing
        if !isnothing(S) && size(S, 1) < n
             S = vcat(S, zeros(Int, n - size(S, 1), size(S, 2)))
        end
        
        # Validation
        n_cat,r_cat = size(S)
        @assert n_cat == n "S should have n rows"
        @assert size(aT, 1) == r_cat "Mismatch in catalysis reaction count"
        @assert size(aT,2) == n "Mismatch catalysis species, aT should have n columns"
        @assert length(k) == r_cat "Mismatch in catalysis reaction count"

        # Create sparse matrices
        _S_sparse = sparse(Float64.(S))
        _aT_sparse = sparse(Float64.(aT))

        new(S, aT, k, cat_x_idx, r_cat, _S_sparse, _aT_sparse)
    end
end


mutable struct Vertex{F,T}
    # --- Initial / Identifying Properties ---
    perm::Vector{T} # The regime vector
    idx::Int # Index of the vertex in the Bnc.vertices list
    real::Bool # Whether the vertex is real or fake vertex.
    
    # --- Basic Calculated Properties ---
    P::SparseMatrixCSC{Int, Int}
    P0::Vector{F} 
    M::SparseMatrixCSC{Int, Int}
    M0::Vector{F} #
    C_x::SparseMatrixCSC{Int, Int}
    C0_x::Vector{F} 

    # LU decomposition of M
    # _M_lu::LU
    # _M_lu::SparseArrays.UMFPACK.UmfpackLU{Float64, Int}

    # --- Expensive Calculated Properties ---
    nullity::T
    H::SparseMatrixCSC{Float64, Int} # Taking inverse, can have Float.
    H0::Vector{F} 
    C_qK::SparseMatrixCSC{Float64, Int}
    C0_qK::Vector{F} 
    
    #--- Neighbors ---
    neighbors_idx::Vector{Int}
    # finite_neighbors_idx::Vector{Int}
    # infinite_neighbors_idx::Vector{Int}

    # The inner constructor also needs to be updated for the parametric type
    function Vertex{F,T}(;perm, P, P0, M, M0, C_x, C0_x, idx,real,nullity) where {F<:Real ,T<:Integer}
        # _M_lu = lu(M, check=false) # It's good practice to ensure M is Float64 for LU
        # Use new{T} to construct an instance of Vertex{T}
        new{F,T}(perm, idx,real, P, P0, M, M0, C_x, C0_x,
            # _M_lu,
            nullity,
            SparseMatrixCSC{Float64, Int}(undef, 0, 0), # H
            Vector{F}(undef, 0),          # H0
            SparseMatrixCSC{Float64, Int}(undef, 0, 0), # C_qK
            Vector{F}(undef, 0),          # C0_qK
            Int[], # neighbors_idx (using empty literal is cleaner)
            # Int[], # finite_neighbors_idx
            # Int[]  # infinite_neighbors_idx
        )
    end
end

mutable struct VertexEdge{T}
    to::Int
    diff_r::Int
    change_dir_x::SparseVector{Int8, T}
    change_dir_qK::Union{Nothing, SparseVector{Float64, T}}
    function VertexEdge(to::Int, diff_r::Int, change_dir_x::SparseVector{Int8, T}) where {T}
        return new{T}(to, diff_r, change_dir_x, nothing)
    end
end

# Adjacency list + optional caches
mutable struct VertexGraph{T} 
    neighbors::Vector{Vector{VertexEdge{T}}}
    change_dir_qK_computed::Bool
    # nullity::Union{Nothing, Vector{T}} # optional cache of nullity for each vertex #Store nullity again.
    function VertexGraph(neighbors::Vector{Vector{VertexEdge{T}}}) where {T}
        new{T}(neighbors,false)
    end
end

mutable struct Bnc{T} # T is the int type to save all the indices
    # ----Parameters of the binding networks------
    N::Matrix{Int} # binding reaction matrix
    L::Matrix{Int} # conservation law matrix

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
    vertices_perm::Vector{Vector{T}} # all feasible regimes.
    vertices_perm_dict::Dict{Vector{T},Int} # map from permutation vector to its idx in the vertices list
    vertices_asymptotic_flag::Vector{Bool} # While this vertice is real
    vertices_nullity::Vector{T} # nullity of one vertex.
    vertices_graph::Union{Any,Nothing} # Using Any for placeholder for VertexGraph

    # vertices_neighbor_mat::SparseMatrixCSC{T, Int} # distance between vertices, upper triangular
    # vertices_change_dir_x::SparseMatrixCSC{SparseVector{Int8,T}, Int} # how the vertices should change under x space to reach its neighbor vertices, upper triangular
    # vertices_change_dir_qK::SparseMatrixCSC{SparseVector{Float64,T}, Int} # how the vertices should change under qK space to reach its neighbor vertices, upper triangular
    # # _vertices_sym_invperm::Vector{Int}

    vertices_data::Dict{Vector{T},Any} # Using Any for placeholder for Vertex
    _vertices_Nρ_inv_dict::Dict{Vector{T}, Tuple{SparseMatrixCSC{Float64, Int},T}} # cache the N_inv for each vertex permutation

    #------other helper parameters------
    direction::Int8 # direction of the binding reactions, determine the ray direction for invertible regime, calculated by sign of det[L;N]

    # Parameters act as the starting points used for qk mapping
    _anchor_log_x::Vector{<:Real}
    _anchor_log_qK::Vector{<:Real}

    #Parameters for mimic calculation process
    _is_change_of_K_involved::Bool  # whether the K is involved in the calculation process

    
    
    # sparse matrix for speeding up the calculation
    _L_sparse::SparseMatrixCSC{Int,Int} # sparse version of L, used for fast calculation
    _L_sparse_val_one::SparseMatrixCSC{Int,Int} # sparse version of L with only non-zero elements set to 1, used for fast calculation
    _valid_L_idx::Vector{Vector{Int}} #record the non-zero column position for each row.

    _N_sparse::SparseMatrixCSC{Int,Int} # sparse version of N transpose, used for fast calculation
    _LN_sparse::SparseMatrixCSC{Float64,Int} # sparse version of [L;N], used for fast calculation

    #------------below are helper parameters for fast updating  value of matrix of the form [L;N] ------------------
    _LN_top_idx::Vector{Int} # first d row index of _LN_sparse
    _LN_top_rows::Vector{Int} # the corresponding row number in L for _LN_top_idx
    _LN_top_cols::Vector{Int} # the corresponding column number in L for _LN_top_idx

    _LN_bottom_idx::Vector{Int} # last r row index of _LN_sparse
    _LN_bottom_rows::Vector{Int} # the corresponding row number in N for _LN_bottom_idx
    _LN_bottom_cols::Vector{Int} # the corresponding column number in N for _LN_bottom_idx
    _LN_top_diag_idx::Vector{Int} # the diagonal index of the top d rows of _LN_sparse, used for fast calculation

    _LN_lu::SparseArrays.UMFPACK.UmfpackLU{Float64,Int} # LU decomposition of _LNt_sparse, used for fast calculation
    # _val_num_L::Int # number of non-zero elements in the sparse matrix L
    

    # Inner constructor 
    function Bnc{T}(N, L, x_sym, q_sym, K_sym, catalysis) where {T<:Integer}
        # get desired values
        r, n = size(N)
        d, n_L = size(L)

        # Validate dimensions for binding network, check if its legal.
        @assert n == d + r "d+r is not equal to n"
        @assert n_L == n "L must have the same number of columns as N"

        @assert length(x_sym) == n "x_sym length must equal number of species (n)"
        @assert length(q_sym) == d "q_sym length must equal number of conserved quantities (d)"
        @assert length(K_sym) == r "K_sym length must equal number of reactions (r)"

        #The direction
        direction = sign(det(Float64.([L;N]))) # Ensure matrix is Float64 for det
        
        # A simplified check for catalysis.S - replace with your actual logic
        _is_change_of_K_involved = !isnothing(catalysis) #&& !all(@view(catalysis.S[r+1:end, :]) .== 0)

        #-------helper parameters-------------
        # paramters for default homotopcontinuous starting point.
        _anchor_log_x = zeros(n)
        _anchor_log_qK = vcat(vec(log10.(sum(L; dims=2))), zeros(r))

        # pre-calculate the non-zero position for L

        _L_sparse = sparse(L) # sparse version of L
        _L_sparse_val_one = sparse(sign.(L)) # sparse version of L with only non-zero elements set to 1
        _valid_L_idx = [findall(!iszero, @view L[i,:]) for i in 1:d] 
        
        _N_sparse = sparse(N) # sparse version of N
        _LN_sparse = Float64.([_L_sparse; _N_sparse])
        (_LN_top_rows, _LN_top_cols, _LN_top_idx) = rowmask_indices(_LN_sparse, 1,d) # record the position of non-zero elements in L within _LN_sparse
        (_LN_bottom_rows, _LN_bottom_cols, _LN_bottom_idx) = rowmask_indices(_LN_sparse, d+1,n) # record the position of non-zero elements in N within _LN_sparse
        _LN_top_diag_idx = diag_indices(_LN_sparse, d)

        _LN_lu = lu(_LN_sparse) # LU decomposition of _LNt_sparse, used for fast calculation
        # _N_sparse = sparse(N) # sparse version of N, used for fast calculation

        new(
            # Fields 1-5
            N, L, r, n, d,
            # Fields 6-9
            x_sym, q_sym, K_sym, catalysis,
            # Fields 10-12 (Initialized empty)
            Vector{T}[],                # vertices_perm
            Dict{Vector{T},Int}(),            # verices_idx
            Bool[],                          # vertices_asymptotic_flag
            T[],                          # vertices_nullity
            nothing,                         # vertices_graph
            # SparseMatrixCSC{Bool, Int}(undef, 0, 0),             # vertices_neighbor_mat
            # SparseMatrixCSC{SparseVector{Int8,T}, Int}(undef, 0, 0),             # vertices_change_dir_x
            # SparseMatrixCSC{SparseVector{Float64,T}, Int}(undef, 0, 0),             # vertices_change_dir_qK
            # Int[],                           # _vertices_sym_invperm
            Dict{Vector{T}, Any}(),              # vertices_data
            Dict{Vector{T}, Tuple{SparseMatrixCSC{Float64, Int},T}}(), # _vertices_perm_Ninv_dict
            # Fields 13-28 (Calculated values)
            direction,
            _anchor_log_x, _anchor_log_qK,
            _is_change_of_K_involved,

            _L_sparse,
            _L_sparse_val_one,
            _valid_L_idx,

            _N_sparse,
            _LN_sparse,

            _LN_top_idx,_LN_top_rows,_LN_top_cols,
            _LN_bottom_idx,_LN_bottom_rows,_LN_bottom_cols,
            _LN_top_diag_idx,

            _LN_lu,
        )
    end
end







# Define a separate outer function for keyword-based construction, needs to be refine later.
function Bnc(;
    N::Union{Matrix{Int},Nothing}=nothing,
    L::Union{Matrix{Int},Nothing}=nothing,
    x_sym::Union{Vector{<:Any},Nothing}=nothing,
    q_sym::Union{Vector{<:Any},Nothing}=nothing,
    K_sym::Union{Vector{<:Any},Nothing}=nothing,
    S::Union{Matrix{Int},Nothing}=nothing,
    aT::Union{Matrix{Int},Nothing}=nothing,
    k::Union{Vector{<:Real},Nothing}=nothing,
    cat_x_idx::Union{Vector{Int},Nothing}=nothing,
)::Bnc
    # if N is not provided, derive it from L, if provided, check its linear indenpendency
    isnothing(N) ? (N = N_from_L(L)) : begin 
        r = size(N,1)
        row_idx = independent_row_idx(N)
        r_new = length(row_idx)
        r != r_new ? @warn("N has been reduced from $r to $r_new rows, for linear dependent.") : nothing
        N = N[row_idx, :] # reduce N to independent rows
        if !isnothing(K_sym) && length(K_sym) == r
            K_sym = K_sym[row_idx] # reduce K_sym to independent rows 
        end
    end

    !isnothing(L) || (L = L_from_N(N)) # if L is not provided, derive it from N

    r,n = size(N)
    d = size(L, 1)

    # Call the inner constructor
    # Number of variables in the binding network
    x_sym = isnothing(x_sym) ? Symbolics.variables(:x, 1:n) : name_converter(x_sym) # convert x_sym to a vector of symbols
    q_sym = isnothing(q_sym) ? Symbolics.variables(:q, 1:d) : name_converter(q_sym) # convert q_sym to a vector of symbols
    K_sym = isnothing(K_sym) ? Symbolics.variables(:K, 1:r) : name_converter(K_sym) # convert K_sym to a vector of symbols

    local catalysis_data::Union{CatalysisData,Nothing}
    if !isnothing(S) || !isnothing(aT) || !isnothing(k) || !isnothing(cat_x_idx)
        catalysis_data = CatalysisData(n, S, aT, k, cat_x_idx)
    else
        catalysis_data = nothing
    end

    T = get_int_type(n) 
    Bnc{T}(N, L, x_sym, q_sym, K_sym, catalysis_data)
end





function update_catalysis!(Bnc::Bnc;
    S::Union{Matrix{Int},Nothing}=nothing,
    aT::Union{Matrix{Int},Nothing}=nothing,
    k::Union{Vector{<:Real},Nothing}=nothing,
    cat_x_idx::Union{Vector{Int},Nothing}=nothing,
    )
    """
    Updates the catalysis data of a `Bnc` object in-place

    # Arguments
    - `bnc::Bnc`: The binding network object to modify.

    # Keyword Arguments
    - `S::Matrix{Int}`: The new catalysis change matrix.
    - `aT::Matrix{Int}`: The new catalysis index and coefficient matrix.
    - `k::Vector{<:Real}`: The new rate constants.
    - `cat_x_idx::Vector{Int}`: The new index of catalytic species.

    Any fields left as `nothing` will not be updated unless a new `CatalysisData`
    object needs to be created.

    """
    if isnothing(bnc.catalysis)
        bnc.catalysis = CatalysisData(bnc.n, S, aT, k, cat_x_idx)
    else
        S = isnothing(S) ? bnc.catalysis.S : S
        aT = isnothing(aT) ? bnc.catalysis.aT : aT
        k = isnothing(k) ? bnc.catalysis.k : k
        cat_x_idx = isnothing(cat_x_idx) ? bnc.catalysis.cat_x_idx : cat_x_idx
        bnc.catalysis = CatalysisData(bnc.n, S, aT, k, cat_x_idx)
    end
end



include("bnc_helperfunctions.jl")
include("bnc_numeric.jl")
include("bnc_regimes.jl")
include("bnc_symbolics.jl")
include("bnc_graphs.jl")




