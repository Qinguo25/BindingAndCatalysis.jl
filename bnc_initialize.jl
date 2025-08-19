
# using GLMakie
# using Plots
using Symbolics
using Parameters
using LinearAlgebra
using DifferentialEquations
using StatsBase
using SparseArrays
using JuMP
# using Interpolations

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


mutable struct Vertex{T}
    # --- Initial / Identifying Properties ---
    perm::Vector{Int} # The regime vector
    idx::Int # Index of the vertex in the Bnc.vertices list
    real::Bool # Whether the vertex is real or fake vertex.
    
    # --- Basic Calculated Properties ---
    P::Matrix{Int}
    P0::Vector{T} # Changed from Num to T
    M::Matrix{Int}
    M0::Vector{T} # Changed from Num to T
    C_x::Matrix{Int}
    C0_x::Vector{T} # Changed from Num to T

    # LU decomposition of M
    _M_lu::LU

    # --- Expensive Calculated Properties ---
    singularity::Int
    H::Matrix{Float64}
    H0::Vector{T} # Changed from Num to T
    C_qK::Matrix{Float64}
    C0_qK::Vector{T} # Changed from Num to T
    
    #--- Neighbors ---
    neighbors_idx::Vector{Int}
    finite_neighbors_idx::Vector{Int}
    infinite_neighbors_idx::Vector{Int}

    # The inner constructor also needs to be updated for the parametric type
    function Vertex{T}(;perm, P, P0, M, M0, C_x, C0_x, idx,real) where {T<:Real}
        _M_lu = lu(Float64.(M), check=false) # It's good practice to ensure M is Float64 for LU
        singularity = issuccess(_M_lu) ? 0 : -1
        
        # Use new{T} to construct an instance of Vertex{T}
        new{T}(perm, idx,real, P, P0, M, M0, C_x, C0_x, _M_lu, singularity,
            Matrix{Float64}(undef, 0, 0), # H
            Vector{T}(undef, 0),          # H0
            Matrix{Float64}(undef, 0, 0), # C_qK
            Vector{T}(undef, 0),          # C0_qK
            Int[], # neighbors_idx (using empty literal is cleaner)
            Int[], # finite_neighbors_idx
            Int[]  # infinite_neighbors_idx
        )
    end
end

mutable struct Bnc
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
    vertices_all::Vector{Vector{Int}} # all feasible regimes.
    vertices_real_idx::Vector{Int} # Index of those are real vertices.
    vertices_singularity::Vector{Int} # All vertices' singularity.
    vertices_idx::Dict{Vector{Int},Int} # map from permutation vector to its idx in the vertices list
    
    vertices_distance::Matrix{Int} # distance between vertices
    vertices_change_dir_x::Matrix{Int} # how the vertices should change under x space to reach its neighbor vertices.
    vertices_change_dir_qK::Matrix{Int} # how the vertices should change under qK space to reach its neighbor vertices.

    vertices_data::Dict{Vector{Int},Any} # Using Any for placeholder for Vertex

    #------other helper parameters------
    direction::Int8 # direction of the binding reactions, determine the ray direction for invertible regime, calculated by sign of det[L;N]

    # Parameters act as the starting points used for qk mapping
    _anchor_log_x::Vector{<:Real}
    _anchor_log_qK::Vector{<:Real}

    #Parameters for mimic calculation process
    _is_change_of_K_involved::Bool  # whether the K is involved in the calculation process

    
    
    # sparse matrix for speeding up the calculation
    _Lt_sparse::SparseMatrixCSC{Float64,Int} # sparse version of L transpose, used for fast calculation
    _LNt_sparse::SparseMatrixCSC{Float64,Int} # sparse version of [L;N]^t, used for fast calculation
    _LNt_lu::SparseArrays.UMFPACK.UmfpackLU{Float64,Int} # LU decomposition of _LNt_sparse, used for fast calculation
    _I::Vector{Int}
    _J::Vector{Int}
    _V::Vector{Float64} # sparse matrix for speeding up the calculation
    _val_num_L::Int # number of non-zero elements in the sparse matrix L

    _Nt_sparse::SparseMatrixCSC{Float64,Int} # sparse version of N transpose, used for fast calculation
    _IN::Vector{Int} # row indices of non-zero elements in _Nt_sparse
    _JN::Vector{Int} # column indices of non-zero elements in _Nt_sparse
    _VN::Vector{Float64} # values of non-zero elements in _Nt_sparse
    _valid_L_idx::Vector{Vector{Int}} #record the non-zero position for L

    _L_sparse_val_one::SparseMatrixCSC{Int,Int} # sparse version of L with only non-zero elements set to 1, used for fast calculation

    # Inner constructor 
    function Bnc(N, L, x_sym, q_sym, K_sym, catalysis)
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
        _valid_L_idx = [findall(!iszero, @view L[i,:]) for i in 1:d] 
        
        # Create Sparse matrices for further fast calculation
        _Lt_sparse = sparse(L') # sparse version of L transpose
        _I,_J,_V = findnz(_Lt_sparse) # get the row, column and value of the sparse matrix
        _val_num_L = length(_V) # number of non-zero elements in the sparse matrix

        _Nt_sparse = sparse(N') # sparse version of N transpose, used for fast calculation
        _IN, _JN, _VN = findnz(_Nt_sparse) # get the row, column and value of the sparse matrix

        # hcat for sparse matrices requires them to be the same type
        _LNt_sparse = hcat(sparse(Float64.(_Lt_sparse)), sparse(Float64.(_Nt_sparse))) 
        _LNt_lu = lu(_LNt_sparse) # LU decomposition of _LNt_sparse, used for fast calculation

        # Create a sparse matrix for L with only non-zero elements set to 1
        _L_sparse_val_one = sparse(sign.(L)) # sparse version of L with

        # --- FIX IS HERE ---
        # Call `new` with positional arguments, not keyword arguments.
        # The order must match the field order in the struct definition.
        new(
            # Fields 1-5
            N, L, r, n, d,
            # Fields 6-9
            x_sym, q_sym, K_sym, catalysis,
            # Fields 10-12 (Initialized empty)
            Vector{Vector{Int}}[],                # vertices_all
            Int[],                          # vertices_real_idx
            Int[],                          # vertices_singularity
            Dict{Vector{Int},Int}(),            # verices_idx
            Matrix{Int}(undef, 0, 0),             # vertices_distance
            Matrix{Int}(undef, 0, 0),             # vertices_change_dir_x
            Matrix{Int}(undef, 0, 0),             # vertices_change_dir_qK
            Dict{Vector{Int}, Any}(),              # vertices_data
            # Fields 13-28 (Calculated values)
            direction,
            _anchor_log_x, _anchor_log_qK,
            _is_change_of_K_involved,
            
            _Lt_sparse, _LNt_sparse, _LNt_lu,
            _I, _J, _V, _val_num_L,
            _Nt_sparse, _IN, _JN, _VN,
            _valid_L_idx,
            _L_sparse_val_one
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

    Bnc(N, L, x_sym, q_sym, K_sym, catalysis_data)
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



