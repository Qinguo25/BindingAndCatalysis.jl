
# using GLMakie
# using Plots
using Symbolics
using Parameters
using LinearAlgebra
using DifferentialEquations
using StatsBase
using SparseArrays
# using Interpolations

include("bnc_numeric.jl")
include("bnc_helperfunctions.jl")
include("bnc_regimes.jl")
include("bnc_symbolics.jl")

# ---------------------Define the struct of binding and catalysis networks----------------------------------

mutable struct Bnc
    # Parameters of the binding networks
    N::Matrix{Int} # binding reaction matrix
    L::Matrix{Int} # conservation law matrix

    r::Int # number of reactions
    n::Int # number of variables
    d::Int # number of conserved quantities

    x_sym::Vector{Num} # species symbols, each column is a species
    q_sym::Vector{Num}
    K_sym::Vector{Num}

    direction::Int8 # direction of the binding reactions, determine the ray direction for invertible regime, calculated by sign of det[L;N]


    # Parameters for the catalysis networks
    S::Union{Matrix{Int},Nothing} # catalysis change in qK space, each column is a reaction
    aT::Union{Matrix{Int},Nothing} # catalysis index and coefficients, rate will be vⱼ=kⱼ∏xᵢ^aT_{j,i}, denote what species catalysis the reaction.

    k::Union{Vector{<:Real},Nothing} # rate constants for catalysis reactions

    r_cat::Union{Int,Nothing} # number of catalysis reactions/species



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
    _val_num::Int # number of non-zero elements in the sparse matrix

    _Nt_sparse::SparseMatrixCSC{Float64,Int} # sparse version of N transpose, used for fast calculation
    _IN::Vector{Int} # row indices of non-zero elements in _Nt_sparse
    _JN::Vector{Int} # column indices of non-zero elements in _Nt_sparse
    _VN::Vector{Float64} # values of non-zero elements in _Nt_sparse
    
    _valid_L_idx::Vector{Vector{Int}} #record the non-zero position for L

    # Inner constructor 
    function Bnc(N, L, x_sym, q_sym, K_sym, S, aT, k)
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
        direction = sign(det([L;N]))

        # Validate dimensions for catalysis network
        #check if catalysis networks paramets legal
        if ~isnothing(S)
            n_S, r_cat = size(S)
            @assert n_S == n "S must have the same number of columns as N"
        else
            r_cat = nothing
        end

        if ~isnothing(aT)
            r_aT, n_aT = size(aT)
            if ~isnothing(r_cat)
                @assert r_aT == r_cat "aT must have the same number of rows as r_cat"
            else
                r_cat = r_aT
            end
            @assert n_aT == n "aT must have the same number of rows, as columns of N"
        end

        if ~isnothing(k)
            @assert isnothing(r_cat) || length(k) == r_cat "k must have the same length as r_cat"
        end


        #helper parameters 
        _anchor_log_x = zeros(n)
        _anchor_log_qK = vcat(vec(log10.(sum(L; dims=2))), zeros(r))
        _is_change_of_K_involved = isnothing(S) || !all(@view(S[r+1:end, :]) .== 0)
        _valid_L_idx = [findall(x->x!=0, @view L[i,:]) for i in 1:d] #record the non-zero position for L
        
        # Create sparse matrices for fast calculation
        _Lt_sparse = sparse(L') # sparse version of L transpose
        _I,_J,_V = findnz(_Lt_sparse) # get the row, column and value of the sparse matrix
        _val_num = length(_V) # number of non-zero elements in the sparse matrix

        _Nt_sparse = sparse(N') # sparse version of N transpose, used for fast calculation
        _IN, _JN, _VN = findnz(_Nt_sparse) # get the row, column and value of the sparse matrix

        _LNt_sparse = sparse_hcat(_Lt_sparse, _Nt_sparse) # sparse version of [L;N]^t
        _LNt_lu = lu(_LNt_sparse) # LU decomposition of _LNt_sparse, used for fast calculation
        # Create the new object with all fields specified
        new(N, L, 
            r, n, d,
            x_sym, q_sym, K_sym,
            direction,
            S, aT, k, r_cat,
            _anchor_log_x,_anchor_log_qK,
            _is_change_of_K_involved,
            _Lt_sparse,
            _LNt_sparse,
            _LNt_lu,
            _I, _J, _V, _val_num,
            _Nt_sparse, _IN, _JN, _VN,
            _valid_L_idx
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
    k::Union{Vector{<:Real},Nothing}=nothing
)::Bnc
    isnothing(N) ? (N = N_from_L(L)) : begin # if N is not provided, derive it from L
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

    #fufill S matrix if S doesn't invovling K 
    if ~isnothing(S)
        n = size(N, 2)
        (nrow_S, ncol_S) = size(S)
        if nrow_S < n
            # If S is not provided or has fewer columns than n, fill it with zeros, make sure S has n rows.
            S = vcat(S, zeros(Int64, n - nrow_S, ncol_S))
        end
    end

    # @show N,L,x_sym,q_sym,K_sym,S,aT,k
    Bnc(N, L, x_sym, q_sym, K_sym, S, aT, k)
end



# fill_name(appendix::String, count::Int=1)::Vector{Symbol} = [Symbol(appendix * string(i)) for i in 1:count]
# fill_name(appendix::String, original_name::Union{Vector{Symbol}}=nothing) = [Symbol(appendix * string(name)) for name in original_name]
# fill_name(appendix::Symbol, args...) = fill_name(string(appendix), args...)





