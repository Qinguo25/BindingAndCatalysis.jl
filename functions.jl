
# using GLMakie
# using Plots
using Symbolics
using Parameters
using LinearAlgebra
using DifferentialEquations
using StatsBase
using SparseArrays

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
        _anchor_log_qK = vcat(vec(log.(sum(L; dims=2))), zeros(r))
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


    !isnothing(L) || (L = L_from_N(N)) # if L is not provided, derive it from N
    !isnothing(N) || (N = N_from_L(L)) # if N is not provided, derive it from L

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

#helper function to convert N matrix to L matrix if L is not provided.
function L_from_N(N::Matrix{Int})::Matrix{Int}
    r, n = size(N)
    d = n - r
    N_1 = @view N[:, 1:d]
    N_2 = @view N[:, d+1:n]
    hcat(Matrix(I, d, d), -(N_2 \ N_1)')
end

function N_from_L(L::Matrix{Int})::Matrix{Int}
    d, n = size(L)
    r = n - d
    L2 = @view L[:,d+1:n]
    hcat(L2',Matrix(-I,r,r))
end

function name_converter(name::Vector{<:T})::Vector{Num} where T 
    if T <: Num
        return name
    else
        return [Symbolics.variable(x; T=Real) for x in name]
    end
end

# fill_name(appendix::String, count::Int=1)::Vector{Symbol} = [Symbol(appendix * string(i)) for i in 1:count]
# fill_name(appendix::String, original_name::Union{Vector{Symbol}}=nothing) = [Symbol(appendix * string(name)) for name in original_name]
# fill_name(appendix::Symbol, args...) = fill_name(string(appendix), args...)

#----------------Functions for calculates the derivative of log(x) with respect to log(qK) and vice versa----------------------

function ∂logqK_∂logx(Bnc::Bnc;
    x::Union{Vector{<:Real},Nothing}=nothing,
    qK::Union{Vector{<:Real},Nothing}=nothing,
    q::Union{Vector{<:Real},Nothing}=nothing)::Matrix{<:Real}

    # 1. Ensure x is defined
    if isnothing(x)
        if isnothing(qK)
            error("Either x or qK must be provided")
        else
            x = qK2x(Bnc, qK) # Derive x from qK
        end
    end

    # 2. Ensure q is defined
    if isnothing(q)
        if isnothing(qK)
            # If qK is not provided but x is (which it must be by this point),
            # derive q from x
            q = Bnc.L * x
        else
            # If qK is provided, q is part of qK
            q = qK[1:Bnc.d]
        end
    end

    return [
        transpose(x) .* Bnc.L ./ q
        Bnc.N
    ]
end


function ∂logx_∂logqK(Bnc::Bnc;
    x::Union{Vector{<:Real},Nothing}=nothing,
    qK::Union{Vector{<:Real},Nothing}=nothing,
    q::Union{Vector{<:Real},Nothing}=nothing)::Matrix{<:Real}

    inv(∂logqK_∂logx(Bnc; x=x, q=q, qK=qK))
end

# ----------------Functions for mapping between qK space and x space----------------------------------

function x2qK(Bnc::Bnc, x::AbstractArray{<:Real,1};
    input_logspace::Bool=false,
    output_logspace::Bool=false,
)::Vector{<:Real}
    if input_logspace
        if output_logspace
            K = Bnc._Nt_sparse' * x
            q = log.(Bnc._Lt_sparse' * exp.(x))
        else
            K = exp.(Bnc._Nt_sparse' * x)
            q = Bnc._Lt_sparse' * exp.(x)
        end
    else
        if output_logspace
            K = Bnc._Nt_sparse' * log.(x)
            q = log.(Bnc._Lt_sparse' * x)
        else
            K = exp.(Bnc._Nt_sparse' * log.(x))
            q = Bnc._Lt_sparse' * x
        end
    end
    return vcat(q, K)
end

function qK2x(Bnc::Bnc, qK::AbstractArray{<:Real,1};
    input_logspace::Bool=false,
    output_logspace::Bool=false,
    startlogx::Union{Vector{<:Real},Nothing}=nothing,
    startlogqK::Union{Vector{<:Real},Nothing}=nothing,
    reltol=1e-8, abstol=1e-9)::Vector{<:Real}

    #---Solve the homotopy ODE to find x from qK.---

    # Define the start point 
    if isnothing(startlogqK) || isnothing(startlogx)
        # If no starting point is provided, use the default
        startlogx = Bnc._anchor_log_x
        startlogqK = Bnc._anchor_log_qK
    end

    endlogqK = input_logspace ? qK : log.(qK) # Convert qK to log space if not already

    sol = logx_traj_with_logqK_change(Bnc,
        startlogqK,
        endlogqK;
        startlogx=startlogx,
        alg=Tsit5(),
        reltol=reltol,
        abstol=abstol,
        save_everystep=false,
        save_start=false,
    )
    x = output_logspace ? sol.u[end] : exp.(sol.u[end])
    return x
end

#----------------Functions using homotopyContinuous to moving across x space along with qK change----------------------
#--------------Initial version------------
# struct HomotopyParams{V<:Vector{<:Real},M<:Matrix{<:Real},
#     SV1<:SubArray,SV2<:SubArray,SV3<:SubArray}
#     #ode required parameters
#     ΔlogqK::V

#     #value buffers
#     x::V
#     q::V
#     J::M  # Buffer for the Jacobian matrix

#     # value views
#     ##outer parameters value views 
#     startlogq::SV1 # View for startlogqK[1:d]
#     ##value buffer views
#     Δlogq::SV2     # View for ΔlogqK[1:d]
#     J_top::SV3     # View for J[1:Bnc.d, :]
# end
# function logx_traj_with_logqK_change(Bnc::Bnc,
#     startlogqK::Vector{<:Real},
#     endlogqK::Vector{<:Real};
#     # Optional parameters for the initial log(x) values
#     startlogx::Union{Vector{<:Real},Nothing}=nothing,
#     # Optional parameters for the ODE solver
#     alg=nothing, # Default to nothing, will use Tsit5() if not provided
#     reltol=1e-8,
#     abstol=1e-9,
#     kwargs... #other Optional arguments for ODE solver
# )::ODESolution

#     #---Solve the homotopy ODE to find x from qK.---

#     # Define the start point for the intergration
#     tspan = (0.0, 1.0)
#     d = Bnc.d
#     n = Bnc.n
#     startlogx = isnothing(startlogx) ? qK2x(Bnc, startlogqK; input_logspace=true, output_logspace=true) : startlogx
#     ΔlogqK = endlogqK - startlogqK

#     # Create the J_buffer here
#     # Assuming Bnc.N is a Matrix{<:Real} and compatible with zeros(d,n)
#     J_buffer = [zeros(d, n); Bnc.N] # This creates a new Matrix{<:Real}

#     # Create the views once here
#     startlogq = @view(startlogqK[1:d])
#     Δlogq = @view(ΔlogqK[1:d])
#     J_top = @view(J_buffer[1:Bnc.d, :])

#     params = HomotopyParams(
#         ΔlogqK, Vector{Float64}(undef, n), #x_buffer
#         Vector{Float64}(undef, d), #q_buffer
#         J_buffer, # Use the pre-created J_buffer
#         startlogq,
#         Δlogq,
#         J_top
#     )

#     function homotopy_ode!(dlogx, logx, p, t)
#         # logx is the current state vector, log(x(t))
#         # params contains constant parameters (startlogqK, dlogqK_vec)
#         # t is the current time/path parameter from 0 to 1
#         @unpack ΔlogqK, x, q, J, startlogq, Δlogq, J_top = p
#         #update q 
#         @. q = exp(startlogq + t * Δlogq)
#         #update x
#         @. x = exp(logx)
#         # Update the Jacobian (only top part needed)
#         @. J_top = x' * Bnc.L / q
#         dlogx .= J \ ΔlogqK
#         # @show dlogx
#         # return dlogx
#     end

#     # Solve the ODE using the DifferentialEquations.jl package
#     prob = ODEProblem(homotopy_ode!, startlogx, tspan, params)
#     sol = solve(prob, alg; reltol=reltol, abstol=abstol, kwargs...)
#     return sol
# end


struct HomotopyParams{V<:Vector{Float64},
    SV1<:SubArray,SV2<:SubArray,SV3<:SubArray,SV4<:SubArray,SV5<:SubArray}

    # Buffers
    ΔlogqK::V
    x::V
    q::V
    Jt::SparseMatrixCSC{Float64,Int}
    Jt_lu::SparseArrays.UMFPACK.UmfpackLU{Float64,Int}

    # Views into buffers
    x_view::SV1
    q_view::SV2
    startlogq::SV3
    Δlogq::SV4
    Jt_left::SV5
end

function logx_traj_with_logqK_change(Bnc::Bnc,
    startlogqK::Vector{<:Real},
    endlogqK::Vector{<:Real};
    # Optional parameters for the initial log(x) values
    startlogx::Union{Vector{<:Real},Nothing}=nothing,
    # Optional parameters for the ODE solver
    alg=nothing, # Default to nothing, will use Tsit5() if not provided
    reltol=1e-8,
    abstol=1e-9,
    kwargs... #other Optional arguments for ODE solver
)::ODESolution

    #---Solve the homotopy ODE to find x from qK.---

    #--Prepare parameters---
    startlogx = isnothing(startlogx) ? qK2x(Bnc, startlogqK; input_logspace=true, output_logspace=true) : startlogx
    
    ΔlogqK = endlogqK - startlogqK
    x = Vector{Float64}(undef, Bnc.n)
    q = Vector{Float64}(undef, Bnc.d)
    Jt = copy(Bnc._LNt_sparse)
    Jt_lu = copy(Bnc._LNt_lu) # LU decomposition of Jt, used for fast calculation

    x_view = @view x[Bnc._I]
    q_view = @view q[Bnc._J]
    startlogq = @view(startlogqK[1:Bnc.d])
    Δlogq = @view(ΔlogqK[1:Bnc.d])
    Jt_left = @view(Jt.nzval[1:Bnc._val_num]) # View for the top part of the Jacobian matrix

    params = HomotopyParams(
        ΔlogqK, 
        x, #x_buffer
        q, #q_buffer
        Jt, #Jt
        Jt_lu, #Jt_lu
        x_view,
        q_view,
        startlogq, #startlogq
        Δlogq, #Δlogq
        Jt_left # Jt_left
    )
    
    # Define the ODE system for the homotopy process
    function homotopy_process!(dlogx, logx, p, t)
        @unpack ΔlogqK, x, q, Jt, Jt_lu,x_view, q_view, startlogq, Δlogq, Jt_left = p
        #update q & x
        @. q = exp(startlogq + t * Δlogq)
        @. x = exp(logx)
        #update Jt(sparse version)
        @. Jt_left = x_view * Bnc._Lt_sparse.nzval / q_view  # Bnc._Lt_sparse.nzval = Bnc._V
        # Update the dlogx
        lu!(Jt_lu, Jt) # recalculate the LU decomposition of Jt
        ldiv!(dlogx, Jt_lu',ΔlogqK)
    end

    # Solve the ODE using the DifferentialEquations.jl package
    tspan = (0.0, 1.0)
    prob = ODEProblem(homotopy_process!, startlogx, tspan, params)
    sol = solve(prob, alg; reltol=reltol, abstol=abstol, kwargs...)
    return sol
end

#----------------Functions for modeling when envolving catalysis reactions----------------------

# original version
# struct TimecurveParam{V<:Vector{<:Real},M<:Matrix{<:Real},
#     SV1<:SubArray,SV2<:SubArray}
#     x::V # Buffer for x values
#     K::V # Buffer for K values
#     v::V # Buffer for the catalysis flux vector
#     Sv::V # Buffer for the catalysis rate vector multiplied by S
#     J::M # Jacobian matrix buffer

#     J_top::SV1 # View for the top part of the Jacobian matrix
#     J_bottom::SV2 # View for the bottom part of the Jacobian matrix
# end

# function time_curve_logx(Bnc::Bnc, logx0::Vector{<:Real}, tspan::Tuple{Real,Real};
#     k::Union{Vector{<:Real},Nothing}=nothing,
#     S::Union{Matrix{Int},Nothing}=nothing,
#     aT::Union{Matrix{Int},Nothing}=nothing,
#     override::Bool=false,
#     alg=nothing, # Default to nothing, will use Tsit5() if not provided
#     reltol=1e-8,
#     abstol=1e-9,
#     kwargs...
# )::ODESolution

#     # ---Solve the ODE to find the time curve of log(x) with respect to qK change.---

#     #--Prepare parameters---
#     if override
#         if ~isnothing(k)
#             Bnc.k = k
#         end
#         if ~isnothing(S)
#             Bnc.S = S
#         end
#         if ~isnothing(aT)
#             Bnc.aT = aT
#         end
#     end
#     k = coalesce(k, Bnc.k)
#     S = coalesce(S, Bnc.S)
#     aT = coalesce(aT, Bnc.aT)

#     #initialize J_buffer
#     J = Matrix{Float64}(undef, Bnc.n, Bnc.n)
#     # create view for the J_buffer
#     J_top = @view J[1:Bnc.d, :]
#     J_bottom = @view J[(Bnc.d+1):end, :]

#     params = TimecurveParam(
#         Vector{Float64}(undef, Bnc.n), # x_buffer
#         Vector{Float64}(undef, Bnc.r), # K_buffer
#         Vector{Float64}(undef, length(k)), # v buffer / flux
#         Vector{Float64}(undef, Bnc.n), # Sv buffer
#         J, # J_buffer
#         #Views for updating J
#         J_top,
#         J_bottom
#     )
#     # Define the ODE system for the time curve
#     if Bnc._is_change_of_K_involved
#         Catalysis_process! = function (dlogx, logx, p, t)
#             @unpack x, K, v, Sv, J, J_top, J_bottom = p
#             #update the values
#             x .= exp.(logx)
#             K .= exp.(Bnc.N * logx)
#             # Calculate the Jacobian matrix J
#             @. J_top = Bnc.L * x'
#             @. J_bottom = K * Bnc.N

#             # dlogx .= J \ (S * (k .* exp.(aT * logx)))
#             mul!(v, aT, logx)
#             @. v = k * exp(v) # calculate the catalysis rate vector
#             mul!(Sv, S, v) # reuse x as a temporary buffer, but need change if x is used in other places, like to act call back for ODESolution
#             dlogx .= J \ Sv
#         end
#     else
#         # If K is not involved, we can skip the K update
#         params.J_bottom .= exp.(Bnc.N * logx0) .* Bnc.N #initialize J_bottom once
#         Catalysis_process! = function (dlogx, logx, p, t)
#             @unpack x, v, Sv, J, J_top = p
#             #update the values
#             x .= exp.(logx)
#             # Update the Jacobian matrix J
#             @. J_top = Bnc.L * x'
#             # dlogx .= J \ (S * (k .* exp.(aT * logx))) # could be optimized by symbolics further.
#             mul!(v, aT, logx) # v = aT * logx
#             @. v = k * exp(v) # calculate the catalysis rate vector
#             mul!(Sv, S, v) # Sv = S * v
#             dlogx .= J \ Sv
#         end
#     end
#     # Create the ODE problem
#     prob = ODEProblem(Catalysis_process!, logx0, tspan, params)
#     sol = solve(prob, alg; reltol=reltol, abstol=abstol, kwargs...)
#     return sol
# end



# # original original version
# struct TimecurveParam2{V<:Vector{<:Real}, M<:Matrix{<:Real},
#                 SV1<:SubArray, SV2<:SubArray}
#     x::V # Buffer for x values
#     K::V # Buffer for K values
#     J::M # Jacobian matrix buffer
#     J_top::SV1 # View for the top part of the Jacobian matrix
#     J_bottom::SV2 # View for the bottom part of the Jacobian matrix
# end

# function time_curve_logx_origin(Bnc::Bnc,logx0::Vector{<:Real}, tspan::Tuple{Real, Real};
#     k::Union{Vector{<:Real},Nothing} = nothing,
#     S::Union{Matrix{Int},Nothing} = nothing,
#     aT::Union{Matrix{Int},Nothing} = nothing,
#     alg=nothing, # Default to nothing, will use Tsit5() if not provided
#     reltol=1e-8,
#     abstol=1e-9,
#     kwargs...
#     )::ODESolution

#     # ---Solve the ODE to find the time curve of log(x) with respect to qK change.---

#     #--Prepare parameters---
#     k = isnothing(k) ? Bnc.k : k
#     S = isnothing(S) ? Bnc.S : S
#     aT = isnothing(aT) ? Bnc.aT : aT

#     #initialize J_buffer
#     J = Matrix{Float64}(undef, Bnc.n, Bnc.n)
#     # create view for the J_buffer
#     J_top = @view J[1:Bnc.d, :]
#     J_bottom = @view J[(Bnc.d+1):end, :]

#     params = TimecurveParam2(
#         Vector{Float64}(undef, Bnc.n), # x_buffer
#         Vector{Float64}(undef, Bnc.r), # K_buffer
#         J, # J_buffer
#         J_top,
#         J_bottom
#     )
#     # Define the ODE system for the time curve
#     if Bnc._is_change_of_K_involved
#         Catalysis_process! = function(dlogx, logx, p, t)
#             @unpack x, K, J, J_top,J_bottom =p
#             #update the values
#             x .= exp.(logx)
#             K .= exp.(Bnc.N * logx)
#             # Calculate the Jacobian matrix J
#             @. J_top = Bnc.L * x'
#             @. J_bottom = K * Bnc.N
#             dlogx .= J \ (S * (k .* exp.(aT * logx)))
#         end
#     else
#         # If K is not involved, we can skip the K update
#         params.J_bottom .= exp.(Bnc.N * logx0) .* Bnc.N #initialize J_bottom once
#         Catalysis_process! = function(dlogx, logx, p, t)
#             @unpack x, J, J_top =p
#             #update the values
#             x .= exp.(logx)
#             @. J_top = Bnc.L * x'
#             dlogx .= J \ (S * (k .* exp.(aT * logx))) # could be optimized by symbolics further.
#         end
#     end
#     # Create the ODE problem
#     prob = ODEProblem(Catalysis_process!, logx0, tspan, params)
#     sol = solve(prob, alg; reltol=reltol, abstol=abstol,kwargs...)
#     return sol    
# end

# struct TimecurveParam{V<:Vector{<:Real},M<:Matrix{<:Real},
#     SV1<:SubArray,SV2<:SubArray}
#     x::V # Buffer for x values
#     K::V # Buffer for K values
#     v::V # Buffer for the catalysis flux vector
#     Sv::V # Buffer for the catalysis rate vector multiplied by S
#     J::M # Jacobian matrix buffer
#     J_top::SV1 # View for the top part of the Jacobian matrix
#     J_bottom::SV2 # View for the bottom part of the Jacobian matrix
# end


# 3rd version, using [LΛₓ;ΛₖN], seems not numerically stable
struct TimecurveParam{V<:Vector{Float64},
    SV1<:SubArray,SV2<:SubArray,SV3<:SubArray,SV4<:SubArray}

    x::V # Buffer for x values
    K::V # Buffer for K values
    v::V # Buffer for the catalysis flux vector
    Sv::V # Buffer for the catalysis rate vector multiplied by S
    Jt::SparseMatrixCSC{Float64,Int} # Jacobian matrix buffer
    Jt_lu::SparseArrays.UMFPACK.UmfpackLU{Float64,Int} # LU decomposition of Jt

    x_view::SV1 # View for x
    K_view::SV2 # View for K
    Jt_left::SV3 # View for the left part of the Jacobian matrix
    Jt_right::SV4 # View for the right part of the Jacobian matrix
end

function catalysis_logx(Bnc::Bnc, logx0::Vector{<:Real}, tspan::Tuple{Real,Real};
    k::Union{Vector{<:Real},Nothing}=nothing,
    S::Union{Matrix{Int},Nothing}=nothing,
    aT::Union{Matrix{Int},Nothing}=nothing,
    override::Bool=false,
    alg=nothing, # Default to nothing, will use Tsit5() if not provided
    reltol=1e-8,
    abstol=1e-9,
    kwargs...
)::ODESolution

    # ---Solve the ODE to find the time curve of log(x) with respect to qK change.---

    #--Prepare parameters---
    if override
        if ~isnothing(k)
            Bnc.k = k
        end
        if ~isnothing(S)
            Bnc.S = S
        end
        if ~isnothing(aT)
            Bnc.aT = aT
        end
    end
    k = isnothing(k) ? Bnc.k : k
    S = isnothing(S) ? Bnc.S : S
    aT = isnothing(aT) ? Bnc.aT : aT
    
    #initialize J_buffer
    # J = Matrix{Float64}(undef, Bnc.n, Bnc.n)
    x = Vector{Float64}(undef, Bnc.n)
    K = Vector{Float64}(undef, Bnc.r)
    v = Vector{Float64}(undef, length(k)) # catalysis flux vector
    Sv = Vector{Float64}(undef, Bnc.n) # catalysis rate vector
    Jt = copy(Bnc._LNt_sparse) # Use the sparse version of the Jacobian matrix
    Jt_lu = copy(Bnc._LNt_lu) # LU decomposition of Jt
    # create view for the J_buffer

    x_view = @view x[Bnc._I]
    K_view = @view K[Bnc._JN]
    Jt_left = @view Jt.nzval[1:Bnc._val_num]
    Jt_right = @view Jt.nzval[Bnc._val_num+1:end]

    
    params = TimecurveParam(
        x, # x_buffer
        K, # K_buffer
        v, # v buffer / flux
        Sv, # Sv buffer
        Jt, # J_buffer
        Jt_lu, # Jt_lu
        #Views for updating J
        x_view,
        K_view,
        Jt_left,
        Jt_right
    )

    # Define the ODE system for the time curve
    if Bnc._is_change_of_K_involved
        Catalysis_process! = function (dlogx, logx, p, t)
            @unpack x, K, v, Sv, Jt, Jt_left, Jt_right = p
            #update the values
            x .= exp.(logx)
            K .= exp.(Bnc._Nt_sparse' * logx)
            
            # Update the Jacobian matrix J
            @. Jt_left = x_view * Bnc._Lt_sparse.nzval 
            @. Jt_right = Bnc._Nt_sparse.nzval * K_view
            lu!(Jt_lu, Jt) # recalculate the LU decomposition of Jt

            # dlogx .= J \ (S * (k .* exp.(aT * logx)))
            mul!(v, aT, logx)
            @. v = k * exp(v) # calculate the catalysis rate vector
            mul!(Sv, S, v) # reuse x as a temporary buffer, but need change if x is used in other places, like to act call back for ODESolution

            ldiv!(dlogx, Jt_lu', Sv) # Use the LU decomposition for fast calculation
        end
    else
        # If K is not involved, we can skip the K update

        params.K .= exp.(Bnc._Nt_sparse' * logx0) #initialize K_view once
        # @show length(Bnc._Nt_sparse.nzval) length(params.K_view) length(params.Jt_right)
        @. params.Jt_right = Bnc._Nt_sparse.nzval * params.K_view #initialize Jt_right once

        Catalysis_process! = function (dlogx, logx, p, t)
            @unpack x, v, Sv, Jt, Jt_left = p
            #update the values
            x .= exp.(logx)
            # Update the Jacobian matrix J
            @. Jt_left = x_view * Bnc._Lt_sparse.nzval 
            lu!(Jt_lu, Jt) # recalculate the LU decomposition of Jt
            
            mul!(v, aT, logx)
            @. v = k * exp(v) # calculate the catalysis rate vector
            mul!(Sv, S, v) # reuse x as a temporary buffer, but need change if x is used in other places, like to act call back for ODESolution

            ldiv!(dlogx, Jt_lu', Sv)
        end
    end
    # Create the ODE problem
    prob = ODEProblem(Catalysis_process!, logx0, tspan, params)
    sol = solve(prob, alg; reltol=reltol, abstol=abstol, kwargs...)
    return sol
end


# # 4th version, using 
# struct TimecurveParam2{V<:Vector{Float64},
#     SV1<:SubArray,SV2<:SubArray,SV3<:SubArray,SV4<:SubArray}

#     x::V # Buffer for x values
#     K::V # Buffer for K values
#     v::V # Buffer for the catalysis flux vector
#     Sv::V # Buffer for the catalysis rate vector multiplied by S
#     Jt::SparseMatrixCSC{Float64,Int} # Jacobian matrix buffer
#     Jt_lu::SparseArrays.UMFPACK.UmfpackLU{Float64,Int} # LU decomposition of Jt

#     x_view::SV1 # View for x
#     K_view::SV2 # View for K
#     Jt_left::SV3 # View for the left part of the Jacobian matrix
#     Jt_right::SV4 # View for the right part of the Jacobian matrix
# end

# function catalysis_logx(Bnc::Bnc, logx0::Vector{<:Real}, tspan::Tuple{Real,Real};
#     k::Union{Vector{<:Real},Nothing}=nothing,
#     S::Union{Matrix{Int},Nothing}=nothing,
#     aT::Union{Matrix{Int},Nothing}=nothing,
#     override::Bool=false,
#     alg=nothing, # Default to nothing, will use Tsit5() if not provided
#     reltol=1e-8,
#     abstol=1e-9,
#     kwargs...
# )::ODESolution

#     # ---Solve the ODE to find the time curve of log(x) with respect to qK change.---

#     #--Prepare parameters---
#     if override
#         if ~isnothing(k)
#             Bnc.k = k
#         end
#         if ~isnothing(S)
#             Bnc.S = S
#         end
#         if ~isnothing(aT)
#             Bnc.aT = aT
#         end
#     end
#     k = isnothing(k) ? Bnc.k : k
#     S = isnothing(S) ? Bnc.S : S
#     aT = isnothing(aT) ? Bnc.aT : aT
    
#     #initialize J_buffer
#     # J = Matrix{Float64}(undef, Bnc.n, Bnc.n)
#     x = Vector{Float64}(undef, Bnc.n)
#     K = Vector{Float64}(undef, Bnc.r)
#     v = Vector{Float64}(undef, length(k)) # catalysis flux vector
#     Sv = Vector{Float64}(undef, Bnc.n) # catalysis rate vector
#     Jt = copy(Bnc._LNt_sparse) # Use the sparse version of the Jacobian matrix
#     Jt_lu = copy(Bnc._LNt_lu) # LU decomposition of Jt
#     # create view for the J_buffer

#     x_view = @view x[Bnc._I]
#     K_view = @view K[Bnc._JN]
#     Jt_left = @view Jt.nzval[1:Bnc._val_num]
#     Jt_right = @view Jt.nzval[Bnc._val_num+1:end]

    
#     params = TimecurveParam(
#         x, # x_buffer
#         K, # K_buffer
#         v, # v buffer / flux
#         Sv, # Sv buffer
#         Jt, # J_buffer
#         Jt_lu, # Jt_lu
#         #Views for updating J
#         x_view,
#         K_view,
#         Jt_left,
#         Jt_right
#     )

#     # Define the ODE system for the time curve
#     if Bnc._is_change_of_K_involved
#         Catalysis_process! = function (dlogx, logx, p, t)
#             @unpack x, K, v, Sv, Jt, Jt_left, Jt_right = p
#             #update the values
#             x .= exp.(logx)
#             K .= exp.(Bnc._Nt_sparse' * logx)
            
#             # Update the Jacobian matrix J
#             @. Jt_left = x_view * Bnc._Lt_sparse.nzval 
#             @. Jt_right = Bnc._Nt_sparse.nzval * K_view
#             lu!(Jt_lu, Jt) # recalculate the LU decomposition of Jt

#             # dlogx .= J \ (S * (k .* exp.(aT * logx)))
#             mul!(v, aT, logx)
#             @. v = k * exp(v) # calculate the catalysis rate vector
#             mul!(Sv, S, v) # reuse x as a temporary buffer, but need change if x is used in other places, like to act call back for ODESolution

#             ldiv!(dlogx, Jt_lu', Sv) # Use the LU decomposition for fast calculation
#         end
#     else
#         # If K is not involved, we can skip the K update

#         params.K .= exp.(Bnc._Nt_sparse' * logx0) #initialize K_view once
#         # @show length(Bnc._Nt_sparse.nzval) length(params.K_view) length(params.Jt_right)
#         # @. params.Jt_right = Bnc._Nt_sparse.nzval * params.K_view #initialize Jt_right once

#         Catalysis_process! = function (dlogx, logx, p, t)
#             @unpack x, v, Sv, Jt, Jt_left = p
#             #update the values
#             x .= exp.(logx)
#             # Update the Jacobian matrix J
#             @. Jt_left = x_view * Bnc._Lt_sparse.nzval 
#             lu!(Jt_lu, Jt) # recalculate the LU decomposition of Jt
            
#             mul!(v, aT, logx)
#             @. v = k * exp(v) # calculate the catalysis rate vector
#             mul!(Sv, S, v) # reuse x as a temporary buffer, but need change if x is used in other places, like to act call back for ODESolution

#             ldiv!(dlogx, Jt_lu', Sv)
#         end
#     end
#     # Create the ODE problem
#     prob = ODEProblem(Catalysis_process!, logx0, tspan, params)
#     sol = solve(prob, alg; reltol=reltol, abstol=abstol, kwargs...)
#     return sol
# end



#----------------Function to finding dominence regime，one of the hardest part to optimize----------------------
# struct Dom_regime
#     Mtd::Matrix{Int}
# end

function have_cyclic_at_node(g::Vector{Vector{Int}}, node::Int, len::Int)::Bool
    # DFS algorithm, 
    # modified from https://github.com/JuliaGraphs/Graphs.jl/blob/2d6f4d56b06cb597ebd5c40c5a8db783f1b83991/src/traversals/dfs.jl#L4-L11
    # 0 if not visited, 1 if in the current dfs path, 2 if fully explored
    vcolor = zeros(UInt8, len)
    vertex_stack = [node]
    while !isempty(vertex_stack)
        u = vertex_stack[end]
        if vcolor[u] == 0
            vcolor[u] = 1
            for n in g[u]
                # we hit a loop when reaching back a vertex of the main path
                if vcolor[n] == 1
                    return true
                elseif vcolor[n] == 0
                    # we store neighbors, but these are not yet on the path
                    push!(vertex_stack, n)
                end
            end
        else
            pop!(vertex_stack)
            if vcolor[u] == 1
                vcolor[u] = 2
            end
        end
    end
    return false
end


function find_valid_regime(idx::Vector{Vector{Int}}, d::Int, n::Int)::Vector{Vector{Int}}
    graph = [Vector{Int}() for _ in 1:n]
    choices = Vector{Int}(undef, d)
    results = Vector{Vector{Int}}()
    function backtrack!(i)
        # All rows are fine
        if i == d + 1
            # @show choices
            push!(results, copy(choices))
            return nothing
        end

        for v in idx[i]
            # add edges for current row. and record.

            target_nodes = [w for w in idx[i] if w != v] # target_nodes

            for node in target_nodes
                push!(graph[node], v) # add edge node -> v
            end

            if ~have_cyclic_at_node(graph, v, n)
                choices[i] = v
                backtrack!(i + 1)
            end

            for node in target_nodes
                pop!(graph[node])
            end
        end
    end
    backtrack!(1)
    return results
end

function find_valid_regime(L::Matrix{Int})
    d, n = size(L)
    idx = [[idx for (idx, value) in enumerate(row) if value != 0] for row in eachrow(L)]
    find_valid_regime(idx, d, n)
end

find_valid_regime(Bnc::Bnc) = find_valid_regime(Bnc._valid_L_idx, Bnc.d, Bnc.n)



# using Base.Threads
# function find_valid_regime_parallel(idx::Vector{Vector{Int}}, d::Int, n::Int)::Vector{Vector{Int}}
#     # A Channel is a thread-safe FIFO queue to collect results from all threads.
#     results_channel = Channel{Vector{Int}}(Inf)

#     # Use @threads to parallelize the loop over the first dimension's choices.
#     # Each thread will handle one or more initial choices from `idx[1]`.
#     @threads for v_initial in idx[1]
#         # --- THREAD-LOCAL STATE ---
#         # Each thread gets its own graph and choices array to prevent race conditions.
#         graph = [Vector{Int}() for _ in 1:n]
#         choices = Vector{Int}(undef, d)

#         # --- INITIAL SETUP FOR THIS THREAD ---
#         # Set the first choice and update the graph accordingly.
#         choices[1] = v_initial
#         target_nodes_initial = [w for w in idx[1] if w != v_initial]
#         for node in target_nodes_initial
#             push!(graph[node], v_initial)
#         end

#         # If the initial choice already creates a cycle, this path is invalid.
#         if have_cyclic_at_node(graph, v_initial, n)
#             continue # This thread moves to its next assigned v_initial.
#         end

#         # --- RECURSIVE BACKTRACKING (within a single thread) ---
#         # This recursive function is defined locally and closes over its
#         # thread-local `graph` and `choices`.
#         function backtrack_from_level2!(i)
#             # Base case: A valid configuration for all `d` dimensions is found.
#             if i == d + 1
#                 put!(results_channel, copy(choices))
#                 return
#             end

#             # Explore choices for the current level `i`.
#             for v in idx[i]
#                 target_nodes = [w for w in idx[i] if w != v]
                
#                 # 1. MAKE MOVE: Add edges for the current choice.
#                 for node in target_nodes
#                     push!(graph[node], v)
#                 end

#                 # 2. CHECK & RECURSE: If no cycle, recurse to the next level.
#                 if !have_cyclic_at_node(graph, v, n)
#                     choices[i] = v
#                     backtrack_from_level2!(i + 1)
#                 end

#                 # 3. BACKTRACK: Revert the graph to its previous state.
#                 for node in target_nodes
#                     pop!(graph[node])
#                 end
#             end
#         end

#         # Start the recursive search from the second level.
#         backtrack_from_level2!(2)
#     end

#     # All threads are finished, so no more results will be added.
#     close(results_channel)

#     # Collect all results from the channel into a final vector.
#     return collect(results_channel)
# end

# using Graphs
# function find_valid_regime(L::Matrix{Int})
#     # Slower may because of repeatedly inherently repeat check. 
#     (d,n) = size(L)
#     idx = [[idx for (idx, value) in enumerate(row) if value != 0] for row in eachrow(L) ] #!!! extremely key, avoid repeated add, or bug when removing it.
#     graph = SimpleDiGraph(n)
#     ict = IncrementalCycleTracker(graph, dir = :out)
#     choices = Vector{Int}(undef, d)
#     results = Vector{Vector{Int}}()
#     function backtrack!(i)
#         if i == d+1 
#             push!(results, copy(choices))  # 使用副本避免后续修改影响结果
#             return nothing
#         end

#         for v in idx[i]
#             target_nodes = [w for w in idx[i] if w != v && ~(w in outneighbors(graph,v))] # target_nodes
#             if add_edge_checked!(ict, v, target_nodes) # add successfully
#                 choices[i] = v
#                 backtrack!(i + 1)
#             end

#             for node in target_nodes
#                 rem_edge!(graph, v, node)
#             end
#         end
#     end
#     backtrack!(1)
#     return results
# end



#-----------------------------------------------fucntions with dominance regime-------------------------------------------------------

#Pure helper functions for converting between matrix and index-value pairs.
function _Mtx2idx_val(Mtx::Matrix{<:T}) where T
    row_num, col_num  = size(Mtx)
    idx = Vector{Int}(undef, row_num)
    val = Vector{T}(undef, row_num)
    for i in 1:row_num
        for j in 1:col_num
            if Mtx[i, j] != 0
                idx[i] = j
                val[i] = Mtx[i, j]
                break 
            end
        end
    end
    return idx,val
end
function _idx_val2Mtx(idx::Vector{Int}, val::T=1, col_num::Union{Int,Nothing}=nothing) where T
    n = length(idx)
    col_num = isnothing(col_num) ? n : col_num # if col_num is not provided, use the maximum idx value
    Mtx = zeros(T, n, col_num)
    for i in 1:n
        if idx[i] != 0
            Mtx[i, idx[i]] = val
        end
    end
    return Mtx
    
end
function _idx_val2Mtx(idx::Vector{Int}, val::Vector{<:T}, col_num::Union{Int,Nothing}=nothing) where T
    # Convert idx and val to a matrix of size (d, n)
    n = length(idx)
    col_num = isnothing(col_num) ? n : col_num
    @assert length(val) == n "val must have the same length as idx"
    Mtx = zeros(T, n, col_num)
    for i in 1:n
        if idx[i] != 0
            Mtx[i, idx[i]] = val[i]
        end
    end
    return Mtx
end

function _check_valid_idx(idx::Vector{Int},Mtx::Matrix{<:Any})
    # Check if the idx is valid for the given Mtx
    @assert length(idx) == size(Mtx, 1) "idx must have the same length as the number of rows in Mtx"
    for i in 1:length(idx)
        @assert Mtx[i, idx[i]] != 0 "Mtx must have non-zero entries at the idx positions"
    end
    return true
end
#----end of helper functions.



function Mtd_a_from_regime(Bnc::Bnc, regime::Vector{Int}; check::Bool=true)::Tuple{Matrix{Int}, Vector{Int}}
    # Generate the ̃M matrix from the given regime
    !check || _check_valid_idx(regime, Bnc.L) # check if the regime is valid for the given L matrix
    M = zeros(Int, Bnc.d, Bnc.n)
    a = zeros(Int, Bnc.d) # buffer for the a values
    for i in 1:Bnc.d
        M[i, regime[i]] = 1
        a[i] = Bnc.L[i, regime[i]]
    end
    return M,a
end

function M_from_regime(Bnc::Bnc, regime::Vector{Int}; check::Bool=true)::Matrix{Int}
    # Generate the ̃M matrix from the given regime
    M = zeros(Int, Bnc.d, Bnc.n)
    !check || _check_valid_idx(regime, Bnc.L)
    for i in 1:Bnc.d
        M[i, regime[i]] = Bnc.L[i, regime[i]]
    end
    M[Bnc.d+1:end, :] .= Bnc.N
    return M
end



function ∂logqK_∂logx_regime(Bnc::Bnc; regime::Union{Vector{Int}, Nothing}=nothing,
    Mtd::Union{Matrix{Int}, Nothing}=nothing,
    M::Union{Matrix{Int}, Nothing}=nothing,
    check::Bool=true)::Matrix{<:Real}
    """
    Calculate the derivative of log(qK) with respect to log(x) given regime
    check: if true, check if the regime is valid for the L matrix
    regime: the regime vector, if not provided , Mtd  will be derived from Mtd or M
    
    Return:
    - logder_qK_x: the derivative of log(qK) with respect to log(x)
    """
    if isnothing(Mtd)
        if isnothing(regime)
            if isnothing(M)
                @error("Either regime or M/Mtd must be provided")
            else
                Mtd = sign.(M)
            end
        else
            (Mtd , _) = Mtd_a_from_regime(Bnc, regime; check=check)
        end
    end

    return vcat(Mtd, Bnc.N)
end

function ∂logx_∂logqK_regime(Bnc::Bnc;
    logder_qK_x::Union{Matrix{<:Real},Nothing}=nothing,
    regime::Union{Vector{Int}, Nothing}=nothing,
    Mtd::Union{Matrix{Int}, Nothing}=nothing,
    M::Union{Matrix{Int}, Nothing}=nothing,
    check::Bool=true)::Tuple{Matrix{<:Real},Int}
    """
    Calculate the derivative of log(qK) with respect to log(x) given regime
    check: if true, check if the regime is valid for the L matrix
    regime: the regime vector, if not provided, will be derived from Mtd or M

    Return:
    - logder_x_qK: the derivative of log(x) with respect to log(qK)
    - singularity: singularity of the logder_qK_x.
    """

    logder_qK_x = isnothing(logder_qK_x) ? ∂logqK_∂logx_regime(Bnc; regime=regime,Mtd=Mtd,M=M, check=check) : logder_qK_x
    logder_qK_x_fac = lu(logder_qK_x,check=false)
    if issuccess(logder_qK_x_fac) # Lu successfully.
        return inv(logder_qK_x_fac),0  # singularity is 0, not singular
    else
        return _adj_singular_matrix(logder_qK_x) .* Bnc.direction # calculate the adj matrix, singularity is calculated and returned,
    end
end



@kwdef struct Regime
    regime::Vector{Int} # The regime vector
    logder_qK_x::Matrix{<:Real} # The derivative of log(qK) with respect to log(x)
    logder_x_qK::Matrix{<:Real} # The derivative of log(x) with respect to log(qK)
    singularity::Int # Whether logder_x_qK is singular
    # direction::Int # If singular, direction defines its direction.
    a::Vector{Int} # Values of L on regime defined position
end



function create_regime(Bnc::Bnc; regime::Vector{Int})::Regime
    Mtd, a = Mtd_a_from_regime(Bnc, regime; check=true)
    logder_qK_x = ∂logqK_∂logx_regime(Bnc; Mtd=Mtd)
    logder_x_qK, singularity = ∂logx_∂logqK_regime(Bnc; logder_qK_x=logder_qK_x)
    return Regime(regime = regime, 
        logder_qK_x = logder_qK_x, 
        logder_x_qK = logder_x_qK, 
        singularity = singularity, 
        # direction = Bnc.direction,
        a = a)
end


x_ineq_mtx(Bnc::Bnc; regime::Regime, check::Bool=true) =  x_ineq_mtx(Bnc; regime=regime.regime, check=check)
function x_ineq_mtx(Bnc::Bnc; regime::Vector{Int}, check::Bool=true)::Matrix{<:Real}
    """
    return a matrix of ineq in x space for regime expressed as Ax < 0
    """
    !check || _check_valid_idx(regime, Bnc.L) # check if the regime is valid for the given L matrix
    idx = Set{Tuple{Int,Int}}()
    for (valid_idx,rgm) in zip(Bnc._valid_L_idx, regime)
        for i in valid_idx
            if i != rgm
            push!(idx, (rgm,i))
            end
        end
    end
    mtx = zeros(eltype(Bnc.L), length(idx), Bnc.n)
    for (i, (rgm, j)) in enumerate(idx)
        mtx[i, rgm] = -1
        mtx[i, j] = 1
    end
    return mtx
end


#----------------------------------------------------------Symbolics calculation fucntions-----------------------------------------------------------


function ∂logqK_∂logx_sym(Bnc::Bnc; show_x_space::Bool=false)::Matrix{Num}
    
    # if any(isnothing, [Bnc.x_sym, Bnc.q_sym])
    #     fill_bnc_symbolic!(Bnc) # Ensure all symbolic variables are defined
    # end

    if show_x_space
        q = Bnc.L * Bnc.x_sym
    else
        q = Bnc.q_sym
    end

    return [
        transpose(Bnc.x_sym) .* Bnc.L ./ q
        Bnc.N
    ]
end

function ∂logx_∂logqK_sym(Bnc::Bnc;show_x_space::Bool=false)::Matrix{Num}
    # Calculate the symbolic derivative of log(qK) with respect to log(x)
    # This function is used for symbolic calculations, not numerical ones.
    return inv(∂logqK_∂logx_sym(Bnc; show_x_space=show_x_space)).|> simplify
end




show_x_space_conditions_for_regime(Bnc::Bnc; regime::Regime) = show_x_space_conditions_for_regime(Bnc; regime=regime.regime)
function show_x_space_conditions_for_regime(Bnc::Bnc; regime::Vector{Int})
    # Show the conditions for the x space for the given regime.
    return x_ineq_mtx(Bnc; regime=regime) * log.(Bnc.x_sym) .< 0
end

function show_qK_space_conditions_for_regime(Bnc::Bnc; regime::Regime)
    regime.singularity != 0 ? @error("Regime is singular, cannot show qK space conditions") : nothing
    A = x_ineq_mtx(Bnc; regime=regime)
    Mtd_N = regime.logder_qK_x
    return simplify.(A * (Mtd_N \ log.([Bnc.q_sym ./ a; Bnc.K_sym])) .< 0)
end

function show_qK_space_conditions_for_regime(Bnc::Bnc; regime::Vector{Int})
    A = x_ineq_mtx(Bnc; regime=regime)
    Mtd, a = Mtd_a_from_regime(Bnc, regime; check=true)
    Mtd_N = [Mtd; Bnc.N]
    @show Mtd
    return simplify.(A / Mtd_N * log.([Bnc.q_sym ./ a; Bnc.K_sym]) .< 0)
end


#-----------------------------------Helper functions---------------------------------------------
# function adjoint_matrix(A)
#     # Calculate the adjoint of a square matrix A ,for calculate the ray if non-feasible regime
#     n = size(A, 1)
#     if size(A, 1) != size(A, 2)
#         error("square matrix required for adjoint calculation")
#     end
#     adj_A = zeros(eltype(A), n, n)
#     for i in 1:n, j in 1:n
#         minor_matrix = @view A[[1:i-1; i+1:end], [1:j-1; j+1:end]]
#         adj_A[j, i] = (-1)^(i + j) * det(minor_matrix)
#     end
#     return adj_A
# end

function _adj_singular_matrix(M::Matrix{<:Real})::Tuple{Matrix{<:Real},Int}
    n, m = size(M)
    @assert n == m "Matrix must be square"
    M_svd = svd(M;full=true)
    singular_pos = findall(M_svd.S .< 1e-7)
    singularity = length(singular_pos)
    adj = zeros(eltype(M), n, n)
    if length(singular_pos) > 1
        @warn("Multiple singular values found")
        return zeros(n, n), singularity
    else # sigularity of 1
        line = M_svd.Vt[singular_pos[1],:] * M_svd.U[:, singular_pos[1]]'
        # @show line
        for i in 1:n
            if abs(line[i,i]) >1e-7
                idx = vcat(1:i-1,i+1:n)
                # @show idx
                amp = det(M[idx,idx])/ line[i,i]
                # @show amp
                @. adj = round(line * amp)
                return adj, singularity
            end
        end
        @error("No non-zero diagonal element found in the singular vector")
    end
end

function randomize(n::Int, log_lower=-5, log_upper=5; log_space::Bool=true)::Vector{<:Real}
    # Generate a random vector of size n with values between 10^log_lower and 10^log_upper

    #turn lowerbound and upperbound into bases of e
    if log_space
        exp10.(rand(n) .* (log_upper - log_lower) .+ log_lower)
    else
        rand(n) .* (exp10(log_upper) - exp10(log_lower)) .+ exp10(log_lower)
    end
end

using JSON3

function arr_to_vector(arr)
    d = ndims(arr)
    if d == 0
        return arr[]  # 处理0维数组（标量）
    elseif d == 1
        return [x for x in arr]  # 1维数组转列表
    else
        # 沿第一维切片，递归处理每个切片（降维后）
        return [arr_to_vector(s) for s in eachslice(arr, dims=1)]
    end
end
function pythonprint(arr)
    txt = JSON3.write(arr_to_vector(arr), pretty=true, indent=4, escape_unicode=false)
    println(txt)
    return nothing
end

function N_generator(r::Int, n::Int; min_binder::Int=2, max_binder::Int=2)::Matrix{Int}
    @assert n > r "n must be greater than r"
    @assert n >= 3 "at least 3 species"
    @assert r >= 1 "at least 1 reactions"
    @assert min_binder >= 1 && max_binder >= min_binder "min_binder and max_binder must be at least 1"
    @assert min_binder <= n - r "min_binder must be smaller than n-r"
    #initialize the matrix
    N = zeros(Int, r, n)
    i_temp = max(max_binder + 1 + r - n, 1) # the row num before which max_binder <= n-r+i-1 

    for i in 1:i_temp # iter across row
        N[i, n-r+i] = -1
        binder_num = sample(min_binder:(n-r+i-1))
        N[i, sample(1:(n-r+i-1), binder_num; replace=false)] .= 1
    end
    for i in (i_temp+1):r # iter acorss row
        N[i, n-r+i] = -1
        binder_num = sample(min_binder:max_binder)
        N[i, sample(1:(n-r+i-1), binder_num; replace=false)] .= 1
    end
    return N
end

# function L_generator(d::Int, n::Int; min_nonatom::Int=0, max_nonatom::Union{Nothing,Int}=nothing, allow_multi::Bool = true)::Matrix{Int}
#     max_nonatom = isnothing(max_nonatom) ? n-d : max_nonatom
#     @assert n >= d "n must be greater than d"
#     @assert min_nonatom >= 0 && max_nonatom >= min_nonatom "min_nonatom and max_nonatom must be at least 0"
#     @assert max_nonatom <= n-d "max_nonatom must be smaller than n-d"
#     #initialize the matrix 
#     L1 = Diagonal(ones(Int, d)) # the first d rows are identity matrix
#     L2 = zeros(Int, d, n-d)
#     for i in 1:d
#         nonatom_num = sample(min_nonatom:max_nonatom)
#         if allow_multi
#             for j in 1:nonatom_num
#                 L2[i, sample(1:(n-d))] += 1
#             end
#         else
#             L2[i, sample(1:(n-d), nonatom_num; replace=false)] .= 1
#         end
#     end
#     return hcat(L1, L2)
# end
function L_generator(d::Int, n::Int; min_binder::Int=2, max_binder::Int=2)::Matrix{Int}
    N = N_generator(n - d, n; min_binder=min_binder, max_binder=max_binder)
    L = L_from_N(N)
    return L
end

# S_generator(n::Int,r_cat::Int; d::Union{Int,nothing} = nothing, transform_only::bool=false, involving_K::Bool=false)::Matrix{Int}
#     S = zeros(Int, n, r_cat)
#     idx = Vector{Int}(undef, 2)
#     row_end = involving_K ? n : n - d
#     for i in 1:r_cat
#         S[sample!(1:row_end,idx),i] = 
#     # assign first 
#     end  
# end 