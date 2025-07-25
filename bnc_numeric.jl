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
            q = log10.(Bnc._Lt_sparse' * exp10.(x))
        else
            K = exp10.(Bnc._Nt_sparse' * x)
            q = Bnc._Lt_sparse' * exp10.(x)
        end
    else
        if output_logspace
            K = Bnc._Nt_sparse' * log10.(x)
            q = log10.(Bnc._Lt_sparse' * x)
        else
            K = exp10.(Bnc._Nt_sparse' * log10.(x))
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

    endlogqK = input_logspace ? qK : log10.(qK) # Convert qK to log space if not already

    sol = _logx_traj_with_logqK_change(Bnc,
        startlogqK,
        endlogqK;
        startlogx=startlogx,
        alg=Tsit5(),
        reltol=reltol,
        abstol=abstol,
        save_everystep=false,
        save_start=false,
    )
    x = output_logspace ? sol.u[end] : exp10.(sol.u[end])
    return x
end



#----------------Functions using homotopyContinuous to moving across x space along with qK change----------------------
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

function x_traj_with_qK_change(
    Bnc::Bnc,
    start_point::Vector{<:Real},
    end_point::Vector{<:Real};
    input_logspace::Bool=false,
    output_logspace::Bool=false,
    alg=nothing, # Default to nothing, will use Tsit5() if not provided
    reltol=1e-8,
    abstol=1e-9,
    kwargs...
)::Tuple{Vector{Float64}, Matrix{Float64}}
    startlogqK = input_logspace ? start_point : log10.(start_point)
    endlogqK = input_logspace ? end_point : log10.(end_point)
    solution = _logx_traj_with_logqK_change(Bnc, startlogqK, endlogqK;
        alg=alg,
        reltol=reltol,
        abstol=abstol,
        dense=false,
        kwargs...
    )
    
    if !output_logspace
        foreach(u -> u .= exp10.(u), solution.u)
    end

    return _ode_solution_wrapper(solution)
end

function x_traj_with_q_chage(
    Bnc::Bnc,
    start_q::Vector{<:Real},
    end_q::Vector{<:Real};
    K::Union{Vector{<:Real},Nothing}=nothing,
    logK::Union{Vector{<:Real},Nothing}=nothing,
    input_logspace::Bool=false,
    output_logspace::Bool=false,
    alg=nothing, # Default to nothing, will use Tsit5() if not provided
    reltol=1e-8,
    abstol=1e-9,
    kwargs...
)::Tuple{Vector{Float64}, Matrix{Float64}}
    # Prepare the start and end points
    K_prepared = input_logspace ? (isnothing(logK) ? log10.(K) : logK) : (isnothing(K) ? K : exp10.(K))
    
    x_traj_with_qK_change(Bnc, [start_q;K_prepared], [end_q;K_prepared]; input_logspace=input_logspace, output_logspace=output_logspace,
        alg=alg, reltol=reltol, abstol=abstol, kwargs...)
end


function _logx_traj_with_logqK_change(Bnc::Bnc,
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
        @. q = exp10(startlogq + t * Δlogq)
        @. x = exp10(logx)
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
            x .= exp10.(logx)
            K .= exp10.(Bnc._Nt_sparse' * logx)
            
            # Update the Jacobian matrix J
            @. Jt_left = x_view * Bnc._Lt_sparse.nzval 
            @. Jt_right = Bnc._Nt_sparse.nzval * K_view
            lu!(Jt_lu, Jt) # recalculate the LU decomposition of Jt

            # dlogx .= J \ (S * (k .* exp10.(aT * logx)))
            mul!(v, aT, logx)
            @. v = k * exp10(v) # calculate the catalysis rate vector
            mul!(Sv, S, v) # reuse x as a temporary buffer, but need change if x is used in other places, like to act call back for ODESolution

            ldiv!(dlogx, Jt_lu', Sv) # Use the LU decomposition for fast calculation
        end
    else
        # If K is not involved, we can skip the K update

        params.K .= exp10.(Bnc._Nt_sparse' * logx0) #initialize K_view once
        # @show length(Bnc._Nt_sparse.nzval) length(params.K_view) length(params.Jt_right)
        @. params.Jt_right = Bnc._Nt_sparse.nzval * params.K_view #initialize Jt_right once

        Catalysis_process! = function (dlogx, logx, p, t)
            @unpack x, v, Sv, Jt, Jt_left = p
            #update the values
            x .= exp10.(logx)
            # Update the Jacobian matrix J
            @. Jt_left = x_view * Bnc._Lt_sparse.nzval 
            lu!(Jt_lu, Jt) # recalculate the LU decomposition of Jt
            
            mul!(v, aT, logx)
            @. v = k * exp10(v) # calculate the catalysis rate vector
            mul!(Sv, S, v) # reuse x as a temporary buffer, but need change if x is used in other places, like to act call back for ODESolution

            ldiv!(dlogx, Jt_lu', Sv)
        end
    end
    # Create the ODE problem
    prob = ODEProblem(Catalysis_process!, logx0, tspan, params)
    sol = solve(prob, alg; reltol=reltol, abstol=abstol, kwargs...)
    return sol
end





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
#         @. q = exp10(startlogq + t * Δlogq)
#         #update x
#         @. x = exp10(logx)
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
#             x .= exp10.(logx)
#             K .= exp10.(Bnc.N * logx)
#             # Calculate the Jacobian matrix J
#             @. J_top = Bnc.L * x'
#             @. J_bottom = K * Bnc.N

#             # dlogx .= J \ (S * (k .* exp10.(aT * logx)))
#             mul!(v, aT, logx)
#             @. v = k * exp10(v) # calculate the catalysis rate vector
#             mul!(Sv, S, v) # reuse x as a temporary buffer, but need change if x is used in other places, like to act call back for ODESolution
#             dlogx .= J \ Sv
#         end
#     else
#         # If K is not involved, we can skip the K update
#         params.J_bottom .= exp10.(Bnc.N * logx0) .* Bnc.N #initialize J_bottom once
#         Catalysis_process! = function (dlogx, logx, p, t)
#             @unpack x, v, Sv, J, J_top = p
#             #update the values
#             x .= exp10.(logx)
#             # Update the Jacobian matrix J
#             @. J_top = Bnc.L * x'
#             # dlogx .= J \ (S * (k .* exp10.(aT * logx))) # could be optimized by symbolics further.
#             mul!(v, aT, logx) # v = aT * logx
#             @. v = k * exp10(v) # calculate the catalysis rate vector
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
#             x .= exp10.(logx)
#             K .= exp10.(Bnc.N * logx)
#             # Calculate the Jacobian matrix J
#             @. J_top = Bnc.L * x'
#             @. J_bottom = K * Bnc.N
#             dlogx .= J \ (S * (k .* exp10.(aT * logx)))
#         end
#     else
#         # If K is not involved, we can skip the K update
#         params.J_bottom .= exp10.(Bnc.N * logx0) .* Bnc.N #initialize J_bottom once
#         Catalysis_process! = function(dlogx, logx, p, t)
#             @unpack x, J, J_top =p
#             #update the values
#             x .= exp10.(logx)
#             @. J_top = Bnc.L * x'
#             dlogx .= J \ (S * (k .* exp10.(aT * logx))) # could be optimized by symbolics further.
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
#             x .= exp10.(logx)
#             K .= exp10.(Bnc._Nt_sparse' * logx)
            
#             # Update the Jacobian matrix J
#             @. Jt_left = x_view * Bnc._Lt_sparse.nzval 
#             @. Jt_right = Bnc._Nt_sparse.nzval * K_view
#             lu!(Jt_lu, Jt) # recalculate the LU decomposition of Jt

#             # dlogx .= J \ (S * (k .* exp10.(aT * logx)))
#             mul!(v, aT, logx)
#             @. v = k * exp10(v) # calculate the catalysis rate vector
#             mul!(Sv, S, v) # reuse x as a temporary buffer, but need change if x is used in other places, like to act call back for ODESolution

#             ldiv!(dlogx, Jt_lu', Sv) # Use the LU decomposition for fast calculation
#         end
#     else
#         # If K is not involved, we can skip the K update

#         params.K .= exp10.(Bnc._Nt_sparse' * logx0) #initialize K_view once
#         # @show length(Bnc._Nt_sparse.nzval) length(params.K_view) length(params.Jt_right)
#         # @. params.Jt_right = Bnc._Nt_sparse.nzval * params.K_view #initialize Jt_right once

#         Catalysis_process! = function (dlogx, logx, p, t)
#             @unpack x, v, Sv, Jt, Jt_left = p
#             #update the values
#             x .= exp10.(logx)
#             # Update the Jacobian matrix J
#             @. Jt_left = x_view * Bnc._Lt_sparse.nzval 
#             lu!(Jt_lu, Jt) # recalculate the LU decomposition of Jt
            
#             mul!(v, aT, logx)
#             @. v = k * exp10(v) # calculate the catalysis rate vector
#             mul!(Sv, S, v) # reuse x as a temporary buffer, but need change if x is used in other places, like to act call back for ODESolution

#             ldiv!(dlogx, Jt_lu', Sv)
#         end
#     end
#     # Create the ODE problem
#     prob = ODEProblem(Catalysis_process!, logx0, tspan, params)
#     sol = solve(prob, alg; reltol=reltol, abstol=abstol, kwargs...)
#     return sol
# end