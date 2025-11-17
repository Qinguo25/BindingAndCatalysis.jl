#----------------Functions for calculates the derivative of log(x) with respect to log(qK) and vice versa----------------------

function ∂logqK_∂logx(Bnc::Bnc;
    x::Union{AbstractVector{<:Real},Nothing}=nothing,
    qK::Union{AbstractVector{<:Real},Nothing}=nothing,
    q::Union{AbstractVector{<:Real},Nothing}=nothing)::Matrix{<:Real}

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
    x::Union{AbstractVector{<:Real},Nothing}=nothing,
    qK::Union{AbstractVector{<:Real},Nothing}=nothing,
    q::Union{AbstractVector{<:Real},Nothing}=nothing)::Matrix{<:Real}

    inv(∂logqK_∂logx(Bnc; x=x, q=q, qK=qK))
end

logder_x_qK(args...;kwargs...) = ∂logx_∂logqK(args...;kwargs...)
logder_qK_x(args...;kwargs...) = ∂logqK_∂logx(args...;kwargs...)

# ----------------Functions for mapping between qK space and x space----------------------------------




function x2qK(Bnc::Bnc, x::AbstractArray{<:Real};
    input_logspace::Bool=false,
    output_logspace::Bool=false,
    only_q::Bool=false,
)::AbstractArray{<:Real}
    if !only_q
        if input_logspace
            if output_logspace
                K = Bnc._N_sparse * x
                q = log10.(Bnc._L_sparse * exp10.(x))
            else
                K = exp10.(Bnc._N_sparse * x)
                q = Bnc._L_sparse * exp10.(x)
            end
        else
            if output_logspace
                K = Bnc._N_sparse * log10.(x)
                q = log10.(Bnc._L_sparse * x)
            else
                K = exp10.(Bnc._N_sparse * log10.(x))
                q = Bnc._L_sparse * x
            end
        end
        return vcat(q, K)
    else
        if input_logspace
            if output_logspace
                q = log10.(Bnc._L_sparse * exp10.(x))
            else
                q = Bnc._L_sparse * exp10.(x)
            end
        else
            if output_logspace
                q = log10.(Bnc._L_sparse * x)
            else
                q = Bnc._L_sparse * x
            end
        end
        return q
    end
end

#----------------------------------------------------------------
# Playground for mapping different methods for solving the nonlinear system
# of equations to find x from qK.
#-----------------------------------------------------------------


function _logqK2logx_nlsolve(Bnc::Bnc, logqK::AbstractArray{<:Real,1};
    startlogx::Union{Vector{<:Real},Nothing}=nothing,
    method ::Union{Symbol,Missing} = missing,
    kwargs...
)::Vector{<:Real}
    n = Bnc.n
    d = Bnc.d
    #---Solve the nonlinear equation to find x from qK.---

    startlogx = isnothing(startlogx) ? copy(Bnc._anchor_log_x) : startlogx

    resid = Vector{Float64}(undef, n)

    logq = @view logqK[1:d]
    logK = @view logqK[d+1:end]

    J = deepcopy(Bnc._LN_sparse)# Make deep copies of sparse matrices to avoid shared state
    x = Vector{Float64}(undef, n)
    q = Vector{Float64}(undef, d)
    x_J_view = @view x[Bnc._LN_top_cols] # view for faster updating J
    q_J_view = @view q[Bnc._LN_top_rows] # view for faster updating J
    J_top = @view J.nzval[Bnc._LN_top_idx] # view for faster updating J
    L_nzval = copy(Bnc._LN_sparse.nzval[Bnc._LN_top_idx])

    params = (; x, q, logq, logK, J, x_J_view, q_J_view, J_top)


    keep_manifold! = function(resid, u, p) 
        logq, logK = p
        resid[1:d] .= log10.(Bnc._L_sparse * exp10.(u)) .- logq
        resid[d+1:end] .= Bnc._N_sparse * u .- logK
        return resid
    end

    manifold_jac! = function(J,u,p) # to have the same signature as keep_manifold!()
        @unpack x,q,logq,J,x_J_view,q_J_view, J_top = p
        # update jac for the current logx     
        @. x = exp10(u) # update x
        q .= Bnc._L_sparse * x #update q
        @. J_top = x_J_view * L_nzval / q_J_view
        return J
    end

    prob = NonlinearProblem(keep_manifold!, startlogx, params; resid_prototype=zeros(n), jac = manifold_jac!, jac_prototype=J)
    
    sol = solve(prob, method; kwargs...)
    if !SciMLBase.successful_retcode(sol.retcode)
        @warn("Nonlinear solver did not converge successfully. Retcode: $(sol.retcode)")
    end
    return sol.u
end

function qK2x(Bnc::Bnc, qK::AbstractArray{<:Real,1};
    K::Union{Vector{<:Real},Nothing}=nothing,
    logK::Union{Vector{<:Real},Nothing}=nothing,
    input_logspace::Bool=false,
    output_logspace::Bool=false,
    startlogx::Union{Vector{<:Real},Nothing}=nothing,
    startlogqK::Union{Vector{<:Real},Nothing}=nothing,
    use_vtx::Bool=false,
    method::Union{Symbol,Missing} = :homotopy,
    reltol = 1e-8,
    abstol = 1e-10,
    kwargs...)::Vector{<:Real}
    """
    Map from qK space to x space.
    Available methods includes: :homotopy, :newton, :manual
    """
    #---Solve the homotopy ODE to find x from qK.---

    # Define the start point 
    if isnothing(startlogqK) || isnothing(startlogx)
        # If no starting point is provided, use the default
        # Make deep copies to avoid shared state in threaded environment
        startlogx = copy(Bnc._anchor_log_x)
        startlogqK = copy(Bnc._anchor_log_qK)
    end

    # Define the end point
    processed_logqK = input_logspace ? qK : log10.(qK)
    local log_K_to_append = nothing
    if !isnothing(logK)
        if !isnothing(K)
            @warn("Both K and logK are provided; using logK.")
        end
        log_K_to_append = logK
    elseif !isnothing(K)
        log_K_to_append = log.(K)
    end
    endlogqK = isnothing(log_K_to_append) ? processed_logqK : vcat(processed_logqK, log_K_to_append)


    if use_vtx
        perm = assign_vertex_qK(Bnc,endlogqK; input_logspace=true,asymptotic=false)
        H,H0 = get_H_H0!(Bnc,perm)
        x = H* endlogqK .+ H0
    elseif ismissing(method) || method != :homotopy
        x = _logqK2logx_nlsolve(Bnc, 
            endlogqK;
            startlogx=startlogx,
            method=method,
            reltol = reltol,
            abstol = abstol,
            kwargs...
        )
    else
        sol = _logx_traj_with_logqK_change(Bnc,
            startlogqK,
            endlogqK;
            startlogx=startlogx,
            alg=ODE.Tsit5(),
            save_everystep=false,
            save_start=false,
            reltol = reltol,
            abstol = abstol,
            kwargs...
        )
        x = sol.u[end]
    end

    x = output_logspace ? x : exp10.(x)
    return x
end

function qK2x(Bnc::Bnc, qK::AbstractArray{<:Real,2};kwargs...)::AbstractArray{<:Real}
    # batch mapping of qK2x for each column of qK and return as matrix.
    # Make thread-safe by creating separate copies for each thread
    f = x -> qK2x(Bnc, x; kwargs...)
    return matrix_iter(f, qK;byrow=false,multithread=true)
end

#----------------Functions using homotopyContinuous to moving across x space along with qK change----------------------

function x_traj_with_qK_change(
    Bnc::Bnc,
    start_point::Vector{<:Real},
    end_point::Vector{<:Real};
    input_logspace::Bool=false,
    output_logspace::Bool=false,
    kwargs...
)
    # println("x_traj_with_qK_change get kwargs: ", kwargs)

    startlogqK = input_logspace ? start_point : log10.(start_point)
    endlogqK = input_logspace ? end_point : log10.(end_point)
    solution = _logx_traj_with_logqK_change(Bnc, startlogqK, endlogqK;
        dense=false,
        kwargs...
    )
    if !output_logspace
        foreach(u -> u .= exp10.(u), solution.u)
    end
    return _ode_solution_wrapper(solution)
end


function x_traj_with_q_change(
    Bnc::Bnc,
    start_q::Vector{<:Real},
    end_q::Vector{<:Real};
    K::Union{Vector{<:Real},Nothing}=nothing,
    logK::Union{Vector{<:Real},Nothing}=nothing,
    input_logspace::Bool=false,
    # output_logspace::Bool=false,
    # alg=nothing, # Default to nothing, will use Tsit5() if not provided
    # reltol=1e-8,
    # abstol=1e-9,
    kwargs...
)
    # Prepare the start and end points
    # println("x_traj_with_q_change get kwargs: ", kwargs)
    K_prepared = input_logspace ? (isnothing(logK) ? log10.(K) : logK) : (isnothing(K) ? K : exp10.(K))

    x_traj_with_qK_change(Bnc, [start_q;K_prepared], [end_q;K_prepared]; input_logspace=input_logspace,kwargs...)
end



struct HomotopyParams{V<:Vector{Float64},SV1<:SubArray,SV2<:SubArray}

    ΔlogqK::V
    logx::V
    logqK::V
    logq::SV1
    logK::SV1

    J::SparseMatrixCSC{Float64,Int} 
    J_lu::SparseArrays.UMFPACK.UmfpackLU{Float64,Int}

    logx_J_view::SV2
    logq_J_view::SV2
    J_top::SV2
    J_top_diag::SV2
    
    # logx_local::V
    # logx_J_view_local::SV2
    # logLx_local::V
    # logLx_J_view_local::SV2
end

function _logx_traj_with_logqK_change(Bnc::Bnc,
    startlogqK::Union{Vector{<:Real},Nothing},
    endlogqK::Vector{<:Real};
    # Optional parameters for the initial log(x) values,act as initial point for ode solving
    startlogx::Union{Vector{<:Real},Nothing}=nothing,
    # Optional parameters for the ODE solver
    alg=nothing, # Default to nothing, will use Tsit5() if not provided
    reltol=1e-8,
    abstol=1e-9,
    ensure_manifold::Bool=true, # Make sure the trajectory stays on the manifold defined by Lx=q and Nlogx=logK
    kwargs... #other Optional arguments for ODE solver
)::ODESolution
    # println("_logx_traj_with_logqK_change get kwargs: ", kwargs)
    #---Solve the homotopy ODE to find x from qK.---

    n = Bnc.n
    d = Bnc.d
    # Prepare starting x if not given
    startlogx = isnothing(startlogx) ? qK2x(Bnc, startlogqK; input_logspace=true, output_logspace=true) : startlogx
    
    #Homotopy path in log-space( a straight line)
    ΔlogqK = Float64.(endlogqK - startlogqK)

    # Create thread-local copies of all mutable data structures
    logx = Vector{Float64}(undef, n)
    logqK = Vector{Float64}(undef, n)
    logq = @view logqK[1:d]
    logK = @view logqK[d+1:end]
    J= deepcopy(Bnc._LN_sparse)# Make deep copies of sparse matrices to avoid shared state
    J_lu = deepcopy(Bnc._LN_lu)

    logx_J_view = @view logx[Bnc._LN_top_cols] # view for faster updating J
    logq_J_view = @view logqK[Bnc._LN_top_rows] # view for faster updating J
    J_top = @view J.nzval[Bnc._LN_top_idx] # view for faster updating J
    J_top_diag = @view J.nzval[Bnc._LN_top_diag_idx] # view for perturb when J is singular

    #Parameters helps for manifold projection
    # logx_local = Vector{Float64}(undef, n)
    # logx_J_view_local = @view logx_local[Bnc._LN_top_cols]
    # logLx_local = Vector{Float64}(undef, d)
    # logLx_J_view_local = @view logLx_local[Bnc._LN_top_rows]

    # Constants helps for updating mutable datas
    L_nzval = copy(Bnc._LN_sparse.nzval[Bnc._LN_top_idx]) # copy the nzval to avoid shared access

    params = HomotopyParams(ΔlogqK, logx, logqK,logq,logK, J, J_lu, logx_J_view, logq_J_view, J_top, J_top_diag,
        # logx_local,logx_J_view_local,logLx_local, logLx_J_view_local
        )

    if !ensure_manifold
        callback = CallbackSet()
    else
        keep_manifold! = function(resid, u, p)  #  Can not write to forms like log_sum_exp10!(logLx_local, Bnc._L_sparse, u) for Autodiff.
            @unpack logq,logK = p
            resid[1:d] .= log10.(Bnc._L_sparse * exp10.(u)) .- logq
            resid[d+1:end] .= Bnc._N_sparse * u .- logK
        end
        # manifold_jac! = function(J,u,p) # to have the same signature as keep_manifold!()
        #     @unpack logx_local, J,logx_J_view_local, J_top, logLx_local,logLx_J_view_local = p
        #     # update jac for the current logx     
        #     @. logx_local = exp10(u) # though name logx , it is actually x here
        #     logLx_local .= Bnc._L_sparse * logx_local # update logLx_local avoid modify logq that involving in "keep_manifold!"
        #     @. J_top = logx_J_view_local * L_nzval / logLx_J_view_local
        #     return J
        # end

        equilibrium_cb = CB.ManifoldProjection(keep_manifold!;
            save=false,
            resid_prototype=zeros(n),
            # manifold_jacobian=manifold_jac!,
            # jac_prototype = [Bnc.L;Bnc.N],
            autodiff = AutoForwardDiff(),
            abstol=1e-12,
            reltol=1e-10
        )
        callback = CallbackSet(equilibrium_cb)
    end

    @inline function eps_lu(J,max_try=100)
        lu!(J_lu, J,check=false) # recalculate the LU decomposition of J
        try_count = 0
        while !issuccess(J_lu) && try_count < max_try
            @.J_top_diag += eps() # perturb the diagonal elements a bit to avoid singularity
            lu!(J_lu, J,check=false)
            try_count += 1
        end
        if try_count == max_try
            @show logx logqK
            @show J
            @warn("Jacobian is still singular after maximum perturbation attempts.")
        end
    end
    homotopy_process! = function(du, u, p, t)
        @unpack ΔlogqK, logx, logqK, J, J_lu, logx_J_view, logq_J_view, J_top,J_top_diag = p
        #update q & x
        clamp!(u,-Inf,20) # make sure not overflow.
        @. logx = u
        @. logqK = startlogqK + t * ΔlogqK
        #update J_top(sparse version) - use the local copy of nzval
        @. J_top = exp10(logx_J_view - logq_J_view) * L_nzval
        # Update the dlogx
        eps_lu(J)
        ldiv!(du, J_lu, ΔlogqK)
    end
    
    # Define the ODE system for the homotopy process
    # Solve the ODE using the DifferentialEquations.jl package
    tspan = (0.0, 1.0)
    prob = ODE.ODEProblem(homotopy_process!, startlogx, tspan, params)
    sol = ODE.solve(prob, alg; reltol=reltol, abstol=abstol, callback=callback, kwargs...)
    return sol
end


#--------------------------------------------------------------------------------
#      Functions for modeling when envolving catalysis reactions, 
#--------------------------------------------------------------------------------



function x_traj_cat(Bnc::Bnc, qK0_or_q0::Vector{<:Real}, tspan::Tuple{Real,Real};
    K::Union{Vector{<:Real},Nothing}=nothing,
    logK::Union{Vector{<:Real},Nothing}=nothing,
    input_logspace::Bool=false,
    output_logspace::Bool=false,
    kwargs...
    )
    # prepare the qK0 and calculate start logx0
    if isnothing(K) && isnothing(logK)
        @assert length(qK0_or_q0) == Bnc.n "qK0 must have length n, or you shall pass K as keyword argument"
        qK0 = qK0_or_q0
    else
        @assert length(qK0_or_q0)== Bnc.d "q0 must have length d"
        K_prepared = input_logspace ? (isnothing(logK) ? log10.(K) : logK) : (isnothing(K) ? K : exp10.(K))
        qK0 = vcat(qK0_or_q0, K_prepared)
    end
    startlogx = qK2x(Bnc, qK0; input_logspace=input_logspace, output_logspace=true)
    
    #---Solve the ODE to find the time curve of log(x) as catalysis happens
    sol = catalysis_logx(Bnc, startlogx, tspan;
        dense = false, #manually handle later
        kwargs...
    )
    if !output_logspace
        foreach(u -> u .= exp10.(u), sol.u)
    end
    
    return _ode_solution_wrapper(sol)
end

function qK_traj_cat(Bnc::Bnc, args...; only_q::Bool=false, output_logspace::Bool=false, kwargs...)::Tuple{Vector{Float64}, Matrix{Float64}}
    t,u = x_traj_cat(Bnc, args...; output_logspace=true, kwargs...)
    u = x2qK(Bnc, u',input_logspace=true, output_logspace=output_logspace, only_q=only_q)'
    return (t,u)
end


struct TimecurveParam{V<:Vector{Float64},
    SV1<:SubArray,SV2<:SubArray,SV3<:SubArray,SV4<:SubArray}

    x::V # Buffer for x values
    K::V # Buffer for K values
    v::V # Buffer for the catalysis flux vector
    Sv::V # Buffer for the catalysis rate vector multiplied by S
    J::SparseMatrixCSC{Float64,Int} # Jacobian matrix buffer
    J_lu::SparseArrays.UMFPACK.UmfpackLU{Float64,Int} # LU decomposition of J

    x_view::SV1 # View for x
    K_view::SV2 # View for K
    J_top::SV3 # View for the left part of the Jacobian matrix
    J_bottom::SV4 # View for the right part of the Jacobian matrix
end

function catalysis_logx(Bnc::Bnc, logx0::Vector{<:Real}, tspan::Tuple{Real,Real};
    alg=nothing, # Default to nothing, will use Tsit5() if not provided
    reltol=1e-8,
    abstol=1e-9,
    kwargs...
)::ODESolution
    # ---Solve the ODE to find the time curve of log(x) with respect to qK change.---
    if isnothing(Bnc.catalysis.S)||isnothing(Bnc.catalysis.aT)||isnothing(Bnc.catalysis.k)
        @error("S or aT or k is not defined, cannot perform catalysis logx calculation")
    end

    k = Bnc.k
    
    x = Vector{Float64}(undef, Bnc.n)
    K = Vector{Float64}(undef, Bnc.r)
    v = Vector{Float64}(undef, length(k)) # catalysis flux vector
    Sv = Vector{Float64}(undef, Bnc.n) # catalysis rate vector
    J = deepcopy(Bnc._LN_sparse) # Use the sparse version of the Jacobian matrix
    J_lu = deepcopy(Bnc._LN_lu) # LU decomposition of J

    x_view = @view x[Bnc._LN_top_cols]
    K_view = @view K[Bnc._LN_bottom_rows]
    J_top = @view J.nzval[Bnc._LN_top_idx]
    J_bottom = @view J.nzval[Bnc._LN_bottom_idx]
    # create view for the J_buffer , for updating [LΛ_x; Λ_KN]
    params = TimecurveParam(
        x, # x_buffer
        K, # K_buffer
        v, # v buffer / flux
        Sv, # Sv buffer
        J, # J_buffer
        J_lu, # Jt_lu
        #Views for updating J
        x_view,
        K_view,
        J_top,
        J_bottom
    )


    L_nzval = copy(Bnc._LN_sparse.nzval[Bnc._LN_top_idx]) # copy the nzval to avoid shared access
    N_nzval = copy(Bnc._LN_sparse.nzval[Bnc._LN_bottom_idx]) # copy the nzval to avoid shared access
    aT_sparse = Bnc.catalysis._aT_sparse # copy the aT_sparse to avoid shared access
    S_sparse = Bnc.catalysis._S_sparse # copy the S_sparse to avoid shared access
    N_sparse = Bnc._N_sparse # copy the N_sparse to avoid shared access
    _is_change_of_K_involved = Bnc._is_change_of_K_involved

    # Define the ODE system for the time curve
    if _is_change_of_K_involved
        Catalysis_process! = function (dlogx, logx, p, t)
            @unpack x, K, v, Sv, J, J_lu, J_top, J_bottom = p
            #update the values
            x .= exp10.(logx)
            K .= exp10.(N_sparse * logx)

            # Update the Jacobian matrix J
            @. J_top = x_view * L_nzval 
            @. J_bottom = N_nzval * K_view
            lu!(J_lu, J) # recalculate the LU decomposition of J

            # dlogx .= J \ (S * (k .* exp10.(aT * logx)))
            mul!(v, aT_sparse, logx)
            @. v = k * exp10(v) # calculate the catalysis rate vector
            mul!(Sv, S_sparse, v) # reuse x as a temporary buffer, but need change if x is used in other places, like to act call back for ODESolution
            ldiv!(dlogx, J_lu, Sv) # Use the LU decomposition for fast calculation
        end
    else
        # If K is not involved, we can skip the K update
        params.K .= exp10.(N_sparse * logx0) #initialize K_view once
        # @show length(Bnc._Nt_sparse.nzval) length(params.K_view) length(params.Jt_right)
        @. params.J_bottom = N_nzval * params.K_view #initialize Jt_right once

        Catalysis_process! = function (dlogx, logx, p, t)
            @unpack x, v, Sv, J, J_lu, J_top = p
            #update the values
            x .= exp10.(logx)
            # Update the Jacobian matrix J
            @. J_top = x_view * L_nzval
            lu!(J_lu, J) # recalculate the LU decomposition of J

            mul!(v, aT_sparse, logx)
            @. v = k * exp10(v) # calculate the catalysis rate vector
            mul!(Sv, S_sparse, v) # reuse x as a temporary buffer, but need change if x is used in other places, like to act call back for ODESolution
            ldiv!(dlogx, J_lu, Sv)
        end
    end

    # Create the ODE problem
    prob = ODE.ODEProblem(Catalysis_process!, logx0, tspan, params)
    sol = ODE.solve(prob, alg; reltol=reltol, abstol=abstol, kwargs...)
    return sol
end


# ---------------------------------------------------------------Get regime data from resulting matrix---------------------------------------

function get_reaction_order(Bnc::Bnc, x_mat::Matrix{<:Real}, q_mat::Union{Matrix{<:Real},Nothing}=nothing;
    x_idx::Union{Vector{Int},Nothing}=nothing,
    qK_idx::Union{Vector{Int},Nothing}=nothing,
    only_q::Bool=false,
)::Array{Float64,3}
    # Get the reaction order from the resulting matrix, where a regime is calculated for each row by formula.
    # x_mat: Matrix of x values, each row is a different time point
    # q_mat: Matrix of qK values, each row is a different time point
    # x_idx: Indices of x to be calculated, default is all indices
    # qK_idx: Indices of qK to be calculated, default is all indices
    q_mat = isnothing(q_mat) ? x2qK(Bnc, x_mat'; input_logspace=false, output_logspace=false, only_q=true)' : q_mat
    x_idx = isnothing(x_idx) ? (1:Bnc.n) : x_idx
    qK_idx = isnothing(qK_idx) ? (1:Bnc.n) : qK_idx
    if only_q
        qK_idx = qK_idx[findall(i->i<=Bnc.d, qK_idx)] # only keep the indices for q
    end

    flag = length(qK_idx) <= length(x_idx)

    A = sparse(_idx_val2Mtx(collect(x_idx),1,Bnc.n)) # x*n
    B = sparse(_idx_val2Mtx(collect(qK_idx),1,Bnc.n)) # q * n

    regimes = Array{Float64,3}(undef,size(x_mat, 1),length(x_idx), length(qK_idx)) # initialize the regimes array to storage. t have to be the last value to make storage continuous
    tmp_regime = flag ? Matrix{Float64}(undef, Bnc.n, length(qK_idx)) : Matrix{Float64}(undef, Bnc.n, length(x_idx))
    # temporary matrix to store the result of regime
    # Get the regimes from the resulting matrix,where a regime is calculated for each row.
    
    Jt = copy(Bnc._LNt_sparse)
    Jt_lu = copy(Bnc._LNt_lu)

    Jt_left = @view(Jt.nzval[1:length(Bnc._L_sparse.nzval)])
    
    function _update_Jt!(Jt_left, x::AbstractArray{<:Real}, q::AbstractArray{<:Real})
        x_view = @view(x[Bnc._I])
        @. Jt_left = x_view * Bnc._Lt_sparse.nzval ./ q[Bnc._J]
        return nothing
    end

    if flag # qK_idx is shorter
        for (i, (x, q)) in enumerate(zip(eachrow(x_mat), eachrow(q_mat)))
            _update_Jt!(Jt_left, x, q)
            lu!(Jt_lu, Jt) # recalculate the LU decomposition of Jt
            ldiv!(tmp_regime, Jt_lu', B')
            regimes[i,:,:] .= A * tmp_regime
        end
    else
        for (i, (x, q)) in enumerate(zip(eachrow(x_mat), eachrow(q_mat)))
            _update_Jt!(Jt_left, x, q)
            lu!(Jt_lu, Jt) # recalculate the LU decomposition of Jt
            ldiv!(tmp_regime, Jt_lu, A')
            regimes[i,:,:] .= (B * tmp_regime)'
        end
    end
    return regimes
end


function assign_vertex_x(Bnc::Bnc{T}, x::AbstractVector{<:Real};input_logspace::Bool=false,asymptotic::Bool=true)::Vector{T} where T
    # x = input_logspace ? exp10.(x) : x
    L = Bnc._L_sparse
    d = Bnc.d
    n = Bnc.n
    max_indices = zeros(T, d)
    max_val = fill(-Inf, d)
    colptr = L.colptr
    rowval = L.rowval

    if asymptotic
        nzval = @view(x[Bnc._LN_top_cols])
    else
        x = input_logspace ? exp10.(x) : x # linear or log space only matters when not asymptotic
        nzval = @view(x[Bnc._LN_top_cols]) .* L.nzval
    end

    @inbounds for col in 1:n
        col_start_idx = colptr[col]
        col_end_idx   = colptr[col+1] - 1
        if col_start_idx <= col_end_idx #escape empty column
            @inbounds for idx in col_start_idx:col_end_idx
                v = nzval[idx]
                row = rowval[idx]
                if v > max_val[row]
                    max_val[row] = v
                    max_indices[row] = col
                end
            end
        end
    end
    return max_indices
end


# function get_vertex_qK(Bnc::Bnc, x::AbstractMatrix{<:Real}; kwargs...) 
#     [get_vertex_qK_slow(Bnc, row; kwargs...) for row in eachrow(x)]
# end

function assign_vertex_qK(Bnc::Bnc ; x::AbstractVector{<:Real}, input_logspace::Bool=false, kwargs...) 
    # @show all_vertice_idx
    logqK = x2qK(Bnc,x; input_logspace=input_logspace, output_logspace=true)
    return assign_vertex_qK(Bnc, logqK; input_logspace=true, kwargs...)
end

function assign_vertex_qK(Bnc::Bnc, qK::AbstractVector{<:Real}; input_logspace::Bool=false, asymptotic::Bool=true, eps=0) 
    real_only = asymptotic ? true : nothing
    all_vertice_idx = get_vertices(Bnc, singular=false, asymptotic = real_only, return_idx = false)
    # @show all_vertice_idx
    logqK = input_logspace ? qK : log10.(qK)
    
    record = Vector{Float64}(undef,length(all_vertice_idx))
    for (i, idx) in enumerate(all_vertice_idx)
        C, C0 = get_C_C0_qK!(Bnc, idx) 
        
        min_val = if !asymptotic
            minimum(C * logqK .+ C0)
        else
            minimum(C * logqK)
        end
        # @show idx
        # @show min_val
        record[i] = min_val

        if record[i] >= -eps
            return idx
        end
    end
    @warn("All vertex conditions failed for logqK=$logqK. Returning the best-fit vertex.")
    return all_vertice_idx[findmax(record)[2]]
end




function have_perm(Bnc::Bnc, perm::Vector{<:Integer})
    """
    Check if the vertex represented by perm is within the Bnc.
    """
    find_all_vertices!(Bnc)
    return haskey(Bnc.vertices_idx, perm)
end

#-----------------------------------------------------------------
# Function of calculating volume of vertices
#-----------------------------------------------------------------
# function calc_volume(C::AbstractMatrix{<:Real}, C0::AbstractVector{<:Real}; 
#     confidence_level::Float64=0.95,
#     N=1_000_000,
#     batch_size::Int=100_000,
#     log_lower=-6,
#     log_upper=6
# )::Tuple{Float64,Float64}
#     N = Int(N)

#     n = size(C, 2)
#     dist = Uniform(log_lower, log_upper)

#     n_batches = cld(N, batch_size)  # 向上取整批次数
#     counts = zeros(Int, n_batches)  # 每批结果

#     Threads.@threads for b in 1:n_batches
#         m = (b == n_batches) ? (N - (n_batches-1)*batch_size) : batch_size
#         samples = rand(dist, n, m)
#         vals = C * samples .+ C0

#         local_count = 0
#         @inbounds for j in 1:m
#             if all(@view(vals[:, j]) .> 0)
#                 local_count += 1
#             end
#         end
#         counts[b] = local_count
#     end

#     count = sum(counts)
#     P_hat = count / N
#     z = quantile(Normal(), (1 + confidence_level) / 2)

#     # Wilson 置信区间
#     denom = 1 + z^2 / N
#     center = (P_hat + z^2/(2N)) / denom
#     margin = (z / denom) * sqrt(P_hat*(1-P_hat)/N + z^2/(4N^2))

#     return (center, margin)
# end

# The core calculate function:
# function calc_volume(Cs::AbstractVector{<:AbstractMatrix{<:Real}}, C0s::AbstractVector{<:AbstractVector{<:Real}};
#     confidence_level::Float64=0.95,
#     N::Int=1_000_000,
#     batch_size::Int=100_000,
#     log_lower=-6,
#     log_upper=6,
#     tol::Float64=1e-10,
# )::Vector{Tuple{Float64,Float64}}

#     n_batches = cld(N, batch_size)
#     dist = Uniform(log_lower, log_upper)
#     n_threads = Threads.nthreads()

#     n = size(Cs[1], 2)

#     # --- 每线程局部计数 ---
#     thread_counts = [zeros(Int, length(Cs)) for _ in 1:n_threads]

#     Threads.@threads for b in 1:n_batches
#         m = (b == n_batches) ? (N - (n_batches-1)*batch_size) : batch_size
#         samples = rand(dist, n, m)
#         local_counts = thread_counts[Threads.threadid()]

#         @inbounds for j in 1:m
#             @views x = samples[:, j]
#             for i in eachindex(Cs)
#                 @views A = Cs[i]
#                 @views b = C0s[i]
#                 if all(A * x .+ b .> - tol)
#                     local_counts[i] += 1
#                 end
#             end
#         end
#     end

#     # --- 汇总 ---
#     total_counts = zeros(Int, length(Cs))
    
#     for c in thread_counts
#         @inbounds total_counts .+= c
#     end
#     # --- Wilson 区间 ---
#     function get_center_margin(count::Int, N::Int)
#         if count == 0
#             return (0.0, 0.0)
#         end
#         P_hat = count / N
#         z = quantile(Normal(), (1 + confidence_level) / 2)
#         denom = 1 + z^2 / N
#         center = (P_hat + z^2/(2*N)) / denom
#         margin = (z / denom) * sqrt(P_hat*(1 - P_hat)/N + z^2/(4*N^2))
#         return center, margin
#     end

#     @show total_counts
#     return [get_center_margin(c, N) for c in total_counts]
# end

function calc_volume(Cs::AbstractVector{<:AbstractMatrix{<:Real}},
                     C0s::AbstractVector{<:AbstractVector{<:Real}};
    confidence_level::Float64 = 0.95,
    contain_overlap::Bool = false,
    batch_size::Int = 100_000,
    log_lower = -6,
    log_upper = 6,
    tol::Float64 = 0.0,
    rel_tol::Float64 = 0.005,  # 相对误差阈值
    time_limit::Float64 = 40.0,
)::Vector{Tuple{Float64, Float64}}

    @assert length(Cs) == length(C0s) "Cs and C0s must have same length"
    n_regimes = length(Cs)
    @info "Number of polyhedra to calc volume: $n_regimes"
    n_threads = Threads.nthreads()
    n_dim = size(Cs[1], 2)
    dist = Uniform(log_lower, log_upper)

    total_counts = zeros(Int, n_regimes)
    total_N = 0
    stats = fill((0.0, 0.0), n_regimes)
    rel_errors = fill(Inf, n_regimes)
    active = trues(n_regimes)  # 哪些 regime 仍在采样中

    # 每线程本地计数，避免锁
    thread_counts = [zeros(Int, n_regimes) for _ in 1:n_threads]

    start_time = time()
    z = quantile(Normal(), (1 + confidence_level) / 2)

    @inline function get_center_margin(count::Int, N::Int)
        if count == 0
            return 0.0, 0.0
        end
        P_hat = count / N
        denom = 1 + z^2 / N
        center = (P_hat + z^2 / (2 * N)) / denom
        margin = (z / denom) * sqrt(P_hat * (1 - P_hat) / N + z^2 / (4 * N^2))
        return center, margin
    end

    # --- 主循环 ---
    while true
        elapsed = time() - start_time
        if elapsed > time_limit
            @info "Reached time limit ($(round(elapsed, digits=2)) s). Stopping."
            break
        end
        if !any(active)
            @info "All regimes converged after $total_N samples."
            break
        end

        # 生成一批样本（列是样本）
        samples = rand(dist, n_dim, batch_size)

       Threads.@threads for j in 1:batch_size
            tid = Threads.threadid()
            local_counts = thread_counts[tid]
            @views x = samples[:, j]

            # 对当前点，只检查 active 的 regimes
            # 如果 contain_overlap == false：当找到一个满足的 regime，记录并 break（一个点只属于一个）
            # 如果 contain_overlap == true：记录所有满足的 regimes（允许多分配）
            for i in eachindex(Cs)
                if !active[i]
                    continue
                end
                @views A = Cs[i]
                @views b = C0s[i]

                # 计算 A*x + b 的每个分量是否都 > -tol
                # 使用 all(...) 简洁直观（可能会分配一个临时数组在某些情况下，但通常 A*x+b 是向量）
                vals = A * x .+ b
                if any(vals .< -tol)
                    # 不满足当前 regime：不管 contain_overlap 与否，都去检查下一个 regime
                    continue
                end

                # 满足当前 regime
                local_counts[i] += 1
                if !contain_overlap
                    break  # 当不允许重叠时，属于一个 regime 后停止检查其它 regime
                else
                    # 若允许重叠，继续检查其它 regimes
                end
            end
        end

        # 汇总线程统计并清零线程本地计数
        for c in thread_counts
            total_counts .+= c
            fill!(c, 0)
        end
        total_N += batch_size

        # 更新置信区间与相对误差，仅对 active 的 regimes
        for i in eachindex(Cs)
            if !active[i]
                continue
            end
            center, margin = get_center_margin(total_counts[i], total_N)
            stats[i] = (center, margin)
            # 相对误差用 margin/center；若 center==0 则看成无穷（还没观测到）
            rel_errors[i] = (center == 0.0) ? Inf : (margin / center)
            if rel_errors[i] <= rel_tol
                active[i] = false
            end
        end
    end

    elapsed = time() - start_time
    @info "Total samples: $total_N, Elapsed: $(round(elapsed, digits=2)) s"
    return stats
end

calc_volume(C::AbstractMatrix{<:Real}, C0::AbstractVector{<:Real}; kwargs...)::Tuple{Float64,Float64} = calc_volume([C], [C0]; kwargs...)[1]

function calc_volume(polys::AbstractVector{<:Polyhedron};
    asymptotic::Bool=true,
    kwargs...
)::Vector{Tuple{Float64,Float64}}
    # --- 提前提取每个 poly 的 (A, b) ---
    full_dims = fulldim(polys[1])

    function repolyhedron(poly::Polyhedron)
        (A,b,linset) = MixedMatHRep(hrep(poly)) |> p->(p.A, p.b,p.linset)
        p_new = hrep(A, zeros(size(b)), linset) |> x-> polyhedron(x,CDDLib.Library())
        return p_new
    end

    if asymptotic
        polys = repolyhedron.(polys)
    end
    full_dim_idx = findall(v -> v == full_dims, dim.(polys))

    reps = polys[full_dim_idx] .|> poly -> MixedMatHRep(hrep(poly)) |> p->(p.A, p.b)
    vals = collect(zip(zeros(Float64, length(polys)), zeros(Float64, length(polys))))

    C = [ -rep[1] for rep in reps ]
    C0 = [ rep[2] for rep in reps ]

    vals[full_dim_idx] .= calc_volume(C, C0; kwargs...)
    return vals
end
calc_volume(poly::Polyhedron;kwargs...)::Tuple{Float64,Float64} = calc_volume([poly]; kwargs...)[1]

function calc_volume(model::Bnc;
    asymptotic::Bool=true,
    kwargs...
)::Vector{Tuple{Float64, Float64}}
    # get index for those worth calculate volume
    idxs = get_vertices(model, singular=false, 
        asymptotic= asymptotic ? true : nothing, return_idx=true)

    # get C and C0 for each vertex
    Cs = [get_C_qK!(model,idx) for idx in idxs]

    if asymptotic # get only asymptotic vertices
        C0s = [zeros(size(C,1)) for C in Cs]
    else # get all non-singular vertices
        C0s = [get_C0_qK!(model,idx) for idx in idxs]
    end

    # calculate volume for each vertex
    # Initialize the value
    vals = collect(zip(zeros(Float64, length(model.vertices_perm)), zeros(Float64, length(model.vertices_perm))))
    # fill the value
    vals[idxs] .= calc_volume(Cs, C0s; kwargs...)
    return vals
end



function calc_volume(Bnc::Bnc, perm;
    asymptotic::Bool=true, 
    kwargs...
)::Tuple{Float64,Float64} # singular/ asymptotic not be put here, as dimensions could reduce and change.
    
    if is_singular(Bnc,perm)
        return (0.0, 0.0)
    end
    
    if asymptotic && !is_asymptotic(Bnc,perm)
        return (0.0, 0.0)
    end

    C, C0 = get_C_C0_qK!(Bnc, perm)
    
    if asymptotic
        C0 .= 0.0
    end
    center, margin = calc_volume(C, C0; kwargs...)
    return (center, margin)
end













# Trying speedup assign_vertex_qK, but not success yet.
function get_i_j(model::Bnc,perm::Vector{<:Integer}, t::Integer)
    i = findfirst(>(t),model._C_partition_idx) - 1
    j1 = perm[i]
    cth = t - model._C_partition_idx[i] + 1
    j2 = model._valid_L_idx[i][cth]
    j2 < j1 ? nothing : j2 += 1
    return i, j1, j2
end




function assign_vertex_qK_test(Bnc::Bnc{T}, qK::AbstractVector{<:Real};
                               input_logspace::Bool=false,
                               asymptotic::Bool=true, eps=0) where T
    logqK = input_logspace ? qK : log10.(qK)
    Perm_tried = Set{UInt64}()  # 存放哈希值

    function try_perm!(perm1)
        (C, C0) = get_C0_qK!(Bnc, perm1)
        err = C * logqK .+ C0
        ts = findall(er -> er <= -eps, err)

        # 没有违反不等式，返回
        if isempty(ts)
            return perm1
        end

        h = hash(perm1)
        if h in Perm_tried
            error("Cyclic permutation detected! Tried permutations: $(collect(Perm_tried))")
        end
        push!(Perm_tried, h)

        # 对所有违反的约束更新 perm
        for t in ts
            i, j1, j2 = get_i_j(Bnc, perm1, t)
            perm1[i] = j2
            if !haskey(Bnc.vertices_perm_dict, perm1) 
                perm1[i] = j1  # 恢复原值
            end
            try_perm!(perm1)
        end
        # else
        #         @show perm1
    end

    # 假设初始 perm1 为 1:Bnc.d 或者外部传入
    perm0 = collect(1:Bnc.d) .|> x->T(x)
    return try_perm!(perm0)
end


