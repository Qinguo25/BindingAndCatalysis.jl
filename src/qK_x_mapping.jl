# ----------------Functions for mapping between qK space and x space----------------------------------

"""
    x2qK(bnc::Bnc, x; input_logspace=false, output_logspace=false, only_q=false)

Map concentrations `x` to totals/binding constants `qK`.

# Arguments
- `bnc`: Binding network model.
- `x`: Species concentrations (vector or matrix).

# Keyword Arguments
- `input_logspace`: Treat `x` as log10 values when `true`.
- `output_logspace`: Return log10 values when `true`.
- `only_q`: Return only `q` (conservation totals) when `true`.

# Returns
- Vector or array containing `q` (and `K` if `only_q=false`).
"""
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



"""
    qK2x(bnc::Bnc, qK; K=nothing, logK=nothing, input_logspace=false, output_logspace=false,
        startlogx=nothing, startlogqK=nothing, use_vtx=false, method=:homotopy,
        reltol=1e-8, abstol=1e-10, kwargs...) -> Vector

Map from totals/binding constants (`qK`) to species concentrations `x`.

# Arguments
- `bnc`: Binding network model.
- `qK`: Vector of totals (and optionally binding constants).

# Keyword Arguments
- `K`: Binding constants in linear space.
- `logK`: Binding constants in log10 space.
- `input_logspace`: Treat inputs as log10 values when `true`.
- `output_logspace`: Return log10 values when `true`.
- `startlogx`: Initial guess for log10(x).
- `startlogqK`: Initial log10(qK) for homotopy.
- `use_vtx`: Use regime-based closed form when `true`.
- `method`: Solver method (`:homotopy` or NonlinearSolve symbol).
- `reltol`, `abstol`: Solver tolerances.
- `kwargs...`: Passed through to the solver.

# Returns
- Vector of `x` values in log10 or linear space.
"""
function qK2x(Bnc::Bnc, qK::AbstractVector{<:Real};
    input_logspace::Bool=false,
    output_logspace::Bool=false,
    startlogx::Union{Vector{<:Real},Nothing}=nothing,
    startlogqK::Union{Vector{<:Real},Nothing}=nothing,
    use_vtx::Bool=false,
    method::Union{Symbol,Missing} = :homotopy,
    reltol = 1e-8,
    abstol = 1e-10,
    kwargs...
)::Vector{Float64}
    # Map from qK space to x space using homotopy or nonlinear solving.
    #---Solve the homotopy ODE to find x from qK.---

    # Define the start point 
    

    endlogqK = input_logspace ? qK : log10.(qK)

    logx = if use_vtx
            perm = assign_vertex_qK(Bnc,endlogqK; input_logspace=true,asymptotic_only=false)
            H,H0 = get_H_H0(Bnc,perm)
            H* endlogqK .+ H0
        elseif ismissing(method) || method != :homotopy
            _logqK2logx_nlsolve(Bnc,
                endlogqK;
                startlogx = isnothing(startlogx) ? copy(Bnc._anchor_log_x) : Float64.(startlogx),
                method=method,
                reltol=reltol,
                abstol=abstol,
                kwargs...
            )
        else
            if isnothing(startlogqK) || isnothing(startlogx)
                # If no starting point is provided, use the default
                # Make deep copies to avoid shared state in threaded environment
                startlogx = copy(Bnc._anchor_log_x)
                startlogqK = copy(Bnc._anchor_log_qK)
            end
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
            sol.u[end]
        end

    logx = output_logspace ? logx : exp10.(logx)
    return logx
end

"""
    qK2x(bnc::Bnc, qK::AbstractArray{<:Real,2}; kwargs...) -> AbstractArray

Batch mapping from qK space to x space for each column of `qK`.
"""
function qK2x(Bnc::Bnc, qK::AbstractArray{<:Real,2};kwargs...)::AbstractArray{<:Real}
    # batch mapping of qK2x for each column of qK and return as matrix.
    # Make thread-safe by creating separate copies for each thread
    f = x -> qK2x(Bnc, x; kwargs...)
    return matrix_iter(f, qK;byrow=false,multithread=true)
end















#----------------------------------------------------------------
# Playground for mapping different methods for solving the nonlinear system
# of equations to find x from qK.
#-----------------------------------------------------------------


"""
    _logqK2logx_nlsolve(bnc::Bnc, logqK; startlogx=nothing, method=missing, kwargs...) -> Vector

Solve for `logx` given `logqK` using a nonlinear solver.

# Arguments
- `bnc`: Binding network model.
- `logqK`: Log10 values of q and K.

# Keyword Arguments
- `startlogx`: Initial guess for log10(x).
- `method`: NonlinearSolve algorithm symbol.
- `kwargs...`: Passed through to `solve`.

# Returns
- Estimated log10(x) vector.
"""
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
    x_M_view = @view x[Bnc._LN_top_cols] # view for faster updating J
    q_M_view = @view q[Bnc._LN_top_rows] # view for faster updating J
    M_top = @view J.nzval[Bnc._LN_top_idx] # view for faster updating J
    L_nzval = copy(Bnc._LN_sparse.nzval[Bnc._LN_top_idx])

    params = (; x, q, logq, logK, J, x_M_view, q_M_view, M_top)


    keep_manifold! = function(resid, u, p) 
        logq, logK = p
        resid[1:d] .= log10.(Bnc._L_sparse * exp10.(u)) .- logq
        resid[d+1:end] .= Bnc._N_sparse * u .- logK
        return resid
    end

    manifold_jac! = function(J,u,p) # to have the same signature as keep_manifold!()
        @unpack x,q,logq,J,x_M_view,q_M_view, M_top = p
        # update jac for the current logx     
        @. x = exp10(u) # update x
        q .= Bnc._L_sparse * x #update q
        @. M_top = x_M_view * L_nzval / q_M_view
        return J
    end

    prob = NonlinearProblem(keep_manifold!, startlogx, params; resid_prototype=zeros(n), jac = manifold_jac!, jac_prototype=J)
    
    sol = solve(prob, method; kwargs...)
    if !SciMLBase.successful_retcode(sol.retcode)
        @warn("Nonlinear solver did not converge successfully. Retcode: $(sol.retcode)")
    end
    return sol.u
end








#----------------Functions using homotopyContinuous to moving across x space along with qK change----------------------

"""
    x_traj_with_qK_change(bnc::Bnc, start_point, end_point; input_logspace=false, output_logspace=false, kwargs...)

Compute a trajectory in `x` space while `qK` changes linearly in log10 space.

# Arguments
- `bnc`: Binding network model.
- `start_point`: Starting `qK` values.
- `end_point`: Ending `qK` values.

# Keyword Arguments
- `input_logspace`: Treat inputs as log10 values when `true`.
- `output_logspace`: Return `x` in log10 space when `true`.
- `kwargs...`: Passed to the ODE solver.

# Returns
- Tuple `(t, x_traj)` containing time points and state vectors.
"""
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

    solution = _logx_traj_with_logqK_change(Bnc, startlogqK, endlogqK; dense=false, kwargs...)

    if !output_logspace
        foreach(u -> u .= exp10.(u), solution.u)
    end

    return _ode_solution_wrapper(solution)
end


"""
    x_traj_with_q_change(bnc::Bnc, start_q, end_q; K=nothing, logK=nothing, input_logspace=false, kwargs...)

Compute an `x` trajectory for a change in `q` while holding `K` fixed.
"""
function x_traj_with_q_change(
    Bnc::Bnc,
    start_q::Vector{<:Real},
    end_q::Vector{<:Real};
    K::Union{Vector{<:Real},Nothing}=nothing,
    logK::Union{Vector{<:Real},Nothing}=nothing,
    input_logspace::Bool=false,
    kwargs...
)
    K_prepared = input_logspace ? (isnothing(logK) ? log10.(K) : logK) : (isnothing(K) ? K : exp10.(K))
    x_traj_with_qK_change(Bnc, [start_q;K_prepared], [end_q;K_prepared]; input_logspace=input_logspace,kwargs...)
end



"""
    HomotopyParams

Cache container for homotopy-based qK→x integration.
"""
struct HomotopyParams{V<:Vector{Float64},SV1<:SubArray,SV2<:SubArray}
    ### Constants
    startlogqK::V
    ΔlogqK::V
    logx::V
    logqK::V
    logq::SV1
    logK::SV1
    logqK_max::Float64

    M::SparseMatrixCSC{Float64,Int} 
    M_lu::SparseArrays.UMFPACK.UmfpackLU{Float64,Int}

    logx_M_view::SV2
    logq_M_view::SV2
    M_top::SV2
    M_top_diag::SV2
end


"""
    get_homotopy_param(bnc::Bnc, startlogqK, endlogqK; startlogx=nothing)

构造 homotopy ODE 所需的参数/缓存（线程局部可变对象）。

返回：
- p::HomotopyParams
- startlogqK0::Vector{Float64}  （用于 ODE 右端项里构造 logqK(t)）
- startlogx0::Vector{Float64}   （ODE 初值）
"""
function get_homotopy_param(Bnc::Bnc, startlogqK::Vector{<:Real}, endlogqK::Vector{<:Real})
    logqK_max = maximum([20.0,maximum(startlogqK), maximum(endlogqK)])
    n = Bnc.n
    d = Bnc.d
    startlogqK = Float64.(startlogqK)
    ΔlogqK = Float64.(endlogqK - startlogqK)
    # Create thread-local copies of all mutable data structures
    logx = Vector{Float64}(undef, n)
    logqK = Vector{Float64}(undef, n)
    logq = @view logqK[1:d]
    logK = @view logqK[d+1:end]
    M= deepcopy(Bnc._LN_sparse)# Make deep copies of sparse matrices to avoid shared state
    M_lu = deepcopy(Bnc._LN_lu)

    logx_M_view = @view logx[Bnc._LN_top_cols] # view for faster updating J
    logq_M_view = @view logqK[Bnc._LN_top_rows] # view for faster updating J
    M_top = @view M.nzval[Bnc._LN_top_idx] # view for faster updating J
    M_top_diag = @view M.nzval[Bnc._LN_top_diag_idx] # view for perturb when J is singular

    p = HomotopyParams(startlogqK, ΔlogqK, logx, logqK,logq,logK, logqK_max, M, M_lu, 
        logx_M_view, logq_M_view, M_top, M_top_diag
        # logx_local,logx_M_view_local,logLx_local, logLx_M_view_local
        )
    return p
end


function get_homotopy_ode(Bnc::Bnc)
    # Constants helps for updating mutable datas
    L_nzval = log10.(Bnc._LN_sparse.nzval[Bnc._LN_top_idx]) # copy the nzval to avoid shared access

    @inline function update_M_lu(M_lu,M,max_try=100)
        lu!(M_lu, M,check=false) # recalculate the LU decomposition of J
        try_count = 0
        while !issuccess(M_lu) && try_count < max_try
            @.M_top_diag += eps() # perturb the diagonal elements a bit to avoid singularity
            lu!(M_lu, M,check=false)
            try_count += 1
        end
        if try_count == max_try
            @error("M is still singular after maximum perturbation attempts.")
            @show M
        end
    end

    function(du, u, p, t)
        @unpack startlogqK, ΔlogqK, logx, logqK,logqK_max, M, M_lu, logx_M_view, logq_M_view, M_top,M_top_diag = p
        #update q & x
        clamp!(u,-Inf,logqK_max) # make sure not overflow.
        @. logx = u
        @. logqK = startlogqK + t * ΔlogqK
        #update M_top(sparse version) - use the local copy of nzval
        @. M_top = exp10(logx_M_view - logq_M_view + L_nzval)
        # Update the dlogx
        update_M_lu(M_lu,M)
        ldiv!(du, M_lu, ΔlogqK)
    end
end


"""
    _logx_traj_with_logqK_change(bnc::Bnc, startlogqK, endlogqK; startlogx=nothing,
        alg=nothing, reltol=1e-8, abstol=1e-9, ensure_manifold=true, npoints=nothing, kwargs...) -> ODESolution

Integrate a homotopy path in log space to map qK changes to x trajectories.
"""
function _logx_traj_with_logqK_change(Bnc::Bnc,
    startlogqK::Vector{<:Real},
    endlogqK::Vector{<:Real};
    # Optional parameters for the initial log(x) values,act as initial point for ode solving
    startlogx::Union{Vector{<:Real},Nothing}=nothing,
    # Optional parameters for the ODE solver
    alg=nothing, # Default to nothing, will use Tsit5() if not provided
    reltol=1e-8,
    abstol=1e-9,
    ensure_manifold::Bool=true, # Make sure the trajectory stays on the manifold defined by Lx=q and Nlogx=logK
    npoints::Union{Nothing, Integer}=nothing,
    kwargs... #other Optional arguments for ODE solver
)::ODESolution
    # println("_logx_traj_with_logqK_change get kwargs: ", kwargs)
    #---Solve the homotopy ODE to find x from qK.---

    
    # Prepare starting x if not given
    u0 = isnothing(startlogx) ? qK2x(Bnc, startlogqK; input_logspace=true, output_logspace=true) : startlogx
    p = get_homotopy_param(Bnc, startlogqK, endlogqK)
    f! = get_homotopy_ode(Bnc)

    callback = if !ensure_manifold
            CB.CallbackSet()
        else
            n = Bnc.n
            d = Bnc.d
            keep_manifold! = function(resid, u, p)  #  Can not write to forms like log_sum_exp10!(logLx_local, Bnc._L_sparse, u) for Autodiff.
                @unpack logq,logK = p
                resid[1:d] .= log10.(Bnc._L_sparse * exp10.(u)) .- logq
                resid[d+1:end] .= Bnc._N_sparse * u .- logK
            end
            equilibrium_cb = CB.ManifoldProjection(keep_manifold!;
                save=false,
                resid_prototype=zeros(n),
                # manifold_jacobian=manifold_jac!,
                # jac_prototype = [Bnc.L;Bnc.N],
                autodiff = AutoForwardDiff(),
                abstol=1e-12,
                reltol=1e-10
            )
            CB.CallbackSet(equilibrium_cb)
        end

    # Solve the ODE using the DifferentialEquations.jl package

    prob = ODE.ODEProblem(f!, u0, (0.0, 1.0), p)

    sol =  if isnothing(npoints) 
                ODE.solve(prob, alg; reltol=reltol, abstol=abstol, callback=callback, kwargs...)
            else
                ODE.solve(prob, alg; reltol=reltol, abstol=abstol, callback=callback,
                saveat=range(0,1,npoints),tstops=range(0,1,npoints),
                 kwargs...)
            end
    return sol
end




#--------------------------------------------------------------------------------
#      Functions for modeling when envolving catalysis reactions, 
#--------------------------------------------------------------------------------



"""
    x_traj_cat(bnc::Bnc, qK0_or_q0, tspan; K=nothing, logK=nothing,
        input_logspace=false, output_logspace=false, kwargs...) -> (Vector, Vector)

Simulate species trajectories under catalysis dynamics.
"""
function x_traj_cat(Bnc::Bnc, x0::Vector{<:Real}, tspan::Tuple{Real,Real};
    input_logspace::Bool=false,
    output_logspace::Bool=false,
    kwargs...
    )
    x0 = input_logspace ? x0 : log10.(x0)
    # startlogx = qK2x(Bnc, qK0; input_logspace=input_logspace, output_logspace=true)
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

"""
    qK_traj_cat(bnc::Bnc, args...; only_q=false, output_logspace=false, kwargs...) -> (Vector{Float64}, Matrix{Float64})

Simulate catalysis dynamics and return trajectories in q/K space.
"""
function qK_traj_cat(Bnc::Bnc, qK0::Vector{<:Real}, args...;
    only_q::Bool=false,
    input_logspace::Bool=false,
    output_logspace::Bool=false,
    )

    logx0 = qK2x(Bnc, qK0; input_logspace=input_logspace, output_logspace=true)
    t,u = x_traj_cat(Bnc, logx0, args...; output_logspace=true, kwargs...)
    u = x2qK.(Ref(Bnc), u;input_logspace=true,output_logspace=output_logspace, only_q=only_q)
    return (t,u)
end
q_traj_cat(args...;kwargs...) = qK_traj_cat(args...;only_q=true,kwargs...)


function have_catalysis(model::Bnc)
    return !isnothing(model.catalysis)
end
"""
    TimecurveParam
    ### Constant
    # k: R^rcat

    ### Cache
    # x: R^d, x
    # v: R^rcat, Λ_k x^aT, 
    # f: R^n, first d value: Sv 
    # f: R^n, first d value: Λ_q^-1 S x^aT
    # M, M_lu

Cache container for catalysis time-course integration.
"""
struct TimecurveParam{V<:Vector{Float64}}
    k::V # catalysis constant  
    x::V 
    v::V 
    f::V 
    M::SparseMatrixCSC{Float64,Int} # Jacobian matrix buffer
    M_lu::SparseArrays.UMFPACK.UmfpackLU{Float64,Int} # LU decomposition of J
end

"""
Get the catalysis parameter for ODE f construction
"""
function get_catalysis_param(model::Bnc,k)
    @assert have_catalysis(Bnc) "Should fill catalysis data first"
    k = Float64.(k)
    x = Vector{Float64}(undef, model.n)
    v = Vector{Float64}(undef, length(k)) # catalysis flux vector
    f = Vector{Float64}(undef, Bnc.n) # catalysis rate vector
    M = deepcopy(Bnc._LN_sparse) # Use the sparse version of the Jacobian matrix
    M_lu = deepcopy(Bnc._LN_lu) # LU decomposition of J
    # create view for the M_buffer , for updating [LΛ_x; Λ_KN]
    TimecurveParam(k, x, v, f, M, M_lu)
end

"""
return the f(du,u,p,t) for ODE solver
"""
function get_catalysis_ode(model::Bnc)
    @assert have_catalysis(Bnc) "Should fill catalysis data first"
    L_nzval = log10.(model._LN_sparse.nzval[model._LN_top_idx]) # copy the nzval to avoid shared access
    
    @inline function update_M_lu(M_lu,M,max_try=100)
        lu!(M_lu, M,check=false) # recalculate the LU decomposition of J
        try_count = 0
        while !issuccess(M_lu) && try_count < max_try
            @.M.nzval[model._LN_top_diag_idx] += eps() # perturb the diagonal elements a bit to avoid singularity
            lu!(M_lu, M,check=false)
            try_count += 1
        end
        if try_count == max_try
            @error("Jacobian is still singular after maximum perturbation attempts.")
            @show M
        end
    end

    function f(du,u,p::TimecurveParam, t) 
        @unpack x, k,  v, f, M, M_lu = p
        ### Constant
        # k: R^rcat

        ### Cache
        # x: R^d
        # v: R^rcat, Λ_k x^aT, 
        # f: R^d, Sv 
        # f: R^d Λ_q^-1 S x^aT
        # M, M_lu
        # using log x to updata the flux vector
        mul!(v, model.catalysis.aT_sparse, u)   # aT * log x 
        @. v = k * exp10(v) # v = Λ_k x^aT
        mul!(f, model.catalysis.S_sparse, v) # Sv = S * Λ_k x^aT

        x .= exp10.(u) # x
        mul!(q, model.L, x)  # q = Lx
        @. v[1:model.d] = v/q # Λ_q^{-1} * S * Λ_k x^aT

        # update M_lu
        q .= log10.(q)
        @. M.nzval[Bnc._LN_top_idx] = exp10(@view(u[model._LN_top_cols]) - @view(q[model._LN_top_rows]) + L_nzval)
        update_M_lu(M_lu, M) # recalculate the LU decomposition of J

        # calculate du
        ldiv!(du, M_lu, v) # Use the LU decomposition for fast calculation
    end
end


"""
    catalysis_logx(bnc::Bnc, logx0, tspan; alg=nothing, reltol=1e-8, abstol=1e-9, kwargs...) -> ODESolution

Solve the catalysis ODE system in log space.
"""
function catalysis_logx(Bnc::Bnc, logx0::Vector{<:Real}, tspan::Tuple{Real,Real};
    k::AbstractVector{<:Real},
    alg=nothing, # Default to nothing, will use Tsit5() if not provided
    reltol=1e-8,
    abstol=1e-9,
    kwargs...
)::ODESolution
    # ---Solve the ODE to find the time curve of log(x) with respect to qK change.---
    p = get_catalysis_param(Bnc, k)
    f = get_catalysis_ode(Bnc)
    # Create the ODE problem
    prob = ODE.ODEProblem(f, logx0, tspan, p)
    sol = ODE.solve(prob, alg; reltol=reltol, abstol=abstol, kwargs...)
    return sol
end
