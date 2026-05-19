export x2qK, qK2x, qK2x_residual, benchmark_qK2x_methods
export x_traj_with_qK_change, x_traj_with_q_change, x_traj_cat, qK_traj_cat, q_traj_cat, catalysis_logx
export qcat_traj_cat, simulate_adaptation

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
                K = Bnc.N * x
                q = log10.(Bnc.L * exp10.(x))
            else
                K = exp10.(Bnc.N * x)
                q = Bnc.L * exp10.(x)
            end
        else
            if output_logspace
                K = Bnc.N * log10.(x)
                q = log10.(Bnc.L * x)
            else
                K = exp10.(Bnc.N * log10.(x))
                q = Bnc.L * x
            end
        end
        return vcat(q, K)
    else
        if input_logspace
            if output_logspace
                q = log10.(Bnc.L * exp10.(x))
            else
                q = Bnc.L * exp10.(x)
            end
        else
            if output_logspace
                q = log10.(Bnc.L * x)
            else
                q = Bnc.L * x
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
- `method`: Solver method. Supported built-ins are `:homotopy`, `:free_energy`,
  `:newton_nullspace`, `:nlsolve`, and `:regime`.
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
    method::Symbol = :homotopy,
    reltol = 1e-8,
    abstol = 1e-10,
    kwargs...
)::Vector{Float64}
    endlogqK = input_logspace ? qK : log10.(qK)

    logx = if use_vtx || method === :regime
            _logqK2logx_regime(Bnc, endlogqK)
        elseif method === :homotopy
            helper = _integration_helper!(Bnc)
            
            if isnothing(startlogqK) || isnothing(startlogx)
                # If no starting point is provided, use the default
                # Make deep copies to avoid shared state in threaded environment
                startlogx = copy(helper._anchor_log_x)
                startlogqK = copy(helper._anchor_log_qK)
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
        elseif method === :free_energy
            _logqK2logx_free_energy(
                Bnc,
                endlogqK;
                startlogx=startlogx,
                reltol=reltol,
                abstol=abstol,
                kwargs...,
            )
        elseif method === :newton_nullspace || method === :nullspace
            _logqK2logx_nullspace_newton(
                Bnc,
                endlogqK;
                startlogx=startlogx,
                reltol=reltol,
                abstol=abstol,
                kwargs...,
            )
        elseif method === :nlsolve
            helper = _integration_helper!(Bnc)
            _logqK2logx_nlsolve(Bnc,
                endlogqK;
                startlogx = isnothing(startlogx) ? copy(helper._anchor_log_x) : Float64.(startlogx),
                reltol=reltol,
                abstol=abstol,
                kwargs...
            )
        else
            throw(ArgumentError("unsupported qK2x method: $method"))
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



"""
    _logqK2logx_nlsolve(bnc::Bnc, logqK; startlogx=nothing, method=missing, kwargs...) -> Vector

Solve for `logx` given `logqK` using a nonlinear solver.

# Arguments
- `bnc`: Binding network model.
- `logqK`: Log10 values of q and K.

# Keyword Arguments
- `startlogx`: Initial guess for log10(x).
- `kwargs...`: Passed through to `solve`.

# Returns
- Estimated log10(x) vector.
"""
function _logqK2logx_nlsolve(Bnc::Bnc, logqK::AbstractArray{<:Real,1};
    startlogx::Union{Vector{<:Real},Nothing}=nothing,
    reltol=1e-10,
    abstol=1e-10,
    maxiters::Integer=80,
    damping::Bool=true,
    kwargs...
)::Vector{<:Real}
    n = Bnc.n
    d = Bnc.d
    helper = _integration_helper!(Bnc)

    u = isnothing(startlogx) ? copy(helper._anchor_log_x) : Float64.(startlogx)
    logqK = Float64.(logqK)
    logq = @view logqK[1:d]
    logK = @view logqK[d+1:end]

    J = copy(helper._LN_sparse)
    x = Vector{Float64}(undef, n)
    q = Vector{Float64}(undef, d)
    resid = Vector{Float64}(undef, n)
    x_M_view = @view x[helper._LN_top_cols]
    q_M_view = @view q[helper._LN_top_rows]
    M_top = @view J.nzval[helper._LN_top_idx]
    L_nzval = copy(helper._LN_sparse.nzval[helper._LN_top_idx])
    target_tol = max(abstol, reltol * max(1.0, norm(logqK, Inf)))
    prev_norm = Inf

    for _ in 1:maxiters
        @. x = exp10(u)
        mul!(q, Bnc.L, x)
        @. q = max(q, 1e-300)
        @views resid[1:d] .= log10.(q) .- logq
        @views resid[d+1:end] .= Bnc.N * u .- logK
        res_norm = norm(resid, Inf)
        res_norm <= target_tol && return u

        @. M_top = x_M_view * L_nzval / q_M_view
        Δ = J \ resid

        step = 1.0
        if damping
            accepted = false
            while step >= 2.0^-40
                u_try = u .- step .* Δ
                trial_resid = qK2x_residual(Bnc, u_try, logqK; input_logspace=true)
                trial_norm = norm(trial_resid, Inf)
                if isfinite(trial_norm) && trial_norm < min(prev_norm, res_norm)
                    u .= u_try
                    prev_norm = trial_norm
                    accepted = true
                    break
                end
                step *= 0.5
            end
            accepted || (u .-= Δ)
        else
            u .-= Δ
        end
    end

    @warn "Full-space qK2x Newton iteration reached maxiters=$maxiters"
    return u
end




function _logqK2logx_regime(Bnc::Bnc, logqK::AbstractArray{<:Real,1})::Vector{Float64}
    perm = assign_regime_qK(Bnc, logqK; input_logspace=true, asymptotic_only=false)
    H, H0 = get_H_H0(Bnc, perm)
    return Vector{Float64}(H * logqK .+ H0)
end

function _logqK2logx_regime_start(Bnc::Bnc, logqK::AbstractArray{<:Real,1})
    try
        return _logqK2logx_regime(Bnc, logqK)
    catch
        return nothing
    end
end





function qK2x_residual(Bnc::Bnc, logx::AbstractVector{<:Real}, qK::AbstractVector{<:Real}; input_logspace::Bool=false)
    logqK = input_logspace ? Float64.(qK) : log10.(Float64.(qK))
    d = Bnc.d
    r = Bnc.r
    resid = Vector{Float64}(undef, d + r)
    resid .= x2qK(Bnc, logx; input_logspace=true, output_logspace=true) .- logqK
    return resid
end





function _logqK2logx_free_energy(
    Bnc::Bnc,
    logqK::AbstractVector{<:Real};
    startlogx::Union{AbstractVector{<:Real},Nothing}=nothing,
    reltol=1e-10,
    abstol=1e-10,
    maxiters::Integer=80,
    robust_start::Bool=true,
    damping::Bool=true,
    warn_on_maxiters::Bool=true,
    kwargs...,
)::Vector{Float64}
    d = Bnc.d
    logqK = Float64.(logqK)
    logq = @view logqK[1:d]
    logK = @view logqK[d+1:end]
    L = Matrix{Float64}(Bnc.L)
    N = Matrix{Float64}(Bnc.N)
    G = -transpose(N) * ((N * transpose(N)) \ Float64.(logK))

    startlogx = if isnothing(startlogx) && robust_start
            _logqK2logx_regime_start(Bnc, logqK) # optional regime-based start
        else
            startlogx
        end

    λ = if isnothing(startlogx) 
        zeros(Float64, d)
    else
        (L * transpose(L)) \ (L * (Float64.(startlogx) .+ G))
    end

    F = Vector{Float64}(undef, d)
    q = Vector{Float64}(undef, d)
    x = Vector{Float64}(undef, Bnc.n)
    logx = Vector{Float64}(undef, Bnc.n)
    J = Matrix{Float64}(undef, d, d)
    target_tol = max(abstol, reltol * max(1.0, norm(logqK, Inf)))
    prev_norm = Inf

    for _ in 1:maxiters
        logx .= transpose(L) * λ .- G
        @. x = exp10(logx)
        mul!(q, L, x)
        @. q = max(q, 1e-300)
        @. F = log10(q) - logq
        res_norm = norm(F, Inf)
        res_norm <= target_tol && return copy(logx)

        J .= (L * Diagonal(x) * transpose(L))
        @inbounds for i in axes(J, 1)
            J[i, :] ./= (log(10.0) * q[i])
        end
        Δ = J \ F

        step = 1.0
        if damping
            accepted = false
            while step >= 2.0^-40
                λ_try = λ .- step .* Δ
                x_try = exp10.(transpose(L) * λ_try .- G)
                q_try = max.(L * x_try, 1e-300)
                F_try = log10.(q_try) .- logq
                trial_norm = norm(F_try, Inf)
                if isfinite(trial_norm) && trial_norm < min(prev_norm, res_norm)
                    λ .= λ_try
                    prev_norm = trial_norm
                    accepted = true
                    break
                end
                step *= 0.5
            end
            accepted || (λ .-= Δ)
        else
            λ .-= Δ
        end
    end

    warn_on_maxiters && @warn "Free-energy qK2x Newton iteration reached maxiters=$maxiters"
    return Vector{Float64}(transpose(L) * λ .- G)
end





function _logqK2logx_nullspace_newton(
    Bnc::Bnc,
    logqK::AbstractVector{<:Real};
    startlogx::Union{AbstractVector{<:Real},Nothing}=nothing,
    reltol=1e-10,
    abstol=1e-10,
    maxiters::Integer=80,
    damping::Bool=true,
    robust_start::Bool=true,
    kwargs...,
)::Vector{Float64}
    d = Bnc.d
    r = Bnc.r
    logqK = Float64.(logqK)
    q = exp10.(@view logqK[1:d])
    logK = Float64.(view(logqK, d + 1:length(logqK)))
    L = Matrix{Float64}(Bnc.L)
    N = Matrix{Float64}(Bnc.N)
    B = transpose(N)
    x0 = transpose(L) * ((L * transpose(L)) \ q)
    logx_start = if isnothing(startlogx) && robust_start
            start = _logqK2logx_regime_start(Bnc, logqK)
            isnothing(start) ? log10.(max.(x0, eps())) : start
        elseif isnothing(startlogx)
            log10.(max.(x0, eps()))
        else
            startlogx
        end
    m = (transpose(B) * B) \ (transpose(B) * (exp10.(logx_start) .- x0))

    F = Vector{Float64}(undef, r)
    J = Matrix{Float64}(undef, r, r)
    target_tol = max(abstol, reltol * max(1.0, norm(logK, Inf)))
    prev_norm = Inf

    for _ in 1:maxiters
        x = x0 .+ B * m
        if any(x .<= 0)
            robust_start || throw(ArgumentError("Nullspace Newton left the positive domain. Pass robust_start=true or a positive startlogx."))
            return _logqK2logx_free_energy(Bnc, logqK; startlogx=logx_start, reltol=reltol, abstol=abstol, maxiters=maxiters)
        end
        F .= N * log10.(x) .- logK
        res_norm = norm(F, Inf)
        res_norm <= target_tol && return log10.(x)

        J .= N * (Diagonal(1.0 ./ (log(10.0) .* x)) * B)
        Δ = J \ F
        step = 1.0
        if damping
            accepted = false
            while step >= 2.0^-40
                m_try = m .- step .* Δ
                x_try = x0 .+ B * m_try
                if all(>(0), x_try)
                    trial_norm = norm(N * log10.(x_try) .- logK, Inf)
                    if isfinite(trial_norm) && trial_norm < min(prev_norm, res_norm)
                        m .= m_try
                        prev_norm = trial_norm
                        accepted = true
                        break
                    end
                end
                step *= 0.5
            end
            accepted || (m .-= Δ)
        else
            m .-= Δ
        end
    end

    x = x0 .+ B * m
    if robust_start
        start = all(x .> 0) ? log10.(x) : logx_start
        return _logqK2logx_free_energy(Bnc, logqK; startlogx=start, reltol=reltol, abstol=abstol, maxiters=maxiters, warn_on_maxiters=false)
    end
    @warn "Nullspace qK2x Newton iteration reached maxiters=$maxiters"
    return log10.(x)
end





# function benchmark_qK2x_methods(
#     Bnc::Bnc,
#     qKs;
#     methods=(:free_energy, :homotopy, :newton_nullspace, :nlsolve, :regime),
#     input_logspace::Bool=true,
#     reference_method::Symbol=:free_energy,
#     kwargs...,
# )
#     cols = qKs isa AbstractMatrix ? [qKs[:, i] for i in axes(qKs, 2)] : collect(qKs)
#     refs = Dict{Int,Vector{Float64}}()
#     for (i, qK) in pairs(cols)
#         refs[i] = qK2x(Bnc, qK; input_logspace=input_logspace, output_logspace=true, method=reference_method, kwargs...)
#     end

#     results = NamedTuple[]
#     for method in methods
#         failures = 0
#         max_resid = 0.0
#         max_ref_err = 0.0
#         elapsed = @elapsed begin
#             for (i, qK) in pairs(cols)
#                 try
#                     logx = qK2x(Bnc, qK; input_logspace=input_logspace, output_logspace=true, method=method, kwargs...)
#                     resid = qK2x_residual(Bnc, logx, qK; input_logspace=input_logspace)
#                     max_resid = max(max_resid, norm(resid, Inf))
#                     max_ref_err = max(max_ref_err, norm(logx .- refs[i], Inf))
#                 catch err
#                     failures += 1
#                 end
#             end
#         end
#         push!(results, (; method, elapsed, failures, max_resid, max_ref_err, n=length(cols)))
#     end
#     return results
# end








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
    helper = _integration_helper!(Bnc)
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
    M = copy(helper._LN_sparse)
    M_lu = deepcopy(helper._LN_lu)

    logx_M_view = @view logx[helper._LN_top_cols] # view for faster updating J
    logq_M_view = @view logqK[helper._LN_top_rows] # view for faster updating J
    M_top = @view M.nzval[helper._LN_top_idx] # view for faster updating J
    M_top_diag = @view M.nzval[helper._LN_top_diag_idx] # top-row entries to perturb when J is singular

    p = HomotopyParams(startlogqK, ΔlogqK, logx, logqK,logq,logK, logqK_max, M, M_lu, 
        logx_M_view, logq_M_view, M_top, M_top_diag
        # logx_local,logx_M_view_local,logLx_local, logLx_M_view_local
        )
    return p
end


function get_homotopy_ode(Bnc::Bnc, p::HomotopyParams)
    # Constants helps for updating mutable datas
    helper = _integration_helper!(Bnc)
    L_nzval = log10.(helper._LN_sparse.nzval[helper._LN_top_idx]) # copy the nzval to avoid shared access

    @inline function update_M_lu(M_lu,M,max_try=100)
        lu!(M_lu, M,check=false) # recalculate the LU decomposition of J
        try_count = 0
        while !issuccess(M_lu) && try_count < max_try
            @.p.M_top_diag += eps() # perturb the diagonal elements a bit to avoid singularity
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
    f! = get_homotopy_ode(Bnc,p)

    callback = if !ensure_manifold
            CB.CallbackSet()
        else
            n = Bnc.n
            d = Bnc.d
            keep_manifold! = function(resid, u, p)  # Can not write to forms like log_sum_exp10!(logLx_local, Bnc.L, u) for Autodiff.
                @unpack logq,logK = p
                resid[1:d] .= log10.(Bnc.L * exp10.(u)) .- logq
                resid[d+1:end] .= Bnc.N * u .- logK
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
    sol = catalysis_logx(Bnc, x0, tspan;
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
    kwargs...
    )

    logx0 = qK2x(Bnc, qK0; input_logspace=input_logspace, output_logspace=true)
    t,u = x_traj_cat(Bnc, logx0, args...; input_logspace=true,output_logspace=true, kwargs...)
    u = x2qK.(Ref(Bnc), u;input_logspace=true,output_logspace=output_logspace, only_q=only_q)
    return (t,u)
end

q_traj_cat(args...;kwargs...) = qK_traj_cat(args...;only_q=true,kwargs...)


function have_catalysis(model::Bnc)
    return !isnothing(model.catalysis)
end

function _logqK_from_logqcat_logwKk(model::Bnc, logqcat::AbstractVector{<:Real}, logwKk::AbstractVector{<:Real})
    cn = model.catalysis
    logwKk = Float64.(logwKk)
    logqK = Vector{Float64}(undef, model.d + model.r)
    logqK[1:cn.r_v] .= Float64.(logqcat)
    logqK[cn.r_v + 1:model.d] .= @view logwKk[1:cn.d_w]
    logqK[model.d + 1:end] .= @view logwKk[cn.d_w + 1:cn.d_w + model.r]
    return logqK
end

function _direct_logx_checked(
    model::Bnc,
    logqK::AbstractVector{<:Real};
    method::Symbol=:free_energy,
    tol::Float64=1e-6,
    qK2x_maxiters::Integer=80,
    startlogx=nothing,
)
    method = method === :homotopy ? :free_energy : method
    logx = try
        qK2x(
            model,
            logqK;
            input_logspace=true,
            output_logspace=true,
            method=method,
            startlogx=startlogx,
            maxiters=qK2x_maxiters,
            warn_on_maxiters=false,
            robust_start=false,
        )
    catch
        return nothing
    end
    maximum(abs.(qK2x_residual(model, logx, logqK; input_logspace=true))) <= tol || return nothing
    return logx
end

function _homotopy_logx_checked(
    model::Bnc,
    from_logqK::AbstractVector{<:Real},
    to_logqK::AbstractVector{<:Real},
    from_logx::AbstractVector{<:Real};
    tol::Float64=1e-6,
    reltol::Real=1e-7,
    abstol::Real=1e-9,
    npoints::Integer=2,
)
    logx = try
        _, xs = x_traj_with_qK_change(
            model,
            Vector{Float64}(from_logqK),
            Vector{Float64}(to_logqK);
            input_logspace=true,
            output_logspace=true,
            startlogx=Vector{Float64}(from_logx),
            reltol=reltol,
            abstol=abstol,
            npoints=npoints,
            ensure_manifold=false,
        )
        Vector{Float64}(xs[end])
    catch
        return nothing
    end
    maximum(abs.(qK2x_residual(model, logx, to_logqK; input_logspace=true))) <= tol || return nothing
    return logx
end

function _logwKk_at(logwKk, t; input_logspace::Bool=true)
    vals = logwKk isa Function ? logwKk(t) : logwKk
    vals = Float64.(vals)
    return input_logspace ? vals : log10.(vals)
end

"""
    qcat_traj_cat(model, logqcat0, logwKk, tspan; kwargs...) -> (t, logqcat)

Simulate reduced catalysis dynamics for `qcat` while `w`, `K`, and `k` are
held fixed or supplied by a time-dependent `logwKk(t)` function.  The ODE state
is `log10(qcat)`.  Inside the RHS, `method=:homotopy` is deliberately treated
as `:free_energy`, because using homotopy there would nest one ODE solve inside
another ODE solve.
"""
function qcat_traj_cat(
    model::Bnc,
    logqcat0::AbstractVector{<:Real},
    logwKk,
    tspan::Tuple{<:Real,<:Real};
    input_logspace::Bool=true,
    output_logspace::Bool=true,
    method::Symbol=:free_energy,
    tol::Float64=1e-6,
    qK2x_maxiters::Integer=80,
    alg=nothing,
    reltol::Real=1e-7,
    abstol::Real=1e-9,
    maxiters::Integer=100_000,
    saveat=range(Float64(tspan[1]), Float64(tspan[2]), length=500),
    max_log10_scale::Real=300.0,
    fail_on_binding_error::Bool=false,
    homotopy_fallback::Bool=true,
    fallback_reltol::Real=reltol,
    fallback_abstol::Real=abstol,
    kwargs...,
)
    have_catalysis(model) || throw(ArgumentError("model has no catalysis data."))
    cn = model.catalysis
    inner_method = method === :homotopy ? :free_energy : method
    method === :homotopy && @warn "qcat_traj_cat does not use homotopy inside the ODE RHS; using :free_energy for qK -> x solves."
    first_logwKk = _logwKk_at(logwKk, Float64(tspan[1]); input_logspace=input_logspace)
    expected_wKk_len = cn.d_w + model.r + cn.n_v
    length(first_logwKk) == expected_wKk_len || throw(ArgumentError("logwKk length must be $expected_wKk_len, got $(length(first_logwKk)). Available wKk symbols: $(wKk_symbol(model))."))

    u0 = input_logspace ? Vector{Float64}(logqcat0) : log10.(Float64.(logqcat0))
    length(u0) == cn.r_v || throw(ArgumentError("logqcat0 length must be $(cn.r_v)."))

    logv = Vector{Float64}(undef, cn.n_v)
    vscaled = Vector{Float64}(undef, cn.n_v)
    qdot_scaled = Vector{Float64}(undef, cn.r_v)
    last_logqK = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    last_logx = Ref{Union{Nothing,Vector{Float64}}}(nothing)

    function rhs!(du, u, _, t)
        current_logwKk = _logwKk_at(logwKk, t; input_logspace=input_logspace)
        logqK = _logqK_from_logqcat_logwKk(model, u, current_logwKk)
        logx = _direct_logx_checked(
            model,
            logqK;
            method=inner_method,
            tol=tol,
            qK2x_maxiters=qK2x_maxiters,
            startlogx=last_logx[],
        )
        if isnothing(logx) && homotopy_fallback && !isnothing(last_logqK[]) && !isnothing(last_logx[])
            logx = _homotopy_logx_checked(
                model,
                last_logqK[],
                logqK,
                last_logx[];
                tol=tol,
                reltol=fallback_reltol,
                abstol=fallback_abstol,
            )
        end
        if isnothing(logx) && homotopy_fallback
            helper = _integration_helper!(model)
            logx = _homotopy_logx_checked(
                model,
                helper._anchor_log_qK,
                logqK,
                helper._anchor_log_x;
                tol=tol,
                reltol=fallback_reltol,
                abstol=fallback_abstol,
                npoints=8,
            )
        end
        if isnothing(logx)
            fail_on_binding_error && error("qK -> x solve failed at t=$t.")
            fill!(du, 0.0)
            return nothing
        end
        last_logqK[] = Vector{Float64}(logqK)
        last_logx[] = Vector{Float64}(logx)

        logk = @view current_logwKk[cn.d_w + model.r + 1:cn.d_w + model.r + cn.n_v]
        mul!(logv, cn._Π_sparse, logx)
        logv .+= logk

        vshift = maximum(logv)
        @. vscaled = exp10(logv - vshift)
        mul!(qdot_scaled, cn.S, vscaled)

        @inbounds for i in 1:cn.r_v
            scale_log = vshift - u[i]
            abs(scale_log) > max_log10_scale && error("qcat ODE scale overflow at t=$t, component=$i, log10 scale=$scale_log.")
            du[i] = qdot_scaled[i] * exp10(scale_log) / log(10.0)
        end
        any(!isfinite, du) && error("qcat ODE produced non-finite derivative at t=$t.")
        return nothing
    end

    prob = ODE.ODEProblem(rhs!, u0, (Float64(tspan[1]), Float64(tspan[2])))
    sol = ODE.solve(
        prob,
        isnothing(alg) ? ODE.Tsit5() : alg;
        saveat=saveat,
        reltol=Float64(reltol),
        abstol=Float64(abstol),
        maxiters=maxiters,
        isoutofdomain=(u, _, _) -> any(!isfinite, u),
        kwargs...,
    )
    us = output_logspace ? sol.u : [exp10.(u) for u in sol.u]
    return collect(sol.t), us
end

function _default_logqcat0_for_adaptation(model::Bnc, logwKk::AbstractVector{<:Real})
    cn = model.catalysis
    out = fill(-3.0, cn.r_v)
    for s in q_cat_symbol(model)
        i = locate_sym_qcat(model, s)
        if s === :tAstar && :tAtotal in wKk_symbol(model)
            out[i] = logwKk[locate_sym_wKk(model, :tAtotal)] - 2
        elseif s === :tBstar && :tBtotal in wKk_symbol(model)
            out[i] = logwKk[locate_sym_wKk(model, :tBtotal)] - 2
        end
    end
    return out
end

function _observe_adaptation_value(model::Bnc, logqcat, logwKk; observe=:Astar, method::Symbol=:free_energy, tol::Float64=1e-6, qK2x_maxiters::Integer=80)
    if observe in q_cat_symbol(model)
        return logqcat[locate_sym_qcat(model, observe)]
    end
    logqK = _logqK_from_logqcat_logwKk(model, logqcat, logwKk)
    logx = _direct_logx_checked(model, logqK; method=method, tol=tol, qK2x_maxiters=qK2x_maxiters)
    isnothing(logx) && return NaN
    return logx[locate_sym_x(model, observe)]
end

"""
    simulate_adaptation(model; p, logtI, logqcat0=nothing, observe=:Astar, kwargs...)

Convenience wrapper for step/input-response simulations where `p` is a log10
`wKk` vector and `logtI(t)` replaces the `:tI` entry over time.  Long dynamic
simulations should use this direct reduced ODE instead of repeatedly calling
`qK2x(...; method=:homotopy)` inside a notebook RHS.
"""
function simulate_adaptation(
    model::Bnc;
    p,
    logtI,
    input_sym=:tI,
    logqcat0=nothing,
    tspan=(0.0, 200.0),
    observe=:Astar,
    method::Symbol=:free_energy,
    tol::Float64=1e-6,
    qK2x_maxiters::Integer=80,
    saveat=range(Float64(tspan[1]), Float64(tspan[2]), length=500),
    kwargs...,
)
    base_logwKk = Vector{Float64}(p)
    input_idx = locate_sym_wKk(model, input_sym)
    logwKk_fun = t -> begin
        vals = copy(base_logwKk)
        vals[input_idx] = Float64(logtI(t))
        vals
    end
    initial_wKk = logwKk_fun(tspan[1])
    q0 = isnothing(logqcat0) ? _default_logqcat0_for_adaptation(model, initial_wKk) : Vector{Float64}(logqcat0)

    t, us = qcat_traj_cat(
        model,
        q0,
        logwKk_fun,
        tspan;
        input_logspace=true,
        output_logspace=true,
        method=method,
        tol=tol,
        qK2x_maxiters=qK2x_maxiters,
        saveat=saveat,
        kwargs...,
    )
    logqcat = reduce(hcat, us)
    obs = Vector{Float64}(undef, length(t))
    logtI_vals = Vector{Float64}(undef, length(t))
    inner_method = method === :homotopy ? :free_energy : method
    observe_is_qcat = observe in q_cat_symbol(model)
    last_logx = nothing
    for (j, tj) in pairs(t)
        current_wKk = logwKk_fun(tj)
        logtI_vals[j] = current_wKk[input_idx]
        if observe_is_qcat
            obs[j] = logqcat[locate_sym_qcat(model, observe), j]
        else
            logqK = _logqK_from_logqcat_logwKk(model, @view(logqcat[:, j]), current_wKk)
            logx = try
                qK2x(
                    model,
                    logqK;
                    input_logspace=true,
                    output_logspace=true,
                    method=inner_method,
                    startlogx=last_logx,
                    maxiters=qK2x_maxiters,
                    warn_on_maxiters=false,
                    robust_start=false,
                )
            catch
                nothing
            end
            if isnothing(logx) || maximum(abs.(qK2x_residual(model, logx, logqK; input_logspace=true))) > tol
                obs[j] = NaN
            else
                last_logx = logx
                obs[j] = logx[locate_sym_x(model, observe)]
            end
        end
    end
    return (; t, logqcat, logtI=logtI_vals, logobserve=obs, observe)
end




#--------------------------------------------------------------------------
#   Below are most AI generated code, which is more experimental and less tested, especially for the catalysis part. Use with caution and report any issues.
#--------------------------------------------------------------------------









"""
    TimecurveParam
    ### Constant
    # logk: R^rcat  # Changed to log10(k) for stability

    ### Cache
    # x: R^n  # Now used as buffer for scaled computations
    # q: R^d  # Buffer for q_scaled or log_q if needed
    # v: R^rcat  # Buffer for log(v_cat) = Π * u + logk
    # f: R^n  # First d values: Λ_q^{-1} Γ v_cat(x)
    # M: SparseMatrixCSC{Float64,Int}  # Jacobian matrix buffer [diag(1/q) L diag(x); N]
    # M_lu: SparseArrays.UMFPACK.UmfpackLU{Float64,Int}  # LU decomposition of M
Cache container for catalysis time-course integration.
"""
struct TimecurveParam{V<:Vector{Float64},SV<:SubArray}
    logk::V
    x_scaled::V
    q_scaled::V
    logv::V
    v_scaled::V
    qdot_scaled::V
    rhs::V
    M::SparseMatrixCSC{Float64,Int}
    M_lu::SparseArrays.UMFPACK.UmfpackLU{Float64,Int}
    M_top::SV
end

"""
Get the catalysis parameter for ODE f construction
"""
function get_catalysis_param(model::Bnc, k)
    @assert have_catalysis(model) "Should fill catalysis data first"
    helper = _integration_helper!(model)
    logk = log10.(Float64.(k))
    cn = model.catalysis
    x_scaled = Vector{Float64}(undef, model.n)
    q_scaled = Vector{Float64}(undef, model.d)
    logv = Vector{Float64}(undef, cn.n_v)
    v_scaled = Vector{Float64}(undef, cn.n_v)
    qdot_scaled = Vector{Float64}(undef, cn.r_v)
    rhs = zeros(Float64, model.n)
    M = copy(helper._LN_sparse)
    M_lu = lu(M)
    M_top = @view M.nzval[helper._LN_top_idx]
    TimecurveParam(logk, x_scaled, q_scaled, logv, v_scaled, qdot_scaled, rhs, M, M_lu, M_top)
end

"""
return the f(du,u,p,t) for ODE solver
"""
function get_catalysis_ode(model::Bnc)
    @assert have_catalysis(model) "Should fill catalysis data first"
    helper = _integration_helper!(model)
    cn = model.catalysis
    L_nzval = Float64.(helper._LN_sparse.nzval[helper._LN_top_idx])

    function f(du, u, p::TimecurveParam, t)
        @unpack logk, x_scaled, q_scaled, logv, v_scaled, qdot_scaled, rhs, M, M_lu, M_top = p

        mul!(logv, cn._Π_sparse, u)
        logv .+= logk

        u_shift = maximum(u)
        @. x_scaled = exp10(u - u_shift)
        mul!(q_scaled, model.L, x_scaled)
        @. q_scaled = max(q_scaled, 1e-300)
        @. M_top = L_nzval * x_scaled[helper._LN_top_cols] / q_scaled[helper._LN_top_rows]

        lu!(M_lu, M, check=false)
        issuccess(M_lu) || error("Catalysis logx Jacobian is singular at t=$t.")

        v_shift = maximum(logv)
        @. v_scaled = exp10(logv - v_shift)
        mul!(qdot_scaled, cn.S, v_scaled)

        fill!(rhs, 0.0)
        scale = exp10(v_shift - u_shift) / log(10.0)
        @views @. rhs[1:cn.r_v] = scale * qdot_scaled / q_scaled[1:cn.r_v]

        ldiv!(du, M_lu, rhs)
        any(!isfinite, du) && error("Catalysis ODE produced non-finite du at t=$t.")
        return nothing
    end
end

"""
Compute  Λ_{L*exp10(a)}^{-1} * Γ*exp10(b) in a stable way
(No changes needed, but included for completeness)
"""
function stable_Linv_Γexp10(L::SparseMatrixCSC{<:Real,Int},
                            Γ::SparseMatrixCSC{<:Real,Int},
                            a::AbstractVector{<:Real},
                            b::AbstractVector{<:Real};
                            q_floor::Float64 = 1e-300)
    # Scale exp10(a) to avoid overflow/underflow: exp10(a) = 10^c * exp10(a-c)
    c = maximum(a)
    xscaled = exp10.(Float64.(a) .- c)               # in (0,1]
    qscaled = Vector{Float64}(undef, size(L,1))
    mul!(qscaled, sparse(Float64.(L)), xscaled)      # qscaled = L * exp10(a-c)
    # q = 10^c * qscaled, so 1/q = 10^(-c) ./ qscaled
    @inbounds @. qscaled = max(qscaled, q_floor)
    # Scale exp10(b): exp10(b) = 10^d * exp10(b-d)
    d = maximum(b)
    vscaled = exp10.(Float64.(b) .- d)               # in (0,1]
    yscaled = Vector{Float64}(undef, size(Γ,1))
    mul!(yscaled, sparse(Float64.(Γ)), vscaled)      # yscaled = Γ * exp10(b-d)
    # Combine scales: (Γ*10^b) ./ (L*10^a) = 10^(d-c) * (yscaled ./ qscaled)
    scale = exp10(d - c)
    out = Vector{Float64}(undef, length(yscaled))
    @inbounds @. out = (yscaled / qscaled) * scale
    return out
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
    sol = ODE.solve(prob, isnothing(alg) ? ODE.Tsit5() : alg; reltol=reltol, abstol=abstol, kwargs...)
    return sol
end






# Helper functions to scale sparse matrices
function scale_columns!(A::SparseMatrixCSC{Float64, Int}, s::Vector{Float64})
    n = size(A, 2)
    @inbounds for j = 1:n
        for p = A.colptr[j]:(A.colptr[j+1]-1)
            A.nzval[p] *= s[j]
        end
    end
    return A
end

function scale_rows!(A::SparseMatrixCSC{Float64, Int}, s::Vector{Float64})
    @inbounds for j = 1:size(A, 2)
        for p = A.colptr[j]:(A.colptr[j+1]-1)
            row = A.rowval[p]
            A.nzval[p] *= s[row]
        end
    end
    return A
end

# The right-hand side function for the ODE: dy/dt = f(y)
function ode_rhs!(dy::Vector{Float64}, y::Vector{Float64}, p, t)
    Lf, Γf, Nf, Πf, k::Vector{Float64}, q_floor::Float64 = p

    n = length(y)
    d = size(Lf, 1)
    r = size(Nf, 1)

    # Scale x = 10.^y
    max_y = maximum(y)
    xscaled = exp10.(y .- max_y)

    # qscaled = L * xscaled
    qscaled = Vector{Float64}(undef, d)
    mul!(qscaled, Lf, xscaled)
    @. qscaled = max(qscaled, q_floor)

    # Build A = Λ_q^{-1} * L * Λ_x (stably: diag(1./qscaled) * L * diag(xscaled))
    L_scaled = copy(Lf)
    scale_columns!(L_scaled, xscaled)
    scale_rows!(L_scaled, 1.0 ./ qscaled)  # Now L_scaled is A

    # Build M = [A; N]
    M = vcat(L_scaled, Nf)

    # Compute v_cat stably: v_cat = k .* exp10.(Π * y)
    u = Πf * y
    c = maximum(u)
    vscaled = exp10.(u .- c)
    v_cat_scaled = k .* vscaled

    # sv_scaled = Γ * v_cat_scaled
    sv_scaled = Vector{Float64}(undef, d)
    mul!(sv_scaled, Γf, v_cat_scaled)

    # zscaled = (1 ./ qscaled) .* sv_scaled
    zscaled = (1.0 ./ qscaled) .* sv_scaled

    # Full scale for z = exp10(c - max_y) * zscaled
    scale = exp10(c - max_y)
    z = scale .* zscaled

    # w = [z; zeros(r)]
    w = vcat(z, zeros(Float64, r))

    # Solve M * dy = w (using factorization for sparse matrix)
    fact = factorize(M)
    dy[:] = fact \ w

    return nothing
end

# Main simulation function
function simulate_ode(L::SparseMatrixCSC{<:Real, Int},
                      Γ::SparseMatrixCSC{<:Real, Int},
                      N::SparseMatrixCSC{<:Real, Int},
                      Π::SparseMatrixCSC{<:Real, Int},
                      k::AbstractVector{<:Real},  # Assuming Lambda_k is a vector k
                      y0::AbstractVector{<:Real},  # Initial log10(x)
                      tspan::Tuple{<:Real, <:Real};
                      q_floor::Float64 = 1e-300,
                      rtol::Float64 = 1e-6,
                      atol::Float64 = 1e-6,
                      solver = ODE.Tsit5())  # Can change to other solvers like Rodas5() for stiff systems
    # Convert to Float64 sparse matrices
    Lf = sparse(Float64.(L))
    Γf = sparse(Float64.(Γ))
    Nf = sparse(Float64.(N))
    Πf = sparse(Float64.(Π))

    # Convert vectors to Float64
    kf = Float64.(k)
    y0f = Float64.(y0)

    # Pack parameters
    p = (Lf, Γf, Nf, Πf, kf, q_floor)

    # Define ODE problem
    prob = ODE.ODEProblem(ode_rhs!, y0f, tspan, p)

    # Solve
    sol = ODE.solve(prob, solver; reltol=rtol, abstol=atol)

    return sol
end

# function catalysis_logx(Bnc::Bnc, logx0::Vector{<:Real}, tspan::Tuple{Real,Real};
#     k::AbstractVector{<:Real},
#     alg=nothing, # Default to nothing, will use Tsit5() if not provided
#     reltol=1e-8,
#     abstol=1e-9,
#     kwargs...
# )::ODESolution
#     return simulate_ode(Bnc.L, 
#             Bnc.catalysis._Γ_sparse, 
#             Bnc.N, 
#             sparse(Bnc.catalysis.Π), 
#             k, 
#             logx0,
#             tspan; 
#             rtol=reltol, atol=abstol, solver=alg === nothing ? ODE.Tsit5() : alg)
# end
