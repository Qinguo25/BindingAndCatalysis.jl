export catalysis_logx,
    qcat_traj_cat, trajectory_matrix, simulate_catalysis_trajectory, simulate_adaptation

#--------------------------------------------------------------------------------
# Catalysis dynamics
#--------------------------------------------------------------------------------

function have_catalysis(model::Bnc)
    return !isnothing(model.catalysis)
end

function _logqK_from_logqcat_logwKk(
    model::Bnc, logqcat::AbstractVector{<:Real}, logwKk::AbstractVector{<:Real}
)
    cn = model.catalysis
    logwKk = Float64.(logwKk)
    logqK = Vector{Float64}(undef, model.d + model.r)
    logqK[1:(cn.r_v)] .= Float64.(logqcat)
    logqK[(cn.r_v + 1):(model.d)] .= @view logwKk[1:(cn.d_w)]
    logqK[(model.d + 1):end] .= @view logwKk[(cn.d_w + 1):(cn.d_w + model.r)]
    return logqK
end

function _direct_logx_checked(
    model::Bnc,
    logqK::AbstractVector{<:Real};
    method::Union{Symbol, Nothing}=nothing,
    tol::Float64=1e-6,
    qK2x_maxiters::Integer=80,
    startlogx=nothing,
)
    method = _resolve_qK2x_method(model, method)
    logx = try
        qK2x(
            model,
            logqK;
            input=:log,
            output=:log,
            method=method,
            startlogx=startlogx,
            maxiters=qK2x_maxiters,
            warn_on_maxiters=false,
            robust_start=false,
        )
    catch
        return nothing
    end
    maximum(abs.(qK2x_residual(model, logx, logqK; input=:log))) <= tol || return nothing
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
            input=:log,
            output=:log,
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
    maximum(abs.(qK2x_residual(model, logx, to_logqK; input=:log))) <= tol || return nothing
    return logx
end

function _logwKk_at(logwKk, t; input_logspace::Bool=true)
    vals = logwKk isa Function ? logwKk(t) : logwKk
    vals = Float64.(vals)
    return input_logspace ? vals : log10.(vals)
end

function _old_logk_from_logk(cn::CatalysisData, logk::AbstractVector{<:Real})
    length(logk) == cn.n_k ||
        throw(ArgumentError("logk length must be $(cn.n_k), got $(length(logk))."))
    logk_old = Vector{Float64}(undef, cn.n_v)
    mul!(logk_old, cn.F, Float64.(logk))
    logk_old .+= Float64.(cn.F0)
    return logk_old
end

function _old_logk_from_logwKk(model::Bnc, logwKk::AbstractVector{<:Real})
    cn = model.catalysis
    first = cn.d_w + model.r + 1
    last = cn.d_w + model.r + cn.n_k
    return _old_logk_from_logk(cn, @view logwKk[first:last])
end

"""
    qcat_traj_cat(model, logqcat0, logwKk, tspan; kwargs...) -> (t, states)

Simulate reduced catalysis dynamics for `qcat` while `w`, `K`, and `k` are
held fixed or supplied by a time-dependent `logwKk(t)` function. The ODE state
is `log10(qcat)`. `states` is a vector of saved state vectors. Use
`trajectory_matrix(states)` for matrix-shaped plotting data.

A failed `qK -> x` binding solve terminates the integration with an exception;
it is never converted into a zero derivative.
"""
function qcat_traj_cat(
    model::Bnc,
    logqcat0::AbstractVector{<:Real},
    logwKk,
    tspan::Tuple{<:Real, <:Real};
    input::Symbol=:log,
    output::Symbol=:log,
    input_logspace::Union{Bool, Nothing}=nothing,
    output_logspace::Union{Bool, Nothing}=nothing,
    method::Union{Symbol, Nothing}=nothing,
    tol::Float64=1e-6,
    qK2x_maxiters::Integer=80,
    alg=nothing,
    reltol::Real=1e-7,
    abstol::Real=1e-9,
    maxiters::Integer=100_000,
    saveat=range(Float64(tspan[1]), Float64(tspan[2]); length=500),
    max_log10_scale::Real=300.0,
    fail_on_binding_error=_UNSET_KEYWORD,
    homotopy_fallback::Bool=true,
    fallback_reltol::Real=reltol,
    fallback_abstol::Real=abstol,
    return_solution::Bool=false,
    kwargs...,
)
    fail_on_binding_error === _UNSET_KEYWORD || throw(
        ArgumentError(
            "keyword `fail_on_binding_error` is no longer supported; " *
            "binding solve failures now always terminate the integration.",
        ),
    )
    have_catalysis(model) || throw(ArgumentError("model has no catalysis data."))
    cn = model.catalysis
    input = _resolve_space_mode(input, input_logspace, :input_logspace)
    output = _resolve_space_mode(output, output_logspace, :output_logspace)
    inner_method = _resolve_qK2x_method(model, method)
    first_logwKk = _logwKk_at(logwKk, Float64(tspan[1]); input_logspace=(input === :log))
    expected_wKk_len = cn.d_w + model.r + cn.n_k
    length(first_logwKk) == expected_wKk_len || throw(
        ArgumentError(
            "logwKk length must be $expected_wKk_len, got $(length(first_logwKk)). Available wKk symbols: $(wKk_symbol(model)).",
        ),
    )

    u0 = input === :log ? Vector{Float64}(logqcat0) : log10.(Float64.(logqcat0))
    length(u0) == cn.r_v || throw(ArgumentError("logqcat0 length must be $(cn.r_v)."))

    logv = Vector{Float64}(undef, cn.n_v)
    vscaled = Vector{Float64}(undef, cn.n_v)
    qdot_scaled = Vector{Float64}(undef, cn.r_v)
    last_logqK = Ref{Union{Nothing, Vector{Float64}}}(nothing)
    last_logx = Ref{Union{Nothing, Vector{Float64}}}(nothing)

    function rhs!(du, u, _, t)
        current_logwKk = _logwKk_at(logwKk, t; input_logspace=(input === :log))
        logqK = _logqK_from_logqcat_logwKk(model, u, current_logwKk)
        logx = _direct_logx_checked(
            model,
            logqK;
            method=inner_method,
            tol=tol,
            qK2x_maxiters=qK2x_maxiters,
            startlogx=last_logx[],
        )
        if isnothing(logx) &&
            homotopy_fallback &&
            !isnothing(last_logqK[]) &&
            !isnothing(last_logx[])
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
            error("qK -> x solve failed at t=$t.")
        end
        last_logqK[] = Vector{Float64}(logqK)
        last_logx[] = Vector{Float64}(logx)

        mul!(logv, cn._Π_sparse, logx)
        logv .+= _old_logk_from_logwKk(model, current_logwKk)

        vshift = maximum(logv)
        @. vscaled = exp10(logv - vshift)
        mul!(qdot_scaled, cn.S, vscaled)

        @inbounds for i in 1:(cn.r_v)
            scale_log = vshift - u[i]
            abs(scale_log) > max_log10_scale && error(
                "qcat ODE scale overflow at t=$t, component=$i, log10 scale=$scale_log."
            )
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
    us = output === :log ? sol.u : [exp10.(u) for u in sol.u]
    return_solution && return collect(sol.t), us, sol
    return collect(sol.t), us
end

"""
    trajectory_matrix(states) -> Matrix

Convert a vector of saved state vectors, such as the second return value of
`qcat_traj_cat`, into a matrix with one column per saved time point.
"""
trajectory_matrix(states::AbstractMatrix) = states

function trajectory_matrix(states)
    isempty(states) && return Matrix{Float64}(undef, 0, 0)
    return reduce(hcat, states)
end

function _solver_successful_retcode(retcode)
    retcode_string = string(retcode)
    return retcode_string in ("Success", "Terminated") ||
           endswith(retcode_string, ".Success") ||
           endswith(retcode_string, ".Terminated")
end

function _trajectory_diagnostics(sol, t, tspan, maxiters::Integer)
    t_start = Float64(tspan[1])
    t_stop = Float64(tspan[2])
    t_final = isempty(t) ? NaN : Float64(t[end])
    return (;
        retcode=sol.retcode,
        successful=_solver_successful_retcode(sol.retcode),
        reached_final_time=!isnan(t_final) &&
                           t_final >= t_stop - 100 * eps(max(abs(t_stop), 1.0)),
        t_start,
        t_final,
        t_stop,
        n_saved=length(t),
        maxiters,
    )
end

function _log_vector_at(source, t; input::Symbol, expected_len::Integer, name::Symbol)
    vals = source isa Function ? source(t) : source
    isnothing(vals) && throw(ArgumentError("`$name` must not evaluate to `nothing`."))
    vals = Float64.(vals)
    length(vals) == expected_len ||
        throw(ArgumentError("`$name` length must be $expected_len, got $(length(vals))."))
    return input === :log ? vals : log10.(vals)
end

function _exactly_one_parameter_pair(
    log_value, linear_value, log_name::Symbol, linear_name::Symbol
)
    provided = (!isnothing(log_value)) + (!isnothing(linear_value))
    provided == 1 ||
        throw(ArgumentError("Provide exactly one of `$log_name` or `$linear_name`."))
    return if !isnothing(log_value)
        (log_value, :log, log_name)
    else
        (linear_value, :linear, linear_name)
    end
end

function _logwKk_source(
    model::Bnc;
    logwKk=nothing,
    wKk=nothing,
    logw=nothing,
    w=nothing,
    logK=nothing,
    K=nothing,
    logk=nothing,
    k=nothing,
)
    have_combined = !isnothing(logwKk) || !isnothing(wKk)
    have_components =
        !isnothing(logw) ||
        !isnothing(w) ||
        !isnothing(logK) ||
        !isnothing(K) ||
        !isnothing(logk) ||
        !isnothing(k)

    have_combined &&
        have_components &&
        throw(
            ArgumentError(
                "Provide either `logwKk`/`wKk` or component `w`, `K`, `k` values, not both."
            ),
        )
    cn = model.catalysis
    expected_wKk_len = cn.d_w + model.r + cn.n_k

    if !isnothing(logwKk) || !isnothing(wKk)
        source, input, name = _exactly_one_parameter_pair(logwKk, wKk, :logwKk, :wKk)
        return t ->
            _log_vector_at(source, t; input=input, expected_len=expected_wKk_len, name=name)
    end

    have_components || throw(
        ArgumentError(
            "Provide either `logwKk`/`wKk`, or one value for each component pair: `logw`/`w`, `logK`/`K`, and `logk`/`k`.",
        ),
    )
    w_source, w_input, w_name = _exactly_one_parameter_pair(logw, w, :logw, :w)
    K_source, K_input, K_name = _exactly_one_parameter_pair(logK, K, :logK, :K)
    k_source, k_input, k_name = _exactly_one_parameter_pair(logk, k, :logk, :k)

    return t -> [
        _log_vector_at(w_source, t; input=w_input, expected_len=cn.d_w, name=w_name)
        _log_vector_at(K_source, t; input=K_input, expected_len=model.r, name=K_name)
        _log_vector_at(k_source, t; input=k_input, expected_len=cn.n_k, name=k_name)
    ]
end

function _initial_logqcat(
    model::Bnc; logqcat0=nothing, qcat0=nothing, default_logqcat0=nothing
)
    provided = (!isnothing(logqcat0)) + (!isnothing(qcat0))
    if provided == 0
        isnothing(default_logqcat0) &&
            throw(ArgumentError("Provide exactly one of `logqcat0` or `qcat0`."))
        logqcat = Vector{Float64}(default_logqcat0)
    elseif provided == 1
        logqcat = isnothing(logqcat0) ? log10.(Float64.(qcat0)) : Vector{Float64}(logqcat0)
    else
        throw(ArgumentError("Provide exactly one of `logqcat0` or `qcat0`."))
    end

    cn = model.catalysis
    length(logqcat) == cn.r_v ||
        throw(ArgumentError("logqcat0 length must be $(cn.r_v), got $(length(logqcat))."))
    return logqcat
end

"""
    simulate_catalysis_trajectory(model; logqcat0, logwKk, tspan, kwargs...) -> NamedTuple

High-level reduced catalysis trajectory wrapper. Parameters can be supplied as a
fixed vector or a function of time, either in combined `wKk` form or split into
`w`, `K`, and reduced `k` blocks:

```julia
simulate_catalysis_trajectory(model; logqcat0, logwKk=t -> p(t), tspan)
simulate_catalysis_trajectory(model; qcat0, w=t -> w(t), K=K0, k=k0, tspan)
```

The returned named tuple includes `t`, `logqcat`, `qcat`, `states`, `output`,
and `diagnostics`. `logqcat` and `qcat` are matrices with one column per saved
time. `diagnostics` records the solver `retcode`, success flag, final-time
reachability, saved-point count, and `maxiters`. A failed inner binding solve
throws instead of returning a trajectory with successful diagnostics.
"""
function simulate_catalysis_trajectory(
    model::Bnc;
    logqcat0=nothing,
    qcat0=nothing,
    logwKk=nothing,
    wKk=nothing,
    logw=nothing,
    w=nothing,
    logK=nothing,
    K=nothing,
    logk=nothing,
    k=nothing,
    tspan=nothing,
    output::Symbol=:log,
    output_logspace::Union{Bool, Nothing}=nothing,
    maxiters::Integer=100_000,
    kwargs...,
)
    have_catalysis(model) || throw(ArgumentError("model has no catalysis data."))
    isnothing(tspan) && throw(ArgumentError("`tspan` is required."))
    output = _resolve_space_mode(output, output_logspace, :output_logspace)
    logqcat = _initial_logqcat(model; logqcat0=logqcat0, qcat0=qcat0)
    logwKk_fun = _logwKk_source(
        model; logwKk=logwKk, wKk=wKk, logw=logw, w=w, logK=logK, K=K, logk=logk, k=k
    )

    t, states, sol = qcat_traj_cat(
        model,
        logqcat,
        logwKk_fun,
        tspan;
        input=:log,
        output=output,
        maxiters=maxiters,
        return_solution=true,
        kwargs...,
    )
    traj = trajectory_matrix(states)
    diagnostics = _trajectory_diagnostics(sol, t, tspan, maxiters)

    if output === :log
        return (; t, logqcat=traj, qcat=exp10.(traj), states, output, diagnostics)
    else
        return (; t, qcat=traj, logqcat=log10.(traj), states, output, diagnostics)
    end
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

"""
    simulate_adaptation(model; p, logtI, logqcat0=nothing, observe=:Astar, kwargs...)

Convenience wrapper for step/input-response simulations where `p` is a log10
`wKk` vector and `logtI(t)` replaces the `:tI` entry over time. Long dynamic
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
    method::Union{Symbol, Nothing}=nothing,
    tol::Float64=1e-6,
    qK2x_maxiters::Integer=80,
    saveat=range(Float64(tspan[1]), Float64(tspan[2]); length=500),
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
    q0 = if isnothing(logqcat0)
        _default_logqcat0_for_adaptation(model, initial_wKk)
    else
        Vector{Float64}(logqcat0)
    end

    traj = simulate_catalysis_trajectory(
        model;
        logqcat0=q0,
        logwKk=logwKk_fun,
        tspan=tspan,
        output=:log,
        method=method,
        tol=tol,
        qK2x_maxiters=qK2x_maxiters,
        saveat=saveat,
        kwargs...,
    )
    t = traj.t
    logqcat = traj.logqcat
    obs = Vector{Float64}(undef, length(t))
    logtI_vals = Vector{Float64}(undef, length(t))
    inner_method = _resolve_qK2x_method(model, method)
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
                    input=:log,
                    output=:log,
                    method=inner_method,
                    startlogx=last_logx,
                    maxiters=qK2x_maxiters,
                    warn_on_maxiters=false,
                    robust_start=false,
                )
            catch
                nothing
            end
            if isnothing(logx) ||
                maximum(abs.(qK2x_residual(model, logx, logqK; input=:log))) > tol
                obs[j] = NaN
            else
                last_logx = logx
                obs[j] = logx[locate_sym_x(model, observe)]
            end
        end
    end
    return (; t, logqcat, logtI=logtI_vals, logobserve=obs, observe)
end

#--------------------------------------------------------------------------------
# Full x-space catalysis ODE
#--------------------------------------------------------------------------------

"""
    TimecurveParam

Cache container for catalysis time-course integration.
"""
struct TimecurveParam{V <: Vector{Float64}, SV <: SubArray}
    logk::V
    x_scaled::V
    q_scaled::V
    logv::V
    v_scaled::V
    qdot_scaled::V
    rhs::V
    M::SparseMatrixCSC{Float64, Int}
    M_lu::SparseArrays.UMFPACK.UmfpackLU{Float64, Int}
    M_top::SV
end

"""
    get_catalysis_param(model, k) -> TimecurveParam

Build cached buffers used by the full `log(x)` catalysis ODE.
"""
function get_catalysis_param(model::Bnc, k)
    @assert have_catalysis(model) "Should fill catalysis data first"
    helper = _integration_helper!(model)
    cn = model.catalysis
    logk = _old_logk_from_logk(cn, log10.(Float64.(k)))
    x_scaled = Vector{Float64}(undef, model.n)
    q_scaled = Vector{Float64}(undef, model.d)
    logv = Vector{Float64}(undef, cn.n_v)
    v_scaled = Vector{Float64}(undef, cn.n_v)
    qdot_scaled = Vector{Float64}(undef, cn.r_v)
    rhs = zeros(Float64, model.n)
    M = copy(helper._LN_sparse)
    M_lu = lu(M)
    M_top = @view M.nzval[helper._LN_top_idx]
    return TimecurveParam(
        logk, x_scaled, q_scaled, logv, v_scaled, qdot_scaled, rhs, M, M_lu, M_top
    )
end

"""
    get_catalysis_ode(model) -> Function

Return the in-place RHS used by `catalysis_logx`.
"""
function get_catalysis_ode(model::Bnc)
    @assert have_catalysis(model) "Should fill catalysis data first"
    helper = _integration_helper!(model)
    cn = model.catalysis
    L_nzval = Float64.(helper._LN_sparse.nzval[helper._LN_top_idx])

    function f(du, u, p::TimecurveParam, t)
        (; logk, x_scaled, q_scaled, logv, v_scaled, qdot_scaled, rhs, M, M_lu, M_top) = p

        mul!(logv, cn._Π_sparse, u)
        logv .+= logk

        u_shift = maximum(u)
        @. x_scaled = exp10(u - u_shift)
        mul!(q_scaled, model.L, x_scaled)
        @. q_scaled = max(q_scaled, 1e-300)
        @. M_top = L_nzval * x_scaled[helper._LN_top_cols] / q_scaled[helper._LN_top_rows]

        lu!(M_lu, M; check=false)
        issuccess(M_lu) || error("Catalysis logx Jacobian is singular at t=$t.")

        v_shift = maximum(logv)
        @. v_scaled = exp10(logv - v_shift)
        mul!(qdot_scaled, cn.S, v_scaled)

        fill!(rhs, 0.0)
        scale = exp10(v_shift - u_shift) / log(10.0)
        @views @. rhs[1:(cn.r_v)] = scale * qdot_scaled / q_scaled[1:(cn.r_v)]

        ldiv!(du, M_lu, rhs)
        any(!isfinite, du) && error("Catalysis ODE produced non-finite du at t=$t.")
        return nothing
    end
end

"""
    catalysis_logx(bnc, logx0, tspan; k, alg=nothing, reltol=1e-8, abstol=1e-9, kwargs...) -> ODESolution

Solve the full catalysis ODE system in log-space species coordinates.
"""
function catalysis_logx(
    Bnc::Bnc,
    logx0::Vector{<:Real},
    tspan::Tuple{Real, Real};
    k::AbstractVector{<:Real},
    alg=nothing,
    reltol=1e-8,
    abstol=1e-9,
    kwargs...,
)::ODESolution
    p = get_catalysis_param(Bnc, k)
    f = get_catalysis_ode(Bnc)
    prob = ODE.ODEProblem(f, logx0, tspan, p)
    return ODE.solve(
        prob, isnothing(alg) ? ODE.Tsit5() : alg; reltol=reltol, abstol=abstol, kwargs...
    )
end
