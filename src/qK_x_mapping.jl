export x2qK, qK2x, qK2x_residual
export x_traj_with_qK_change, x_traj_with_q_change, x_traj_cat, qK_traj_cat, q_traj_cat

#========================================================================================#
# Public qK/x mappings
#========================================================================================#

const _SPACE_MODES = (:linear, :log)

@inline function _resolve_space_mode(
    mode::Symbol, legacy_logspace::Union{Bool, Nothing}, legacy_name::Symbol
)
    if !isnothing(legacy_logspace)
        legacy_mode = legacy_logspace ? :log : :linear
        if mode !== :linear && mode !== legacy_mode
            throw(ArgumentError("conflicting `$legacy_name` and new coordinate-space mode"))
        end
        return legacy_mode
    end

    mode in _SPACE_MODES || throw(
        ArgumentError("coordinate-space mode must be `:linear` or `:log`, got `$mode`")
    )
    return mode
end

"""
    x2qK(bnc::Bnc, x; input=:linear, output=:linear, only_q=false)

Map concentrations `x` to totals/binding constants `qK`.

# Arguments

  - `bnc`: Binding network model.
  - `x`: Species concentrations (vector or matrix).

# Keyword Arguments

  - `input`: Coordinate space of `x`; supported values are `:linear` and `:log`.
  - `output`: Coordinate space of returned values; supported values are `:linear` and `:log`.
  - `only_q`: Return only `q` (conservation totals) when `true`.

# Returns

  - Vector or array containing `q` (and `K` if `only_q=false`).
"""
function x2qK(
    Bnc::Bnc,
    x::AbstractArray{<:Real};
    input::Symbol=:linear,
    output::Symbol=:linear,
    input_logspace::Union{Bool, Nothing}=nothing,
    output_logspace::Union{Bool, Nothing}=nothing,
    only_q::Bool=false,
)::AbstractArray{<:Real}
    input = _resolve_space_mode(input, input_logspace, :input_logspace)
    output = _resolve_space_mode(output, output_logspace, :output_logspace)

    logx = input === :log ? x : log10.(x)
    x_linear = input === :log ? exp10.(x) : x

    q = Bnc.L * x_linear
    q_out = output === :log ? log10.(q) : q
    only_q && return q_out

    logK = Bnc.N * logx
    K_out = output === :log ? logK : exp10.(logK)
    return vcat(q_out, K_out)
end

#========================================================================================#
# qK2x solver dispatch
#========================================================================================#

"""
    qK2x(bnc::Bnc, qK; input=:linear, output=:linear,
        startlogx=nothing, startlogqK=nothing, use_vtx=false, method=nothing,
        reltol=1e-8, abstol=1e-10, kwargs...) -> Vector

Map from totals/binding constants (`qK`) to species concentrations `x`.

# Arguments

  - `bnc`: Binding network model.
  - `qK`: Vector of totals (and optionally binding constants).

# Keyword Arguments

  - `input`: Coordinate space of `qK`; supported values are `:linear` and `:log`.
  - `output`: Coordinate space of returned `x`; supported values are `:linear` and `:log`.
  - `startlogx`: Initial guess for log10(x).
  - `startlogqK`: Initial log10(qK) for homotopy.
  - `use_vtx`: Use regime-based closed form when `true`.
  - `method`: Solver method. Supported built-ins are `:homotopy`, `:free_energy`,
    `:newton_nullspace`, `:nlsolve`, and `:regime`. If omitted, `_default_method`
    chooses `:free_energy` only when the conservation and reaction matrices are
    orthogonal; otherwise it chooses `:homotopy`.
  - `reltol`, `abstol`: Solver tolerances.
  - `kwargs...`: Passed through to the solver.

# Returns

  - Vector of `x` values in log10 or linear space.
"""
function qK2x(
    Bnc::Bnc,
    qK::AbstractVector{<:Real};
    input::Symbol=:linear,
    output::Symbol=:linear,
    input_logspace::Union{Bool, Nothing}=nothing,
    output_logspace::Union{Bool, Nothing}=nothing,
    startlogx::Union{Vector{<:Real}, Nothing}=nothing,
    startlogqK::Union{Vector{<:Real}, Nothing}=nothing,
    use_vtx::Bool=false,
    method::Union{Symbol, Nothing}=nothing,
    reltol=1e-8,
    abstol=1e-10,
    kwargs...,
)::Vector{Float64}
    input = _resolve_space_mode(input, input_logspace, :input_logspace)
    output = _resolve_space_mode(output, output_logspace, :output_logspace)
    method = _resolve_qK2x_method(Bnc, method)
    endlogqK = input === :log ? qK : log10.(qK)

    logx = if use_vtx || method === :regime
        _logqK2logx_regime(Bnc, endlogqK)
    elseif method === :homotopy
        helper = _integration_helper!(Bnc)

        if isnothing(startlogqK) || isnothing(startlogx)
            startlogx = copy(helper._anchor_log_x)
            startlogqK = copy(helper._anchor_log_qK)
        end

        sol = _logx_traj_with_logqK_change(
            Bnc,
            startlogqK,
            endlogqK;
            startlogx=startlogx,
            alg=ODE.Tsit5(),
            save_everystep=false,
            save_start=false,
            reltol=reltol,
            abstol=abstol,
            kwargs...,
        )
        sol.u[end]
    elseif method === :free_energy
        _logqK2logx_free_energy(
            Bnc, endlogqK; startlogx=startlogx, reltol=reltol, abstol=abstol, kwargs...
        )
    elseif method === :newton_nullspace || method === :nullspace
        _logqK2logx_nullspace_newton(
            Bnc, endlogqK; startlogx=startlogx, reltol=reltol, abstol=abstol, kwargs...
        )
    elseif method === :nlsolve
        helper = _integration_helper!(Bnc)
        _logqK2logx_nlsolve(
            Bnc,
            endlogqK;
            startlogx=if isnothing(startlogx)
                copy(helper._anchor_log_x)
            else
                Float64.(startlogx)
            end,
            reltol=reltol,
            abstol=abstol,
            kwargs...,
        )
    else
        throw(ArgumentError("unsupported qK2x method: $method"))
    end

    logx = output === :log ? logx : exp10.(logx)
    return logx
end

"""
    qK2x(bnc::Bnc, qK::AbstractArray{<:Real,2}; kwargs...) -> AbstractArray

Batch mapping from qK space to x space for each column of `qK`.
"""
function qK2x(Bnc::Bnc, qK::AbstractArray{<:Real, 2}; kwargs...)::AbstractArray{<:Real}
    # batch mapping of qK2x for each column of qK and return as matrix.
    # Make thread-safe by creating separate copies for each thread
    f = x -> qK2x(Bnc, x; kwargs...)
    return matrix_iter(f, qK; byrow=false, multithread=true)
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
function _logqK2logx_nlsolve(
    Bnc::Bnc,
    logqK::AbstractArray{<:Real, 1};
    startlogx::Union{Vector{<:Real}, Nothing}=nothing,
    reltol=1e-10,
    abstol=1e-10,
    maxiters::Integer=80,
    damping::Bool=true,
    kwargs...,
)::Vector{<:Real}
    n = Bnc.n
    d = Bnc.d
    helper = _integration_helper!(Bnc)

    u = isnothing(startlogx) ? copy(helper._anchor_log_x) : Float64.(startlogx)
    logqK = Float64.(logqK)
    logq = @view logqK[1:d]
    logK = @view logqK[(d + 1):end]

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
        @views resid[(d + 1):end] .= Bnc.N * u .- logK
        res_norm = norm(resid, Inf)
        res_norm <= target_tol && return u

        @. M_top = x_M_view * L_nzval / q_M_view
        Δ = J \ resid

        step = 1.0
        if damping
            accepted = false
            while step >= 2.0^-40
                u_try = u .- step .* Δ
                trial_resid = qK2x_residual(Bnc, u_try, logqK; input=:log)
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

function _logqK2logx_regime(Bnc::Bnc, logqK::AbstractArray{<:Real, 1})::Vector{Float64}
    perm = assign_regime_qK(Bnc, logqK; input=:log, asymptotic_only=false)
    H, H0 = get_H_H0(Bnc, perm)
    return Vector{Float64}(H * logqK .+ H0)
end

function _logqK2logx_regime_start(Bnc::Bnc, logqK::AbstractArray{<:Real, 1})
    try
        return _logqK2logx_regime(Bnc, logqK)
    catch
        return nothing
    end
end

function qK2x_residual(
    Bnc::Bnc,
    logx::AbstractVector{<:Real},
    qK::AbstractVector{<:Real};
    input::Symbol=:linear,
    input_logspace::Union{Bool, Nothing}=nothing,
)
    input = _resolve_space_mode(input, input_logspace, :input_logspace)
    logqK = input === :log ? Float64.(qK) : log10.(Float64.(qK))
    d = Bnc.d
    r = Bnc.r
    resid = Vector{Float64}(undef, d + r)
    resid .= x2qK(Bnc, logx; input=:log, output=:log) .- logqK
    return resid
end

function _logqK2logx_free_energy(
    Bnc::Bnc,
    logqK::AbstractVector{<:Real};
    startlogx::Union{AbstractVector{<:Real}, Nothing}=nothing,
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
    logK = @view logqK[(d + 1):end]
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
    startlogx::Union{AbstractVector{<:Real}, Nothing}=nothing,
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
    logK = Float64.(view(logqK, (d + 1):length(logqK)))
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
            robust_start || throw(
                ArgumentError(
                    "Nullspace Newton left the positive domain. Pass robust_start=true or a positive startlogx.",
                ),
            )
            return _logqK2logx_free_energy(
                Bnc,
                logqK;
                startlogx=logx_start,
                reltol=reltol,
                abstol=abstol,
                maxiters=maxiters,
            )
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
        return _logqK2logx_free_energy(
            Bnc,
            logqK;
            startlogx=start,
            reltol=reltol,
            abstol=abstol,
            maxiters=maxiters,
            warn_on_maxiters=false,
        )
    end
    @warn "Nullspace qK2x Newton iteration reached maxiters=$maxiters"
    return log10.(x)
end

# function benchmark_qK2x_methods(
#     Bnc::Bnc,
#     qKs;
#     methods=(:free_energy, :homotopy, :newton_nullspace, :nlsolve, :regime),
#     input::Symbol=:log,
#     reference_method::Symbol=:free_energy,
#     kwargs...,
# )
#     cols = qKs isa AbstractMatrix ? [qKs[:, i] for i in axes(qKs, 2)] : collect(qKs)
#     refs = Dict{Int,Vector{Float64}}()
#     for (i, qK) in pairs(cols)
#         refs[i] = qK2x(Bnc, qK; input=input, output=:log, method=reference_method, kwargs...)
#     end

#     results = NamedTuple[]
#     for method in methods
#         failures = 0
#         max_resid = 0.0
#         max_ref_err = 0.0
#         elapsed = @elapsed begin
#             for (i, qK) in pairs(cols)
#                 try
#                     logx = qK2x(Bnc, qK; input=input, output=:log, method=method, kwargs...)
#                     resid = qK2x_residual(Bnc, logx, qK; input=input)
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
    x_traj_with_qK_change(bnc::Bnc, start_point, end_point; input=:linear, output=:linear, kwargs...)

Compute a trajectory in `x` space while `qK` changes linearly in log10 space.

# Arguments

  - `bnc`: Binding network model.
  - `start_point`: Starting `qK` values.
  - `end_point`: Ending `qK` values.

# Keyword Arguments

  - `input`: Coordinate space of inputs; supported values are `:linear` and `:log`.
  - `output`: Coordinate space of returned `x`; supported values are `:linear` and `:log`.
  - `kwargs...`: Passed to the ODE solver.

# Returns

  - Tuple `(t, x_traj)` containing time points and state vectors.
"""
function x_traj_with_qK_change(
    Bnc::Bnc,
    start_point::Vector{<:Real},
    end_point::Vector{<:Real};
    input::Symbol=:linear,
    output::Symbol=:linear,
    input_logspace::Union{Bool, Nothing}=nothing,
    output_logspace::Union{Bool, Nothing}=nothing,
    kwargs...,
)
    input = _resolve_space_mode(input, input_logspace, :input_logspace)
    output = _resolve_space_mode(output, output_logspace, :output_logspace)

    startlogqK = input === :log ? start_point : log10.(start_point)
    endlogqK = input === :log ? end_point : log10.(end_point)

    solution = _logx_traj_with_logqK_change(
        Bnc, startlogqK, endlogqK; dense=false, kwargs...
    )

    if output === :linear
        foreach(u -> u .= exp10.(u), solution.u)
    end

    return _ode_solution_wrapper(solution)
end

"""
    x_traj_with_q_change(bnc::Bnc, start_q, end_q; K=nothing, logK=nothing, input=:linear, kwargs...)

Compute an `x` trajectory for a change in `q` while holding `K` fixed.
"""
function x_traj_with_q_change(
    Bnc::Bnc,
    start_q::Vector{<:Real},
    end_q::Vector{<:Real};
    K::Union{Vector{<:Real}, Nothing}=nothing,
    logK::Union{Vector{<:Real}, Nothing}=nothing,
    input::Symbol=:linear,
    input_logspace::Union{Bool, Nothing}=nothing,
    kwargs...,
)
    input = _resolve_space_mode(input, input_logspace, :input_logspace)
    K_prepared = if input === :log
        (isnothing(logK) ? log10.(K) : logK)
    else
        (isnothing(K) ? K : exp10.(K))
    end
    return x_traj_with_qK_change(
        Bnc, [start_q; K_prepared], [end_q; K_prepared]; input=input, kwargs...
    )
end

"""
    HomotopyParams

Cache container for homotopy-based qK→x integration.
"""
struct HomotopyParams{V <: Vector{Float64}, SV1 <: SubArray, SV2 <: SubArray}
    ### Constants
    startlogqK::V
    ΔlogqK::V
    logx::V
    logqK::V
    logq::SV1
    logK::SV1
    logqK_max::Float64

    M::SparseMatrixCSC{Float64, Int}
    M_lu::SparseArrays.UMFPACK.UmfpackLU{Float64, Int}

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
    logqK_max = maximum([20.0, maximum(startlogqK), maximum(endlogqK)])
    n = Bnc.n
    d = Bnc.d
    startlogqK = Float64.(startlogqK)
    ΔlogqK = Float64.(endlogqK - startlogqK)
    # Create thread-local copies of all mutable data structures
    logx = Vector{Float64}(undef, n)
    logqK = Vector{Float64}(undef, n)
    logq = @view logqK[1:d]
    logK = @view logqK[(d + 1):end]
    M = copy(helper._LN_sparse)
    M_lu = deepcopy(helper._LN_lu)

    logx_M_view = @view logx[helper._LN_top_cols] # view for faster updating J
    logq_M_view = @view logqK[helper._LN_top_rows] # view for faster updating J
    M_top = @view M.nzval[helper._LN_top_idx] # view for faster updating J
    M_top_diag = @view M.nzval[helper._LN_top_diag_idx] # top-row entries to perturb when J is singular

    p = HomotopyParams(
        startlogqK,
        ΔlogqK,
        logx,
        logqK,
        logq,
        logK,
        logqK_max,
        M,
        M_lu,
        logx_M_view,
        logq_M_view,
        M_top,
        M_top_diag,
        # logx_local,logx_M_view_local,logLx_local, logLx_M_view_local
    )
    return p
end

function get_homotopy_ode(Bnc::Bnc, p::HomotopyParams)
    # Constants helps for updating mutable datas
    helper = _integration_helper!(Bnc)
    L_nzval = log10.(helper._LN_sparse.nzval[helper._LN_top_idx]) # copy the nzval to avoid shared access

    @inline function update_M_lu(M_lu, M, max_try=100)
        lu!(M_lu, M; check=false) # recalculate the LU decomposition of J
        try_count = 0
        while !issuccess(M_lu) && try_count < max_try
            @.p.M_top_diag += eps() # perturb the diagonal elements a bit to avoid singularity
            lu!(M_lu, M; check=false)
            try_count += 1
        end
        if try_count == max_try
            @error("M is still singular after maximum perturbation attempts.")
            @show M
        end
    end

    function (du, u, p, t)
        @unpack startlogqK,
        ΔlogqK, logx, logqK, logqK_max, M, M_lu, logx_M_view, logq_M_view, M_top,
        M_top_diag = p
        #update q & x
        clamp!(u, -Inf, logqK_max) # make sure not overflow.
        @. logx = u
        @. logqK = startlogqK + t * ΔlogqK
        #update M_top(sparse version) - use the local copy of nzval
        @. M_top = exp10(logx_M_view - logq_M_view + L_nzval)
        # Update the dlogx
        update_M_lu(M_lu, M)
        return ldiv!(du, M_lu, ΔlogqK)
    end
end

"""
    _logx_traj_with_logqK_change(bnc::Bnc, startlogqK, endlogqK; startlogx=nothing,
        alg=nothing, reltol=1e-8, abstol=1e-9, ensure_manifold=true, npoints=nothing, kwargs...) -> ODESolution

Integrate a homotopy path in log space to map qK changes to x trajectories.
"""
function _logx_traj_with_logqK_change(
    Bnc::Bnc,
    startlogqK::Vector{<:Real},
    endlogqK::Vector{<:Real};
    # Optional parameters for the initial log(x) values,act as initial point for ode solving
    startlogx::Union{Vector{<:Real}, Nothing}=nothing,
    # Optional parameters for the ODE solver
    alg=nothing, # Default to nothing, will use Tsit5() if not provided
    reltol=1e-8,
    abstol=1e-9,
    ensure_manifold::Bool=true, # Make sure the trajectory stays on the manifold defined by Lx=q and Nlogx=logK
    npoints::Union{Nothing, Integer}=nothing,
    kwargs..., #other Optional arguments for ODE solver
)::ODESolution
    # println("_logx_traj_with_logqK_change get kwargs: ", kwargs)
    #---Solve the homotopy ODE to find x from qK.---

    # Prepare starting x if not given
    u0 = isnothing(startlogx) ? qK2x(Bnc, startlogqK; input=:log, output=:log) : startlogx
    p = get_homotopy_param(Bnc, startlogqK, endlogqK)
    f! = get_homotopy_ode(Bnc, p)

    callback = if !ensure_manifold
        CB.CallbackSet()
    else
        n = Bnc.n
        d = Bnc.d
        keep_manifold! = function (resid, u, p)  # Can not write to forms like log_sum_exp10!(logLx_local, Bnc.L, u) for Autodiff.
            @unpack logq, logK = p
            resid[1:d] .= log10.(Bnc.L * exp10.(u)) .- logq
            return resid[(d + 1):end] .= Bnc.N * u .- logK
        end
        equilibrium_cb = CB.ManifoldProjection(
            keep_manifold!;
            save=false,
            resid_prototype=zeros(n),
            # manifold_jacobian=manifold_jac!,
            # jac_prototype = [Bnc.L;Bnc.N],
            autodiff=AutoForwardDiff(),
            abstol=1e-12,
            reltol=1e-10,
        )
        CB.CallbackSet(equilibrium_cb)
    end

    # Solve the ODE using the DifferentialEquations.jl package

    prob = ODE.ODEProblem(f!, u0, (0.0, 1.0), p)

    sol = if isnothing(npoints)
        ODE.solve(prob, alg; reltol=reltol, abstol=abstol, callback=callback, kwargs...)
    else
        ODE.solve(
            prob,
            alg;
            reltol=reltol,
            abstol=abstol,
            callback=callback,
            saveat=range(0, 1, npoints),
            tstops=range(0, 1, npoints),
            kwargs...,
        )
    end
    return sol
end

#--------------------------------------------------------------------------------
#      Functions for modeling when envolving catalysis reactions, 
#--------------------------------------------------------------------------------

"""
    x_traj_cat(bnc::Bnc, x0, tspan; input=:linear, output=:linear, kwargs...) -> (Vector, Vector)

Simulate species trajectories under catalysis dynamics.
"""
function x_traj_cat(
    Bnc::Bnc,
    x0::Vector{<:Real},
    tspan::Tuple{Real, Real};
    input::Symbol=:linear,
    output::Symbol=:linear,
    input_logspace::Union{Bool, Nothing}=nothing,
    output_logspace::Union{Bool, Nothing}=nothing,
    kwargs...,
)
    input = _resolve_space_mode(input, input_logspace, :input_logspace)
    output = _resolve_space_mode(output, output_logspace, :output_logspace)
    x0 = input === :log ? x0 : log10.(x0)
    #---Solve the ODE to find the time curve of log(x) as catalysis happens
    sol = catalysis_logx(
        Bnc,
        x0,
        tspan;
        dense=false, #manually handle later
        kwargs...,
    )
    if output === :linear
        foreach(u -> u .= exp10.(u), sol.u)
    end

    return _ode_solution_wrapper(sol)
end

"""
    qK_traj_cat(bnc::Bnc, args...; only_q=false, input=:linear, output=:linear, kwargs...) -> (Vector{Float64}, Matrix{Float64})

Simulate catalysis dynamics and return trajectories in q/K space.
"""
function qK_traj_cat(
    Bnc::Bnc,
    qK0::Vector{<:Real},
    args...;
    only_q::Bool=false,
    input::Symbol=:linear,
    output::Symbol=:linear,
    input_logspace::Union{Bool, Nothing}=nothing,
    output_logspace::Union{Bool, Nothing}=nothing,
    kwargs...,
)
    input = _resolve_space_mode(input, input_logspace, :input_logspace)
    output = _resolve_space_mode(output, output_logspace, :output_logspace)
    logx0 = qK2x(Bnc, qK0; input=input, output=:log)
    t, u = x_traj_cat(Bnc, logx0, args...; input=:log, output=:log, kwargs...)
    u = x2qK.(Ref(Bnc), u; input=:log, output=output, only_q=only_q)
    return (t, u)
end

q_traj_cat(args...; kwargs...) = qK_traj_cat(args...; only_q=true, kwargs...)
