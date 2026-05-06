export calc_volume


#=====================================================================================#
# Statistical functions and utilities
#=====================================================================================#
# 根据命中次数 count、总样本数 N 和 z 值，计算 Wilson 置信区间的中心和半宽。
@inline function _wilson_center_margin(count::Int, N::Int, z::Float64)
    p̂ = count / N
    z2 = z * z
    denom = 1 + z2 / N
    center = (p̂ + z2 / (2N)) / denom
    # Guard against tiny negative roundoff near 0 in the Wilson half-width.
    radicand = max(0.0, p̂ * (1 - p̂) / N + z2 / (4N * N))
    margin = (z / denom) * sqrt(radicand)
    return center, margin
end

# 把采样参数整理成统一的配置对象。
function _prepare_sampling_config(
    sampler::Symbol, # should be :gaussian or :uniform_box
    n_dim::Integer;
    μ::Union{Nothing,AbstractVector{<:Real}}=nothing,
    σ::Float64=1.0,
    log_lower::Float64=-6.0,
    log_upper::Float64=6.0,
    μ_length_message::AbstractString="length(μ) must equal the sample dimension",
)
    μ64 = fill(0.0, n_dim)

    if sampler === :gaussian
        if !isnothing(μ)
            @assert length(μ) == n_dim μ_length_message
            @inbounds for k in 1:n_dim
                μ64[k] = Float64(μ[k])
            end
        end
        @assert σ > 0 "σ must be > 0"
    elseif sampler === :uniform_box
        @assert log_upper > log_lower "log_upper must be > log_lower"
    else
        error("sampler must be :gaussian or :uniform_box, got $sampler")
    end

    return (;
        sampler,
        μ64,
        σ,
        log_lower,
        box_width=log_upper - log_lower,
    )
end


# 按采样配置，往向量 x 里写入一个随机样本。
@inline function _draw_sample!(x::AbstractVector{Float64}, rng, sampling)
    if sampling.sampler === :gaussian
        @inbounds @simd for k in eachindex(x)
            x[k] = sampling.μ64[k] + sampling.σ * randn(rng)
        end
    else
        @inbounds @simd for k in eachindex(x)
            x[k] = sampling.log_lower + sampling.box_width * rand(rng)
        end
    end
    return x
end



#=====================================================================================#
# Volume calculation for direct C,C0
#=====================================================================================#

# 检查一个点是否满足线性不等式约束。
@inline function _satisfies_constraints(
    y::AbstractVector{<:Real},
    b::AbstractVector{<:Real},
    tol::Float64,
)
    @inbounds for k in eachindex(y, b)
        y[k] + b[k] >= -tol || return false
    end
    return true
end


# 把各线程本地计数器累加到全局计数器里，然后把线程局部计数清零。
function _flush_thread_counts!(
    total_counts::Vector{Int},
    thread_counts::AbstractVector{<:Vector{Int}},
    active_ids::AbstractVector{<:Integer},
)
    @inbounds for local_counts in thread_counts
        for idx in active_ids
            total_counts[idx] += local_counts[idx]
            local_counts[idx] = 0
        end
    end
    return nothing
end

# 根据当前累计采样结果更新每个 regime 的 Volume 统计值，并决定哪些 regime 还需要继续采样。
function _update_volume_stats!(
    stats::Vector{Volume},
    total_counts::Vector{Int},
    active_ids::AbstractVector{<:Integer},
    total_N::Int,
    z::Float64,
    rel_tol::Float64,
    abs_tol::Float64,
)
    new_active = Int[]
    sizehint!(new_active, length(active_ids))

    @inbounds for idx in active_ids
        center, margin = _wilson_center_margin(total_counts[idx], total_N, z)
        stats[idx] = Volume(center, margin^2)

        rel_error = center == 0.0 ? Inf : (margin / center)
        if rel_error > rel_tol && margin > abs_tol
            push!(new_active, idx)
        end
    end

    return new_active
end

function _estimate_volumes(
    accumulate!,
    n_regimes::Int,
    n_dim::Int;
    sampler::Symbol = :gaussian,
    μ::Union{Nothing,AbstractVector{<:Real}} = nothing,
    σ::Float64 = 1.0,
    log_lower::Float64 = -6.0,
    log_upper::Float64 = 6.0,
    confidence_level::Float64 = 0.95,
    batch_size::Int = 100_000,
    abs_tol::Float64 = 1.0e-8,
    rel_tol::Float64 = 0.005,
    time_limit::Float64 = 120.0,
    show_progress::Bool = false,
    rng_seed::Integer = 0x12345678,
    μ_length_message::AbstractString = "length(μ) must equal the sample dimension",
)::Vector{Volume}
    n_regimes == 0 && return Volume[]

    z = quantile(Normal(), (1 + confidence_level) / 2)
    sampling = _prepare_sampling_config(
        sampler,
        n_dim;
        μ=μ,
        σ=σ,
        log_lower=log_lower,
        log_upper=log_upper,
        μ_length_message=μ_length_message,
    )

    total_counts = zeros(Int, n_regimes)
    total_N = 0
    stats = zeros(Volume, n_regimes)
    active_ids = collect(1:n_regimes)

    n_slots = Threads.maxthreadid()
    thread_counts = [zeros(Int, n_regimes) for _ in 1:n_slots]
    seed = Int(rng_seed)
    thread_rng = [Random.MersenneTwister(seed + tid) for tid in 1:n_slots]
    thread_x = [Vector{Float64}(undef, n_dim) for _ in 1:n_slots]

    p = show_progress ? Progress(n_regimes, desc="Calculating...", dt=1.0) : nothing
    start_time = time()

    while true
        elapsed = time() - start_time
        if elapsed > time_limit
            @info "Reached time limit ($(round(elapsed, digits=2)) s). Stopping."
            break
        elseif isempty(active_ids)
            @info "All regimes converged after $total_N samples."
            break
        end

        accumulate!(thread_counts, thread_rng, thread_x, batch_size, active_ids, sampling)
        _flush_thread_counts!(total_counts, thread_counts, active_ids)
        total_N += batch_size

        new_active = _update_volume_stats!(
            stats,
            total_counts,
            active_ids,
            total_N,
            z,
            rel_tol,
            abs_tol,
        )

        show_progress && next!(p, step = length(active_ids) - length(new_active))
        active_ids = new_active
    end

    show_progress && finish!(p)
    return stats
end


# 把一组 (C, C0) 约束整理成后续 Monte Carlo 可直接使用的形式。
function _prepare_hrep_volume_problem(
    Cs::AbstractVector{<:AbstractMatrix{<:Real}},
    C0s::AbstractVector{<:AbstractVector{<:Real}};
    rebase_mat::Union{AbstractMatrix{<:Real},Nothing}=nothing,
)
    @assert length(Cs) == length(C0s) "Cs and C0s must have same length"
    n_regimes = length(Cs)
    n_regimes == 0 && return (; Cs, b64=Vector{Vector{Float64}}(), n_dim=0)

    n_dim = size(Cs[1], 2)
    for i in eachindex(Cs)
        @assert size(Cs[i], 2) == n_dim "All Cs must have same column dimension"
        @assert size(Cs[i], 1) == length(C0s[i]) "size(Cs[$i],1) must match length(C0s[$i])"
    end

    rebased_Cs =
        if isnothing(rebase_mat)
            Cs
        else
            [Cs[i] * rebase_mat for i in eachindex(Cs)]
        end

    b64 = Vector{Vector{Float64}}(undef, n_regimes)
    for i in eachindex(C0s)
        b = C0s[i]
        b64[i] = b isa Vector{Float64} ? b : Float64.(b)
    end

    return (; Cs=rebased_Cs, b64, n_dim)
end


# 把 [(C, C0), (C, C0), ...] 这种结构拆成 Cs 和 C0s 两个数组。
# 如果 asymptotic=true，则 C0s 用零向量替代。
function _split_C_C0(C_C0s; asymptotic::Bool)
    Cs = getindex.(C_C0s, 1)
    C0s = asymptotic ? [zeros(size(rep[2])) for rep in C_C0s] : getindex.(C_C0s, 2)
    return Cs, C0s
end


# 对一组对象（只要支持 get_C_C0）提取约束并调用 calc_volume(Cs, C0s; ...)。
function _calc_selected_constraint_volumes(items; asymptotic::Bool, kwargs...)
    C_C0s = items .|> get_C_C0
    Cs, C0s = _split_C_C0(C_C0s; asymptotic=asymptotic)
    return calc_volume(Cs, C0s; kwargs...)
end


# 并行采样 batch_size 个点，统计每个点命中了哪些 polyhedron。
# contain_overlap=false：一个样本最多记到一个 regime。
# contain_overlap=true：一个样本可同时给多个 regime 计数。
function _accumulate_polyhedron_hits!(
    thread_counts,
    thread_rng,
    thread_x,
    thread_y,
    batch_size::Int,
    active_ids::AbstractVector{<:Integer},
    Cs,
    b64,
    sampling,
    regime_judge_tol::Float64,
    contain_overlap::Bool,
)
    Threads.@threads for _ in 1:batch_size
        tid = Threads.threadid()
        rng = thread_rng[tid]
        x = thread_x[tid]
        local_counts = thread_counts[tid]
        ywork = thread_y[tid]

        _draw_sample!(x, rng, sampling)

        @inbounds for idx in active_ids
            y = ywork[idx]
            mul!(y, Cs[idx], x)

            _satisfies_constraints(y, b64[idx], regime_judge_tol) || continue
            local_counts[idx] += 1
            contain_overlap || break
        end
    end
    return nothing
end


"""
    calc_volume(Cs, C0s; kwargs...) -> Vector{Volume}

Monte Carlo 估计：默认使用 N 维高斯抽样 `x ~ 𝒩(μ, σ²I)` 估计各 polyhedron 的概率质量；
若 `sampler=:uniform_box` 则估计盒上均匀抽样下的体积（概率×盒体积）。

polyhedron 约束：A*x + b >= -tol
"""
function calc_volume(
    Cs::AbstractVector{<:AbstractMatrix{<:Real}},
    C0s::AbstractVector{<:AbstractVector{<:Real}};
    # --- sampling ---
    sampler::Symbol = :gaussian,               # :gaussian (default) or :uniform_box
    μ::Union{Nothing,AbstractVector{<:Real}} = nothing,
    σ::Float64 = 1.0,                          # for gaussian: std (isotropic)
    log_lower::Float64 = -6.0,                 # for uniform_box
    log_upper::Float64 = 6.0,                  # for uniform_box

    # --- estimation control ---
    confidence_level::Float64 = 0.95,
    contain_overlap::Bool = false,
    regime_judge_tol::Float64 = 0.0,
    batch_size::Int = 100_000,
    abs_tol::Float64 = 1.0e-8,
    rel_tol::Float64 = 0.005,
    time_limit::Float64 = 120.0,

    # --- perf/UX ---
    show_progress::Bool = false,

    # --- rebase---
    rebase_mat:: Union{AbstractMatrix{<:Real},Nothing} = nothing
)::Vector{Volume}

    problem = _prepare_hrep_volume_problem(Cs, C0s; rebase_mat=rebase_mat)
    n_regimes = length(problem.Cs)
    @info "Number of polyhedra to calc volume: $n_regimes"
    n_regimes == 0 && return Volume[]

    regime_judge_tol = abs(regime_judge_tol)
    n_slots = Threads.maxthreadid()
    thread_y = [
        [Vector{Float64}(undef, size(problem.Cs[i], 1)) for i in 1:n_regimes]
        for _ in 1:n_slots
    ]

    return _estimate_volumes(
        n_regimes,
        problem.n_dim;
        sampler=sampler,
        μ=μ,
        σ=σ,
        log_lower=log_lower,
        log_upper=log_upper,
        confidence_level=confidence_level,
        batch_size=batch_size,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        time_limit=time_limit,
        show_progress=show_progress,
        rng_seed=0x12345678,
        μ_length_message="length(μ) must equal n_dim",
    ) do thread_counts, thread_rng, thread_x, batch_size, active_ids, sampling
        _accumulate_polyhedron_hits!(
            thread_counts,
            thread_rng,
            thread_x,
            thread_y,
            batch_size,
            active_ids,
            problem.Cs,
            problem.b64,
            sampling,
            regime_judge_tol,
            contain_overlap,
        )
    end
end


"""
    calc_volume(C, C0; kwargs...) -> Volume

Compute volume for a single polyhedron.
"""
calc_volume(C::AbstractMatrix{<:Real}, C0::AbstractVector{<:Real}; kwargs...)::Volume =
    calc_volume([C], [C0]; kwargs...)[1]

# calc_vertex_volume(Bnc::Bnc, perm;kwargs...) = calc_vertices_volume(Bnc,[perm]; kwargs...)[1]


#=====================================================================================#
# Volume calculation from qK-space classifier
#=====================================================================================#

_bind_volume_route(
    ::Bnc,
    ::AbstractVector{<:Integer};
    contain_overlap::Bool=false,
    kwargs...,
) = contain_overlap ? :polyhedra : :classifier

function _calc_bind_regime_volumes(
    Bnc::Bnc,
    regime_ids::AbstractVector{<:Integer};
    asymptotic::Bool=true,
    contain_overlap::Bool=false,
    rebase_mat::Union{AbstractMatrix{<:Real},Nothing}=nothing,
    kwargs...,
)
    vals = zeros(Volume, length(regime_ids))

    rgm_ids, rgm_mask = filter_regimes(
        Bnc,
        regime_ids;
        singular=false,
        asymptotic=asymptotic,
        return_mask=true,
    )

    positions = findall(rgm_mask)

    isempty(rgm_ids) && return vals

    route = _bind_volume_route(Bnc, rgm_ids; contain_overlap=contain_overlap)
    vals[positions] .= if route === :classifier
        _calc_volume_via_classifier(
            Bnc,
            rgm_ids;
            asymptotic=asymptotic,
            rebase_mat=rebase_mat,
            kwargs...,
        )
    elseif route === :polyhedra
        rgms = [get_regime(Bnc, idx; inv_info=true) for idx in rgm_ids]
        _calc_selected_constraint_volumes(
            rgms;
            asymptotic=asymptotic,
            contain_overlap=contain_overlap,
            rebase_mat=rebase_mat,
            kwargs...,
        )
    else
        error("unknown bind volume route: $route")
    end
    return vals
end

function _accumulate_classifier_hits!(
    thread_counts,
    thread_rng,
    thread_x,
    batch_size::Int,
    classifier::CompiledClassifier,
    idx_to_pos::AbstractDict{<:Integer,<:Integer},
    sampling;
    asymptotic::Bool = false,
    regime_judge_tol::Float64 = 0.0,
)
    Threads.@threads for _ in 1:batch_size
        tid = Threads.threadid()
        rng = thread_rng[tid]
        x = thread_x[tid]
        local_counts = thread_counts[tid]

        _draw_sample!(x, rng, sampling)
        regime_idx, sides = _classifier_candidates(
            classifier,
            x;
            asymptotic_only=asymptotic,
            tol=regime_judge_tol
        )

        for regime_id in regime_idx
            pos = get(idx_to_pos, regime_id, 0)
            pos == 0 && continue
            local_counts[pos] += 1
        end
    end
    return nothing
end



function _calc_volume_via_classifier(
    Bnc::Bnc,
    regime_ids::AbstractVector{<:Integer};
    sampler::Symbol = :gaussian,
    μ::Union{Nothing,AbstractVector{<:Real}} = nothing,
    σ::Float64 = 1.0,
    log_lower::Float64 = -6.0,
    log_upper::Float64 = 6.0,
    confidence_level::Float64 = 0.95,
    regime_judge_tol::Float64 = 0.0,
    batch_size::Int = 100_000,
    abs_tol::Float64 = 1.0e-8,
    rel_tol::Float64 = 0.005,
    time_limit::Float64 = 120.0,
    show_progress::Bool = true,
    asymptotic::Bool = false,
    rebase_mat::Union{AbstractMatrix{<:Real},Nothing} = nothing,
)::Vector{Volume}
    n_regimes = length(regime_ids)
    n_regimes == 0 && return Volume[]

    n_dim = isnothing(rebase_mat) ? Bnc.n : size(rebase_mat, 2)

    resolved_rebase = if isnothing(rebase_mat)
        nothing
    else
        @assert size(rebase_mat,1) == Bnc.n "size(rebase_mat) must equal (qK dimension, qK dimension)"
        Float64.(rebase_mat)
    end

    grh = get_regimes_graph!(Bnc; full=true)
    qK_hp_data = grh.hp_data[_space(grh, :qK)]
    classifier = compile_classifier(
        qK_hp_data.hyperplanes,
        qK_hp_data.hp_to_poly.M,
        regime_ids;
        rebase_mat=resolved_rebase,
    )

    idx_to_pos = Dict(Int(regime_ids[i]) => i for i in eachindex(regime_ids))
    regime_judge_tol = abs(regime_judge_tol)

    return _estimate_volumes(
        n_regimes,
        n_dim;
        sampler=sampler,
        μ=μ,
        σ=σ,
        log_lower=log_lower,
        log_upper=log_upper,
        confidence_level=confidence_level,
        batch_size=batch_size,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        time_limit=time_limit,
        show_progress=show_progress,
        rng_seed=0x5eed1234,
        μ_length_message="length(μ) must equal qK dimension",
    ) do thread_counts, thread_rng, thread_x, batch_size, active_ids, sampling
        _accumulate_classifier_hits!(
            thread_counts,
            thread_rng,
            thread_x,
            batch_size,
            classifier,
            idx_to_pos,
            sampling;
            asymptotic=asymptotic,
            regime_judge_tol=regime_judge_tol,
        )
    end
end


#-------------------------------------------------------------------------------------
# Volume calculation for polyhedra
#--------------------------------------------------------------------------------------

"""
    _remove_poly_intersect(poly::Polyhedron) -> Polyhedron

Remove intersection offsets to test asymptoticity in polyhedra.
"""
function _remove_poly_intersect(poly::Polyhedron)
    rep = MixedMatHRep(hrep(poly))
    return polyhedron(hrep(rep.A, zeros(eltype(rep.b), size(rep.b)), rep.linset), POLY_BACK_END)
end

"""
    _get_mask(polys; singular=nothing, asymptotic=nothing) -> Vector{Bool}

Return a boolean mask for polyhedra matching singularity/asymptotic filters.
"""
function _get_mask(polys::AbstractVector{<:Polyhedron};
     singular::Union{Bool,Integer,Nothing}=nothing, 
     asymptotic::Union{Bool,Nothing}=nothing)::Vector{Bool}
    n = length(polys)
    full_dim = fulldim(polys[1])
    nullities = full_dim .- dim.(polys)

    asym_flags =
        if isnothing(asymptotic)
            fill(false, n)
        else
            stripped_polys = _remove_poly_intersect.(polys)
            stripped_nullities = full_dim .- dim.(stripped_polys)
            stripped_nullities .== nullities
        end

    matches_singular(nullity) = isnothing(singular) || (
        (singular === true  && nullity > 0) ||
        (singular === false && nullity == 0) ||
        (singular isa Int   && nullity ≤ singular)
    )

    matches_asymptotic(flag) = isnothing(asymptotic) || (asymptotic == flag)
    
    return [
        matches_singular(nullities[i]) && matches_asymptotic(asym_flags[i])
        for i in 1:n
    ]
end

"""
    filter_polys(polys; return_idx=false, kwargs...) -> Vector

Filter polyhedra by singularity/asymptotic criteria.
"""
function filter_polys(polys; return_idx::Bool=false, kwargs...)
    mask = _get_mask(polys; kwargs...)
    return return_idx ? findall(mask) : polys[mask]
end

#------------------------------------------------------------------------------------------------
# calculate volume for Bnc regimes,
#------------------------------------------------------------------------------------------------

"""
    calc_volume(rgms::Union{AbstractVector{<:BindRegime}, AbstractVector{<:Polyhedron}}; asymptotic=true, kwargs...) -> Vector{Volume}

Compute volumes for a collection of polyhedra or vertices.

    calc_volume(model::Bnc, perms=nothing; asymptotic=true, kwargs...) -> Vector{Volume}

Compute volumes for selected regimes in a model.
"""
function calc_volume(rgms::AbstractVector{<:BindRegime};
    asymptotic::Bool=true,
    contain_overlap::Bool=false,
    rebase_mat::Union{AbstractMatrix{<:Real},Nothing}=nothing,
    kwargs...
) # singular/ asymptotic not be put here, as dimensions could reduce and change.
    n_all = length(rgms)
    vals = zeros(Volume, n_all)
    n_all == 0 && return vals

    idxs = findall(_get_mask(rgms;
        singular=false,
        asymptotic=asymptotic ? true : nothing))
    isempty(idxs) && return vals

    same_model = all(get_binding_network(rgm) === get_binding_network(rgms[1]) for rgm in rgms)
    if same_model
        Bnc = get_binding_network(rgms[1])
        regime_ids = get_idx.(rgms)
        return _calc_bind_regime_volumes(
            Bnc,
            regime_ids;
            asymptotic=asymptotic,
            contain_overlap=contain_overlap,
            rebase_mat=rebase_mat,
            kwargs...,
        )
    end

    vals[idxs] .= _calc_selected_constraint_volumes(
        rgms[idxs];
        asymptotic=asymptotic,
        contain_overlap=contain_overlap,
        rebase_mat=rebase_mat,
        kwargs...,
    )
    return vals
end

function calc_volume(rgms::AbstractVector{<:Polyhedron};
    # model::Bnc, perms=nothing;
    asymptotic::Bool=true,
    kwargs...
) # singular/ asymptotic not be put here, as dimensions could reduce and change.
    n_all = length(rgms)
    vals = zeros(Volume, n_all)
    n_all == 0 && return vals

    idxs = filter_polys(
        rgms;
        return_idx=true,
        singular=false,
        asymptotic=asymptotic ? true : nothing,
    )
    isempty(idxs) && return vals

    vals[idxs] .= _calc_selected_constraint_volumes(
        rgms[idxs];
        asymptotic=asymptotic,
        kwargs...,
    )
    return vals
end

"""
    calc_volume(poly::Polyhedron; kwargs...) -> Volume

Compute the volume for a single polyhedron.
"""
calc_volume(poly::Polyhedron;kwargs...) = calc_volume([poly]; kwargs...)[1]
