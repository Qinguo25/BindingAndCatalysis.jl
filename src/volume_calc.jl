export calc_volume

@inline _zero_volume() = Volume(0.0, 0.0)
_zero_volumes(n::Integer) = fill(_zero_volume(), n)

@inline function _wilson_center_margin(count::Int, N::Int, z::Float64)
    p̂ = count / N
    z2 = z * z
    denom = 1 + z2 / N
    center = (p̂ + z2 / (2N)) / denom
    margin = (z / denom) * sqrt(p̂ * (1 - p̂) / N + z2 / (4N * N))
    return center, margin
end

function _prepare_sampling_config(
    sampler::Symbol,
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

function _split_C_C0(C_C0s; asymptotic::Bool)
    Cs = getindex.(C_C0s, 1)
    C0s = asymptotic ? [zeros(size(rep[2])) for rep in C_C0s] : getindex.(C_C0s, 2)
    return Cs, C0s
end

function _calc_selected_constraint_volumes(items; asymptotic::Bool, kwargs...)
    C_C0s = items .|> get_C_C0
    Cs, C0s = _split_C_C0(C_C0s; asymptotic=asymptotic)
    return calc_volume(Cs, C0s; kwargs...)
end

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

@inline function _bind_volume_route(
    Bnc::Bnc,
    regime_ids::AbstractVector{<:Integer};
    asymptotic::Bool=true,
    contain_overlap::Bool=false,
    rebase_mat::Union{AbstractMatrix{<:Real},Nothing}=nothing,
)
    return isnothing(rebase_mat) && !contain_overlap ? :classifier : :polyhedra
end

function _select_regular_bind_regimes(
    Bnc::Bnc,
    regime_ids::AbstractVector{<:Integer};
    asymptotic::Bool=true,
)
    vertices = _bind_regimes_data(Bnc)
    positions = Int[]
    selected_ids = Int[]
    sizehint!(positions, length(regime_ids))
    sizehint!(selected_ids, length(regime_ids))

    for (pos, idx_any) in enumerate(regime_ids)
        idx = Int(idx_any)
        rgm = vertices[idx]
        if rgm.nullity == 0 && (!asymptotic || rgm.is_asymptotic)
            push!(positions, pos)
            push!(selected_ids, idx)
        end
    end

    return positions, selected_ids
end

function _calc_bind_regime_volumes(
    Bnc::Bnc,
    regime_ids::AbstractVector{<:Integer};
    asymptotic::Bool=true,
    contain_overlap::Bool=false,
    rebase_mat::Union{AbstractMatrix{<:Real},Nothing}=nothing,
    kwargs...,
)
    vals = _zero_volumes(length(regime_ids))
    positions, selected_ids = _select_regular_bind_regimes(Bnc, regime_ids; asymptotic=asymptotic)
    isempty(selected_ids) && return vals

    if _bind_volume_route(
        Bnc,
        selected_ids;
        asymptotic=asymptotic,
        contain_overlap=contain_overlap,
        rebase_mat=rebase_mat,
    ) === :classifier
        vals[positions] .= _calc_volume_via_classifier(
            Bnc,
            selected_ids;
            asymptotic_only=asymptotic,
            kwargs...,
        )
        return vals
    end

    rgms = @view _bind_regimes_data(Bnc)[selected_ids]
    vals[positions] .= _calc_selected_constraint_volumes(
        rgms;
        asymptotic=asymptotic,
        contain_overlap=contain_overlap,
        rebase_mat=rebase_mat,
        kwargs...,
    )
    return vals
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

    n_dim = problem.n_dim
    z = quantile(Normal(), (1 + confidence_level) / 2)
    sampling = _prepare_sampling_config(
        sampler,
        n_dim;
        μ=μ,
        σ=σ,
        log_lower=log_lower,
        log_upper=log_upper,
        μ_length_message="length(μ) must equal n_dim",
    )

    regime_judge_tol = abs(regime_judge_tol)

    total_counts = zeros(Int, n_regimes)
    total_N = 0
    stats = _zero_volumes(n_regimes)
    active_ids = collect(1:n_regimes)

    n_slots = Threads.maxthreadid()
    thread_counts = [zeros(Int, n_regimes) for _ in 1:n_slots]
    thread_rng = [Random.MersenneTwister(0x12345678 + tid) for tid in 1:n_slots]
    thread_x = [Vector{Float64}(undef, n_dim) for _ in 1:n_slots]
    thread_y = [
        [Vector{Float64}(undef, size(problem.Cs[i], 1)) for i in 1:n_regimes]
        for _ in 1:n_slots
    ]

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

        if show_progress
            next!(p, step = length(active_ids) - length(new_active))
        end
        active_ids = new_active
    end

    show_progress && finish!(p)
    return stats
end


"""
    calc_volume(C, C0; kwargs...) -> Volume

Compute volume for a single polyhedron.
"""
calc_volume(C::AbstractMatrix{<:Real}, C0::AbstractVector{<:Real}; kwargs...)::Volume =
    calc_volume([C], [C0]; kwargs...)[1]

# calc_vertex_volume(Bnc::Bnc, perm;kwargs...) = calc_vertices_volume(Bnc,[perm]; kwargs...)[1]


function _classify_sampled_regime(
    Bnc::Bnc,
    classifier,
    x::AbstractVector{<:Real};
    asymptotic_only::Bool=false,
    regime_judge_tol::Float64=0.0,
)
    candidate_ids, _ = _candidate_regimes(classifier, x; tol=regime_judge_tol)
    if length(candidate_ids) == 1
        return candidate_ids[1]
    end

    if !isempty(candidate_ids)
        regime_idx = _assign_regime_qK_from_candidates(
            Bnc,
            x,
            candidate_ids;
            asymptotic_only=asymptotic_only,
            eps=regime_judge_tol,
            return_idx=true,
        )
        !isnothing(regime_idx) && return regime_idx
    end

    return _assign_regime_qK_idx_fallback(
        Bnc,
        x;
        asymptotic_only=asymptotic_only,
        eps=regime_judge_tol,
        warn_on_fallback=false,
    )
end

function _accumulate_classifier_hits!(
    thread_counts,
    thread_rng,
    thread_x,
    batch_size::Int,
    Bnc::Bnc,
    classifier,
    idx_to_pos::AbstractDict{<:Integer,<:Integer},
    sampling;
    asymptotic_only::Bool=false,
    regime_judge_tol::Float64=0.0,
)
    Threads.@threads for _ in 1:batch_size
        tid = Threads.threadid()
        rng = thread_rng[tid]
        x = thread_x[tid]
        local_counts = thread_counts[tid]

        _draw_sample!(x, rng, sampling)
        regime_idx = _classify_sampled_regime(
            Bnc,
            classifier,
            x;
            asymptotic_only=asymptotic_only,
            regime_judge_tol=regime_judge_tol,
        )

        pos = get(idx_to_pos, regime_idx, 0)
        pos == 0 && continue
        local_counts[pos] += 1
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
    show_progress::Bool = false,
    asymptotic_only::Bool = false,
)::Vector{Volume}
    n_regimes = length(regime_ids)
    n_regimes == 0 && return Volume[]

    classifier = _get_qK_hyperplane_classifier(Bnc; asymptotic_only=asymptotic_only)
    n_dim = Bnc.d + Bnc.r
    idx_to_pos = Dict(regime_ids[i] => i for i in eachindex(regime_ids))
    z = quantile(Normal(), (1 + confidence_level) / 2)
    sampling = _prepare_sampling_config(
        sampler,
        n_dim;
        μ=μ,
        σ=σ,
        log_lower=log_lower,
        log_upper=log_upper,
        μ_length_message="length(μ) must equal qK dimension",
    )
    regime_judge_tol = abs(regime_judge_tol)

    total_counts = zeros(Int, n_regimes)
    total_N = 0
    stats = _zero_volumes(n_regimes)
    active_ids = collect(1:n_regimes)

    n_slots = Threads.maxthreadid()
    thread_counts = [zeros(Int, n_regimes) for _ in 1:n_slots]
    thread_rng = [Random.MersenneTwister(0x5eed1234 + tid) for tid in 1:n_slots]
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

        _accumulate_classifier_hits!(
            thread_counts,
            thread_rng,
            thread_x,
            batch_size,
            Bnc,
            classifier,
            idx_to_pos,
            sampling;
            asymptotic_only=asymptotic_only,
            regime_judge_tol=regime_judge_tol,
        )

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

        if show_progress
            next!(p, step = length(active_ids) - length(new_active))
        end
        active_ids = new_active
    end

    show_progress && finish!(p)
    return stats
end


#-------------------------------------------------------------------------------------
# Volume calculation for polyhedra
#--------------------------------------------------------------------------------------

"""
    _remove_poly_intersect(poly::Polyhedron) -> Polyhedron

Remove intersection offsets to test asymptoticity in polyhedra.
"""
function _remove_poly_intersect(poly::Polyhedron)
    rep = hrep(poly)
    return polyhedron(hrep(rep.A, zeros(eltype(rep.b), size(rep.b)), rep.linset))
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
    vals = _zero_volumes(n_all)
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
    vals = _zero_volumes(n_all)
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
