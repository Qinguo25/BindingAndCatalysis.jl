"""
    calc_volume(Cs, C0s; kwargs...) -> Vector{Tuple{Float64,Float64}}

Monte Carlo estimate for each polyhedron/regime defined by `A*x + b >= -tol`
(where `A = Cs[i]`, `b = C0s[i]`), using Wilson score interval to stop per-regime
when relative error <= `rel_tol` (or `time_limit` hit).

Returns `stats[i] = (center, margin)` for each regime.
"""
function calc_volume(
    Cs::AbstractVector{<:AbstractMatrix{<:Real}},
    C0s::AbstractVector{<:AbstractVector{<:Real}};
    confidence_level::Float64 = 0.95,
    contain_overlap::Bool = false,
    regime_judge_tol::Float64 = 0.0,
    batch_size::Int = 100_000,
    log_lower::Float64 = -6.0,
    log_upper::Float64 = 6.0,
    abs_tol::Float64 = 1.0e-8,
    rel_tol::Float64 = 0.005,
    time_limit::Float64 = 120.0,
)::Vector{Volume}


    @assert length(Cs) == length(C0s) "Cs and C0s must have same length"
    n_regimes = length(Cs)
    @info "Number of polyhedra to calc volume: $n_regimes"
    n_regimes == 0 && return Tuple{Float64,Float64}[]

    # Dimensions & sanity
    n_dim = size(Cs[1], 2)
    for i in 1:n_regimes
        @assert size(Cs[i], 2) == n_dim "All Cs must have same column dimension"
        @assert size(Cs[i], 1) == length(C0s[i]) "size(Cs[$i],1) must match length(C0s[$i])"
    end

    # Wilson interval parameter
    z = quantile(Normal(), (1 + confidence_level) / 2)

    @inline function wilson_center_margin(count::Int, N::Int)
        count == 0 && return 0.0, 0.0
        P̂ = count / N
        denom = 1 + z^2 / N
        center = (P̂ + z^2 / (2N)) / denom
        margin = (z / denom) * sqrt(P̂ * (1 - P̂) / N + z^2 / (4N^2))
        return center, margin
    end

    # Global stats
    total_counts = zeros(Int, n_regimes)
    total_N = 0
    
    stats = [Volume(0.0, 0.0) for _ in 1:n_regimes]

    active_ids = collect(1:n_regimes)

    # Thread-local slot count (important: maxthreadid, not nthreads)
    n_slots = Threads.maxthreadid()
    thread_counts = [zeros(Int, n_regimes) for _ in 1:n_slots]

    # Thread-local RNG + x workspace
    # Use a stable seed per thread to avoid contention and keep reproducibility-ish.
    thread_rng = [Random.MersenneTwister(0x12345678 + tid) for tid in 1:n_slots]
    thread_x = [Vector{Float64}(undef, n_dim) for _ in 1:n_slots]

    # Thread-local y workspaces: one vector per regime (length = m_i)
    # Use Float64 for speed; if your b is Float64 this is perfect.
    thread_y = [
        [Vector{Float64}(undef, size(Cs[i], 1)) for i in 1:n_regimes]
        for _ in 1:n_slots
    ]

    # Pre-grab b as Float64 vectors if possible (avoids repeated Real->Float64 conversions)
    # If b is already Vector{Float64}, this is just a cheap reference.
    b64 = Vector{Vector{Float64}}(undef, n_regimes)
    for i in 1:n_regimes
        bi = C0s[i]
        if bi isa Vector{Float64}
            b64[i] = bi
        else
            b64[i] = Float64.(bi)
        end
    end

    start_time = time()
    width = log_upper - log_lower

    p = Progress(n_regimes, desc="Calculating volumes...", dt=1.0)

    regime_judge_tol = abs(regime_judge_tol) # ensure non-negative

    while true
        (time() - start_time > time_limit) && (@info "Reached time limit ($(round(time() - start_time, digits=2)) s). Stopping.";break)
        isempty(active_ids) && (@info "All regimes converged after $total_N samples.";break)

        # Monte Carlo batch
        Threads.@threads for _ in 1:batch_size
            tid = Threads.threadid()
            rng = thread_rng[tid]
            x = thread_x[tid]
            local_counts = thread_counts[tid]
            ywork = thread_y[tid]

            # x ~ Uniform(log_lower, log_upper)^n_dim
            @inbounds @simd for k in 1:n_dim
                x[k] = log_lower + width * rand(rng)
            end

            # test regimes
            for idx in active_ids
                A = Cs[idx]
                b = b64[idx]
                y = ywork[idx]

                # y = A*x  (sparse gemv)
                mul!(y, A, x)

                # check y + b >= -tol  (fuse add+check)
                ok = true
                @inbounds for k in 1:length(y)
                    if y[k] + b[k] < -regime_judge_tol
                        ok = false
                        break
                    end
                end
                ok || continue

                local_counts[idx] += 1
                contain_overlap || break
            end
        end

        # Reduce and reset counts; update N
        for c in thread_counts
            @inbounds for i in active_ids
                total_counts[i] += c[i]
                c[i] = 0
            end
        end
        total_N += batch_size

        # Update CI and prune active_ids
        new_active = Int[]
        sizehint!(new_active, length(active_ids))
        for i in active_ids
            center, margin = wilson_center_margin(total_counts[i], total_N)
            stats[i] = Volume(center, margin^2)
            re = (center == 0.0) ? Inf : (margin / center)
            if re > rel_tol && margin > abs_tol
                push!(new_active, i)
            end
        end
        next!(p, step = length(active_ids) - length(new_active))
        active_ids = new_active
    end
    finish!(p)
    return stats
end


calc_volume(C::AbstractMatrix{<:Real}, C0::AbstractVector{<:Real}; kwargs...)::Tuple{Float64,Float64} = calc_volume([C], [C0]; kwargs...)[1]





#------------------------------------------------------------------------------------------------
# calculate volume for Bnc regimes,
#------------------------------------------------------------------------------------------------


function calc_volume(model::Bnc, perms=nothing;
    asymptotic::Union{Bool,Nothing}=true, 
    kwargs...
) # singular/ asymptotic not be put here, as dimensions could reduce and change.


    # filter out those perms expect to give zero volume
    if isnothing(perms)
        find_all_vertices!(model)  # ensure vertices_perm is populated
        n_all  = length(model.vertices_perm)
        # get index for those worth calculate volume
        idxs = get_vertices(model, singular=false, asymptotic= asymptotic; return_idx=true) # Are both index and perms!!!!!!
        perms_to_calc = idxs
    else
        n_all = length(perms)
        idxs = findall(perms) do perm # not perms!!!!!!!!!!!
                !is_singular(model,perm) && (isnothing(asymptotic) || is_asymptotic(model,perm) == asymptotic)
            end
        perms_to_calc = perms[idxs]
    end


    # initialize the data
    vals = [Volume(0.0, 0.0) for _ in 1:n_all]
    
    if isempty(perms_to_calc) # No perms to calc
        return vals
    end

    CC0s = [get_C_C0_qK(model,perm) for perm in perms_to_calc]
    Cs = [ rep[1] for rep in CC0s ]
    C0s = asymptotic ? [zeros(size(rep[2])) for rep in CC0s] : [ rep[2] for rep in CC0s ]
    
    vals[idxs] .= calc_volume(Cs, C0s; kwargs...)
    return vals
end
# calc_vertex_volume(Bnc::Bnc, perm;kwargs...) = calc_vertices_volume(Bnc,[perm]; kwargs...)[1]







#-------------------------------------------------------------------------------------
# Volume calculation for polyhedras
#--------------------------------------------------------------------------------------

# filter and then calculate volumes for polyhedra
function _remove_poly_intersect(poly::Polyhedron)
    (A,b,linset) = MixedMatHRep(hrep(poly)) |> p->(p.A, p.b,p.linset)
    p_new = hrep(A, zeros(size(b)), linset) |> x-> polyhedron(x,CDDLib.Library())
    return p_new
end

function _get_polys_mask(polys::AbstractVector{<:Polyhedron};
     singular::Union{Bool,Integer,Nothing}=nothing, 
     asymptotic::Union{Bool,Nothing}=nothing)::Vector{Bool}
    # ensure nullity and asymptotic flags are calculated

    n = length(polys)

    full_dim = fulldim(polys[1])
    dims = dim.(polys)
    nlt = full_dim .- dims

    flag_asym =
        if isnothing(asymptotic)
            fill(false, n)               # 不使用 asym 标准
        else
            # only compute if needed
            polys_asym = _remove_poly_intersect.(polys)
            nlt_new = full_dim .- dim.(polys_asym)
            nlt_new .== nlt               # asym condition
        end

    check_singular(nlt) = isnothing(singular) || (
        (singular === true  && nlt > 0) ||
        (singular === false && nlt == 0) ||
        (singular isa Int   && nlt ≤ singular)
    )

    check_asym(flag_asym) = isnothing(asymptotic) || (asymptotic == flag_asym)
    
    return [ check_singular(nlt[i]) && check_asym(flag_asym[i]) for i in 1:n ]
end

function filter_polys(polys; return_idx::Bool=false, kwargs...)
    mask = _get_polys_mask(polys; kwargs...)
    return return_idx ? findall(mask) : polys[mask]
end


function calc_volume(polys::AbstractVector{<:Polyhedron};
    asymptotic::Bool=true,
    kwargs...
)
    n_all = length(polys)
    idxs = filter_polys(polys; singular=false, asymptotic= asymptotic ? true : nothing, return_idx=true)
    reps = polys[idxs] .|> poly -> MixedMatHRep(hrep(poly)) |> p->(p.A, p.b)
    
    vals = [Volume(0.0, 0.0) for _ in 1:n_all]

    if isempty(reps)
        return vals
    end    

    Cs = [ -rep[1] for rep in reps ]
    C0s = asymptotic ? [zeros(size(rep[2])) for rep in reps] : [ rep[2] for rep in reps ]
    
    
    vals[idxs] .= calc_volume(Cs, C0s; kwargs...)
    return vals
end

calc_volume(poly::Polyhedron;kwargs...) = calc_volume([poly]; kwargs...)[1]