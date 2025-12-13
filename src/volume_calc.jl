
# Core function of calc volume:
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
            # 如果 contain_overlap == true：记录所有满足的 regimes（允许多分配
            vals = [similar(b) for b in C0s] # 提前分配内存
            for i in eachindex(Cs)
                if !active[i]
                    continue
                end
                @views A = Cs[i]
                @views b = C0s[i]

                # 计算 A*x + b 的每个分量是否都 > -tol
                # 使用 all(...) 简洁直观（可能会分配一个临时数组在某些情况下，但通常 A*x+b 是向量）
                mul!(vals[i], A,x)
                vals[i] .+b
                # vals = A * x .+ b
                if any(vals[i] .< -tol)
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





#------------------------------------------------------------------------------------------------
# calculate volume for Bnc regimes,
#------------------------------------------------------------------------------------------------


function calc_volume(model::Bnc, perms=nothing;
    asymptotic::Union{Bool,Nothing}=true, 
    kwargs...
)::Vector{Tuple{Float64, Float64}} # singular/ asymptotic not be put here, as dimensions could reduce and change.


    # filter out those perms expect to give zero volume
    if isnothing(perms)
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
    vals = collect(zip(zeros(Float64, n_all), zeros(Float64, n_all)))
    
    if isempty(perms_to_calc) # No perms to calc
        return vals
    end

    CC0s = [get_C_C0_qK!(model,perm) for perm in perms_to_calc]
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
)::Vector{Tuple{Float64,Float64}}
    n_all = length(polys)
    idxs = filter_polys(polys; singular=false, asymptotic= asymptotic ? true : nothing, return_idx=true)
    reps = polys[idxs] .|> poly -> MixedMatHRep(hrep(poly)) |> p->(p.A, p.b)
    
    vals = collect(zip(zeros(Float64, n_all), zeros(Float64, n_all)))

    if isempty(reps)
        return vals
    end    

    Cs = [ -rep[1] for rep in reps ]
    C0s = asymptotic ? [zeros(size(rep[2])) for rep in reps] : [ rep[2] for rep in reps ]
    
    
    vals[idxs] .= calc_volume(Cs, C0s; kwargs...)
    return vals
end

calc_volume(poly::Polyhedron;kwargs...) = calc_volume([poly]; kwargs...)[1]