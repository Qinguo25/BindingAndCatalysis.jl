@inline function _normalize_simo_observe_x(model::Bnc, observe_x)
    idxs = if isnothing(observe_x)
        collect(1:model.n)
    elseif observe_x isa AbstractVector
        Int.(locate_sym_x.(Ref(model), observe_x))
    else
        [Int(locate_sym_x(model, observe_x))]
    end
    return idxs, x_sym(model)[idxs], !(observe_x isa AbstractVector) && !isnothing(observe_x)
end

function _calc_RO_for_single_path(
    model::Bnc,
    path::AbstractVector{<:Integer},
    change_qK_idx::Integer,
    observe_x_idx::AbstractVector{<:Integer},
)::Matrix{Float64}
    out = Matrix{Float64}(undef, length(path), length(observe_x_idx))
    for i in eachindex(path)
        rgm_idx = Int(path[i])
        H = get_H(model, rgm_idx)
        for (j, x_idx) in enumerate(observe_x_idx)
            if !is_singular(model, rgm_idx)
                out[i, j] = round(Float64(H[x_idx, change_qK_idx]); digits=3)
            else
                ord = H[x_idx, change_qK_idx]
                out[i, j] = abs(ord) < 1e-6 ? NaN : Float64(ord) * Inf
            end
        end
    end
    return out
end

function _dedup(ord_path::AbstractVector{T})::Vector{T} where T<:Real
    isempty(ord_path) && return T[]
    out = T[ord_path[1]]
    pending_nan = false
    last_out = out[1]
    @assert !isnan(last_out) "The first element cannot be NaN for deduplication."

    for x in @view ord_path[2:end]
        if isnan(x)
            pending_nan = true
            continue
        end
        if x != last_out
            if pending_nan
                push!(out, NaN)
                pending_nan = false
            end
            push!(out, x)
            last_out = x
        else
            pending_nan = false
        end
    end
    return out
end

function _dedup_rows(ord_path::AbstractMatrix{<:Real})
    size(ord_path, 1) <= 1 && return copy(ord_path)
    keep = Int[1]
    for i in 2:size(ord_path, 1)
        prev = @view ord_path[keep[end], :]
        curr = @view ord_path[i, :]
        all(isequal.(curr, prev)) || push!(keep, i)
    end
    return ord_path[keep, :]
end

function get_RO_path(
    model::Bnc,
    rgm_idx_shift_pth::AbstractVector;
    change_qK,
    observe_x=nothing,
    deduplicate::Bool=false,
    keep_singular::Bool=true,
    keep_nonasymptotic::Bool=true,
)
    rgm_idx_shift_pth = get_idx.(Ref(model), rgm_idx_shift_pth)
    observe_x_idx, _, scalar_observe = _normalize_simo_observe_x(model, observe_x)

    ord_path = let
        change_qK_idx = locate_sym_qK(model, change_qK)
        _calc_RO_for_single_path(model, rgm_idx_shift_pth, change_qK_idx, observe_x_idx)
    end

    mask = _get_mask(
        model,
        rgm_idx_shift_pth;
        singular=keep_singular ? nothing : false,
        asymptotic=keep_nonasymptotic ? nothing : true,
    )
    ord_path = ord_path[mask, :]

    ord_path = if deduplicate
        scalar_observe ? _dedup(vec(ord_path[:, 1])) : _dedup_rows(ord_path)
    else
        scalar_observe ? vec(ord_path[:, 1]) : ord_path
    end

    return ord_path
end

function _ensure_ro_regimes_materialized!(
    model::Bnc,
    rgm_idx_for_each_paths::AbstractVector{<:AbstractVector{<:Integer}},
)
    seen = Set{Int}()
    ordered_idxs = Int[]

    for path in rgm_idx_for_each_paths
        for idx in path
            idx = Int(idx)
            if !(idx in seen)
                push!(ordered_idxs, idx)
                push!(seen, idx)
            end
        end
    end

    for idx in ordered_idxs
        get_binding_regime(model, idx; inv_info=true)
    end

    return nothing
end

function get_RO_paths(
    model::Bnc,
    rgm_paths::AbstractVector{<:AbstractVector},
    args...;
    kwargs...,
)
    rgm_idx_for_each_paths = rgm_paths .|> x -> get_idx.(Ref(model), x)
    _ensure_ro_regimes_materialized!(model, rgm_idx_for_each_paths)

    ord_for_each_paths = Vector{Any}(undef, length(rgm_idx_for_each_paths))
    if Threads.nthreads() == 1 || length(rgm_idx_for_each_paths) <= 1
        for i in eachindex(rgm_idx_for_each_paths)
            ord_for_each_paths[i] = get_RO_path(model, rgm_idx_for_each_paths[i], args...; kwargs...)
        end
    else
        Threads.@threads for i in eachindex(rgm_idx_for_each_paths)
            ord_for_each_paths[i] = get_RO_path(model, rgm_idx_for_each_paths[i], args...; kwargs...)
        end
    end
    return ord_for_each_paths
end

function get_RO_paths(model::SIMOPaths, pth_idx::Union{Nothing,AbstractVector}=nothing; observe_x=nothing, kwargs...)
    path_idxs = get_indices(model, pth_idx)
    rgm_paths = get_path.(Ref(model), path_idxs; return_idx=true)
    return get_RO_paths(
        model.bn,
        rgm_paths;
        change_qK=model.change_qK_idx,
        observe_x=observe_x,
        kwargs...,
    )
end

get_RO_path(model::SIMOPaths, pth_idx, args...; kwargs...) = get_RO_paths(model, [get_idx(model, pth_idx)], args...; kwargs...)[1]
