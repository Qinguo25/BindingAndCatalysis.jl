function get_expression_path(grh::SISOPaths, pth; observe_x=nothing)
    bn = get_binding_network(grh)
    rgm_pth = get_path(grh, pth; return_idx=true)
    rgm_nlt = get_nullities(bn, rgm_pth)

    change_qK_idx = grh.change_qK_idx
    observe_x_idx = isnothing(observe_x) ? (1:bn.n) : locate_sym_x.(Ref(bn), observe_x)
    rgm_interface = get_interface.(Ref(bn), rgm_pth[1:end-1], rgm_pth[2:end])

    H_H0 = Vector{Any}(undef, length(rgm_pth))
    for i in eachindex(rgm_pth)
        rgm = rgm_pth[i]
        nlt = rgm_nlt[i]
        if nlt == 0
            H, H0 = get_H_H0(bn, rgm)
            H_H0[i] = (H[observe_x_idx, :], H0[observe_x_idx])
        elseif nlt == 1
            H = get_H(bn, rgm)
            H_H0[i] = (H[observe_x_idx, change_qK_idx], nothing)
        else
            error("Nullity > 1 is not supported for expression path.")
        end
    end
    return H_H0, rgm_interface
end

function _calc_RO_for_single_path(model, path::AbstractVector{<:Integer}, change_qK_idx, observe_x_idx)::Vector{<:Real}
    r_ord = Vector{Float64}(undef, length(path))
    for i in eachindex(path)
        if !is_singular(model, path[i])
            r_ord[i] = round(Float64(get_H(model, path[i])[observe_x_idx, change_qK_idx]); digits=3)
        else
            ord = get_H(model, path[i])[observe_x_idx, change_qK_idx]
            r_ord[i] = abs(ord) < 1e-6 ? NaN : Float64(ord) * Inf
        end
    end
    return r_ord
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

function get_RO_path(
    model::Bnc,
    rgm_idx_shift_pth::AbstractVector;
    change_qK,
    observe_x,
    deduplicate::Bool=false,
    keep_singular::Bool=true,
    keep_nonasymptotic::Bool=true,
)::Vector{<:Real}
    rgm_idx_shift_pth = get_idx.(Ref(model), rgm_idx_shift_pth)

    ord_path = let
        change_qK_idx = locate_sym_qK(model, change_qK)
        observe_x_idx = locate_sym_x(model, observe_x)
        _calc_RO_for_single_path(model, rgm_idx_shift_pth, change_qK_idx, observe_x_idx)
    end

    mask = _get_mask(
        model,
        rgm_idx_shift_pth;
        singular=keep_singular ? nothing : false,
        asymptotic=keep_nonasymptotic ? nothing : true,
    )
    ord_path = ord_path[mask]

    return deduplicate ? _dedup(ord_path) : ord_path
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
        get_regime(model, idx; inv_info=true)
    end

    return nothing
end

function get_RO_paths(
    model::Bnc,
    rgm_paths::AbstractVector{<:AbstractVector},
    args...;
    kwargs...,
)::Vector{Vector{<:Real}}
    rgm_idx_for_each_paths = rgm_paths .|> x -> get_idx.(Ref(model), x)
    _ensure_ro_regimes_materialized!(model, rgm_idx_for_each_paths)

    ord_for_each_paths = Vector{Vector{<:Real}}(undef, length(rgm_idx_for_each_paths))
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

function get_RO_paths(model::SISOPaths, pth_idx::Union{Nothing,AbstractVector}=nothing; observe_x, kwargs...)
    path_idxs = _normalize_siso_path_selection(model, pth_idx)
    rgm_paths = get_path.(Ref(model), path_idxs; return_idx=true)
    observe_x_idx = locate_sym_x(model.bn, observe_x)
    return get_RO_paths(
        model.bn,
        rgm_paths;
        change_qK=model.change_qK_idx,
        observe_x=observe_x_idx,
        kwargs...,
    )
end

get_RO_path(model::SISOPaths, pth_idx, args...; kwargs...) = get_RO_paths(model, [get_idx(model, pth_idx)], args...; kwargs...)[1]
