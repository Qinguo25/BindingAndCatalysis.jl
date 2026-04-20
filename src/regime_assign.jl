export assign_regime, assign_regime_qK, assign_regime_x

#-----------------------------------------------------------------
# Functions for assigning vertices
#-----------------------------------------------------------------

struct QKHyperplaneClassifier
    regime_ids::Vector{Int}
    dirs::Vector{SparseVector{Float64,Int}}
    bias::Vector{Float64}
    allow_pos::Vector{BitVector}
    allow_neg::Vector{BitVector}
end

@inline function _hyperplane_side(val::Real, tol::Real)
    return val > tol ? Int8(1) : val < -tol ? Int8(-1) : Int8(0)
end

function _qK_signature(
    dirs::AbstractVector{<:SparseVector},
    bias::AbstractVector{<:Real},
    logqK::AbstractVector{<:Real};
    tol::Real=0,
)
    signs = Vector{Int8}(undef, length(dirs))
    @inbounds for i in eachindex(dirs)
        signs[i] = _hyperplane_side(dot(dirs[i], logqK) + bias[i], tol)
    end
    return signs
end

@inline _qK_classifier_point_repr(logqK::AbstractVector{<:Real}) = repr(collect(logqK))
@inline _qK_classifier_sig_repr(sig::AbstractVector{Int8}) = repr(collect(sig))

function _throw_qK_classifier_boundary_error(
    logqK::AbstractVector{<:Real},
    sig::AbstractVector{Int8},
)
    boundary_ids = findall(iszero, sig)
    error(
        "qK hyperplane classifier hit hyperplane boundary: " *
        "logqK=$(_qK_classifier_point_repr(logqK)), " *
        "signature=$(_qK_classifier_sig_repr(sig)), " *
        "boundary_hyperplane_ids=$(repr(boundary_ids))",
    )
end

function _throw_qK_classifier_nonunique_error(
    logqK::AbstractVector{<:Real},
    sig::AbstractVector{Int8},
    candidate_ids::AbstractVector{<:Integer},
)
    error(
        "qK hyperplane classifier is not unique: " *
        "logqK=$(_qK_classifier_point_repr(logqK)), " *
        "signature=$(_qK_classifier_sig_repr(sig)), " *
        "candidate_ids=$(repr(Int.(candidate_ids)))",
    )
end

function _throw_qK_classifier_no_candidate_error(
    logqK::AbstractVector{<:Real},
    sig::AbstractVector{Int8},
)
    error(
        "qK hyperplane classifier found no candidate regime: " *
        "logqK=$(_qK_classifier_point_repr(logqK)), " *
        "signature=$(_qK_classifier_sig_repr(sig)), " *
        "candidate_ids=Int[]",
    )
end

function _candidate_regime_positions(
    classifier::QKHyperplaneClassifier,
    sig::AbstractVector{Int8},
    logqK::Union{AbstractVector{<:Real},Nothing}=nothing;
    strict::Bool=false,
)
    if strict
        any(iszero, sig) && _throw_qK_classifier_boundary_error(something(logqK, Int[]), sig)
    end
    candidates = trues(length(classifier.regime_ids))
    @inbounds for i in eachindex(sig)
        allow =
            if sig[i] > 0
                classifier.allow_pos[i]
            elseif sig[i] < 0
                classifier.allow_neg[i]
            else
                strict && _throw_qK_classifier_boundary_error(something(logqK, Int[]), sig)
                continue
            end
        candidates .&= allow
        any(candidates) || return Int[]
    end
    return findall(candidates)
end

function _candidate_regimes(
    classifier::QKHyperplaneClassifier,
    logqK::AbstractVector{<:Real};
    tol::Real=0,
    strict::Bool=false,
)
    sig = _qK_signature(classifier.dirs, classifier.bias, logqK; tol=tol)
    pos = _candidate_regime_positions(classifier, sig, logqK; strict=strict)
    return classifier.regime_ids[pos], sig
end

function _resolve_unique_qK_candidate(
    classifier::QKHyperplaneClassifier,
    logqK::AbstractVector{<:Real};
    tol::Real=0,
)
    candidate_ids, sig = _candidate_regimes(classifier, logqK; tol=tol, strict=true)
    isempty(candidate_ids) && _throw_qK_classifier_no_candidate_error(logqK, sig)
    length(candidate_ids) == 1 || _throw_qK_classifier_nonunique_error(logqK, sig, candidate_ids)
    return Int(candidate_ids[1]), sig
end

function _classifier_point(Bnc::Bnc, idx::Int; asymptotic_only::Bool=false)
    return get_one_inner_point(Bnc, idx)
end

function _intern_constraint_hyperplane!(
    dirs::Vector{SparseVector{Float64,Int}},
    bias::Vector{Float64},
    key_to_id::Dict,
    row::SparseVector,
    β::Real;
    key_mode::Symbol=:float,
    atol::Float64=1e-10,
    round_digits::Int=10,
)
    canon = _canonicalize_qK_interface(row, β; key_mode=key_mode, atol=atol, round_digits=round_digits)
    canon === nothing && return 0, Int8(0)
    dir_norm, bnorm, sign, key = canon
    hid = get!(key_to_id, key) do
        push!(dirs, dir_norm)
        push!(bias, bnorm)
        length(dirs)
    end
    return hid, sign
end

function _merge_required_sign(old::Int8, new::Int8)
    old == new && return old
    return Int8(0)
end

function _get_regime_qK_hyperplane_id_signs(
    grh::RegimeGraph,
    regime,
)
    Bnc = get_binding_network(grh)
    idx = get_idx(Bnc, regime)
    info = Dict{Int,Int8}()
    for edge in grh.neighbors[idx]
        _edge_has_qK_interface(edge) || continue
        hid = edge.qK_interface_idx
        sign = Int8(-edge.qK_interface_sign)
        sign == 0 && error(
            "RegimeGraph returned zero qK edge sign: regime=$idx, neighbor=$(edge.to), hyperplane_id=$hid",
        )
        old = get(info, hid, sign)
        old == sign || error(
            "Inconsistent qK hyperplane sign in RegimeGraph: regime=$idx, hyperplane_id=$hid, " *
            "existing_sign=$old, new_sign=$sign",
        )
        info[hid] = sign
    end
    return info
end

function _build_qK_hyperplane_classifier(grh::RegimeGraph; asymptotic_only::Bool=false)
    Bnc = get_binding_network(grh)
    regimes = get_regimes(
        Bnc;
        singular=false,
        asymptotic=asymptotic_only ? true : nothing,
        return_idx=true,
    )

    n_hps = length(grh.qK_interface_pool)
    dirs = Vector{SparseVector{Float64,Int}}(undef, n_hps)
    bias = Vector{Float64}(undef, n_hps)
    for hid in 1:n_hps
        hp = grh.qK_interface_pool[hid]
        dirs[hid] = hp.change_dir_qK
        bias[hid] = hp.intersect_qK
    end

    n_regimes = length(regimes)
    allow_pos = [trues(n_regimes) for _ in 1:n_hps]
    allow_neg = [trues(n_regimes) for _ in 1:n_hps]

    for (pos, idx) in enumerate(regimes)
        info = _get_regime_qK_hyperplane_id_signs(grh, idx)
        for (hid, required) in info
            if required == Int8(1)
                allow_neg[hid][pos] = false
            elseif required == Int8(-1)
                allow_pos[hid][pos] = false
            else
                error(
                    "RegimeGraph returned zero qK hyperplane sign for regime=$idx and hyperplane_id=$hid",
                )
            end
        end
    end

    return QKHyperplaneClassifier(regimes, dirs, bias, allow_pos, allow_neg)
end

function _get_qK_hyperplane_classifier(Bnc::Bnc; asymptotic_only::Bool=false)
    grh = get_regimes_graph!(Bnc; full=true)
    field = asymptotic_only ? :qK_classifier_asymptotic : :qK_classifier_full
    classifier = getfield(grh, field)
    if isnothing(classifier)
        classifier = _build_qK_hyperplane_classifier(grh; asymptotic_only=asymptotic_only)
        setfield!(grh, field, classifier)
    end
    return classifier::QKHyperplaneClassifier
end

function _assign_regime_qK_fallback(
    Bnc::Bnc,
    logqK::AbstractVector{<:Real};
    asymptotic_only::Bool=false,
    eps=0,
    return_idx::Bool=false,
    warn_on_fallback::Bool=true,
)
    real_only = asymptotic_only ? true : nothing
    all_vertice_idx = get_regimes(Bnc, singular=false, asymptotic = real_only, return_idx = true)

    record = Vector{Float64}(undef,length(all_vertice_idx))
    for (i, idx) in enumerate(all_vertice_idx)
        C, C0 = get_C_C0_qK(Bnc, idx)
        min_val = if !asymptotic_only
            minimum(C * logqK .+ C0)
        else
            minimum(C * logqK)
        end
        record[i] = min_val

        if record[i] >= -eps
            return return_idx ? idx : get_perm(Bnc, idx)
        end
    end
    warn_on_fallback && @warn("All vertex conditions failed for logqK=$logqK. Returning the best-fit vertex.")
    idx = all_vertice_idx[findmax(record)[2]]
    return return_idx ? idx : get_perm(Bnc, idx)
end

function _assign_regime_qK_from_candidates(
    Bnc::Bnc,
    logqK::AbstractVector{<:Real},
    candidate_ids::AbstractVector{<:Integer};
    asymptotic_only::Bool=false,
    eps=0,
    return_idx::Bool=false,
)
    isempty(candidate_ids) && return _assign_regime_qK_fallback(
        Bnc,
        logqK;
        asymptotic_only=asymptotic_only,
        eps=eps,
        return_idx=return_idx,
        warn_on_fallback=false,
    )

    classifier = _get_qK_hyperplane_classifier(Bnc; asymptotic_only=asymptotic_only)
    pos_map = Dict(classifier.regime_ids[i] => i for i in eachindex(classifier.regime_ids))
    selected_ids = Int[]
    selected_pos = Int[]
    for idx_any in candidate_ids
        idx = Int(idx_any)
        haskey(pos_map, idx) || continue
        push!(selected_ids, idx)
        push!(selected_pos, pos_map[idx])
    end
    isempty(selected_ids) && return _assign_regime_qK_fallback(
        Bnc,
        logqK;
        asymptotic_only=asymptotic_only,
        eps=eps,
        return_idx=return_idx,
        warn_on_fallback=false,
    )

    relevant_hids = Int[]
    sizehint!(relevant_hids, length(classifier.dirs))
    for hid in eachindex(classifier.dirs)
        relevant = false
        @inbounds for pos in selected_pos
            if !classifier.allow_pos[hid][pos] || !classifier.allow_neg[hid][pos]
                relevant = true
                break
            end
        end
        relevant && push!(relevant_hids, hid)
    end

    sub_classifier = QKHyperplaneClassifier(
        selected_ids,
        classifier.dirs[relevant_hids],
        classifier.bias[relevant_hids],
        [copy(classifier.allow_pos[hid][selected_pos]) for hid in relevant_hids],
        [copy(classifier.allow_neg[hid][selected_pos]) for hid in relevant_hids],
    )

    idx, _ = _resolve_unique_qK_candidate(sub_classifier, logqK; tol=abs(eps))
    return return_idx ? idx : get_perm(Bnc, idx)
end

@inline function _assign_regime_qK_idx_fallback(
    Bnc::Bnc,
    logqK::AbstractVector{<:Real};
    asymptotic_only::Bool=false,
    eps=0,
    warn_on_fallback::Bool=false,
)
    return _assign_regime_qK_fallback(
        Bnc,
        logqK;
        asymptotic_only=asymptotic_only,
        eps=eps,
        return_idx=true,
        warn_on_fallback=warn_on_fallback,
    )
end

"""
    assign_regime_x(bnc::Bnc, x; input_logspace=false, asymptotic_only=true, return_idx=false)

Assign a regime given a point in x space.
"""
function assign_regime_x(Bnc::Bnc{T}, x::AbstractVector{<:Real};
    input_logspace::Bool=false,
    asymptotic_only::Bool=true,
    return_idx::Bool=false) where T
    # x = input_logspace ? exp10.(x) : x
    helper = _integration_helper!(Bnc)
    L = Bnc.L
    d = Bnc.d
    n = Bnc.n
    max_indices = zeros(T, d)
    max_val = fill(-Inf, d)
    colptr = L.colptr
    rowval = L.rowval

    if asymptotic_only
        nzval = @view(x[helper._LN_top_cols])
    else
        x = input_logspace ? exp10.(x) : x # linear or log space only matters when not asymptotic
        nzval = @view(x[helper._LN_top_cols]) .* L.nzval
    end

    @inbounds for col in 1:n
        col_start_idx = colptr[col]
        col_end_idx   = colptr[col+1] - 1
        if col_start_idx <= col_end_idx #escape empty column
            @inbounds for idx in col_start_idx:col_end_idx
                v = nzval[idx]
                row = rowval[idx]
                if v > max_val[row]
                    max_val[row] = v
                    max_indices[row] = col
                end
            end
        end
    end
    return return_idx ? get_idx(Bnc,max_indices) : max_indices
end
# function get_vertex_qK(Bnc::Bnc, x::AbstractMatrix{<:Real}; kwargs...) 
#     [get_vertex_qK_slow(Bnc, row; kwargs...) for row in eachrow(x)]
# end

"""
    assign_regime_qK(bnc::Bnc; x, input_logspace=false, kwargs...) -> Vector

Assign a regime given a point in x space by first mapping to qK.
"""
function assign_regime_qK(Bnc::Bnc ; x::AbstractVector{<:Real}, input_logspace::Bool=false, kwargs...) 
    # @show all_vertice_idx
    logqK = x2qK(Bnc,x; input_logspace=input_logspace, output_logspace=true)
    return assign_regime_qK(Bnc, logqK; input_logspace=true, kwargs...)
end
"""
    assign_regime_qK(bnc::Bnc, qK; input_logspace=false, asymptotic_only=false, eps=0, return_idx=false, strict=true)

Assign a regime given qK coordinates.
"""
function assign_regime_qK(
    Bnc::Bnc,
    qK::AbstractVector{<:Real};
    input_logspace::Bool=false,
    asymptotic_only::Bool=false,
    eps=0,
    return_idx::Bool=false,
    strict::Bool=true,
)
    logqK = input_logspace ? qK : log10.(qK)
    classifier = _get_qK_hyperplane_classifier(Bnc; asymptotic_only=asymptotic_only)
    if strict
        regime_idx, _ = _resolve_unique_qK_candidate(classifier, logqK; tol=abs(eps))
        return return_idx ? regime_idx : get_perm(Bnc, regime_idx)
    end

    candidate_ids, _ = _candidate_regimes(classifier, logqK; tol=abs(eps), strict=false)
    length(candidate_ids) == 1 && return return_idx ? candidate_ids[1] : get_perm(Bnc, candidate_ids[1])
    !isempty(candidate_ids) && return _assign_regime_qK_from_candidates(
        Bnc,
        logqK,
        candidate_ids;
        asymptotic_only=asymptotic_only,
        eps=eps,
        return_idx=return_idx,
    )

    return _assign_regime_qK_fallback(
        Bnc,
        logqK;
        asymptotic_only=asymptotic_only,
        eps=eps,
        return_idx=return_idx,
        warn_on_fallback=true,
    )
end

"""
    assign_regime(args...; kwargs...) -> Vector

Alias for `assign_regime_qK`.
"""
assign_regime(args...;kwargs...)=assign_regime_qK(args...;kwargs...)


#-------------------------------------------------------------------------------------------------------------------------------------------------------
