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
    allow_zero::Vector{BitVector}
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

function _candidate_regime_positions(
    classifier::QKHyperplaneClassifier,
    sig::AbstractVector{Int8},
)
    candidates = trues(length(classifier.regime_ids))
    @inbounds for i in eachindex(sig)
        allow =
            if sig[i] > 0
                classifier.allow_pos[i]
            elseif sig[i] < 0
                classifier.allow_neg[i]
            else
                classifier.allow_zero[i]
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
)
    sig = _qK_signature(classifier.dirs, classifier.bias, logqK; tol=tol)
    pos = _candidate_regime_positions(classifier, sig)
    return classifier.regime_ids[pos], sig
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

function _build_qK_hyperplane_classifier(Bnc::Bnc; asymptotic_only::Bool=false)
    regimes = get_regimes(
        Bnc;
        singular=false,
        asymptotic=asymptotic_only ? true : nothing,
        return_idx=true,
    )
    dirs = SparseVector{Float64,Int}[]
    bias = Float64[]
    key_to_id = Dict{Any,Int}()
    regime_constraints = Vector{Dict{Int,Int8}}(undef, length(regimes))
    key_mode = _affine_is_exact(Bnc) ? :exact : :float

    for (pos, idx) in enumerate(regimes)
        C, C0, nlt = get_C_C0_nullity_qK(Bnc, idx)
        info = Dict{Int,Int8}()
        for row_idx in 1:size(C, 1)
            I, V = findnz(C[row_idx, :])
            isempty(I) && continue
            row = SparseArrays.sparsevec(Int.(I), Float64.(V), size(C, 2))
            β = asymptotic_only ? 0.0 : Float64(C0[row_idx])
            hid, sign = _intern_constraint_hyperplane!(dirs, bias, key_to_id, row, β; key_mode=key_mode)
            hid == 0 && continue
            required = row_idx <= nlt ? Int8(0) : sign
            info[hid] = haskey(info, hid) ? _merge_required_sign(info[hid], required) : required
        end
        regime_constraints[pos] = info
    end

    n_regimes = length(regimes)
    n_hps = length(dirs)
    allow_pos = [trues(n_regimes) for _ in 1:n_hps]
    allow_neg = [trues(n_regimes) for _ in 1:n_hps]
    allow_zero = [trues(n_regimes) for _ in 1:n_hps]

    for (pos, info) in enumerate(regime_constraints)
        for (hid, required) in info
            if required == Int8(1)
                allow_neg[hid][pos] = false
                allow_zero[hid][pos] = false
            elseif required == Int8(-1)
                allow_pos[hid][pos] = false
                allow_zero[hid][pos] = false
            else
                allow_pos[hid][pos] = false
                allow_neg[hid][pos] = false
            end
        end
    end

    return QKHyperplaneClassifier(regimes, dirs, bias, allow_pos, allow_neg, allow_zero)
end

function _get_qK_hyperplane_classifier(Bnc::Bnc; asymptotic_only::Bool=false)
    grh = get_regimes_graph!(Bnc; full=true)
    field = asymptotic_only ? :qK_classifier_asymptotic : :qK_classifier_full
    classifier = getfield(grh, field)
    if isnothing(classifier)
        classifier = _build_qK_hyperplane_classifier(Bnc; asymptotic_only=asymptotic_only)
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

    record = Vector{Float64}(undef, length(candidate_ids))
    for (i, idx) in enumerate(candidate_ids)
        C, C0 = get_C_C0_qK(Bnc, idx)
        min_val = if !asymptotic_only
            minimum(C * logqK .+ C0)
        else
            minimum(C * logqK)
        end
        record[i] = min_val
        if min_val >= -eps
            return return_idx ? Int(idx) : get_perm(Bnc, idx)
        end
    end

    idx = Int(candidate_ids[findmax(record)[2]])
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
    assign_regime_qK(bnc::Bnc, qK; input_logspace=false, asymptotic_only=false, eps=0, return_idx=false)

Assign a regime given qK coordinates.
"""
function assign_regime_qK(Bnc::Bnc, qK::AbstractVector{<:Real}; input_logspace::Bool=false, asymptotic_only::Bool=false, eps=0, return_idx::Bool=false) 
    logqK = input_logspace ? qK : log10.(qK)
    classifier = _get_qK_hyperplane_classifier(Bnc; asymptotic_only=asymptotic_only)
    candidate_ids, _ = _candidate_regimes(classifier, logqK; tol=abs(eps))
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

# Trying speedup assign_regime_qK, but not success yet.
"""
    get_i_j(model::Bnc, perm, t) -> (Int, Int, Int)

Return row/column indices for a constraint index `t`.
"""
function get_i_j(model::Bnc,perm::Vector{<:Integer}, t::Integer)
    i = findfirst(>(t),model._C_partition_idx) - 1
    j1 = perm[i]
    cth = t - model._C_partition_idx[i] + 1
    j2 = model._valid_L_idx[i][cth]
    j2 < j1 ? nothing : j2 += 1
    return i, j1, j2
end

"""
    assign_regime_qK_test(bnc::Bnc, qK; input_logspace=false, asymptotic=true, eps=0)

Experimental qK regime assignment using constraint violation updates.
"""
function assign_regime_qK_test(Bnc::Bnc{T}, qK::AbstractVector{<:Real};
                               input_logspace::Bool=false,
                               asymptotic::Bool=true, eps=0) where T
    logqK = input_logspace ? qK : log10.(qK)
    Perm_tried = Set{UInt64}()  # 存放哈希值

    function try_perm!(perm1)
        (C, C0) = get_C_C0_qK(Bnc, perm1)
        err = C * logqK .+ C0
        ts = findall(er -> er <= -eps, err)

        # 没有违反不等式，返回
        if isempty(ts)
            return perm1
        end

        h = hash(perm1)
        if h in Perm_tried
            error("Cyclic permutation detected! Tried permutations: $(collect(Perm_tried))")
        end
        push!(Perm_tried, h)

        # 对所有违反的约束更新 perm
        for t in ts
            i, j1, j2 = get_i_j(Bnc, perm1, t)
            perm1[i] = j2
            if !haskey(_bind_regimes_perm_dict(Bnc), perm1) 
                perm1[i] = j1  # 恢复原值
            end
            try_perm!(perm1)
        end
        # else
        #         @show perm1
    end

    # 假设初始 perm1 为 1:Bnc.d 或者外部传入
    perm0 = collect(1:Bnc.d) .|> x->T(x)
    return try_perm!(perm0)
end
