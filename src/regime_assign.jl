export assign_regime, assign_regime_qK, assign_regime_x

#-----------------------------------------------------------------
# Functions for assigning vertices
#-----------------------------------------------------------------

struct QKHyperplaneClassifier
    regime_ids::Vector{Int}
    dirs::Vector{SparseVector{Float64,Int}}
    bias::Vector{Float64}
    signature_to_regimes::Dict{Tuple{Vararg{Int8}},Vector{Int}}
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
    return Tuple(signs)
end

function _lookup_qK_signature(
    classifier::QKHyperplaneClassifier,
    sig::Tuple{Vararg{Int8}},
)
    direct = get(classifier.signature_to_regimes, sig, Int[])
    length(direct) == 1 && return direct[1]

    any(iszero, sig) || return nothing

    matches = Int[]
    for (key, ids) in classifier.signature_to_regimes
        ok = true
        @inbounds for i in eachindex(sig)
            s = sig[i]
            s == 0 && continue
            if key[i] != s
                ok = false
                break
            end
        end
        ok && append!(matches, ids)
    end

    unique!(matches)
    return length(matches) == 1 ? matches[1] : nothing
end

function _classifier_point(Bnc::Bnc, idx::Int; asymptotic_only::Bool=false)
    return get_one_inner_point(Bnc, idx)
end

function _build_qK_hyperplane_classifier(Bnc::Bnc; asymptotic_only::Bool=false)
    grh = get_regimes_graph!(Bnc; full=true)
    regimes = get_regimes(
        Bnc;
        singular=false,
        asymptotic=asymptotic_only ? true : nothing,
        return_idx=true,
    )

    dirs = getfield.(grh.qK_interface_pool, :change_dir_qK)
    bias = asymptotic_only ? zeros(Float64, length(dirs)) : Float64.(getfield.(grh.qK_interface_pool, :intersect_qK))
    signature_to_regimes = Dict{Tuple{Vararg{Int8}},Vector{Int}}()

    for idx in regimes
        pt = _classifier_point(Bnc, idx; asymptotic_only=asymptotic_only)
        sig = _qK_signature(dirs, bias, pt; tol=1.0e-8)
        push!(get!(signature_to_regimes, sig) do
            Int[]
        end, idx)
    end

    return QKHyperplaneClassifier(regimes, dirs, bias, signature_to_regimes)
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
    @warn("All vertex conditions failed for logqK=$logqK. Returning the best-fit vertex.")
    idx = all_vertice_idx[findmax(record)[2]]
    return return_idx ? idx : get_perm(Bnc, idx)
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
    sig = _qK_signature(classifier.dirs, classifier.bias, logqK; tol=abs(eps))
    idx = _lookup_qK_signature(classifier, sig)
    !isnothing(idx) && return return_idx ? idx : get_perm(Bnc, idx)

    return _assign_regime_qK_fallback(
        Bnc,
        logqK;
        asymptotic_only=asymptotic_only,
        eps=eps,
        return_idx=return_idx,
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
