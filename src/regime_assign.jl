export assign_regime, assign_regime_qK, assign_regime_x

#-----------------------------------------------------------------
# Functions for assigning vertices
#-----------------------------------------------------------------

struct QKHyperplaneClassifier
    regime_ids::Vector{Int}
    dirs::Vector{SparseVector{Float64,Int}}
    bias::Vector{Float64} 
    allow_pos::Vector{BitVector} # Should be caring about the growing 
    allow_neg::Vector{BitVector}
end

@inline _hyperplane_side(val::Real, tol::Real) = val >= tol ? Int8(1) : val < -tol ? Int8(-1) : Int8(0)


# Given a point decide the signature of which side of each hyperplane it is on

function _classifier_error(kind::Symbol, logqK, sig; candidate_ids=Int[])
    msg =
        kind === :nonunique ? "qK hyperplane classifier is not unique" :
        kind === :nocandidate ? "qK hyperplane classifier found no candidate regime" :
        error("unknown classifier error kind: $kind")

    error(
        msg * ": " *
        "logqK=$(repr(collect(logqK))), " *
        "signature=$(repr(collect(sig))), " *
        "candidate_ids=$(repr(Int.(candidate_ids)))"
    )
end



function _classifier_candidates(
    classifier::QKHyperplaneClassifier,
    logqK::AbstractVector{<:Real};
    tol::Real = 0,
    asymptotic_only::Bool=false,
)
    sides = let
        sides = Vector{Int8}(undef, length(C_qKs))
        
        if asymptotic_only
            @inbounds for i in eachindex(C_qKs)
                sides[i] = _hyperplane_side(dot(C_qKs[i], logqK), tol)
            end
        else 
            @inbounds for i in eachindex(C_qKs)
                sides[i] = _hyperplane_side(dot(C_qKs[i], logqK) + C0_qKs[i], tol)
            end
        end

        sides
    end

    alive = trues(length(classifier.regime_ids))

    @inbounds for hid in eachindex(sides)
        s = sides[hid]
        if s > 0
            alive .&= classifier.allow_pos[hid]
        elseif s < 0
            alive .&= classifier.allow_neg[hid]
        else
            continue
        end
        any(alive) || break
    end

    return classifier.regime_ids[findall(alive)],  sides
end


```
shrink a classifier to only the candidates.
```
function _restrict_classifier(
    classifier::QKHyperplaneClassifier,
    candidate_ids::AbstractVector{<:Integer},
)
    pos_map = Dict(classifier.regime_ids[i] => i for i in eachindex(classifier.regime_ids))
    selected_pos = [pos_map[Int(idx)] for idx in candidate_ids if haskey(pos_map, Int(idx))]
    isempty(selected_pos) && return nothing
    return QKHyperplaneClassifier(
        classifier.regime_ids[selected_pos],
        classifier.dirs,
        classifier.bias,
        [classifier.allow_pos[i][selected_pos] for i in eachindex(classifier.allow_pos)],
        [classifier.allow_neg[i][selected_pos] for i in eachindex(classifier.allow_neg)],
    )
end


```
Get the hyperplane id and sign info from RegimeGraph.
```
function _get_regime_qK_hyperplane_id_signs(grh::RegimeGraph, regime)
    Bnc = get_binding_network(grh)
    idx = get_idx(Bnc, regime)
    Hpid_dir = Dict{Int,Int8}()

    for edge in grh.neighbors[idx]
        _edge_has_qK_interface(edge) || continue
        hid = edge.qK_interface_idx
        dir = -edge.qK_interface_sign

        old = get(Hpid_dir, hid, dir)
        old == dir || error("Inconsistent qK hyperplane sign, BUGGY code: regime=$idx, hyperplane_id=$hid")
        Hpid_dir[hid] = dir
    end
    return Hpid_dir
end



function _build_qK_hyperplane_classifier(
    grh::RegimeGraph;
    candidates::Union{Nothing,AbstractVector}=nothing,
)
    model = get_binding_network(grh)

    regimes = if isnothing(candidates)
        get_indices(model; singular=false)
    else
        [get_idx(model, rgm) for rgm in filter_regimes(model, candidates; singular=false)]
    end

    n_regimes = length(regimes)
    regime_signs = Vector{Dict{Int,Int8}}(undef, n_regimes)
    active_hids = Set{Int}()

    for (pos, idx) in enumerate(regimes)
        info = _get_regime_qK_hyperplane_id_signs(grh, idx)
        regime_signs[pos] = info
        union!(active_hids, keys(info))
    end

    compact_hids = sort!(collect(active_hids))
    hid_to_pos = Dict(hid => pos for (pos, hid) in enumerate(compact_hids))

    pool = grh.qK_interface_pool
    dirs = [pool[hid].change_dir_qK for hid in compact_hids]
    bias = [pool[hid].intersect_qK for hid in compact_hids]

    n_hps = length(compact_hids)
    allow_pos = [trues(n_regimes) for _ in 1:n_hps]
    allow_neg = [trues(n_regimes) for _ in 1:n_hps]

    for (pos, signs) in enumerate(regime_signs)
        for (hid, sgn) in signs
            local_hid = hid_to_pos[hid]
            if sgn > 0
                allow_neg[local_hid][pos] = false
            elseif sgn < 0
                allow_pos[local_hid][pos] = false
            else
                error("zero qK hyperplane sign for regime=$(regimes[pos]) and hyperplane_id=$hid")
            end
        end
    end

    return QKHyperplaneClassifier(regimes, dirs, bias, allow_pos, allow_neg)
end


function _get_qK_hyperplane_classifier(Bnc::Bnc;)
    grh = get_regimes_graph!(Bnc; full=true)
    if isnothing(classifier)
        classifier = _build_qK_hyperplane_classifier(grh)
        grh.qK_classifier_full = classifier
    end
    return grh.qK_classifier_full
end




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
)
    logqK = input_logspace ? qK : log10.(qK)
    classifier = _get_qK_hyperplane_classifier(Bnc)

    candidate_ids, _ = _classifier_candidates(classifier, logqK; tol=abs(eps), asymptotic_only= asymptotic_only)

    if length(candidate_ids) == 1
        idx = Int(candidate_ids[1])
        return return_idx ? idx : get_perm(Bnc, idx)
    elseif isempty(candidate_ids) 
        _classifier_error(:nocandidate, logqK, sig)
    else
        _classifier_error(:nonunique, logqK, sig; candidate_ids=candidate_ids)
    end

    return _assign_regime_qK_fallback(
        Bnc,
        logqK;
        asymptotic_only=asymptotic_only,
        eps=eps,
        return_idx=return_idx,
        warn_on_fallback=true,
    )
end


#-------------------------------------------------------------------------------------------------------------------------------------------------------

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

"""
    assign_regime(args...; kwargs...) -> Vector

Alias for `assign_regime_qK`.
"""
assign_regime(args...;kwargs...)=assign_regime_qK(args...;kwargs...)

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
