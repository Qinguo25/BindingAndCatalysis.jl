export assign_regime, assign_regime_qK, assign_regime_x
export condition_contains, solve_logx_checked, assign_bnc_regime_wKk

#-----------------------------------------------------------------
# Functions for assigning vertices
#-----------------------------------------------------------------

"""
    CompiledClassifier

Hot-loop-friendly qK hyperplane classifier.

For each hyperplane h:
- `allow_pos[h]` is a BitVector of regimes still possible if a point is on the positive side.
- `allow_neg[h]` is a BitVector of regimes still possible if a point is on the negative side.

Boundary points keep both sides.
"""
struct CompiledClassifier
    regime_ids::Vector{Int}
    dirs::Vector{SparseVector{Float64,Int}}
    bias::Vector{Float64} 
    allow_pos::Vector{BitVector} # Should be caring about the growing 
    allow_neg::Vector{BitVector}
end

Base.length(c::CompiledClassifier) = length(c.regime_ids)

@inline _hyperplane_side(val::Real, tol::Real) = val >= tol ? Int8(1) : val < -tol ? Int8(-1) : Int8(0)


# Given a point decide the signature of which side of each hyperplane it is on





function _classifier_candidates(
    classifier::CompiledClassifier,
    logqK::AbstractVector{<:Real};
    tol::Real = 0,
    asymptotic_only::Bool=false,
)
    sides = let
        sides = Vector{Int8}(undef, length(classifier.dirs))
        
        if asymptotic_only
            @inbounds for i in eachindex(classifier.dirs)
                sides[i] = _hyperplane_side(dot(classifier.dirs[i], logqK), tol)
            end
        else 
            @inbounds for i in eachindex(classifier.dirs)
                sides[i] = _hyperplane_side(dot(classifier.dirs[i], logqK) + classifier.bias[i], tol)
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

# shrink a classifier to only the candidates.
function _restrict_classifier(
    classifier::CompiledClassifier,
    candidate_ids::AbstractVector{<:Integer},
)
    pos_map = Dict(classifier.regime_ids[i] => i for i in eachindex(classifier.regime_ids))
    selected_pos = [pos_map[Int(idx)] for idx in candidate_ids if haskey(pos_map, Int(idx))]
    isempty(selected_pos) && return nothing
    return CompiledClassifier(
        classifier.regime_ids[selected_pos],
        classifier.dirs,
        classifier.bias,
        [classifier.allow_pos[i][selected_pos] for i in eachindex(classifier.allow_pos)],
        [classifier.allow_neg[i][selected_pos] for i in eachindex(classifier.allow_neg)],
    )
end

function _allow_masks_from_incidence(M::SparseMatrixCSC{Int8, Int})
    n_regimes, n_hps = size(M)

    allow_pos = [trues(n_regimes) for _ in 1:n_hps]
    allow_neg = [trues(n_regimes) for _ in 1:n_hps]

    rows = rowvals(M)
    vals = nonzeros(M)

    @inbounds for h in 1:n_hps
        for k in nzrange(M, h)
            r = rows[k]
            sgn = vals[k]

            if sgn == Int8(1)
                allow_neg[h][r] = false
            elseif sgn == Int8(-1)
                allow_pos[h][r] = false
            else
                allow_neg[h][r] = false
                allow_pos[h][r] = false
            end
        end
    end

    return allow_pos, allow_neg
end

function _C_C0_from_pool(
    hyperplanes::AbstractVector{RegimeHyperplane},
    active_hids::Union{AbstractVector{<:Integer}, Nothing}=nothing;
    rebase_mat::Union{AbstractMatrix{<:Real}, Nothing}=nothing,
)
    active_hids = isnothing(active_hids) ? eachindex(hyperplanes) : active_hids
    n = length(active_hids)

    dirs = Vector{SparseVector{Float64, Int}}(undef, n)
    bias = Vector{Float64}(undef, n)

    if isnothing(rebase_mat)
        @inbounds for (j, hid0) in pairs(active_hids)
            hid = Int(hid0)
            dirs[j] = SparseVector{Float64,Int}(hyperplanes[hid].change_dir_qK)
            bias[j] = Float64(hyperplanes[hid].intersect_qK)
        end
    else
        Rt = transpose(Float64.(rebase_mat))
        @inbounds for (j, hid0) in pairs(active_hids)
            hid = Int(hid0)
            dirs[j] = sparse(Rt * hyperplanes[hid].change_dir_qK)
            bias[j] = Float64(hyperplanes[hid].intersect_qK)
        end
    end

    return dirs, bias
end

"""
    compile_classifier(hps, M, regime_ids=nothing; rebase_mat=nothing)

Compile qK hyperplanes and incidence rows into a classifier.

`M` is the polyhedron-regime by hyperplane incidence matrix. `regime_ids`
selects rows of `M`, using global regime indices.
"""
function compile_classifier(
    hps::AbstractVector{<:RegimeHyperplane},
    M::SparseMatrixCSC{Int8, Int},
    regime_ids::Union{AbstractVector{<:Integer}, Nothing}=nothing;
    rebase_mat::Union{Nothing, AbstractMatrix{<:Real}}=nothing,
)
    rows = if isnothing(regime_ids)
        collect(1:size(M, 1))
    else
        collect(Int.(regime_ids))
    end

    if isempty(rows)
        return CompiledClassifier(
            Int[],
            SparseVector{Float64, Int}[],
            Float64[],
            BitVector[],
            BitVector[],
        )
    end

    Msub = M[rows, :]
    _, active_hids_raw, _ = findnz(Msub)
    active_hids = sort!(unique(active_hids_raw))
    Mactive = Msub[:, active_hids]

    dirs, bias = _C_C0_from_pool(hps, active_hids; rebase_mat=rebase_mat)
    allow_pos, allow_neg = _allow_masks_from_incidence(Mactive)

    return CompiledClassifier(rows, dirs, bias, allow_pos, allow_neg)
end

# Get the hyperplane id and sign info from RegimeGraph.
function _get_regime_qK_hyperplane_id_signs(grh::RegimeGraph, regime)
    Bnc = get_binding_network(grh)
    idx = get_idx(Bnc, regime)
    Hpid_dir = Dict{Int,Int8}()

    for edge in grh.neighbors[idx]
        _edge_has_qK_interface(grh, edge) || continue
        hid, sign = _edge_idx_sign(edge, grh, :qK)
        dir = -sign

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

    qK_hp_data = grh.hp_data[_space(grh, :qK)]
    return compile_classifier(
        qK_hp_data.hyperplanes,
        qK_hp_data.hp_to_poly.M,
        regimes,
    )
end


function _get_qK_hyperplane_classifier(Bnc::Bnc)
    grh = get_regimes_graph!(Bnc; full=true)
    classifier = grh.qK_classifier_full
    if isnothing(classifier)
        classifier = _build_qK_hyperplane_classifier(grh)
        grh.qK_classifier_full = classifier
    end
    return classifier
end




"""
    assign_regime_qK(bnc::Bnc; x, input_logspace=false, kwargs...) -> Vector

Assign a regime given a point in x space by first mapping to qK.
"""
function assign_regime_qK(Bnc::Bnc ; x::AbstractVector{<:Real}, input_logspace::Bool=false, kwargs...) 
    logqK = x2qK(Bnc,x; input_logspace=input_logspace, output_logspace=true)
    return assign_regime_qK(Bnc, logqK; input_logspace=true, kwargs...)
end
"""
    assign_regime_qK(bnc::Bnc, qK; input_logspace=false, asymptotic_only=false, eps=0, return_idx=false)

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

    candidate_ids, sig = _classifier_candidates(
        classifier,
        logqK;
        tol=abs(eps),
        asymptotic_only=asymptotic_only,
    )

    if length(candidate_ids) == 1
        idx = Int(candidate_ids[1])
        return return_idx ? idx : get_perm(Bnc, idx)
    elseif isempty(candidate_ids) 
        msg = "qK hyperplane classifier found no candidate regime"
        # @error(msg * ": logqK=$(repr(collect(logqK))), signature=$(repr(collect(sig)))")
        return _assign_regime_qK_fallback(
            Bnc,
            logqK;
            asymptotic_only=asymptotic_only,
            eps=eps,
            return_idx=return_idx,
            warn_on_fallback=false,
        )
    else
        msg = "qK hyperplane classifier is not unique"
        error(msg * ": logqK=$(repr(collect(logqK))), signature=$(repr(collect(sig))), candidate_ids=$(repr(Int.(candidate_ids)))")
        return nothing
    end
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

function condition_contains(C, C0, nullity::Integer, z::AbstractVector{<:Real}; tol::Float64=1e-8)
    vals = Vector{Float64}(C * z .+ C0)
    nullity > 0 && any(abs.(vals[1:nullity]) .> tol) && return false
    length(vals) > nullity && any(vals[nullity + 1:end] .< -tol) && return false
    return true
end

function solve_logx_checked(
    model::Bnc,
    logqK::AbstractVector{<:Real};
    method::Union{Symbol,Nothing}=nothing,
    tol::Float64=1e-6,
)
    method = _resolve_qK2x_method(model, method)
    logx = try
        if method === :free_energy
            qK2x(
                model,
                logqK;
                input_logspace=true,
                output_logspace=true,
                method=method,
                warn_on_maxiters=false,
            )
        else
            qK2x(model, logqK; input_logspace=true, output_logspace=true, method=method)
        end
    catch
        return nothing
    end

    method === :regime && return logx
    maximum(abs.(qK2x_residual(model, logx, logqK; input_logspace=true))) <= tol || return nothing
    return logx
end

function assign_bnc_regime_wKk(model::Bnc, logwKk::AbstractVector{<:Real}; tol::Float64=1e-8, max_nullity::Integer=0)
    rgms = get_bnc_regimes(model)
    for (idx, rgm) in pairs(rgms)
        C, C0, nlt = get_C_C0_nullity_wKk(rgm)
        nlt <= max_nullity || continue
        condition_contains(C, C0, nlt, logwKk; tol=tol) && return idx
    end
    return 0
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
    all_regime_idx = get_regimes(Bnc, singular=false, asymptotic = real_only, return_idx = true)

    record = Vector{Float64}(undef,length(all_regime_idx))
    for (i, idx) in enumerate(all_regime_idx)
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
    warn_on_fallback && @warn("All regime conditions failed for logqK=$logqK. Returning the best-fit regime.")
    idx = all_regime_idx[findmax(record)[2]]
    return return_idx ? idx : get_perm(Bnc, idx)
end
