export find_all_regimes!,
    get_binding_regimes_dict, get_bind_regimes_dict, get_nullities, get_volumes, have_perm
export get_regimes, get_perms, get_indices, get_regimes_neighbor_mat
export is_singular, is_asymptotic, n_regimes
export get_idx, get_perm, get_regime, get_neighbors, get_nullity
export get_P_P0, get_P, get_P0
export get_M_M0, get_M, get_M0
export get_H_H0, get_H, get_H0
export get_affine_x2q, get_affine_x2qK, get_affine_x2qcat, get_affine_x2w, get_affine_x2K
export get_affine_qK2x, get_affine_xk2qKk, get_affine_qKk2xk
export get_C_C0_x, get_C_x, get_C0_x
export get_C_C0_nullity_qK, get_C_C0_qK, get_C_qK, get_C0_qK
export get_C_C0_nullity, get_C_C0, get_C, get_C0
export check_feasibility_with_constraint, feasible_regimes_with_constraint
export feasible_vertices_with_constraint, feasible_vertieces_with_constraint
export get_polyhedron, get_volume, get_polyhedra
export is_neighbor, get_interface, get_change_dir
export get_function

#========================================================================================#
#--------------Core computation functions-------------------------
#========================================================================================#

"""
    _calc_C_C0_qK_singular(bnc::Bnc, regime) -> (SparseMatrixCSC, Vector, Int)

Build qK-space constraints `(C_qK, C0_qK, nullity)` for singular regimes via affine mapping.
"""

function _calc_C_C0_qK_singular(Bnc::Bnc, vtx)
    M, M0 = get_M_M0(Bnc, vtx)
    C, C0 = get_C_C0_x(Bnc, vtx)
    C, C0, nlt = _affine_mapping_polyhedra(C, C0, 0, M, M0)
    # C = sparse(C)
    # droptol!(C, 1e-10)
    return C, C0, nlt
end

function _affine_mapping_polyhedra(C, C0, nullity, M, M0)
    poly_x = _build_polyhedron_from_C_C0(C, C0, nullity)

    if isempty(poly_x)
        return _canonical_empty_condition(C, C0; ncols=size(M, 1))
    end

    poly_elim = M * poly_x  # If for convenience, one can write `translate(M * poly_x, M0)`, and then C0qK = b
    if isempty(poly_elim)
        return _canonical_empty_condition(C, C0; ncols=size(M, 1))
    end
    rlt = MixedMatHRep(hrep(poly_elim))

    A, b, linset = (rlt.A, rlt.b, rlt.linset)
    # @show linset
    @assert isempty(linset) || linset == BitSet(1:maximum(linset)) "linear rows are not the first top n rows, code fix is needed"
    # perm = [collect(linset) ; [i for i in 1:size(A,1) if i ∉ linset]]
    CqK = (x -> droptol!(x, 1e-10))(sparse(-A))
    C0qK = (b + A * M0)
    return CqK, C0qK, length(linset)
end

#------------------Helper functions -------------------------------------------
"""
    _regime_graph_to_sparse(g::RegimeGraph; weight_fn=e->1) -> SparseMatrixCSC

Convert a `RegimeGraph` to a sparse adjacency matrix.
"""
function _regime_graph_to_sparse(G::RegimeGraph; weight_fn=e -> 1)
    n = length(G.neighbors)
    sample_edge = nothing
    for edges in G.neighbors
        if !isempty(edges)
            sample_edge = first(edges)
            break
        end
    end
    isnothing(sample_edge) && return spzeros(Int, n, n)
    Ty = typeof(weight_fn(sample_edge))
    # 预分配估计：平均度 × n
    nnz = sum(length(v) for v in G.neighbors)
    I = Vector{Int}(undef, nnz)
    J = Vector{Int}(undef, nnz)
    V = Vector{Ty}(undef, nnz)
    idx = 0
    for i in 1:n
        for e in G.neighbors[i] #Edge
            idx += 1
            I[idx] = i
            J[idx] = e.to
            V[idx] = weight_fn(e)
        end
    end
    return dropzeros!(sparse(I, J, V, n, n))
end

#------------------------------------------------------------------------------
#             1. Functions find all regimes and return properties
# ------------------------------------------------------------------------------

find_bind_regimes!(model::Bnc{T}) where {T} = find_all_regimes!(model)
"""
    find_all_regimes!(bnc::Bnc) -> Vector{Vector{Int}}

Compute and cache all regime permutations, the x-neighbor graph, and regime
objects. Low-nullity (`0/1`) affine data are inferred directly from the graph;
only deferred high-nullity perms are sent to `_calc_nullity`.

Affine propagation for `H`/`H0` and regular-regime qK conditions is always
computed in exact arithmetic. Polyhedral projection and returned polyhedron
objects are materialized through the floating-point `CDDLib` backend.
"""
function find_all_regimes!(model::Bnc{T}) where {T}
    return _with_regime_cache_lock(model) do
        is_bind_regimes_built(model) && return nothing
        _remove_regime_data!(model)
        @info "---------------------Start finding all regimes--------------------"

        (all_perms, is_asymptotic) = let
            perms, is_asymp = _enumerate_all_regimes(model._L_helper)
            perms = [Vector{T}(v) for v in perms]
            (perms, is_asymp)
        end

        n_regimes = length(all_perms)
        n_asym_rgms = sum(is_asymptotic)
        @info "Finished, with $(n_regimes) regimes found and $(n_asym_rgms) asymptotic regimes."

        @info "2.Building x-neighbor regime graph..."
        model.vertices_graph = let
            grh = _calc_regimes_graph(model._L_helper, all_perms)
            grh.bn = model
            grh
        end

        @info "3.Building regime objects..."
        model.BindRegimes = let
            regimes = _build_bind_regimes(
                model, all_perms, is_asymptotic, fill(T(-1), n_regimes)
            )
            regimes_perm_dict = Dict(perm => idx for (idx, perm) in enumerate(all_perms))
            Regimes(regimes_perm_dict, regimes)
        end

        @info "4.Propagating affine data and deferred nullity labels..."
        _prefill_affine_cache!(model; ensure_built=false)

        @info "5.Calculating qK change directions on the regime graph..."
        _fulfill_regimes_graph!(model.vertices_graph)

        @info "Finished."
        return nothing
    end
end

@inline function _build_bind_regimes(
    model::Bnc{T}, all_perms, is_asymptotic, nullity
) where {T}
    n_regimes = length(all_perms)
    regimes = Vector{BindRegime}(undef, n_regimes)
    Threads.@threads for i in 1:n_regimes
        regimes[i] = BindRegime(;
            network=model,
            perm=all_perms[i],
            idx=i,
            is_asymptotic=is_asymptotic[i],
            nullity=nullity[i],
        )
    end
    return regimes
end

"""
    _initialize_regime!(vtx::BindRegime) -> BindRegime

Fill the basic linear-algebra fields of a lazily-created `BindRegime`.
"""
function _initialize_regime!(vtx::BindRegime)::BindRegime
    if !isnothing(vtx.P)
        return vtx
    end
    Bnc = vtx.network
    helper = Bnc._L_helper
    perm = vtx.perm

    P, P0 = _calc_P_P0(perm, helper)
    C_x, C0_x = _calc_C_C0(perm, helper)
    M = vcat(P, Bnc.N)
    M0 = vcat(P0, zeros(eltype(P0), Bnc.r))

    vtx.P0 = P0
    vtx.C_x = C_x
    vtx.C0_x = C0_x
    vtx.M0 = M0
    vtx.M = M
    vtx.P = P
    return vtx
end

function _materialize_qK_conditions!(rgm::BindRegime)
    _initialize_regime!(rgm)
    (!isnothing(rgm.C_qK) && !isnothing(rgm.C0_qK)) && return nothing

    if rgm.nullity == 0
        C_qK = rgm.C_x * rgm.H

        dropzeros!(C_qK)
        rgm.C_qK = C_qK
        rgm.C0_qK = rgm.C0_x + rgm.C_x * rgm.H0

    else
        rgm.C_qK, rgm.C0_qK, nlt = _calc_C_C0_qK_singular(rgm.network, rgm.perm)
        @assert nlt == rgm.nullity "Calculated nullity does not match regime's nullity, code fix is needed"
    end

    return nothing
end

"""
    _fill_all_info!(vtx::BindRegime) -> nothing

Ensure a `BindRegime` has `H/H0` and qK constraints computed and cached.
"""
function _fill_all_info!(vtx::BindRegime)
    _initialize_regime!(vtx)
    _materialize_qK_conditions!(vtx)
    return nothing
end

function _prefill_qK_conditions!(model::Bnc, regime_idxs::AbstractVector{<:Integer};)
    find_all_regimes!(model)
    regimes = _bind_regimes_data(model)

    @showprogress desc = "Prefilling qK conditions" Threads.@threads for pos in eachindex(
        regime_idxs
    )
        _materialize_qK_conditions!(regimes[regime_idxs[pos]])
    end

    return nothing
end

#===============================================================================================================#
# Binding Regime related APIs.
#===============================================================================================================#

#---------------------------------------------------------------------------------------------
#   Functions involving regime relationships, (neighbors finding and changedir finding)
#---------------------------------------------------------------------------------------------
"""
    get_regimes_neighbor_mat_x(bnc::Bnc) -> SparseMatrixCSC

Return the x-space adjacency matrix of the regime graph.
"""
function get_regimes_neighbor_mat_x(Bnc::Bnc)
    grh = get_regimes_graph!(Bnc; full=false)
    spmat = _regime_graph_to_sparse(grh; weight_fn=e -> 1)
    return spmat
end

"""
    get_regimes_neighbor_mat_qK(bnc::Bnc) -> SparseMatrixCSC

Return the qK-space adjacency matrix of the regime graph.
"""
function get_regimes_neighbor_mat_qK(Bnc::Bnc)
    grh = get_regimes_graph!(Bnc; full=true)
    f(x::RegimeEdge) = _edge_has_qK_interface(grh, x) ? 1 : 0
    spmat = _regime_graph_to_sparse(grh; weight_fn=f)
    return spmat
end

function get_regimes_neighbor_mat(args...; kwargs...)
    return get_regimes_neighbor_mat_qK(args...; kwargs...)
end

"""
    get_volumes(bnc::Bnc, regimes=nothing; recompute=false, kwargs...) -> Vector{Volume}

Return volumes for selected regimes, computing missing volumes as needed.
"""
function get_volumes(
    Bnc::Bnc,
    regimes::Union{AbstractVector, Nothing}=nothing;
    recompute::Bool=false,
    rebase_K::Bool=false,
    rebase_mat::Union{AbstractMatrix{<:Real}, Nothing}=nothing,
    kwargs...,
)
    _reject_renamed_keywords(kwargs)
    all_rgms = if isnothing(regimes)
        get_binding_indices(Bnc)
    else
        [get_idx(Bnc, rgm) for rgm in regimes]
    end

    regimes_to_calc = if recompute
        all_rgms
    else
        rgm_data = _bind_regimes_data(Bnc)
        filter(i -> isnothing(rgm_data[i].volume), all_rgms)
    end

    if !isempty(regimes_to_calc)
        rebase_mat = if !isnothing(rebase_mat)
            @assert !rebase_K "Cannot specify both rebase_K and providing rebase_mat"
            rebase_mat
        elseif rebase_K
            Q = rebase_mat_lgK(Bnc.N)
            blockdiag(spdiagm(fill(Rational(1), Bnc.d)), Q)
        else
            nothing
        end

        # #ensure conditions for volume calculation are calced, may further replaced by other functions
        # Threads.@threads for idx in vtxs_to_calc
        #    get_binding_regime(Bnc,idx; inv_info=true)
        # end

        rlts = _calc_bind_regime_volumes(
            Bnc, regimes_to_calc; rebase_mat=rebase_mat, kwargs...
        )
        for (i, idx) in enumerate(regimes_to_calc)
            rgm = get_binding_regime(Bnc, idx; inv_info=false)
            rgm.volume = rlts[i]
        end
    end
    return [rgm.volume for rgm in _bind_regimes_data(Bnc)[all_rgms]]
end

#---------------------------------These properties are calculate when creating BindRegime object---------------------------------------

"""
    get_volume(args...; kwargs...) -> Volume

Return the volume for a single regime.
"""
function get_volume(args...; kwargs...)
    model = get_binding_network(args...)
    idx = get_idx(args...)
    return get_volumes(model, [idx]; kwargs...)[1]
end

#--------------------------------------------------------------------------------------------------------------------------------------
#          relationships between two regimes, based on regime graphs.
#----------------------------------------------------------------------------------------------------------------------------------------

"""
    get_interface_qK(bnc, from, to) -> (SparseVector, Float64)

Return the interface hyperplane between two regimes in qK space.
"""
function get_interface_qK(Bnc, from, to)::Tuple{SparseVector{Float64, Int}, Float64}
    grh = get_regimes_graph!(Bnc; full=true)
    edge = get_edge(grh, from, to; full=true)
    if edge === nothing
        @info "No direct regime-graph edge found; falling back to direct interface reconstruction."
        return get_interface_direct(Bnc, from, to)
    elseif !_edge_has_qK_interface(grh, edge)
        @error(
            "Regimes $get_perm(Bnc, from) and $get_perm(Bnc, to) are neighbors in x space but not in qK space"
        )
    else
        return _edge_qK_interface(grh, edge)
    end
end

"""
    get_interface(args...; kwargs...) -> (SparseVector, Float64)

Convenience wrapper for `get_interface_qK`.
"""
get_interface(args...; kwargs...) = get_interface_qK(args...; kwargs...)
"""
    get_change_dir_qK(args...; kwargs...) -> SparseVector

Return the qK change direction between neighboring regimes.
"""
get_change_dir_qK(args...; kwargs...) = get_interface(args...; kwargs...)[1] # relys on the inner behavior of get_interface,
"""
    get_change_dir(args...; kwargs...) -> SparseVector

Alias for `get_change_dir_qK`.
"""
get_change_dir(args...; kwargs...) = get_change_dir_qK(args...; kwargs...)

"""
    is_neighbor_qK(bnc, regime1, regime2) -> Bool

Return `true` if two regimes are neighbors in qK space.
"""
function is_neighbor_qK(Bnc, vtx1, vtx2)::Bool
    try
        get_interface_qK(Bnc, vtx1, vtx2)
        return true
    catch
        return false
    end
end

"""
    is_neighbor(args...; kwargs...) -> Bool

Alias for `is_neighbor_qK`.
"""
is_neighbor(args...; kwargs...) = is_neighbor_qK(args...; kwargs...)

"""
    get_interface_x(bnc::Bnc, from, to) -> (SparseVector, Float64)

Return the interface hyperplane between two regimes in x space.
"""
function get_interface_x(Bnc::Bnc, from, to)
    edge = get_edge(Bnc, from, to)
    if edge === nothing
        @error(
            "Regimes $get_perm(Bnc, from) and $get_perm(Bnc, to) are not neighbors in x space."
        )
    else
        grh = get_regimes_graph!(Bnc; full=false)
        x_space = _space(grh, :x)
        x_idx, x_sign = _edge_idx_sign(edge, x_space)
        hp = get_hyperplane(grh.hp_data[x_space], x_idx)
        c, c0 = _calc_c_c0(hp, Bnc.n, x_sign)
        return c[:, 1], c0
    end
end

"""
    get_change_dir_x(args...; kwargs...) -> SparseVector

Return the x-space change direction between neighboring regimes.
"""
get_change_dir_x(args...; kwargs...) = get_interface_x(args...; kwargs...)[1]

"""
    get_neighbors(args...; singular=nothing, asymptotic=nothing, return_idx=false) -> Vector

Return neighbors of a regime filtered by singularity and asymptotic flags.

# Keyword Arguments

  - `singular`: `true`, `false`, integer threshold, or `nothing`.
  - `asymptotic`: `true`, `false`, or `nothing`.
  - `return_idx`: Return indices when `true`; otherwise return permutations.
"""
function get_neighbors(args...; return_idx::Bool=false, kwargs...)
    model = get_binding_network(args...)
    grh = get_regimes_graph!(model; full=true)
    rgm_idx = get_idx(args...)

    idx = collect(keys(grh.edge_pos[rgm_idx]))

    regime_data = _bind_regimes_data(model)
    rgms = regime_data[idx]
    idx = get_indices(rgms; kwargs...)
    sort!(idx)
    return return_idx ? idx : getfield.(regime_data[idx], :perm)
end

#-------------------------------------------------------------
#Other higher lever functions
#----------------------------------------------------------------
"""
    summary_regime(args...; compute_volume=false) -> nothing

Print a detailed summary for a single regime.
"""
function summary_regime(args...; compute_volume::Bool=false, volume_kwargs...)
    idx = get_idx(args...)
    perm = get_perm(args...)
    is_real = is_asymptotic(args...)
    nullity = get_nullity(args...)
    println("idx=$idx,perm=$perm, asymptotic=$is_real, nullity=$nullity")
    if compute_volume
        volume = get_volume(args...; volume_kwargs...)
        println("volume=$(volume.mean) +- $(sqrt(volume.var))")
    else
        if !isempty(volume_kwargs)
            @warn "summary_regime received volume keyword arguments but compute_volume=false; ignoring them."
        end
        println("volume=<not computed>; pass compute_volume=true to calculate")
    end
    println("Dominant Relation")
    display.(show_dominant_condition(args...; log_space=false))
    println("Expression")
    try
        display.(show_expression_x(args...; log_space=false))
    catch
    end
    println("Condition:")
    display.(show_condition_qK(args...; log_space=false))

    return nothing
end

"""
    summary(bnc::Bnc, perm) -> nothing

Alias for `summary_regime`.
"""
summary(Bnc::Bnc, perm; kwargs...) = summary_regime(Bnc, perm; kwargs...)
"""
    summary(vtx::BindRegime) -> nothing

Alias for `summary_regime`.
"""
summary(vtx::BindRegime; kwargs...) = summary_regime(vtx; kwargs...)

@inline function _regime_display_dominant_mode(rgm::BindRegime)
    return "perm=$(get_perm(rgm))"
end

function Base.show(io::IO, rgm::BindRegime)
    return print(
        io,
        "BindRegime(",
        _regime_display_dominant_mode(rgm),
        ", nullity=",
        get_nullity(rgm),
        ", asymptotic=",
        is_asymptotic(rgm),
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", rgm::BindRegime)
    println(io, "BindRegime")
    println(io, "  dominant mode: ", _regime_display_dominant_mode(rgm))
    println(io, "  nullity: ", get_nullity(rgm))
    return print(io, "  asymptotic: ", is_asymptotic(rgm))
end

function get_function(vtx::BindRegime)
    H, H0 = get_H_H0(vtx)
    f = function (
        qK::AbstractArray{<:Real};
        input::Symbol=:linear,
        output::Symbol=:linear,
        input_logspace::Union{Bool, Nothing}=nothing,
        output_logspace::Union{Bool, Nothing}=nothing,
    )
        input = _resolve_space_mode(input, input_logspace, :input_logspace)
        output = _resolve_space_mode(output, output_logspace, :output_logspace)
        lgqK = input === :log ? qK : log10.(qK)
        lgx = H * lgqK .+ H0
        return output === :log ? lgx : exp10.(lgx)
    end
    return f
end

is_stable(rgm::BindRegime) = true
stability_code(rgm::BindRegime) = 1
get_affine_x2K(model::Bnc) = (model.N, zeros(eltype(model.N), model.r)) # will the intersection fine ?

#===============================================================================================================#
# Polyhedron-related helper functions
#===============================================================================================================#

"""
    get_C_C0_nullity(poly::Polyhedron) -> (Matrix, Vector, Int)

Extract `(C, C0, nullity)` from a polyhedron in H-representation.
"""
function get_C_C0_nullity(poly::Polyhedron) #Have to make sure the polyhedron has been already detecthlinearity.
    return _polyhedron_to_C_C0_nullity(poly)
end
"""
    get_nullity(poly::Polyhedron, args...; kwargs...) -> Int

Return the nullity encoded in a polyhedron's linear constraints.
"""
get_nullity(poly::Polyhedron, args...; kwargs...) = get_C_C0_nullity(poly)[3]

"""
    get_polyhedron(C, C0, nullity=0) -> Polyhedron

Construct a polyhedron from inequality constraints in qK space.
"""
function get_polyhedron(
    C::AbstractMatrix{<:Real},
    C0::AbstractVector,
    nullity::Integer=0;
    canonicalize::Bool=true,
)::Polyhedron
    return _build_polyhedron_from_C_C0(C, C0, nullity; canonicalize=canonicalize)
end

function get_polyhedron(rgm::BindRegime; chart::Symbol=:qK, canonicalize::Bool=true)
    C, C0, nullity = _regime_C_C0_nullity(rgm, chart)
    return _build_polyhedron_from_C_C0(C, C0, nullity; canonicalize=canonicalize)
end

"""
    get_polyhedron(args...; kwargs...) -> Polyhedron

Convenience wrapper that pulls constraints from a regime or model.
"""
get_polyhedron(args...; kwargs...) =
    get_polyhedron(get_C_C0_nullity_qK(args...)...; kwargs...)

function get_polyhedra(
    model::Bnc,
    vtxs::Union{AbstractVector{T}, Nothing}=nothing;
    canonicalize::Bool=false,
    kwargs...,
) where {T}
    selected = isnothing(vtxs) ? get_regimes(model; kwargs...) : vtxs
    isempty(selected) && return Polyhedron[]

    selected_idxs = [get_idx(model, vtx) for vtx in selected]

    _prefill_qK_conditions!(model, selected_idxs)

    regimes = _bind_regimes_data(model)

    out = Vector{Polyhedron}(undef, length(selected))

    Threads.@threads for i in eachindex(selected)
        rgm = regimes[selected_idxs[i]]
        out[i] = get_polyhedron(rgm.C_qK, rgm.C0_qK, rgm.nullity; canonicalize=canonicalize)
    end

    return out
end
"""
    get_intersect(bnc, vtx1, vtx2) -> Polyhedron

Return the intersection polyhedron between two regimes in qK space.
"""
function get_intersect(Bnc, vtx1, vtx2)::Polyhedron
    p1 = get_polyhedron(Bnc, vtx1)
    dim1 = dim(p1)
    p2 = get_polyhedron(Bnc, vtx2)
    dim2 = dim(p2)
    status = _poly_intersection_status(p1, p2; canonicalize=false, detect_linearities=true)
    # @show dim1, dim2, dim(p)
    if status.dim < max(dim1, dim2) - 1
        error(
            "Regimes $(get_perm(Bnc, vtx1)) and $(get_perm(Bnc, vtx2)) do not have dim-1 intersect.",
        )
    end
    return status.poly
end

"""
    get_interface_direct(bnc::Bnc, from, to) -> (SparseVector, Float64)

Compute the interface hyperplane directly from polyhedral intersection.
"""
function get_interface_direct(
    Bnc::Bnc, from, to
)::Tuple{SparseVector{Float64, Int}, Float64}
    p = get_intersect(Bnc, from, to)
    interface = _poly_interface_from_intersection(p)
    isnothing(interface) && return spzeros(Float64, fulldim(p)), 0.0
    return interface
end

"""
    get_one_inner_point(args...; kwargs...) -> Vector

Convenience wrapper that builds a polyhedron from a regime/model.
"""
get_one_inner_point(args...; kwargs...) =
    get_one_inner_point(get_polyhedron(args...); kwargs...)

"""
    check_feasibility_with_constraint(args...; C, C0, nullity=0) -> Bool

Check whether a regime/polyhedron remains feasible under extra constraints.
"""
function check_feasibility_with_constraint(
    args...; C::AbstractMatrix{<:Real}, C0::AbstractVector{<:Real}, nullity::Int=0
)
    poly_additional = get_polyhedron(C, C0, nullity)
    poly = get_polyhedron(args...)
    return _poly_intersection_status(poly, poly_additional).feasible
end

"""
    feasible_regimes_with_constraint(bnc::Bnc; C, C0, nullity=0, kwargs...) -> Vector

Return regimes feasible under additional constraints.
"""
function feasible_regimes_with_constraint(
    Bnc::Bnc;
    C::AbstractMatrix{<:Real},
    C0::AbstractVector{<:Real},
    nullity::Int=0,
    kwargs...,
)
    all_rgms = get_regimes(Bnc; kwargs...)
    feasible_rgms = Vector{eltype(all_rgms)}()
    for perm in all_rgms
        if check_feasibility_with_constraint(Bnc, perm; C=C, C0=C0, nullity=nullity)
            push!(feasible_rgms, perm)
        end
    end
    return feasible_rgms
end

function feasible_vertices_with_constraint(args...; kwargs...)
    Base.depwarn(
        "`feasible_vertices_with_constraint` is deprecated; use `feasible_regimes_with_constraint`.",
        :feasible_vertices_with_constraint,
    )
    return feasible_regimes_with_constraint(args...; kwargs...)
end

function feasible_vertieces_with_constraint(args...; kwargs...)
    Base.depwarn(
        "`feasible_vertieces_with_constraint` is deprecated; use `feasible_regimes_with_constraint`.",
        :feasible_vertieces_with_constraint,
    )
    return feasible_regimes_with_constraint(args...; kwargs...)
end

#==========================================================================#
