#--------------Core computation functions-------------------------

"""
    _calc_C_C0_qK_singular(bnc::Bnc, vtx) -> (SparseMatrixCSC, Vector)

Build qK-space constraints `(C_qK, C0_qK)` for singular vertices via affine mapping.
"""
function _calc_C_C0_qK_singular(Bnc::Bnc, vtx)
    M,M0 = get_M_M0(Bnc,vtx)
    C,C0 = get_C_C0_x(Bnc,vtx)
    _calc_C_C0_qK_singular(C,C0,M,M0)
end

function _calc_C_C0_qK_singular(C,C0,M,M0)
    # n = Bnc.n
    poly_x = hrep(-C,C0) |> x->polyhedron(x,CDDLib.Library())
    poly_elim = M * poly_x  # If for convenience, one can write `translate(M * poly_x, M0)`, and then C0qK = b
    rlt = MixedMatHRep(hrep(poly_elim))
    A, b, linset = (rlt.A, rlt.b, rlt.linset)
    # @show linset
    @assert linset == BitSet(1:maximum(linset)) "linear rows are not the first top n rows, code fix is needed"
    # perm = [collect(linset) ; [i for i in 1:size(A,1) if i ∉ linset]]
    CqK = sparse(-A) |> x->droptol!(x,1e-10)
    C0qK = (b+A*M0)
    return CqK, C0qK
end

#------------------Helper functions -------------------------------------------
"""
    _regime_graph_to_sparse(g::VertexGraph; weight_fn=e->1) -> SparseMatrixCSC

Convert a `VertexGraph` to a sparse adjacency matrix.
"""
function _regime_graph_to_sparse(G::VertexGraph{T}; weight_fn = e -> 1) where T
    n = length(G.neighbors)
    Ty = eltype(weight_fn(first(G.neighbors[1]))) # infer the type of weights from the first edge
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
    return sparse(I,J,V, n, n) |> dropzeros!
end


# function _fill_neighbor_info!(vtx::BindRegime)
#     """
#     Fill the neighbor info for a given vertex.
#     """
#     Bnc = vtx.network
#     if isempty(vtx.neighbors_idx)
#         vtx_grh = get_regimes_graph!(Bnc;full=false)
#         vtx.neighbors_idx = vtx_grh.neighbors[vtx.idx] .|> e -> e.to
#     end
#     return nothing
# end


@inline is_bind_regimes_built(model::Bnc) = !isnothing(model.BindRegimes)

#------------------------------------------------------------------------------
#             1. Functions find all regimes and return properties
# ------------------------------------------------------------------------------

"""
    find_all_regimes!(bnc::Bnc) -> Vector{Vector{Int}}

Compute and cache all vertex permutations, asymptotic flags, Nρ inverse cache,
and vertex nullities.
"""
function find_all_regimes!(model::Bnc{T};) where T
    if is_bind_regimes_built(model)
        return nothing
    end

    @info "---------------------Start finding all vertices--------------------"
    all_vertices, is_asymptotic =  _enumerate_all_regimes(model._L_helper)
    all_vertices = [Vector{T}(v) for v in all_vertices]

    n_vertices = length(all_vertices)
    n_asym_rgms = sum(is_asymptotic)
    @info "Finished, with $(n_vertices) vertices found and $(n_asym_rgms) asymptotic vertices."
    
    @info "2.Calculating nullity for each vertex..."
    nullity = _calc_nullity(all_vertices, model)
    
    @info "3.Building Regimes..."
    model.BindRegimes = let
        regimes = _build_bind_regimes(model, all_vertices, is_asymptotic, nullity)    
        vertices_perm_dict = Dict(perm => idx for (idx, perm) in enumerate(all_vertices))
        Regimes(vertices_perm_dict, regimes)
    end
    @info "Finished."
    return nothing
end


@inline function _calc_perm_nullity(perm, n::Integer)
    perm_nullity = 0
    seen = falses(n)
    @inbounds for p in perm
        if seen[p]
            perm_nullity += 1
        else
            seen[p] = true
        end
    end
    return perm_nullity
end



# @inline function _calc_nullity(perms, model::Bnc{T}) where T
#     _build_Nρ_cache_parallel!(model, perms) # build Nρ_inv cache in parallel
#     nullity = Vector{T}(undef, length(perms))
    
#     Threads.@threads for i in  eachindex(perms)
#         perm = perms[i]
#         nullity_P = _calc_perm_nullity(perm, model.n)
#         _, nullity_N = _get_Nρ_inv_from_perm!(model, perm)
#         nullity[i] = nullity_P + nullity_N # this is true as we can permute the matrix into diagnal block matrix.
#     end
#     return nullity
# end


@inline function _build_bind_regimes(model::Bnc{T}, all_vertices, is_asymptotic, nullity) where T
    n_vertices = length(all_vertices)
    regimes = Vector{BindRegime}(undef, n_vertices)
    for i in 1:n_vertices
        regimes[i] = BindRegime(
            network = model,
            perm = all_vertices[i],
            idx = i,
            is_asymptotic = is_asymptotic[i],
            nullity = nullity[i]
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

    vtx.P = P
    vtx.P0 = P0
    vtx.C_x = C_x
    vtx.C0_x = C0_x
    vtx.M = vcat(P, Bnc.N)
    vtx.M0 = vcat(P0, zeros(eltype(P0), Bnc.r))
    return vtx
end

@inline _affine_info_ready(rgm::BindRegime) = !isnothing(rgm.H) && !isnothing(rgm.H0)

function _masked_components(grh::SimpleGraph, mask::BitVector)
    n = nv(grh)
    seen = falses(n)
    comps = Vector{Vector{Int}}()

    for src in 1:n
        (!mask[src] || seen[src]) && continue
        comp = Int[]
        stack = [src]
        seen[src] = true
        while !isempty(stack)
            v = pop!(stack)
            push!(comp, v)
            for nb in neighbors(grh, v)
                if mask[nb] && !seen[nb]
                    seen[nb] = true
                    push!(stack, nb)
                end
            end
        end
        push!(comps, comp)
    end

    return comps
end

function _edge_rank1_data(model::Bnc, from_idx::Int, to_idx::Int, edge::VertexEdge)
    i = edge.diff_r
    perm_from = get_perm(model, from_idx)
    perm_to = get_perm(model, to_idx)
    j_from = perm_from[i]
    j_to = perm_to[i]
    δ0 = log10(Float64(model.L[i, j_to])) - log10(Float64(model.L[i, j_from]))
    return i, j_from, j_to, δ0
end

function _propagate_regular_component!(model::Bnc, grh::VertexGraph, comp::Vector{Int})
    isempty(comp) && return nothing

    regimes = model.vertices_data
    seed = comp[1]
    seed_rgm = _initialize_regime!(regimes[seed])
    H_seed = _calc_H(model, seed_rgm.perm)
    H0_seed = vec(-(H_seed * seed_rgm.M0))
    seed_rgm.H = H_seed
    seed_rgm.H0 = H0_seed

    in_comp = falses(length(regimes))
    in_comp[comp] .= true
    seen = falses(length(regimes))
    seen[seed] = true
    queue = [seed]

    while !isempty(queue)
        from_idx = popfirst!(queue)
        from_rgm = regimes[from_idx]
        H_from = from_rgm.H
        H0_from = from_rgm.H0

        for edge in grh.neighbors[from_idx]
            to_idx = edge.to
            (!in_comp[to_idx] || seen[to_idx]) && continue

            to_rgm = _initialize_regime!(regimes[to_idx])
            i, j_from, j_to, δ0 = _edge_rank1_data(model, from_idx, to_idx, edge)
            H_to, H0_to, δ = _rank1_update_H_H0(H_from, H0_from, i, j_from, j_to, δ0)

            if isnothing(H_to)
                # Numerical fallback: the graph update should stay in the regular class,
                # but direct construction is safer than failing hard here.
                H_to = _calc_H(model, to_rgm.perm)
                H0_to = vec(-(H_to * to_rgm.M0))
            end

            to_rgm.H = H_to
            to_rgm.H0 = H0_to
            seen[to_idx] = true
            push!(queue, to_idx)
        end
    end

    return nothing
end

function _ensure_regular_affine_cache!(model::Bnc)
    find_all_regimes!(model)
    model._regimes_affine_ready && return nothing

    lock(model._regimes_affine_lock)
    try
        model._regimes_affine_ready && return nothing

        regimes = model.vertices_data
        grh = get_regimes_graph!(model; full=false)
        regular_mask = falses(length(regimes))
        @inbounds for i in eachindex(regimes)
            regular_mask[i] = regimes[i].nullity == 0
        end

        comps = _masked_components(grh.x_grh, regular_mask)
        Threads.@threads for cid in eachindex(comps)
            _propagate_regular_component!(model, grh, comps[cid])
        end

        model._regimes_affine_ready = true
    finally
        unlock(model._regimes_affine_lock)
    end

    return nothing
end

function _fill_affine_info!(rgm::BindRegime)
    _initialize_regime!(rgm)
    _affine_info_ready(rgm) && return nothing

    model = rgm.network
    if rgm.nullity == 0
        _ensure_regular_affine_cache!(model)
        return nothing
    end

    if rgm.nullity == 1
        H = if allunique(rgm.perm)
            _calc_H(model, rgm.perm)
        else
            H_tmp = _adj_singular_matrix(rgm.M)[1]
            droptol!(sparse(H_tmp), 1e-10) .* model.direction
        end
        rgm.H = H
        rgm.H0 = vec(-(H * rgm.M0))
    end

    return nothing
end

function _materialize_qK_conditions!(rgm::BindRegime)
    _initialize_regime!(rgm)
    (!isnothing(rgm.C_qK) && !isnothing(rgm.C0_qK)) && return nothing

    if rgm.nullity == 0
        _fill_affine_info!(rgm)
        rgm.C_qK = droptol!(sparse(rgm.C_x * rgm.H), 1e-10)
        rgm.C0_qK = rgm.C0_x + rgm.C_x * rgm.H0
    else
        rgm.C_qK, rgm.C0_qK = _calc_C_C0_qK_singular(rgm.network, rgm.perm)
    end

    return nothing
end

"""
    _fill_inv_info!(vtx::BindRegime) -> nothing

Ensure a `BindRegime` has `H/H0` and qK constraints computed and cached.
"""
function _fill_inv_info!(vtx::BindRegime)
    _initialize_regime!(vtx)
    _fill_affine_info!(vtx)
    _materialize_qK_conditions!(vtx)
    return nothing
end



"""
    get_regimes_perm_dict(bnc::Bnc) -> Dict

Return a dictionary mapping permutation vectors to vertex indices.
"""
get_regimes_dict(model::Regimes) = model.vertices_perm_dict

get_bind_regimes_dict(Bnc::Bnc) = let 
    find_all_regimes!(Bnc)
    get_regimes_dict(Bnc.BindRegimes)
end


"""
    get_nullities(bnc::Bnc, rgms=nothing) -> Vector

Return nullity values for selected regimes.
"""
function get_nullities(Bnc::Bnc, rgms::Union{AbstractVector,Nothing}=nothing)
    """
    Calculate the nullity of all vertices in Bnc.
    """
    find_all_regimes!(Bnc)
    if isnothing(rgms)
        return getfield.(Bnc.vertices_data, :nullity)
    else
        idxs = get_idx.(Ref(Bnc), rgms)
        return getfield.(Bnc.vertices_data[idxs], :nullity)
    end

end

"""
    get_volumes(bnc::Bnc, vtxs=nothing; recalculate=false, kwargs...) -> Vector{Volume}

Return volumes for selected vertices, computing missing volumes as needed.
"""
function get_volumes(Bnc::Bnc,vtxs::Union{AbstractVector,Nothing}=nothing; 
    recalculate::Bool=false, 
    rebase_K::Bool = false, 
    rebase_mat:: Union{AbstractMatrix{<:Real},Nothing} = nothing,
    kwargs...)

    all_vtxs = isnothing(vtxs) ? get_regimes(Bnc;return_idx=true) : [get_idx(Bnc, vtx) for vtx in vtxs]

    vtxs_to_calc = 
        if recalculate
            all_vtxs
        else
            filter(i -> isnothing(Bnc.vertices_data[i].volume), all_vtxs)
        end
    
    if !isempty(vtxs_to_calc)

        rebase_mat = if  !isnothing(rebase_mat)
                    @assert !rebase_K "Cannot specify both rebase_K and providing rebase_mat"
                    rebase_mat
                elseif rebase_K
                    Q = rebase_mat_lgK(Bnc.N)
                    blockdiag(spdiagm(fill(Rational(1), Bnc.d)), Q)
                else
                    nothing
                end
        
        #ensure conditions for volume calculation are calced, may further replaced by other functions
        Threads.@threads for idx in vtxs_to_calc
           get_regime(Bnc,idx; inv_info=true)
        end
        
        vtx_data = @view Bnc.vertices_data[vtxs_to_calc]
        rlts = calc_volume(vtx_data; rebase_mat=rebase_mat, kwargs...)
        for (i,idx) in enumerate(vtxs_to_calc)
            vtx = get_regime(Bnc,idx; inv_info=false)
            vtx.volume = rlts[i]
        end
    end
    return [vtx.volume for vtx in Bnc.vertices_data[all_vtxs]]
end

#---------------------------------------------------------------------------------------------
#   Functions involving vertices relationships, (neighbors finding and changedir finding)
#---------------------------------------------------------------------------------------------
"""
    get_regimes_neighbor_mat_x(bnc::Bnc) -> SparseMatrixCSC

Return the x-space adjacency matrix of the vertex graph.
"""
function get_regimes_neighbor_mat_x(Bnc::Bnc)
    grh = get_regimes_graph!(Bnc;full=false)
    spmat = _regime_graph_to_sparse(grh; weight_fn = e -> 1)
    return spmat
end

"""
    get_regimes_neighbor_mat_qK(bnc::Bnc) -> SparseMatrixCSC

Return the qK-space adjacency matrix of the vertex graph.
"""
function get_regimes_neighbor_mat_qK(Bnc::Bnc)
    grh = get_regimes_graph!(Bnc;full=true)
    f(x::VertexEdge) = _edge_has_qK_interface(x) ? 1 : 0
    spmat = _regime_graph_to_sparse(grh; weight_fn = f)
    return spmat
end

get_regimes_neighbor_mat(args...;kwargs...) =  get_regimes_neighbor_mat_qK(args...;kwargs...)


#-------------------------------------------------------------------------------------
#         functions involving single vertex and lazy calculate  its properties, act as keys for higher level functions
# ------------------------------------------------------------------------------------
"""
    get_idx(bnc::Bnc, idx::Integer; check=false) -> Integer

Return the vertex index, optionally validating it.
"""
function get_idx(Bnc::Bnc, idx::T;check::Bool=false) where T<:Integer
    if check
        find_all_regimes!(Bnc)
        @assert idx ≥ 1 && idx ≤ n_regimes(Bnc) "The given index is out of range."
    end
   return idx
end
get_idx(Bnc::Bnc,perm::AbstractVector;kwargs...)=(get_bind_regimes_dict(Bnc)[get_perm(Bnc, perm)])
get_idx(vtx::BindRegime) = vtx.idx
get_idx(Bnc::Bnc, vtx::BindRegime;kwargs...)= get_idx(vtx)


"""
    get_perm(bnc::Bnc, perm; check=false) -> Vector

Return the permutation vector, optionally validating it.
"""
function get_perm(Bnc::Bnc,perm::Vector{<:Integer};check::Bool=false)
    if check
        @assert haskey(get_bind_regimes_dict(Bnc), perm) "The given perm is not in Bnc"
    end
    return perm
end
get_perm(Bnc::Bnc, perm::AbstractVector) = get_perm(Bnc, locate_sym_x.(Ref(Bnc), perm))
get_perm(Bnc::Bnc, idx::Integer; kwargs...)=(find_all_regimes!(Bnc); Bnc.vertices_data[idx].perm)
get_perm(vtx::BindRegime) = vtx.perm
get_perm(Bnc::Bnc, vtx::BindRegime;kwargs...)= get_perm(vtx)


"""
    get_regime(bnc::Bnc, perm; check=false, kwargs...) -> BindRegime

Retrieve a vertex from cache or create it if missing.
"""
function get_regime(Bnc::Bnc, perm; check::Bool=false, kwargs...)::BindRegime
    find_all_regimes!(Bnc) #initialize perm_data
    
    vtx = begin
        idx = get_idx(Bnc, perm; check=check)          
        _initialize_regime!(Bnc.vertices_data[idx])
    end
    return get_regime(vtx; kwargs...)
end
"""
    get_regime(vtx::BindRegime; inv_info=true, kwargs...) -> BindRegime

Ensure a vertex has requested cached fields and return it.
"""
function get_regime(vtx::BindRegime; inv_info::Bool=true,kwargs...)::BindRegime
    _initialize_regime!(vtx)
    if inv_info
        _fill_inv_info!(vtx)
    end
    return vtx
end
#-------------------------------------------------------------------------------------------------------------


"""
    get_binding_network(bnc_or_vertex, args...) -> Bnc

Return the binding network associated with a vertex or the model itself.
"""
get_binding_network(Bnc::Bnc,args...)=Bnc
get_binding_network(vtx::BindRegime,args...)=vtx.network

"""
    have_perm(bnc::Bnc, perm_or_idx) -> Bool

Return `true` when a permutation or index exists in the model.
"""
have_perm(Bnc::Bnc, perm::AbstractVector) = haskey(get_bind_regimes_dict(Bnc), get_perm(Bnc, perm))
have_perm(Bnc::Bnc, idx::Integer) = (find_all_regimes!(Bnc); idx ≥ 1 && idx ≤ n_regimes(Bnc))
have_perm(Bnc::Bnc, vtx::BindRegime) = have_perm(Bnc, get_perm(vtx))


"""
    get_neighbors(args...; singular=nothing, asymptotic=nothing, return_idx=false) -> Vector

Return neighbors of a vertex filtered by singularity and asymptotic flags.

# Keyword Arguments
- `singular`: `true`, `false`, integer threshold, or `nothing`.
- `asymptotic`: `true`, `false`, or `nothing`.
- `return_idx`: Return indices when `true`; otherwise return permutations.
"""
function get_neighbors(args...; singular::Union{Bool,Int,Nothing}=nothing, asymptotic::Union{Bool,Nothing}=nothing, return_idx::Bool=false)
    Bnc = get_binding_network(args...)
    grh = get_regimes_graph!(Bnc;full=true)
    rgm_idx = get_idx(args...)

    idx = keys(grh.edge_pos[rgm_idx]) |> collect
    
    vertices = Bnc.vertices_data
    idx = filter(idx) do i
        vtx = vertices[i]
        nlt = vtx.nullity
        flag_asym = vtx.is_asymptotic

        ok_singular = isnothing(singular) || (
            (singular === true  && nlt > 0) ||
            (singular === false && nlt == 0) ||
            (singular isa Int   && nlt ≤ singular)
        )

        ok_asym = isnothing(asymptotic) || (asymptotic == flag_asym)
        return ok_singular && ok_asym 
    end

    sort!(idx)

    return return_idx ? idx : getfield.(vertices[idx], :perm)
end


# --------------------------These properties are stored in Bnc as vector form when finding regimes, so we can access them directly.----------------------------
# """
# Gets the nullity of a vertex
# eg: get_nullity(model,perm)
#     get_nullity(vtx)
# """
# get_nullity(args...) = begin
#     model = get_binding_network(args...)
#     find_all_regimes!(model)
#     return model.vertices_nullity[get_idx(args...)]
# end::Integer

"""
    is_singular(args...) -> Bool

Return `true` if the vertex has nonzero nullity.
"""
is_singular(args...)= get_nullity(args...) > 0


"""
    is_asymptotic(args...) -> Bool

Return `true` if the vertex is asymptotic (real).
"""
is_asymptotic(args...) = begin
    model = get_binding_network(args...)
    find_all_regimes!(model)
    return model.vertices_data[get_idx(args...)].is_asymptotic
end::Bool
is_asymptotic(vtx::BindRegime) = vtx.is_asymptotic



#---------------------------------These properties are calculate when creating BindRegime object---------------------------------------
"""
    get_P_P0(args...) -> (SparseMatrixCSC, Vector)

Return `(P, P0)` for a vertex, creating it if needed.
"""
get_P_P0(args...) = get_regime(args...; inv_info=false) |> vtx -> (vtx.P, vtx.P0)
"""
    get_P(args...) -> SparseMatrixCSC

Return `P` for a vertex.
"""
get_P(args...) = get_P_P0(args...)[1]
"""
    get_P0(args...) -> Vector

Return `P0` for a vertex.
"""
get_P0(args...) = get_P_P0(args...)[2]

"""
    get_M_M0(args...) -> (SparseMatrixCSC, Vector)

Return `(M, M0)` for a vertex, creating it if needed.
"""
get_M_M0(args...) = get_regime(args...; inv_info=false) |> vtx -> (vtx.M, vtx.M0)
"""
    get_M(args...) -> SparseMatrixCSC

Return `M` for a vertex.
"""
get_M(args...) = get_M_M0(args...)[1]
"""
    get_M0(args...) -> Vector

Return `M0` for a vertex.
"""
get_M0(args...) = get_M_M0(args...)[2]

"""
    get_C_C0_x(args...) -> (SparseMatrixCSC, Vector)

Return `(C_x, C0_x)` for a vertex.
"""
get_C_C0_x(args...) = get_regime(args...; inv_info=false) |> vtx -> (vtx.C_x, vtx.C0_x)
"""
    get_C_x(args...) -> SparseMatrixCSC

Return `C_x` for a vertex.
"""
get_C_x(args...) = get_C_C0_x(args...)[1]
"""
    get_C0_x(args...) -> Vector

Return `C0_x` for a vertex.
"""
get_C0_x(args...) = get_C_C0_x(args...)[2]


"""
    get_C_C0_nullity_qK(args...) -> (SparseMatrixCSC, Vector, Int)

Return `(C_qK, C0_qK, nullity)` for a vertex.
"""
get_C_C0_nullity_qK(args...) = get_regime(args...; inv_info=true) |> vtx -> (vtx.C_qK, vtx.C0_qK, vtx.nullity)
"""
    get_C_C0_qK(args...) -> (SparseMatrixCSC, Vector)

Return `(C_qK, C0_qK)` for a vertex.
"""
get_C_C0_qK(args...) = get_C_C0_nullity_qK(args...)[1:2]
"""
    get_C_qK(args...) -> SparseMatrixCSC

Return `C_qK` for a vertex.
"""
get_C_qK(args...) = get_C_C0_nullity_qK(args...)[1]
"""
    get_C0_qK(args...) -> Vector

Return `C0_qK` for a vertex.
"""
get_C0_qK(args...) = get_C_C0_nullity_qK(args...)[2]


"""
    get_H_H0(args...) -> (SparseMatrixCSC, Vector)

Return `(H, H0)` for a regime with nullity at most 1.
"""
get_H_H0(args...) = get_nullity(args...) > 1 ? @error("BindRegime's nullity is bigger than 1, cannot get H0") : get_regime(args...; inv_info=true) |> rgm -> (rgm.H, rgm.H0)
"""
    get_H(args...) -> SparseMatrixCSC

Return `H` for a vertex when nullity <= 1.
"""
get_H(args...) = get_nullity(args...) > 1 ? @error("BindRegime's nullity is bigger than 1, cannot get H") : get_regime(args...; inv_info=true).H
"""
    get_H0(args...) -> Vector

Return `H0` for a vertex.
"""
get_H0(args...) = get_H_H0(args...)[2]


"""
    get_polyhedron(C, C0, nullity=0) -> Polyhedron

Construct a polyhedron from inequality constraints in qK space.
"""
function get_polyhedron(C::AbstractMatrix{<:Real}, C0::AbstractVector{<:Real}, nullity::Integer=0)::Polyhedron 
    if nullity ==0
        return hrep(-C,C0) |> x-> polyhedron(x,CDDLib.Library())
    else
        linset = BitSet(1:nullity)
        return hrep(-C,C0,linset) |> x-> polyhedron(x,CDDLib.Library())
    end
end
"""
    get_polyhedron(args...) -> Polyhedron

Convenience wrapper that pulls constraints from a vertex or model.
"""
get_polyhedron(args...)=get_polyhedron(get_C_C0_nullity_qK(args...)...)



"""
    get_C_C0_nullity(poly::Polyhedron) -> (Matrix, Vector, Int)

Extract `(C, C0, nullity)` from a polyhedron in H-representation.
"""
function get_C_C0_nullity(poly::Polyhedron) #Have to make sure the polyhedron has been already detecthlinearity.
    p = MixedMatHRep(hrep(poly))
    C = -p.A
    C0 = p.b
    nullity = begin
        linset = p.linset
        if !isempty(linset)
            nty = maximum(linset)
            @assert linset == BitSet(1:nty)
        else
            nty = 0
        end
        nty
    end
    return (C, C0, nullity)
end
"""
    get_C_C0_nullity(args...; kwargs...) -> (Matrix, Vector, Int)

Return `(C, C0, nullity)` for a vertex or polyhedron.
"""
get_C_C0_nullity(args...;kwargs...) = get_C_C0_nullity_qK(args...;kwargs...)
"""
    get_C_C0(args...; kwargs...) -> (Matrix, Vector)

Return `(C, C0)` for a vertex or polyhedron.
"""
get_C_C0(args...;kwargs...) = get_C_C0_nullity(args...;kwargs...) |> x->(x[1], x[2]) 
"""
    get_C(args...; kwargs...) -> Matrix

Return `C` for a vertex or polyhedron.
"""
get_C(args...;kwargs...) = get_C_C0_nullity(args...;kwargs...)[1]
"""
    get_C0(args...; kwargs...) -> Vector

Return `C0` for a vertex or polyhedron.
"""
get_C0(args...;kwargs...) = get_C_C0_nullity(args...;kwargs...)[2]

"""
    get_nullity(poly::Polyhedron, args...; kwargs...) -> Int

Return the nullity encoded in a polyhedron's linear constraints.
"""
get_nullity(poly::Polyhedron,args...;kwargs...) = get_C_C0_nullity(poly::Polyhedron,args...;kwargs...)[3]
"""
    get_nullity(args...) -> Int

Return the nullity of a vertex.
"""
get_nullity(args...) = begin
    model = get_binding_network(args...)
    find_all_regimes!(model)
    return model.vertices_data[get_idx(args...)].nullity
end::Integer
get_nullity(vtx::BindRegime) = vtx.nullity

"""
    n_regimes(bnc::Bnc) -> Int

Return the number of vertices in the model.
"""
n_regimes(Bnc::Bnc) = (find_all_regimes!(Bnc); length(Bnc.vertices_data))

"""
    get_volume(args...; kwargs...) -> Volume

Return the volume for a single vertex.
"""
function get_volume(args...;  kwargs...)
    model = get_binding_network(args...)
    idx = get_idx(args...)
    return get_volumes(model, [idx]; kwargs...)[1]
end


#--------------------------------------------------------------------------------------------------------------------------------------
#          Naive code for figuring out  relationships between two vertices 
#----------------------------------------------------------------------------------------------------------------------------------------

"""
    _is_regime_graph_neighbor(bnc, vtx1, vtx2) -> Bool

Return `true` if vertices are neighbors in the vertex graph.
"""
function _is_regime_graph_neighbor(Bnc, vtx1, vtx2)::Bool
    grh = get_regimes_graph!(Bnc; full=true)
    edge = get_edge(grh, vtx1, vtx2; full=true)
    if edge === nothing || !_edge_has_qK_interface(edge)
        return false
    else
        return true
    end
end

"""
    get_intersect(bnc, vtx1, vtx2) -> Polyhedron

Return the intersection polyhedron between two vertices in qK space.
"""
function get_intersect(Bnc,vtx1,vtx2)::Polyhedron
    p1 = get_polyhedron(Bnc, vtx1)
    dim1 = dim(p1)
    p2 = get_polyhedron(Bnc, vtx2)
    dim2 = dim(p2)

    p = intersect(p1,p2)
    detecthlinearity!(p)
    # @show dim1, dim2, dim(p)
    if dim(p)< max(dim1,dim2)-1
        error("Vertices $(get_perm(Bnc, vtx1)) and $(get_perm(Bnc, vtx2)) do not have dim-1 intersect.")
    end
    return p
end


"""
    get_interface_direct(bnc::Bnc, from, to) -> (SparseVector, Float64)

Compute the interface hyperplane directly from polyhedral intersection.
"""
function get_interface_direct(Bnc::Bnc, from, to)::Tuple{SparseVector{Float64,Int}, Float64}
    p = get_intersect(Bnc, from, to)
    hplanes = hyperplanes(p)
    # @show hplanes
    hp = collect(hplanes)[end]
    a = droptol!(sparse(hp.a), 1e-10)
    b = -hp.β
    return a, b
end

"""
    get_interface_qK(bnc, from, to) -> (SparseVector, Float64)

Return the interface hyperplane between two vertices in qK space.
"""
function get_interface_qK(Bnc, from, to)::Tuple{SparseVector{Float64,Int}, Float64}
    grh = get_regimes_graph!(Bnc; full=true)
    edge = get_edge(grh, from, to; full=true)
    if edge === nothing
        @info "no directly edge found, judge using Polyhedra.jl, could be problematic if you concerning changing direction"
        return get_interface_direct(Bnc, from, to)
    elseif !_edge_has_qK_interface(edge)
        @error("Vertices $get_perm(Bnc, from) and $get_perm(Bnc, to) are neighbors in x space but not in qK space")
    else
        return _edge_qK_interface(grh, edge)
    end   
end

"""
    get_interface(args...; kwargs...) -> (SparseVector, Float64)

Convenience wrapper for `get_interface_qK`.
"""
get_interface(args...;kwargs...) = get_interface_qK(args...;kwargs...)
"""
    get_change_dir_qK(args...; kwargs...) -> SparseVector

Return the qK change direction between neighboring vertices.
"""
get_change_dir_qK(args...;kwargs...) = get_interface(args...;kwargs...)[1] # relys on the inner behavior of get_interface, 
"""
    get_change_dir(args...; kwargs...) -> SparseVector

Alias for `get_change_dir_qK`.
"""
get_change_dir(args...;kwargs...) = get_change_dir_qK(args...;kwargs...)

"""
    is_neighbor_qK(bnc, vtx1, vtx2) -> Bool

Return `true` if two vertices are neighbors in qK space.
"""
function is_neighbor_qK(Bnc, vtx1, vtx2)::Bool
    try get_interface_qK(Bnc, vtx1, vtx2)
        return true
    catch
        return false
    end
end

"""
    is_neighbor(args...; kwargs...) -> Bool

Alias for `is_neighbor_qK`.
"""
is_neighbor(args...;kwargs...) = is_neighbor_qK(args...;kwargs...)


"""
    get_interface_x(bnc::Bnc, from, to) -> (SparseVector, Float64)

Return the interface hyperplane between two vertices in x space.
"""
function get_interface_x(Bnc::Bnc, from, to)
    edge = get_edge(Bnc, from, to)
    if edge === nothing 
        @error("Vertices $get_perm(Bnc, from) and $get_perm(Bnc, to) are not neighbors in x space.")
    else 
        return edge.change_dir_x, edge.intersect_x
    end
end

"""
    get_change_dir_x(args...; kwargs...) -> SparseVector

Return the x-space change direction between neighboring vertices.
"""
get_change_dir_x(args...;kwargs...) = get_interface_x(args...;kwargs...)[1]


#-------------------------------------------------------------------------------------
#         functions of getting vertices with certain properties
# -------------------------------------------------------------------------------------
"""
    get_regimes(bnc::Bnc; singular=nothing, asymptotic=nothing, return_idx=false) -> Vector

Return vertices that satisfy singularity/asymptotic filters.
"""
function get_regimes(Bnc::Bnc; return_idx::Bool=false, kwargs...)
    find_all_regimes!(Bnc)
    idx_all = eachindex(Bnc.vertices_data)
    masks = _get_mask(Bnc, idx_all; kwargs...)
    return return_idx ? findall(masks) : getfield.(Bnc.vertices_data[masks], :perm)
end


"""
    _get_mask(model::Bnc, vtxs; singular=nothing, asymptotic=nothing) -> Vector{Bool}

Return a boolean mask for vertices matching filter criteria.
"""
function _get_mask(model::Bnc,vtxs::AbstractVector{<:Integer};
     singular::Union{Bool,Integer,Nothing}=nothing, 
     asymptotic::Union{Bool,Nothing}=nothing)::Vector{Bool}
    find_all_regimes!(model)
    vertices = model.vertices_data

    @inline f(nlt) = isnothing(singular) || (
        (singular === true  && nlt > 0) ||
        (singular === false && nlt == 0) ||
        (singular isa Int   && nlt ≤ singular)
    )

    @inline g(flag_asym) = isnothing(asymptotic) || (asymptotic == flag_asym)
    
    return map(vtxs) do i
        vtx = vertices[i]
        f(vtx.nullity) && g(vtx.is_asymptotic)
    end
end
function _get_mask(rgms::AbstractVector{<:BindRegime};
     singular::Union{Bool,Integer,Nothing}=nothing, 
     asymptotic::Union{Bool,Nothing}=nothing)::Vector{Bool}
    
    @inline f(nlt) = isnothing(singular) || (
        (singular === true  && nlt > 0) ||
        (singular === false && nlt == 0) ||
        (singular isa Int   && nlt ≤ singular)
    )

    @inline g(flag_asym) = isnothing(asymptotic) || (asymptotic == flag_asym)

    return map(rgms) do vtx
        f(get_nullity(vtx)) && g(is_asymptotic(vtx))
    end
end

_get_regimes_mask(args...; kwargs...) = _get_mask(args...; kwargs...)



#-------------------------------------------------------------
# Functions using Polyhedra.jl  to calculate and fufill the 
#polyhedron helper functions
"""
    hyperplane_project_func(polyhedra::Polyhedron) -> Function

Return a projection function onto the affine subspace defined by polyhedron hyperplanes.
"""
function hyperplane_project_func(polyhedra::T)::Function where T<:Polyhedron
    if !hashyperplanes(polyhedra)
        error("polyhedra doesn't have hyperplanes")
    end
    # A^⊤y =b to project to this subspace   
    A = stack([i.a for i in hyperplanes(polyhedra)])
    b = stack([i.β for i in hyperplanes(polyhedra)])
    @show A,b
    # Now we need to generate a function to project a point into this hyperplanes
    AAtA_inv = A*pinv(A'*A)
    b0 = AAtA_inv*b
    P0 = I(size(A,1))-AAtA_inv*A'
    return x -> P0*x+b0
end



"""
    get_one_inner_point(poly::Polyhedron; rand_line=true, rand_ray=true, extend=3) -> Vector

Return a point guaranteed to lie inside the polyhedron.
"""
function get_one_inner_point(poly::T;rand_line=true,rand_ray=true,extend=3) where T<:Polyhedron
    vrep_poly = MixedMatVRep(vrep(poly))
    point = [mean(p) for p in eachcol(vrep_poly.V)]
    ray_avg = zeros(size(point,1))
    for (i, ray) in enumerate(eachrow(vrep_poly.R))
        if i ∉ vrep_poly.Rlinset
            norm_ray = norm(ray)
            sigma = rand_ray ? (rand()+0.5)*extend : extend
            ray_avg .+= (ray ./ norm_ray .* sigma )
        else
            if rand_line
                norm_ray = norm(ray)
                sigma = (rand()-0.5)*extend
                ray_avg .+= (ray ./ norm_ray * sigma)
            end
        end
    end
    return (point.+ ray_avg)
end
"""
    get_one_inner_point(args...; kwargs...) -> Vector

Convenience wrapper that builds a polyhedron from a vertex/model.
"""
get_one_inner_point(args...;kwargs...)=get_one_inner_point(get_polyhedron(args...);kwargs...)


"""
    check_feasibility_with_constraint(args...; C, C0, nullity=0) -> Bool

Check whether a vertex/polyhedron remains feasible under extra constraints.
"""
function check_feasibility_with_constraint(args...;C::AbstractMatrix{<:Real},C0::AbstractVector{<:Real},nullity::Int=0)
    poly_additional = get_polyhedron(C,C0,nullity)
    poly = get_polyhedron(args...)
    ins = intersect(poly,poly_additional)
    @info "The dimension of the intersected polyhedra is $(dim(ins))"
    return !isempty(ins)
end

"""
    feasible_vertieces_with_constraint(bnc::Bnc; C, C0, nullity=0, kwargs...) -> Vector

Return vertices feasible under additional constraints.
"""
function feasible_vertieces_with_constraint(Bnc::Bnc; C::AbstractMatrix{<:Real},C0::AbstractVector{<:Real},nullity::Int=0,kwargs...)
    all_vtx = get_regimes(Bnc;kwargs...)
    feasible_vtx = Vector{eltype(all_vtx)}()
    for perm in all_vtx
        if check_feasibility_with_constraint(Bnc, perm; C=C, C0=C0, nullity=nullity)
            push!(feasible_vtx, perm)
        end
    end
    return feasible_vtx
end

#-------------------------------------------------------------
#Other higher lever functions
#----------------------------------------------------------------
"""
    summary_regime(args...) -> nothing

Print a detailed summary for a single vertex.
"""
function summary_regime(args...)
    idx= get_idx(args...)
    perm = get_perm(args...)
    is_real = is_asymptotic(args...)
    nullity = get_nullity(args...)
    volume = get_volume(args...)
    println("idx=$idx,perm=$perm, asymptotic=$is_real, nullity=$nullity")
    println("volume=$(volume.mean) +- $(sqrt(volume.var))")
    println("Dominante Relation")
    display.(show_dominant_condition(args...;log_space=false))
    println("Expression")
    try
        display.(show_expression_x(args...;log_space=false))
    catch
    end
    println("Condition:")
    display.(show_condition_qK(args...;log_space=false))
    
    return nothing
end

"""
    summary(bnc::Bnc, perm) -> nothing

Alias for `summary_regime`.
"""
summary(Bnc::Bnc, perm)= summary_regime(Bnc, perm)
"""
    summary(vtx::BindRegime) -> nothing

Alias for `summary_regime`.
"""
summary(vtx::BindRegime)= summary_regime(vtx)


# function summary_vertices(Bnc::Bnc;kwargs...)
#     vtx = get_regimes(Bnc;kwargs...)
#     vtx .|> x->summary_regime(Bnc,x)
#     return nothing
# end


function get_function(vtx::BindRegime)
    H,H0 = get_H_H0(vtx)
    
    f = function(qK::AbstractArray{<:Real}; input_logspace::Bool=false, output_logspace::Bool=false)
            lgqK = input_logspace ? qK : log10.(qK)
            lgx = H * lgqK .+ H0
        return output_logspace ? lgx : exp10.(lgx)
    end

    return f
end
