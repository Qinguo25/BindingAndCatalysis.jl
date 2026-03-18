#--------------Core computation functions-------------------------

"""
    _calc_C_C0_qK_singular(bnc::Bnc, vtx) -> (SparseMatrixCSC, Vector)

Build qK-space constraints `(C_qK, C0_qK)` for singular vertices via affine mapping.
"""
function _calc_C_C0_qK_singular(Bnc::Bnc, vtx)
    M,M0 = get_M_M0(Bnc,vtx)
    C,C0 = get_C_C0_x(Bnc,vtx)
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

"""
    _calc_change_col(from, to) -> Tuple

Compute the change direction between two permutations.
"""
function _calc_change_col(from::Vector{T},to::Vector{T}) where T<:Integer
    j1 = 0
    j2 = 0
    inconsis = Int[]
    for (i , (val_a,val_b)) in enumerate(zip(from,to))
        if val_a == val_b
            continue
        else
            push!(inconsis, i)
        end
    end
    target_inconsis = Set(to[inconsis])
    if target_inconsis |> length == 1
        j1,j2 = from[inconsis[1]], to[inconsis[1]]
        return j1,j2    
    end
    for (val1,i1) in zip(from[inconsis], inconsis)
        if val1 ∈ target_inconsis
            j2 = to[i1]
            i2 = inconsis[findfirst(x -> x == val1, to[inconsis])]
            j1 = from[i2]
            return j1,j2
        end
    end
end

"""
    _get_i_j_perms(from, to) -> (Int, Int, Int, Int)

Return indices and column selections that differ between two permutations.
"""
function _get_i_j_perms(from::Vector{T},to::Vector{T}) where T<:Integer
    inconsis_idx = findall(from .!= to)
    if length(inconsis_idx) == 1
        i1 = inconsis_idx[1]
        i2 = i1
    else
        intersect_val = Set(from[inconsis_idx]) ∩ Set(to[inconsis_idx])
        @assert length(intersect_val) == 1 "More than one intersected value found in inconsistent positions."
        for i in inconsis_idx
            if from[i] ∈ intersect_val
                i2 = i
            end
            if to[i] ∈ intersect_val
                i1 = i
            end
        end
    end
    j1 = from[i1]
    j2 = to[i2]
    return i1,i2,j1,j2
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

"""
    _fill_inv_info!(vtx::BindRegime) -> nothing

Ensure a `BindRegime` has `H/H0` and qK constraints computed and cached.
"""
function _fill_inv_info!(vtx::BindRegime)
    _initialize_regime!(vtx)
    Bnc = vtx.network
    if !isnothing(vtx.H)
        return nothing
    end
    if vtx.nullity == 0
        H = _calc_H(Bnc, vtx.perm) 
        vtx.H = H # Calculate the inverse matrix from pre-computed LU decomposition of M H=M^-1
        vtx.H0 = - H * vtx.M0  # H0 = -M^-1 * M0
        vtx.C_qK = droptol!(sparse(vtx.C_x * H),1e-10) # C_qK = C_x * H
        vtx.C0_qK = vtx.C0_x + vtx.C_x * vtx.H0 # C0_qK = C0_x + C_x * H0 
    else
        if vtx.nullity ==1
            # we need to check where this nullity comes from.
            if length(Set(vtx.perm)) == Bnc.d # the nullity comes from N
                vtx.H = _calc_H(Bnc, vtx.perm).* Bnc.direction 
            else # the nullity comes from P
                H = _adj_singular_matrix(vtx.M)[1]
                vtx.H = droptol!(sparse(H),1e-10).* Bnc.direction
            end
        else # nullity>1 , H, HO is nolonger avaliable
            vtx.H = spzeros(Bnc.n, Bnc.n) # fill value as a sign that this regime is fully computed
        end
        vtx.C_qK, vtx.C0_qK = _calc_C_C0_qK_singular(Bnc, vtx.perm)
    end
    return nothing
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


#------------------------------------------------------------------------------
#             1. Functions find all regimes and return properties
# ------------------------------------------------------------------------------

"""
    find_all_regimes!(bnc::Bnc) -> Vector{Vector{Int}}

Compute and cache all vertex permutations, asymptotic flags, Nρ inverse cache,
and vertex nullities.
"""

function find_all_regimes!(model::Bnc{T};) where T
    if _bind_regimes_built(model)
        return _bind_regimes_perm(model)
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
    regimes = _build_regimes(model, all_vertices, is_asymptotic, nullity)
    vertices_perm_dict = Dict(perm => idx for (idx, perm) in enumerate(all_vertices))
    model.BindRegimes = BindRegimes(vertices_perm_dict, regimes)

    return all_vertices
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



@inline function _calc_nullity(all_vertices, model::Bnc{T}) where T
    _build_Nρ_cache_parallel!(model, all_vertices) # build Nρ_inv cache in parallel
    nullity = Vector{T}(undef, length(all_vertices))
    
    Threads.@threads for i in  eachindex(all_vertices)
        perm = all_vertices[i]
        nullity_P = _calc_perm_nullity(perm, model.n)
        _, nullity_N = _get_Nρ_inv_from_perm!(model, perm)
        nullity[i] = nullity_P + nullity_N # this is true as we can permute the matrix into diagnal block matrix.
    end
    return nullity
end


@inline function _build_regimes(model::Bnc{T}, all_vertices, is_asymptotic, nullity) where T
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



# function find_all_regimes!(model::Bnc{T};) where T # cheap enough for now
#     if isempty(model.vertices_perm) 
#         @info "---------------------Start finding all vertices--------------------"
#         # all vertices
#         all_vertices, is_asymptotic =  _enumerate_all_regimes(model._L_helper)

#         all_vertices = [Vector{T}(v) for v in all_vertices]
        
#         n_vertices = length(all_vertices)
#         # finding asymptotic vertices, which is the real vertices.
#         n_asym_rgms = sum(is_asymptotic)
#         @info "Finished, with $(n_vertices) vertices found and $(n_asym_rgms) asymptotic vertices."
#         @info "-------------Start calculating nullity for each vertex, it also takes a while.------------"
        
#         @info "1.Building Nρ_inv cache in parallel..."
        
#         # Caltulate the nullity for each vertices
#         nullity = Vector{T}(undef, length(all_vertices))

#         model.vertices_perm = all_vertices
#         model.vertices_asymptotic_flag = is_asymptotic
#         model.vertices_perm_dict = Dict(a=>idx for (idx, a) in enumerate(model.vertices_perm)) # Map from vertex to its index
#         model.vertices_nullity = nullity
#         model.vertices_data = Vector{BindRegime}(undef, n_vertices)
#         model._vertices_is_initialized = falses(n_vertices)
#         model._vertices_volume_is_calced = falses(n_vertices)
#     end
#     return model.vertices_perm
# end


"""
    get_regimes_perm_dict(bnc::Bnc) -> Dict

Return a dictionary mapping permutation vectors to vertex indices.
"""
function get_regimes_perm_dict(Bnc::Bnc)
    find_all_regimes!(Bnc) # Ensure vertices are calculated
    return _bind_regimes_perm_dict(Bnc)
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
    f(x::VertexEdge) = isnothing(x.change_dir_qK) ? 0 : 1
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
get_idx(Bnc::Bnc,perm::AbstractVector;kwargs...)=(find_all_regimes!(Bnc);Bnc.vertices_perm_dict[get_perm(Bnc, perm)])
get_idx(vtx::BindRegime) = vtx.idx
get_idx(Bnc::Bnc, vtx::BindRegime;kwargs...)= get_idx(vtx)


"""
    get_perm(bnc::Bnc, perm; check=false) -> Vector

Return the permutation vector, optionally validating it.
"""
function get_perm(Bnc::Bnc,perm::Vector{<:Integer};check::Bool=false)
    if check
        find_all_regimes!(Bnc)
        @assert haskey(Bnc.vertices_perm_dict, perm) "The given perm is not in Bnc"
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
have_perm(Bnc::Bnc, perm::AbstractVector) = (find_all_regimes!(Bnc); haskey(Bnc.vertices_perm_dict, get_perm(Bnc, perm)))
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

Return `(H, H0)` for a non-singular vertex.
"""
get_H_H0(args...) = is_singular(args...) ? @error("BindRegime is singular, cannot get H0") : get_regime(args...; inv_info=true) |> vtx -> (vtx.H, vtx.H0)
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
    edge = get_edge(Bnc,vtx1,vtx2) 
    if edge === nothing || edge.change_dir_qK === nothing
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
    edge = get_edge(Bnc, from, to)
    if edge === nothing
        @info "no directly edge found, judge using Polyhedra.jl, could be problematic if you concerning changing direction"
        return get_interface_direct(Bnc, from, to)
    elseif edge.change_dir_qK === nothing
        @error("Vertices $get_perm(Bnc, from) and $get_perm(Bnc, to) are neighbors in x space but not in qK space")
    else
        a = edge.change_dir_qK
        b = edge.intersect_qK
        return a, b
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
