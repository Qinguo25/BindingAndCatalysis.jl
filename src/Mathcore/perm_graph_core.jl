
"""
    RegimeEdge

Edge metadata connecting neighboring vertices in a regime graph.
"""
mutable struct RegimeEdge
    to::Int
    i ::Int # different row index

    idx_sign::Vector{Tuple{Int,Int8}}

    function RegimeEdge(to::Int, i::Int, c_c0_x_idx::Int, c_c0_x_sign::Int8)
        return new(to, i, Tuple{Int,Int8}[(c_c0_x_idx, c_c0_x_sign), (0, 0)])
    end

    function RegimeEdge(to::Int, i::Int, idx_sign::Vector{Tuple{Int,Int8}})
        return new(to, i, idx_sign)
    end

end

const _EDGE_SPACE_PRIMARY = 1
const _EDGE_SPACE_SECONDARY = 2
const _EDGE_SPACE_TERTIARY = 3

# Integer aliases kept for low-level callers. Each RegimeGraph now records the
# meaning of these slots in `space_idx`, e.g. Dict(:x => 1, :qK => 2).
const _EDGE_SPACE_X = _EDGE_SPACE_PRIMARY
const _EDGE_SPACE_V = _EDGE_SPACE_PRIMARY
const _EDGE_SPACE_XK = _EDGE_SPACE_SECONDARY
const _EDGE_SPACE_QK = _EDGE_SPACE_SECONDARY
const _EDGE_SPACE_BNC_XK = _EDGE_SPACE_PRIMARY
const _EDGE_SPACE_QKK = _EDGE_SPACE_SECONDARY
const _EDGE_SPACE_WKK = _EDGE_SPACE_TERTIARY

@inline function _ensure_edge_space!(edge::RegimeEdge, space::Int)
    while length(edge.idx_sign) < space
        push!(edge.idx_sign, (0, Int8(0)))
    end
    return edge
end

@inline _edge_idx_sign(edge::RegimeEdge, space::Int) =
    length(edge.idx_sign) < space ? (0, Int8(0)) : edge.idx_sign[space]

@inline function _set_edge_idx_sign!(edge::RegimeEdge, space::Int, idx::Int, sign::Integer)
    _ensure_edge_space!(edge, space)
    edge.idx_sign[space] = (idx, Int8(sign))
    return edge
end

@inline _edge_has_space(edge::RegimeEdge, space::Int) = _edge_idx_sign(edge, space)[1] != 0

function Base.getproperty(edge::RegimeEdge, sym::Symbol)
    if sym === :c_c0_x_idx
        return _edge_idx_sign(edge, _EDGE_SPACE_X)[1]
    elseif sym === :c_c0_x_sign
        return _edge_idx_sign(edge, _EDGE_SPACE_X)[2]
    elseif sym === :qK_interface_idx
        return _edge_idx_sign(edge, _EDGE_SPACE_QK)[1]
    elseif sym === :qK_interface_sign
        return _edge_idx_sign(edge, _EDGE_SPACE_QK)[2]
    end
    return getfield(edge, sym)
end

function Base.setproperty!(edge::RegimeEdge, sym::Symbol, value)
    if sym === :c_c0_x_idx
        _, sign = _edge_idx_sign(edge, _EDGE_SPACE_X)
        _set_edge_idx_sign!(edge, _EDGE_SPACE_X, value, sign)
    elseif sym === :c_c0_x_sign
        idx, _ = _edge_idx_sign(edge, _EDGE_SPACE_X)
        _set_edge_idx_sign!(edge, _EDGE_SPACE_X, idx, value)
    elseif sym === :qK_interface_idx
        _, sign = _edge_idx_sign(edge, _EDGE_SPACE_QK)
        _set_edge_idx_sign!(edge, _EDGE_SPACE_QK, value, sign)
    elseif sym === :qK_interface_sign
        idx, _ = _edge_idx_sign(edge, _EDGE_SPACE_QK)
        _set_edge_idx_sign!(edge, _EDGE_SPACE_QK, idx, value)
    else
        setfield!(edge, sym, value)
    end
    return value
end




# Adjacency list + optional caches
"""
    RegimeGraph

Adjacency structure for vertices with optional caches for change directions.
"""
mutable struct RegimeGraph{Tv}
    bn::Union{AbstractBnc, Nothing}
    neighbors::Vector{Vector{RegimeEdge}}

    edge_pos::Vector{Dict{Int, Int}}  # (u,v) -> (u,edge_pos[u][v]) to locate the RegimeEdge.

    hp_data::Vector{Any}
    space_idx::Dict{Symbol,Int}

    qK_classifier_full::Any # numeric classifier for classifying regimes based on qK hyperplanes, to be filled when needed.

    function RegimeGraph{Tv}(bn, neighbors, edge_pos, hp_data, space_idx, qK_classifier_full) where {Tv}
        return new{Tv}(bn, neighbors, edge_pos, hp_data, Dict{Symbol,Int}(space_idx), qK_classifier_full)
    end

    function RegimeGraph(
        L_helper::MatrixHelper{Tv},
        neighbors::Vector{Vector{RegimeEdge}};
        space_idx::Dict{Symbol,Int}=Dict(:x => _EDGE_SPACE_X, :qK => _EDGE_SPACE_QK),
    ) where {Tv}
        
        edge_pos = let
            edge_pos = Vector{Dict{Int,Int}}(undef, length(neighbors))
                for i in eachindex(neighbors)
                    edges = neighbors[i]
                    d = Dict{Int,Int}()
                    sizehint!(d, length(edges))
                    for (k, e) in enumerate(edges)
                        d[e.to] = k
                    end
                    edge_pos[i] = d
                end
                edge_pos
            end

        return new{Tv}(
            nothing,
            neighbors,
            edge_pos,
            Any[L_helper, RegimeToHyperplanePool(L_helper.n)],
            space_idx,
            nothing,
        )
    end
end

function RegimeGraph(
    neighbors::Vector{Vector{RegimeEdge}},
    hp_data::Vector{Any};
    bn=nothing,
    space_idx::Dict{Symbol,Int}=Dict(Symbol("space$i") => i for i in eachindex(hp_data)),
)
    edge_pos = Vector{Dict{Int,Int}}(undef, length(neighbors))
    for i in eachindex(neighbors)
        d = Dict{Int,Int}()
        sizehint!(d, length(neighbors[i]))
        for (k, e) in enumerate(neighbors[i])
            d[e.to] = k
        end
        edge_pos[i] = d
    end
    return RegimeGraph{Int}(bn, neighbors, edge_pos, hp_data, space_idx, nothing)
end

function Base.getproperty(grh::RegimeGraph, sym::Symbol)
    if sym === :x_hp_data
        return getfield(grh, :hp_data)[get(getfield(grh, :space_idx), :x, _EDGE_SPACE_X)]
    elseif sym === :qK_hp_data
        return getfield(grh, :hp_data)[get(getfield(grh, :space_idx), :qK, _EDGE_SPACE_QK)]
    end
    return getfield(grh, sym)
end

@inline function _edge_space_index(grh::RegimeGraph, space::Symbol)
    return get(grh.space_idx, space) do
        throw(ArgumentError("RegimeGraph does not have edge space :$space. Available spaces: $(sort!(collect(keys(grh.space_idx))))"))
    end
end
@inline _edge_space_index(::RegimeGraph, space::Integer) = Int(space)
@inline _edge_idx_sign(edge::RegimeEdge, grh::RegimeGraph, space) = _edge_idx_sign(edge, _edge_space_index(grh, space))
@inline _set_edge_idx_sign!(edge::RegimeEdge, grh::RegimeGraph, space, idx::Int, sign::Integer) =
    _set_edge_idx_sign!(edge, _edge_space_index(grh, space), idx, sign)
@inline _edge_has_space(edge::RegimeEdge, grh::RegimeGraph, space) =
    _edge_has_space(edge, _edge_space_index(grh, space))
@inline _has_edge_space(grh::RegimeGraph, space::Symbol) = haskey(grh.space_idx, space)

function _default_edge_space(grh::RegimeGraph)
    for space in (:qK, :qKk, :xk, :wKk, :v, :x)
        _has_edge_space(grh, space) && return space
    end
    return first(keys(grh.space_idx))
end

function _default_layout_edge_space(grh::RegimeGraph)
    for space in (:x, :v, :xk, :qK, :qKk, :wKk)
        _has_edge_space(grh, space) && return space
    end
    return _default_edge_space(grh)
end



Base.display(io::IO, grh::RegimeGraph) = print(io, "RegimeGraph with $(length(grh.neighbors)) vertices and $(sum(length.(grh.neighbors))) edges")
Base.show(io::IO, grh::RegimeGraph) = print(io, "RegimeGraph with $(length(grh.neighbors)) vertices and $(sum(length.(grh.neighbors))) edges")


#-----------------------------------------------------------------------------------------------
#This is graph associated functions for Bnc models and archetyple behaviors associated code
#-----------------------------------------------------------------------------------------------
"""
    _calc_regimes_graph(bnc::Bnc, perms) -> RegimeGraph

Build a `RegimeGraph` from regime permutations, connecting regimes that differ
in exactly one row.
"""
function _calc_regimes_graph(
    helper::MatrixHelper,
    perms::Vector{<:AbstractVector{T}};
    primary_space::Symbol=:x,
    secondary_space::Symbol=:qK,
) where {T<:Integer}
    # n = helper.n
    n_vtxs = length(perms)
    d = length(perms[1])
    thread_edges = [Vector{Tuple{Int, RegimeEdge}}() for _ in 1:Threads.maxthreadid()]

    # 按行分桶：key 为去掉该行后的签名（Tuple），值为该签名下的 (regime idx, row choice)
    @showprogress for i in 1:d
        buckets = Dict{Tuple{Vararg{T}}, Vector{Tuple{Int,T}}}()
        @inbounds for q in 1:n_vtxs
            v = perms[q]
            sig = if i == 1
                    Tuple(v[2:end])
                elseif i == d
                    Tuple(v[1:end-1])
                else
                    Tuple((v[1:i-1]..., v[i+1:end]...))
                end
            push!(get!(buckets, sig) do
                Vector{Tuple{Int,T}}()
            end, (q, v[i]))
        end

        groups = collect(values(buckets))

        # 同桶内两两相连：沿边方向表示“增加 target dominant term，减少 source dominant term”
        Threads.@threads for gi in eachindex(groups)
            tid = Threads.threadid()
            local_edges = thread_edges[tid]
            group = groups[gi]
            m = length(group)
            m <= 1 && continue

            @inbounds for a in 1:m-1
                from_idx, j_from = group[a]
                for b in a+1:m
                    to_idx, j_to = group[b]
                    j_from == j_to && continue

                    hid, sign = locate_halfspace(helper, i, j_to, j_from)
                    push!(local_edges, (from_idx, RegimeEdge(to_idx, i, hid, sign)))
                    push!(local_edges, (to_idx, RegimeEdge(from_idx, i, hid, -sign)))
                end
            end
        end
    end

    all_edges = reduce(vcat, thread_edges; init=Tuple{Int, RegimeEdge}[])
    neighbors = [Vector{RegimeEdge}() for _ in 1:n_vtxs]
    for (from, e) in all_edges
        push!(neighbors[from], e)
    end
    return RegimeGraph(
        helper,
        neighbors;
        space_idx=Dict(primary_space => _EDGE_SPACE_PRIMARY, secondary_space => _EDGE_SPACE_SECONDARY),
    )
end




#=============================================================================================#
#          Calc qK-space change directions for edges with nullity <= 1 regimes
#=============================================================================================#




@inline _edge_has_qK_interface(edge::RegimeEdge) = _edge_has_space(edge, _EDGE_SPACE_QK)

function _edge_interface(grh::RegimeGraph, edge::RegimeEdge, space::Int)
    idx, dir = _edge_idx_sign(edge, space)
    idx == 0 && return nothing
    
    hp = get_hyperplane(grh.hp_data[space], idx)

    return _calc_c_c0(hp, dir)
end
_edge_interface(grh::RegimeGraph, edge::RegimeEdge, space::Symbol) =
    _edge_interface(grh, edge, _edge_space_index(grh, space))

_edge_qK_interface(grh::RegimeGraph, edge::RegimeEdge) = _edge_interface(grh, edge, :qK)





"""
    _fulfill_regimes_graph!(vtx_graph::RegimeGraph) -> nothing

Compute qK-space change directions for edges in the vertex graph.
"""
function _fulfill_regimes_graph!(vtx_graph::RegimeGraph)
    Bnc = vtx_graph.bn
    regimes = _bind_regimes_data(Bnc)
    qK_space = _edge_space_index(vtx_graph, :qK)
    x_space = _edge_space_index(vtx_graph, :x)
    db = vtx_graph.hp_data[qK_space]

    I = Int[]     # row indices: polyhedron id
    J = Int[]     # col indices: hyperplane id
    V = Int8[]    # values: +1 or -1

    @showprogress for p1 in eachindex(vtx_graph.neighbors)
        nlt1 = regimes[p1].nullity

        if nlt1 > 1 #skip regimes with nullity >1
            continue
        end

        edges = vtx_graph.neighbors[p1]
        for e in edges
            p2 = e.to
            p1 < p2 || continue
            
            nlt2 = regimes[p2].nullity
            nlt2 > 1 && continue  #skip regimes with nullity >1
        


            rev_pos = vtx_graph.edge_pos[p2][p1]
            
            e_rev = vtx_graph.neighbors[p2][rev_pos]

            src_rgm = regimes[p1]

            x_idx, dir_x = _edge_idx_sign(e, x_space)
            c_c0 = get_hyperplane(vtx_graph.hp_data[x_space], x_idx)

            c_qK, c0_qK = _calc_dir(
                src_rgm.nullity,
                src_rgm.H,
                src_rgm.H0,
                c_c0
            ) # already dropped zeros in c_qK

            hid, dir = add_halfspace!(db, c_qK, c0_qK, dir_x; canonicalize=true)

            hid == 0 && continue

            push!(I, p1)
            push!(J, hid)
            push!(V, -dir) #hid, dir define the halfspace for p2, so p1 is on the opposite side

            push!(I, p2)
            push!(J, hid)
            push!(V, dir)

            _set_edge_idx_sign!(e, qK_space, hid, dir)

            _set_edge_idx_sign!(e_rev, qK_space, hid, -dir)
        end
    end

    M = sparse(I, J, V, length(vtx_graph.neighbors), length(db.hyperplanes))
    MT = sparse(J, I, V, length(db.hyperplanes), length(vtx_graph.neighbors))
    db.hp_to_poly = FacetIncidence(M, MT)
    return nothing
end


@inline function _calc_dir(
    nlt::Int,
    H::SparseMatrixCSC{<:Real,Int},
    H0::AbstractVector{<:Real},
    c_c0::Hyperplane_perm,
)
    c_qK = c_c0 * H 
    c0_qK = nlt ==0 ? c_c0 * H0  : mul(c_c0, H0; with_c0=false) 
    dropzeros!(c_qK)
    return c_qK, c0_qK
end
