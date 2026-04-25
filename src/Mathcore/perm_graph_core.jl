
"""
    RegimeEdge

Edge metadata connecting neighboring vertices in a regime graph.
"""
mutable struct RegimeEdge
    to::Int
    i ::Int # different row index
    c_c0_x_idx::Int
    c_c0_x_sign::Int8
      
    qK_interface_idx::Int
    qK_interface_sign::Int8

    function RegimeEdge(to::Int, i::Int, c_c0_x_idx::Int, c_c0_x_sign::Int8)
        return new(to, i, c_c0_x_idx, c_c0_x_sign, 0, 0)
    end

end

struct RegimeHyperplane
    change_dir_qK::SparseVector{Rational{Int},Int}
    intersect_qK::ExactLogExpr
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

    qK_interface_pool::Vector{RegimeHyperplane}
    x_interface_pool::Vector{Hyperplane_perm{Tv}}
    qK_classifier_full::Any # numeric classifier for classifying regimes based on qK hyperplanes, to be filled when needed.

    function RegimeGraph(L_helper::MatrixHelper{Tv}, neighbors::Vector{Vector{RegimeEdge}}) where {Tv}
        
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
            RegimeHyperplane[],
            L_helper.hyperplanes,
            nothing,
        )
    end
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
function _calc_regimes_graph(helper::MatrixHelper, perms::Vector{<:AbstractVector{T}}) where {T<:Integer}
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

                    hid, sign = choiceineq_between(helper, i, j_to, j_from)
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
    return RegimeGraph(helper, neighbors)
end
#=============================================================================================#
#          Calc qK-space change directions for edges with nullity <= 1 regimes
#=============================================================================================#


@inline _edge_has_qK_interface(edge::RegimeEdge) = edge.qK_interface_idx != 0

function _edge_qK_interface(grh::RegimeGraph, edge::RegimeEdge)
    edge.qK_interface_idx == 0 && return nothing

    hp = grh.qK_interface_pool[edge.qK_interface_idx]
    if edge.qK_interface_sign >= 0
        return hp.change_dir_qK, hp.intersect_qK
    else
        return -hp.change_dir_qK, -hp.intersect_qK
    end
end


function _canonicalize_qK_interface(
    c_qK::SparseVector{<:Rational},
    c0_qK::ExactLogExpr,
)
    dir, scale = let
        v = nonzeros(c_qK)[1]
        (v >= 0 ? Int8(1) : Int8(-1)), abs(v)
    end

    # normalize
    c_qK.nzval .= (c_qK.nzval .* dir) ./ scale
    c0_qK = (c0_qK * dir) / scale

    key = (Tuple(c_qK.nzind), Tuple(c_qK.nzval), c0_qK)

    return dir, key, c_qK, c0_qK
end


function _intern_qK_interface!(
    pool::Vector{RegimeHyperplane},
    key_to_id::Dict,
    c_qK::SparseVector{<:Rational},
    c0_qK::ExactLogExpr,
    dir::Int8,
)
    dropzeros!(c_qK)
    nnz(c_qK) == 0 && return 0, Int8(0)

    dir_inner, key, c_qK, c0_qK = _canonicalize_qK_interface(c_qK, c0_qK)

    hid = get!(key_to_id, key) do
        push!(pool, RegimeHyperplane(c_qK, c0_qK))
        length(pool)
    end

    return hid, sign(dir*dir_inner)
end





"""
    _fulfill_regimes_graph!(vtx_graph::RegimeGraph) -> nothing

Compute qK-space change directions for edges in the vertex graph.
"""
function _fulfill_regimes_graph!(vtx_graph::RegimeGraph)
    Bnc = vtx_graph.bn
    regimes = _bind_regimes_data(Bnc)
    empty!(vtx_graph.qK_interface_pool)

    key_to_id = Dict{Any,Int}()

    for edges in vtx_graph.neighbors
        for e in edges
            e.qK_interface_idx = 0
            e.qK_interface_sign = 0
        end
    end

    @showprogress for p1 in eachindex(vtx_graph.neighbors)
        nlt1 = regimes[p1].nullity

        if nlt1 > 1
            continue
        end

        edges = vtx_graph.neighbors[p1]
        for e in edges
            p2 = e.to
            p1 < p2 || continue
            
            nlt2 = regimes[p2].nullity
            nlt2 > 1 && continue
        
            rev_pos = vtx_graph.edge_pos[p2][p1]
            
            e_rev = vtx_graph.neighbors[p2][rev_pos]

            src_rgm = regimes[p1]




            c_c0 = vtx_graph.x_interface_pool[e.c_c0_x_idx]
            dir_x = e.c_c0_x_sign



            c_qK, c0_qK = _calc_dir(
                src_rgm.nullity,
                src_rgm.H,
                src_rgm.H0,
                c_c0
            )

            hid, dir = _intern_qK_interface!(
                    vtx_graph.qK_interface_pool, 
                    key_to_id, 
                    c_qK, 
                    c0_qK,
                    dir_x)

            hid == 0 && continue

            e.qK_interface_idx = hid
            e.qK_interface_sign = dir

            e_rev.qK_interface_idx = hid
            e_rev.qK_interface_sign = -dir

        end
    end
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
    # dropzero!(c_qK)
    return c_qK, c0_qK
end
