mutable struct SIMOPaths{T}
    bn::Bnc{T}
    qK_grh::SimpleDiGraph
    change_qK_idx::T

    sources::Vector{Int}
    sinks::Vector{Int}
    paths_dict::Union{Nothing, Dict{Vector{Int}, Int}}
    rgm_paths::Vector{Vector{Int}}
    path_edge_idxs::Vector{Vector{Int}}
    edge_keys::Vector{Tuple{Int, Int}}
    path_node_mask::BitVector

    node_polys::Vector{Polyhedron}
    node_polys_is_calc::BitVector
    edge_polys::Vector{Polyhedron}
    edge_polys_is_calc::BitVector

    path_polys::Vector{Polyhedron}
    path_volume::Vector{Volume}

    path_volume_is_calc::BitVector
    path_polys_is_calc::BitVector

    function SIMOPaths(
        model::Bnc{T}, qK_grh, change_qK_idx, sources, sinks, rgm_paths
    ) where {T}
        edge_keys, path_edge_idxs = _build_path_edge_index(rgm_paths)
        path_node_mask = falses(n_regimes(model))
        for path in rgm_paths
            for idx in path
                path_node_mask[Int(idx)] = true
            end
        end
        node_polys = Vector{Polyhedron}(undef, n_regimes(model))
        node_polys_is_calc = falses(length(node_polys))
        edge_polys = Vector{Polyhedron}(undef, length(edge_keys))
        edge_polys_is_calc = falses(length(edge_keys))
        path_polys = Vector{Polyhedron}(undef, length(rgm_paths))
        path_volume = Vector{Volume}(undef, length(rgm_paths))
        path_volume_is_calc = falses(length(rgm_paths))
        path_polys_is_calc = falses(length(rgm_paths))
        return new{T}(
            model,
            qK_grh,
            change_qK_idx,
            sources,
            sinks,
            nothing,
            rgm_paths,
            path_edge_idxs,
            edge_keys,
            path_node_mask,
            node_polys,
            node_polys_is_calc,
            edge_polys,
            edge_polys_is_calc,
            path_polys,
            path_volume,
            path_volume_is_calc,
            path_polys_is_calc,
        )
    end
end

@inline function _is_isolated_singular_simo_regime(
    model::Bnc, qK_grh::AbstractGraph, idx::Integer
)::Bool
    v = Int(idx)
    return indegree(qK_grh, v) == 0 &&
           outdegree(qK_grh, v) == 0 &&
           get_nullity(model, v) > 0
end

function SIMOPaths(model::Bnc{T}, change_qK; rgm_paths=nothing) where {T}
    change_qK_idx = locate_sym_qK(model, change_qK)

    if rgm_paths === nothing
        qK_grh = get_SIMO_graph(model, change_qK)
        sources, sinks = get_sources_sinks(model, qK_grh)
        rgm_paths = _enumerate_paths(qK_grh; sources, sinks)
    else
        qK_grh = graph_from_paths(rgm_paths, n_regimes(model))
        sources, sinks = get_sources_sinks(qK_grh)
    end

    filter!(rgm_paths) do path
        length(path) > 1 || get_nullity(model, only(path)) == 0
    end

    if isempty(rgm_paths)
        sources = Int[]
        sinks = Int[]
    else
        sources = unique(Int(first(p)) for p in rgm_paths)
        sinks = unique(Int(last(p)) for p in rgm_paths)
    end

    return SIMOPaths(model, qK_grh, change_qK_idx, sources, sinks, rgm_paths)
end

function _ensure_paths_dict!(grh::SIMOPaths)
    isnothing(grh.paths_dict) || return grh.paths_dict
    grh.paths_dict = Dict(p => idx for (idx, p) in enumerate(grh.rgm_paths))
    return grh.paths_dict
end

"""
    _build_path_edge_index(rgm_paths::AbstractVector{<:AbstractVector{<:Integer}})

为一组路径构建“去重后的无向边索引”。

该函数会遍历每条路径中相邻节点组成的边，并将边 `(u, v)` 与 `(v, u)`
视为同一条无向边，统一映射到一个唯一的整数编号。

返回值包含两部分：
- `edge_keys::Vector{Tuple{Int,Int}}`：  所有去重后的边，按首次出现顺序存储。
- `path_edge_idxs::Vector{Vector{Int}}`：  每条路径对应的边编号序列，其中每个编号指向 `edge_keys` 中的一条边。

# 参数
- `rgm_paths`：路径集合；每条路径由一串节点编号组成。

# 返回
- `(edge_keys, path_edge_idxs)`

# 说明
- 路径长度小于 2 时，不产生边，对应返回空的边编号列表。
- 边按无向方式处理，因此路径 `[..., u, v, ...]` 和 `[..., v, u, ...]`
  中对应边会共享同一个索引。

# 示例
```julia
rgm_paths = [[1, 3, 5], [5, 3, 1], [1, 2]]
edge_keys, path_edge_idxs = _build_path_edge_index(rgm_paths)

# edge_keys == [(1, 3), (3, 5), (1, 2)]
# path_edge_idxs == [[1, 2], [2, 1], [3]]
```
"""

function _build_path_edge_index(rgm_paths::AbstractVector{<:AbstractVector{<:Integer}})
    total_refs = sum(max(length(path) - 1, 0) for path in rgm_paths)
    edge_keys = Tuple{Int, Int}[]
    sizehint!(edge_keys, total_refs)
    edge_dict = Dict{Tuple{Int, Int}, Int}()
    path_edge_idxs = Vector{Vector{Int}}(undef, length(rgm_paths))

    for (path_idx, path) in enumerate(rgm_paths)
        n_edges = max(length(path) - 1, 0)
        idxs = Vector{Int}(undef, n_edges)
        @inbounds for i in 1:n_edges
            u = Int(path[i])
            v = Int(path[i + 1])
            a, b = u < v ? (u, v) : (v, u)
            edge_key = (a, b)
            edge_idx = get!(edge_dict, edge_key) do
                push!(edge_keys, edge_key)
                length(edge_keys)
            end
            idxs[i] = edge_idx
        end
        path_edge_idxs[path_idx] = idxs
    end

    return edge_keys, path_edge_idxs
end

function get_indices(
    grh::SIMOPaths, pth_idx::Union{Nothing, Integer, AbstractVector}=nothing
)
    return if isnothing(pth_idx)
        collect(1:length(grh.rgm_paths))
    else
        Int.(get_idx.(Ref(grh), pth_idx))
    end
end

@inline function _path_indices_to_calculate(
    is_calc::BitVector, pth_idx::AbstractVector{<:Integer}; recompute::Bool=false
)
    idxs = Int.(pth_idx)
    return recompute ? idxs : filter(i -> !is_calc[i], idxs)
end

get_neighbor_graph_qK(grh::SIMOPaths; kwargs...) = grh.qK_grh
get_SIMO_graph(grh::SIMOPaths) = grh.qK_grh
function get_SIMO_graph(model::Bnc, change_qK)
    return get_SIMO_graph(get_regimes_graph!(model; full=true), change_qK)
end

function get_SIMO_graph(grh::RegimeGraph, change_qK)::SimpleDiGraph
    bn = get_binding_network(grh)
    change_qK_idx = locate_sym_qK(bn, change_qK)

    n = length(grh.neighbors)
    g = SimpleDiGraph(n)
    for (i, edges) in enumerate(grh.neighbors)
        get_nullity(bn, i) > 1 && continue
        for e in edges
            if !_edge_has_qK_interface(grh, e) || e.to < i
                continue
            end
            val = _edge_qK_interface(grh, e)[1][change_qK_idx]
            if val > 1e-6
                add_edge!(g, i, e.to)
            elseif val < -1e-6
                add_edge!(g, e.to, i)
            end
        end
    end

    return g
end

function get_path(grh::SIMOPaths, pth_idx::Integer; return_idx::Bool=false)
    rgm_idxs = grh.rgm_paths[pth_idx]
    return return_idx ? rgm_idxs : get_perm.(Ref(get_binding_network(grh)), rgm_idxs)
end

function get_path(grh::SIMOPaths, pth::AbstractVector; return_idx::Bool=false)
    bn = get_binding_network(grh)
    return return_idx ? get_idx.(Ref(bn), pth) : get_perm.(Ref(bn), pth)
end

get_binding_network(grh::SIMOPaths, args...) = grh.bn
function get_C_C0_nullity_qK(grh::SIMOPaths, pth_idx; remove_h_redundancy::Bool=false)
    C, C0, nullity = get_C_C0_nullity(get_polyhedron(grh, pth_idx))
    return _maybe_remove_h_redundancy(
        C, C0, nullity; remove_h_redundancy=remove_h_redundancy
    )
end

get_idx(grh::SIMOPaths, pth::AbstractVector) =
    let
        bn = get_binding_network(grh)
        idxs = get_idx.(Ref(bn), pth)
        _ensure_paths_dict!(grh)[idxs]
    end
get_idx(grh::SIMOPaths, pth::Integer) = pth
