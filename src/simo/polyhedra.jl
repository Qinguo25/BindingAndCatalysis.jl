# For SIMO path conditions we only need a canonical H-rep, not an LP-minimal one.
# Using the light pass here matches cddlib's "canonicalize after projection" style
# much better than running a full LP-based strong reduction on every path.
_clean_polyhedron!(p::Polyhedron) = backend_from_fastpath(p; canonicalize=true)

function _ensure_node_polyhedra!(grh::SIMOPaths, rgm_idxs::AbstractVector{<:Integer})
    bn = get_binding_network(grh)
    regimes = _bind_regimes_data(bn)
    unique_idxs = unique(Int.(rgm_idxs))

    function build_one!(idx::Int)
        if !grh.node_polys_is_calc[idx]
            rgm = regimes[idx]
            _materialize_qK_conditions!(rgm)
            grh.node_polys[idx] = get_polyhedron(rgm.C_qK, rgm.C0_qK, rgm.nullity; canonicalize=false)
            grh.node_polys_is_calc[idx] = true
        end
        return nothing
    end

    if Threads.nthreads() == 1 || length(unique_idxs) <= 1
        for idx in unique_idxs
            build_one!(idx)
        end
    else
        Threads.@threads for pos in eachindex(unique_idxs)
            build_one!(unique_idxs[pos])
        end
    end
    return nothing
end

function _ensure_edge_polyhedra!(grh::SIMOPaths, edge_idxs::AbstractVector{<:Integer})
    edge_idxs_unique = unique(Int.(edge_idxs))
    edge_idxs_to_calc = filter(idx -> !grh.edge_polys_is_calc[idx], edge_idxs_unique)
    isempty(edge_idxs_to_calc) && return nothing

    bn = get_binding_network(grh)
    prefer_fastpath = backend_prefers_fastpath(_affine_is_exact(bn))
    rgm_idxs = Int[]
    sizehint!(rgm_idxs, 2 * length(edge_idxs_to_calc))
    for edge_idx in edge_idxs_to_calc
        u, v = grh.edge_keys[edge_idx]
        push!(rgm_idxs, u)
        push!(rgm_idxs, v)
    end
    _ensure_node_polyhedra!(grh, rgm_idxs)

    el_dim = BitSet((grh.change_qK_idx,))
    @info "Start building polyhedra for edges (total: $(length(edge_idxs_to_calc)))"
    @showprogress Threads.@threads for pos in eachindex(edge_idxs_to_calc)
        edge_idx = edge_idxs_to_calc[pos]
        u, v = grh.edge_keys[edge_idx]
        grh.edge_polys[edge_idx] = backend_intersect_eliminate(
            grh.node_polys[u],
            grh.node_polys[v],
            el_dim;
            canonicalize=false,
            prefer_fastpath=prefer_fastpath,
        )
        grh.edge_polys_is_calc[edge_idx] = true
    end

    return nothing
end

function _build_path_polyhedron(
    grh::SIMOPaths,
    path::AbstractVector{<:Integer},
    edge_idxs::AbstractVector{<:Integer},
)::Polyhedron
    prefer_fastpath = backend_prefers_fastpath(_affine_is_exact(get_binding_network(grh)))
    if isempty(edge_idxs)
        _ensure_node_polyhedra!(grh, [Int(first(path))])
        poly = backend_eliminate(
            grh.node_polys[Int(first(path))],
            BitSet((grh.change_qK_idx,));
            canonicalize=false,
            prefer_fastpath=prefer_fastpath,
        )
        return prefer_fastpath ? poly : _clean_polyhedron!(poly)
    end

    poly = backend_intersect_many(grh.edge_polys[Int.(edge_idxs)]; canonicalize=false, prefer_fastpath=prefer_fastpath)
    return prefer_fastpath ? poly : _clean_polyhedron!(poly)
end

function _calc_polyhedra_for_paths_bulk_suffix_dag!(
    grh::SIMOPaths,
    path_idxs::AbstractVector{<:Integer},
)::Vector{Polyhedron}
    path_idxs = Int.(path_idxs)
    isempty(path_idxs) && return Polyhedron[]
    prefer_fastpath = backend_prefers_fastpath(_affine_is_exact(get_binding_network(grh)))

    edge_idxs = unique(vcat(grh.path_edge_idxs[path_idxs]...))
    _ensure_edge_polyhedra!(grh, edge_idxs)

    sink_vertices = unique(Int.(last.(grh.rgm_paths[path_idxs])))
    _ensure_node_polyhedra!(grh, sink_vertices)
    el_dim = BitSet((grh.change_qK_idx,))

    child_of = Int[]
    vertex_of = Int[]
    edge_of = Int[]
    poly_of = Vector{Any}()
    is_calc = Bool[]
    key_to_node = Dict{Tuple{Int,Int},Int}()

    function make_node(child::Int, vertex::Int, edge_idx::Int)
        push!(child_of, child)
        push!(vertex_of, vertex)
        push!(edge_of, edge_idx)
        push!(poly_of, nothing)
        push!(is_calc, false)
        return length(child_of)
    end

    function get_base_node(v::Int)
        return get!(key_to_node, (0, v)) do
            make_node(0, v, 0)
        end
    end

    path_nodes = Vector{Int}(undef, length(path_idxs))
    for (i, path_idx) in enumerate(path_idxs)
        path = grh.rgm_paths[path_idx]
        edge_path = grh.path_edge_idxs[path_idx]
        node = get_base_node(Int(last(path)))
        @inbounds for pos in length(edge_path):-1:1
            u = Int(path[pos])
            edge_idx = Int(edge_path[pos])
            node = get!(key_to_node, (node, u)) do
                make_node(node, u, edge_idx)
            end
        end
        path_nodes[i] = node
    end

    n_nodes = length(child_of)
    depth_of = zeros(Int, n_nodes)
    max_depth = 0
    @inbounds for node in 1:n_nodes
        depth = child_of[node] == 0 ? 0 : depth_of[child_of[node]] + 1
        depth_of[node] = depth
        max_depth = max(max_depth, depth)
    end

    nodes_by_depth = [Int[] for _ in 0:max_depth]
    @inbounds for node in 1:n_nodes
        push!(nodes_by_depth[depth_of[node] + 1], node)
    end

    @info "Start building polyhedra for paths (total: $(length(path_idxs))) via suffix DAG with $(n_nodes) unique suffix states across $(max_depth + 1) layers"
    if prefer_fastpath
        edge_fast = Vector{Any}(undef, length(grh.edge_polys))
        for edge_idx in edge_idxs
            edge_fast[edge_idx] = backend_prepare_fastpath(grh.edge_polys[edge_idx]; prefer_fastpath=true)
        end
        sink_fast = Vector{Any}(undef, length(grh.node_polys))
        for v in sink_vertices
            sink_fast[v] = backend_fast_eliminate(
                backend_prepare_fastpath(grh.node_polys[v]; prefer_fastpath=true),
                el_dim;
                prefer_fastpath=true,
            )
        end

        @showprogress dt=0.1 desc="Building polyhedra via suffix DAG" for depth in 0:max_depth
            layer_nodes = nodes_by_depth[depth + 1]
            isempty(layer_nodes) && continue
            @info "Suffix DAG layer $(depth + 1)/$(max_depth + 1): $(length(layer_nodes)) states"

            Threads.@threads for pos in eachindex(layer_nodes)
                node = layer_nodes[pos]
                poly_of[node] = if child_of[node] == 0
                    sink_fast[vertex_of[node]]
                else
                    backend_fast_intersect(edge_fast[edge_of[node]], poly_of[child_of[node]]; prefer_fastpath=true)
                end
                is_calc[node] = true
            end
        end
        return [backend_from_fastpath(poly_of[node]; canonicalize=false, prefer_fastpath=true) for node in path_nodes]
    end

    @showprogress dt=0.1 desc="Building polyhedra via suffix DAG" for depth in 0:max_depth
        layer_nodes = nodes_by_depth[depth + 1]
        isempty(layer_nodes) && continue
        @info "Suffix DAG layer $(depth + 1)/$(max_depth + 1): $(length(layer_nodes)) states"

        Threads.@threads for pos in eachindex(layer_nodes)
            node = layer_nodes[pos]
            poly = if child_of[node] == 0
                backend_eliminate(
                    grh.node_polys[vertex_of[node]],
                    el_dim;
                    canonicalize=false,
                    prefer_fastpath=false,
                )
            else
                backend_intersect_many(
                    [grh.edge_polys[edge_of[node]], poly_of[child_of[node]]::Polyhedron];
                    canonicalize=false,
                    prefer_fastpath=false,
                )
            end
            poly_of[node] = poly
            is_calc[node] = true
        end
    end

    return [_clean_polyhedron!(poly_of[node]::Polyhedron) for node in path_nodes]
end

function _calc_polyhedra_for_path(
    model::Bnc,
    paths::AbstractVector{<:AbstractVector{<:Integer}},
    change_qK_idx::Integer,
)::Vector{Union{Nothing, Polyhedron}}
    el_dim = BitSet((change_qK_idx,))
    prefer_fastpath = backend_prefers_fastpath(_affine_is_exact(model))

    node_polyhedra = let
        unique_rgms = unique(vcat(paths...))
        dic = Dict{Int,Polyhedron}()
        for r in unique_rgms
            dic[Int(r)] = get_polyhedron(model, r)
        end
        dic
    end

    edges, edge_paths = _build_path_edge_index(paths)
    edge_poly = Vector{Polyhedron}(undef, length(edges))
    @info "Start building polyhedra for edges (total: $(length(edges)))"
    @showprogress Threads.@threads for i in eachindex(edges)
        (u, v) = edges[i]
        edge_poly[i] = if prefer_fastpath
            backend_intersect_eliminate(
                node_polyhedra[u],
                node_polyhedra[v],
                el_dim;
                canonicalize=false,
                prefer_fastpath=prefer_fastpath,
            )
        else
            backend_intersect_eliminate(
                node_polyhedra[u],
                node_polyhedra[v],
                el_dim;
                canonicalize=false,
                prefer_fastpath=false,
            )
        end
    end

    out = Vector{Polyhedron}(undef, length(edge_paths))
    @info "Start building polyhedra for paths (total: $(length(edge_paths)))"
    @showprogress Threads.@threads for i in eachindex(edge_paths)
        if isempty(edge_paths[i])
            out[i] = if prefer_fastpath
                backend_eliminate(
                    node_polyhedra[Int(first(paths[i]))],
                    el_dim;
                    canonicalize=false,
                    prefer_fastpath=true,
                )
            else
                backend_eliminate(
                    node_polyhedra[Int(first(paths[i]))],
                    el_dim;
                    canonicalize=false,
                    prefer_fastpath=false,
                ) |> _clean_polyhedron!
            end
        else
            out[i] = backend_intersect_many(
                edge_poly[edge_paths[i]];
                canonicalize=false,
                prefer_fastpath=false,
            ) |> _clean_polyhedron!
        end
    end
    return out
end

function _calc_polyhedra_for_path(
    model::Bnc,
    path::AbstractVector{<:Integer},
    change_qK,
)::Polyhedron
    change_qK_idx = change_qK isa Integer ? Int(change_qK) : locate_sym_qK(model, change_qK)
    return _calc_polyhedra_for_path(model, [Int.(path)], change_qK_idx)[1]
end

function get_polyhedra(grh::SIMOPaths, pth_idx::Union{AbstractVector,Nothing}=nothing)::Vector{Polyhedron}
    path_idxs = _normalize_simo_path_selection(grh, pth_idx)
    path_idxs_to_calc = _path_indices_to_calculate(grh.path_polys_is_calc, path_idxs)

    if !isempty(path_idxs_to_calc)
        if length(path_idxs_to_calc) == 1
            idx = only(path_idxs_to_calc)
            edge_idxs_to_calc = grh.path_edge_idxs[idx]
            _ensure_edge_polyhedra!(grh, edge_idxs_to_calc)
            grh.path_polys[idx] = _build_path_polyhedron(grh, grh.rgm_paths[idx], edge_idxs_to_calc)
            grh.path_polys_is_calc[idx] = true
        else
            polys = _calc_polyhedra_for_paths_bulk_suffix_dag!(grh, path_idxs_to_calc)
            grh.path_polys[path_idxs_to_calc] .= polys
            grh.path_polys_is_calc[path_idxs_to_calc] .= true
        end
    end

    return grh.path_polys[path_idxs]
end

get_polyhedron(grh::SIMOPaths, pth) = get_polyhedra(grh, [get_idx(grh, pth)])[1]

function _resolve_simo_rebase_mat(
    grh::SIMOPaths;
    rebase_K::Bool=false,
    rebase_mat=nothing,
)
    if !isnothing(rebase_mat)
        @assert !rebase_K "Cannot specify both rebase_K and providing rebase_mat"
        return rebase_mat
    elseif rebase_K
        bn = get_binding_network(grh)
        Q = rebase_mat_lgK(bn.N)
        return blockdiag(spdiagm(fill(Rational(1), bn.d - 1)), Q)
    else
        return nothing
    end
end

function get_volumes(
    grh::SIMOPaths,
    pth_idx::Union{AbstractVector,Nothing}=nothing;
    rebase_K=false,
    rebase_mat=nothing,
    recalculate=false,
    kwargs...,
)
    path_idxs = _normalize_simo_path_selection(grh, pth_idx)
    path_idxs_to_calculate = _path_indices_to_calculate(grh.path_volume_is_calc, path_idxs; recalculate=recalculate)

    if !isempty(path_idxs_to_calculate)
        resolved_rebase_mat = _resolve_simo_rebase_mat(grh; rebase_K=rebase_K, rebase_mat=rebase_mat)
        polys = get_polyhedra(grh, path_idxs_to_calculate)
        rlts = calc_volume(polys; rebase_mat=resolved_rebase_mat, kwargs...)
        for (i, idx) in enumerate(path_idxs_to_calculate)
            grh.path_volume[idx] = rlts[i]
            grh.path_volume_is_calc[idx] = true
        end
    end
    return grh.path_volume[path_idxs]
end

get_volume(grh::SIMOPaths, pth; kwargs...) = get_volumes(grh, [get_idx(grh, pth)]; kwargs...)[1]
