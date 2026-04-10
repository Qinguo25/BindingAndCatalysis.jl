
mutable struct SISOPaths{T} 
    bn::Bnc{T}   # binding Newtork
    qK_grh::SimpleDiGraph # SimpleDiGraph in qK space
    change_qK_idx::T  # which qK is changing in this SISO graph

    sources::Vector{Int}  # source vertices in the graph
    sinks::Vector{Int}    # sink vertices in the graph
    paths_dict::Union{Nothing,Dict{Vector{Int},Int}} # lazily-built map from path to its idx in rgm_paths
    rgm_paths::Vector{Vector{Int}} #All paths from sources to sinks, each path is represented as a vector of vertex idx. Grows exponentially
    path_edge_idxs::Vector{Vector{Int}} # undirected edge ids for each path
    edge_keys::Vector{Tuple{Int,Int}} # global undirected edge pool shared by all paths

    node_polys::Vector{Polyhedron} # vertex polyhedra cache
    node_polys_is_calc::BitVector
    edge_polys::Vector{Polyhedron} # projected edge polyhedra cache
    edge_polys_is_calc::BitVector

    path_polys::Vector{Polyhedron} # the polyhedron for each path, lazily calculated when needed, stored in the same order as rgm_paths
    path_volume::Vector{Volume}# the volume for each path, lazily calculated when needed, stored in the same order as rgm_paths

    path_volume_is_calc::BitVector # whether the volume for each path is calculated, stored in the same order as rgm_paths
    path_polys_is_calc::BitVector # whether the polyhedron for each path is calculated, stored in the same order as rgm_paths
    
     function SISOPaths(model::Bnc{T}, qK_grh, change_qK_idx, sources, sinks, rgm_paths) where T
        edge_keys, path_edge_idxs = _build_path_edge_index(rgm_paths)
        node_polys = Vector{Polyhedron}(undef, n_regimes(model))
        node_polys_is_calc = falses(length(node_polys))
        edge_polys = Vector{Polyhedron}(undef, length(edge_keys))
        edge_polys_is_calc = falses(length(edge_keys))
        path_polys = Vector{Polyhedron}(undef, length(rgm_paths))
        path_volume = Vector{Volume}(undef, length(rgm_paths))
        path_volume_is_calc = falses(length(rgm_paths))
        path_polys_is_calc = falses(length(rgm_paths))
        new{T}(model, qK_grh, change_qK_idx, 
            sources, sinks, 
            nothing,
            rgm_paths, path_edge_idxs, edge_keys,
            node_polys, node_polys_is_calc, edge_polys, edge_polys_is_calc,
            path_polys, path_volume,
            path_volume_is_calc, path_polys_is_calc)
    end
end

function _build_paths_dict(rgm_paths::AbstractVector{<:AbstractVector{<:Integer}})
    paths_dict = Dict{Vector{Int},Int}()
    sizehint!(paths_dict, length(rgm_paths))
    for (i, p) in enumerate(rgm_paths)
        paths_dict[p] = i
    end
    return paths_dict
end

function _ensure_paths_dict!(grh::SISOPaths)
    isnothing(grh.paths_dict) || return grh.paths_dict
    grh.paths_dict = _build_paths_dict(grh.rgm_paths)
    return grh.paths_dict
end

# For SISO path conditions we only need a canonical H-rep, not an LP-minimal one.
# Using the light pass here matches cddlib's "canonicalize after projection" style
# much better than running a full LP-based strong reduction on every path.
_clean_polyhedron!(p::Polyhedron) = (removehredundancy!(p; strong=false); p)

function _build_path_edge_index(rgm_paths::AbstractVector{<:AbstractVector{<:Integer}})
    total_refs = sum(max(length(path) - 1, 0) for path in rgm_paths)
    edge_keys = Tuple{Int,Int}[]
    sizehint!(edge_keys, total_refs)
    edge_dict = Dict{Tuple{Int,Int},Int}()
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

function _ensure_node_polyhedra!(grh::SISOPaths, rgm_idxs::AbstractVector{<:Integer})
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

    if Threads.nthreads() == 1
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

function _ensure_edge_polyhedra!(grh::SISOPaths, edge_idxs::AbstractVector{<:Integer})
    edge_idxs_unique = unique(Int.(edge_idxs))
    edge_idxs_to_calc = filter(edge_idxs_unique) do idx
        !grh.edge_polys_is_calc[idx]
    end
    isempty(edge_idxs_to_calc) && return nothing

    bn = get_binding_network(grh)
    use_cdd = _can_use_cdd_fastpath(_affine_is_exact(bn))
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
        grh.edge_polys[edge_idx] =
            if use_cdd
                cdd_intersect_eliminate(grh.node_polys[u], grh.node_polys[v], el_dim; canonicalize=false)
            else
                p = intersect(grh.node_polys[u], grh.node_polys[v]; canonicalize=false)
                eliminate(p, el_dim; canonicalize=false)
            end
        grh.edge_polys_is_calc[edge_idx] = true
    end

    return nothing
end

function _build_path_polyhedron(
    grh::SISOPaths,
    path::AbstractVector{<:Integer},
    edge_idxs::AbstractVector{<:Integer},
)::Polyhedron
    use_cdd = _can_use_cdd_fastpath(_affine_is_exact(get_binding_network(grh)))
    if isempty(edge_idxs)
        _ensure_node_polyhedra!(grh, [Int(first(path))])
        return if use_cdd
            cdd_eliminate(grh.node_polys[Int(first(path))], BitSet((grh.change_qK_idx,)); canonicalize=false)
        else
            eliminate(grh.node_polys[Int(first(path))], BitSet((grh.change_qK_idx,)); canonicalize=false) |> _clean_polyhedron!
        end
    end
    return if use_cdd
        cdd_intersect_many(grh.edge_polys[Int.(edge_idxs)]; canonicalize=false)
    else
        intersect(grh.edge_polys[Int.(edge_idxs)]...; canonicalize=false) |> _clean_polyhedron!
    end
end

function _calc_polyhedra_for_paths_bulk_suffix_dag!(
    grh::SISOPaths,
    path_idxs::AbstractVector{<:Integer},
)::Vector{Polyhedron}
    path_idxs = Int.(path_idxs)
    isempty(path_idxs) && return Polyhedron[]
    use_cdd = _can_use_cdd_fastpath(_affine_is_exact(get_binding_network(grh)))

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
    sizehint!.(nodes_by_depth, 0)
    @inbounds for node in 1:n_nodes
        push!(nodes_by_depth[depth_of[node] + 1], node)
    end

    @info "Start building polyhedra for paths (total: $(length(path_idxs))) via suffix DAG with $(n_nodes) unique suffix states across $(max_depth + 1) layers"
    if use_cdd
        del_axes = sort!(collect(el_dim))
        edge_cdd = Vector{Any}(undef, length(grh.edge_polys))
        for edge_idx in edge_idxs
            edge_cdd[edge_idx] = CddBridge._native_to_cdd(grh.edge_polys[edge_idx])
        end
        sink_cdd = Vector{Any}(undef, length(grh.node_polys))
        for v in sink_vertices
            sink_cdd[v] = PolyhedraExt.eliminate(CddBridge._native_to_cdd(grh.node_polys[v]), del_axes)
        end

        @showprogress dt=0.1 desc="Building polyhedra via suffix DAG" for depth in 0:max_depth
            layer_nodes = nodes_by_depth[depth + 1]
            isempty(layer_nodes) && continue
            @info "Suffix DAG layer $(depth + 1)/$(max_depth + 1): $(length(layer_nodes)) states"

            Threads.@threads for pos in eachindex(layer_nodes)
                node = layer_nodes[pos]
                poly_of[node] = if child_of[node] == 0
                    sink_cdd[vertex_of[node]]
                else
                    Base.intersect(edge_cdd[edge_of[node]], poly_of[child_of[node]])
                end
                is_calc[node] = true
            end
        end
        return [CddBridge._cdd_to_native(poly_of[node]; canonicalize=false) for node in path_nodes]
    end

    @showprogress dt=0.1 desc="Building polyhedra via suffix DAG" for depth in 0:max_depth
        layer_nodes = nodes_by_depth[depth + 1]
        isempty(layer_nodes) && continue
        @info "Suffix DAG layer $(depth + 1)/$(max_depth + 1): $(length(layer_nodes)) states"

        Threads.@threads for pos in eachindex(layer_nodes)
            node = layer_nodes[pos]
            poly = if child_of[node] == 0
                eliminate(grh.node_polys[vertex_of[node]], el_dim; canonicalize=false)
            else
                intersect(grh.edge_polys[edge_of[node]], poly_of[child_of[node]]::Polyhedron; canonicalize=false)
            end
            poly_of[node] = poly
            is_calc[node] = true
        end
    end

    return [_clean_polyhedron!(poly_of[node]::Polyhedron) for node in path_nodes]
end


"""
    _calc_polyhedra_for_path(model::Bnc, paths, change_qK_idx) -> Vector{Polyhedron}

Compute qK-space polyhedra for each regime path.
"""
function _calc_polyhedra_for_path(
    model::Bnc,
    paths::AbstractVector{<:AbstractVector{<:Integer}},
    change_qK_idx::Integer,
)::Vector{Union{Nothing, Polyhedron}}

    el_dim = BitSet((change_qK_idx,))
    use_cdd = _can_use_cdd_fastpath(_affine_is_exact(model))
    #dict: node: polyhedron 
    node_polyhedra = let
                        unique_rgms = unique(vcat(paths...))
                        dic = Dict{Int,Polyhedron}()
                        for r in unique_rgms
                            pr = get_polyhedron(model, r)
                            dic[Int(r)] = pr        
                        end
                        dic
                    end
    # -------------------------
    # 2) Build unique undirected edges and edge index map
    # key = (min(u,v), max(u,v))
    # -------------------------
    
    #dict: (u,v): edge_idx
    edges, edge_paths = _build_path_edge_index(paths)

    # -------------------------
    # 3) Compute poly for each edge = intersect(poly_of[u], poly_of[v])
    # -------------------------

    edge_poly = let 
        edge_poly = Vector{Polyhedron}(undef, length(edges))
        @info "Start building polyhedra for edges (total: $(length(edges)))"
        @showprogress Threads.@threads  for i in eachindex(edges)
            (u, v) = edges[i]
            edge_poly[i] =
                if use_cdd
                    cdd_intersect_eliminate(node_polyhedra[u], node_polyhedra[v], el_dim; canonicalize=false)
                else
                    p = intersect(node_polyhedra[u], node_polyhedra[v]; canonicalize=false)
                    eliminate(p, el_dim; canonicalize=false)
                end
        end
        edge_poly
    end

    

    out = Vector{Polyhedron}(undef, length(edge_paths))
    @info "Start building polyhedra for paths (total: $(length(edge_paths)))"
    @showprogress Threads.@threads for i in eachindex(edge_paths)
        if isempty(edge_paths[i])
            out[i] = eliminate(node_polyhedra[Int(first(paths[i]))], el_dim; canonicalize=false) |> _clean_polyhedron!
        else
            out[i] = intersect(edge_poly[edge_paths[i]]...; canonicalize=false) |> _clean_polyhedron!
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
"""
    get_neighbor_graph_qK(grh::SISOPaths; kwargs...) -> SimpleDiGraph

Return the qK neighbor graph for a SISO path object.
"""
get_neighbor_graph_qK(grh::SISOPaths; kwargs...) = grh.qK_grh



"""
    get_SISO_graph(grh::SISOPaths) -> SimpleDiGraph

Return the SISO graph stored in a `SISOPaths` object.
"""

get_SISO_graph(grh::SISOPaths) = grh.qK_grh
"""
    get_SISO_graph(model::Bnc, change_qK) -> SimpleDiGraph

Return a SISO graph for a chosen qK coordinate.
"""
get_SISO_graph(model::Bnc, change_qK) = get_SISO_graph(get_regimes_graph!(model; full=true), change_qK)
"""
    get_SISO_graph(grh::VertexGraph, change_qK) -> SimpleDiGraph

Build a SISO graph from a vertex graph for a chosen qK coordinate.
"""
function get_SISO_graph(grh::VertexGraph, change_qK)::SimpleDiGraph
    bn = get_binding_network(grh)
    change_qK_idx = locate_sym_qK(bn, change_qK)
    _ensure_full_regimes_graph!(grh)

    n = length(grh.neighbors)

    g = let 
        g = SimpleDiGraph(n)
        for (i, edges) in enumerate(grh.neighbors)
            nlt = get_nullity(bn,i)
            if nlt >1
                continue
            end
            for e in edges
                if !_edge_has_qK_interface(e) || e.to < i
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
        g
    end

    return g
end



#------------------------------------------------------------------------------
# Higher wrapper for regime graph paths
#------------------------------------------------------------------------------------------

"""
    SISOPaths(model::Bnc, change_qK; rgm_paths=nothing) -> SISOPaths

Construct a `SISOPaths` object for a chosen qK coordinate.
"""
function SISOPaths(model::Bnc{T}, change_qK; rgm_paths=nothing) where {T}
    change_qK_idx = locate_sym_qK(model, change_qK)

    if rgm_paths === nothing
        qK_grh = get_SISO_graph(model, change_qK)
        sources, sinks = get_sources_sinks(model, qK_grh)
        rgm_paths = _enumerate_paths(qK_grh; sources, sinks)
    else
        qK_grh = graph_from_paths(rgm_paths, n_regimes(model))
        sources, sinks = get_sources_sinks(qK_grh)
    end

    return SISOPaths(model, qK_grh, change_qK_idx, sources, sinks, rgm_paths)
end

"""
    get_path(grh::SISOPaths, pth_idx; return_idx=false) -> Vector

Return a path by index, optionally as vertex indices.
"""
function get_path(grh::SISOPaths, pth_idx::Integer; return_idx::Bool=false)
    rgm_idxs = grh.rgm_paths[pth_idx]
    if return_idx
        return rgm_idxs
    else
        bn = get_binding_network(grh)
        return get_perm.(Ref(bn), rgm_idxs)
    end
    return perms
end
"""
    get_path(grh::SISOPaths, pth::AbstractVector; return_idx=false) -> Vector

Normalize a path representation to indices or permutations.
"""
function get_path(grh::SISOPaths, pth::AbstractVector; return_idx::Bool=false)
    bn = get_binding_network(grh)
    return return_idx ? get_idx.(Ref(bn), pth) : get_perm.(Ref(bn), pth)
end

"""
    get_binding_network(grh::SISOPaths, args...) -> Bnc

Return the model backing a SISO path object.
"""
get_binding_network(grh::SISOPaths,args...)= grh.bn
"""
    get_C_C0_nullity_qK(grh::SISOPaths, pth_idx) -> (Matrix, Vector, Int)

Return constraints for a SISO path polyhedron.
"""
get_C_C0_nullity_qK(grh::SISOPaths, pth_idx) = get_polyhedron(grh, pth_idx) |> get_C_C0_nullity



"""
    get_idx(grh::SISOPaths, pth) -> Int

Return the index for a SISO path specification.
"""
get_idx(grh::SISOPaths, pth::AbstractVector) = let
    bn = get_binding_network(grh)
    idxs = get_idx.(Ref(bn), pth)
    _ensure_paths_dict!(grh)[idxs] 
end
"""
    get_idx(grh::SISOPaths, pth::Integer) -> Int

Return the provided path index.
"""
get_idx(grh::SISOPaths, pth::Integer) = pth





"""
    get_polyhedra(grh::SISOPaths, pth_idx=nothing) -> Vector{Polyhedron}

Return polyhedra for selected SISO paths.
"""
function get_polyhedra(grh::SISOPaths, pth_idx::Union{AbstractVector,Nothing} = nothing)::Vector{Polyhedron}
    pth_idx = let 
            if isnothing(pth_idx)
                1:length(grh.rgm_paths)
            else
                get_idx.(Ref(grh), pth_idx)
            end
        end
    
    pth_poly_to_calc = filter(x -> !grh.path_polys_is_calc[x], pth_idx)
    
    if !isempty(pth_poly_to_calc)
        if length(pth_poly_to_calc) == 1
            idx = only(pth_poly_to_calc)
            edge_idxs_to_calc = grh.path_edge_idxs[idx]
            _ensure_edge_polyhedra!(grh, edge_idxs_to_calc)
            grh.path_polys[idx] = _build_path_polyhedron(grh, grh.rgm_paths[idx], edge_idxs_to_calc)
            grh.path_polys_is_calc[idx] = true
        else
            polys = _calc_polyhedra_for_paths_bulk_suffix_dag!(grh, pth_poly_to_calc)
            grh.path_polys[pth_poly_to_calc] .= polys
            grh.path_polys_is_calc[pth_poly_to_calc] .= true
        end
    end

    return grh.path_polys[pth_idx]
end
"""
    get_polyhedron(grh::SISOPaths, pth) -> Polyhedron

Return the polyhedron for a single SISO path.
"""
get_polyhedron(grh::SISOPaths, pth)= get_polyhedra(grh, [get_idx(grh, pth)])[1]



"""
    get_volumes(grh::SISOPaths, pth_idx=nothing; asymptotic=true, recalculate=false, kwargs...) -> Vector{Volume}

Compute volumes for SISO paths.
"""
function get_volumes(grh::SISOPaths, pth_idx::Union{AbstractVector,Nothing}=nothing; 
    rebase_K = false,
    rebase_mat = nothing,
    recalculate=false, kwargs...)

    pth_idx = let 
            if isnothing(pth_idx)
                1:length(grh.rgm_paths)
            else
                get_idx.(Ref(grh), pth_idx)
            end
        end
    
    idxes_to_calculate = recalculate ? pth_idx : filter(x -> !grh.path_volume_is_calc[x], pth_idx)
    
    if !isempty(idxes_to_calculate)

        rebase_mat = if  !isnothing(rebase_mat)
                    @assert !rebase_K "Cannot specify both rebase_K and providing rebase_mat"
                    rebase_mat
                elseif rebase_K
                    Bnc = get_binding_network(grh) 
                    Q = rebase_mat_lgK(Bnc.N)
                    blockdiag(spdiagm(fill(Rational(1), Bnc.d-1)), Q)
                else
                    nothing
                end

        polys = get_polyhedra(grh, idxes_to_calculate)

        rlts = calc_volume(polys; rebase_mat=rebase_mat, kwargs...)
        for (i, idx) in enumerate(idxes_to_calculate)
            grh.path_volume[idx] = rlts[i]
            grh.path_volume_is_calc[idx] = true
        end
    end
    return grh.path_volume[pth_idx]
end

"""
    get_volume(grh::SISOPaths, pth; kwargs...) -> Volume

Return the volume for a single SISO path.
"""
get_volume(grh::SISOPaths, pth; kwargs...) = get_volumes(grh, [get_idx(grh, pth)]; kwargs...)[1]



#-------------------------------------------------------------------------------------
# Regime shifting associated functions
#-------------------------------------------------------------------------------------

"""
    show_regime_path(grh::SISOPaths, pth) -> nothing

Print a formatted regime path with optional volume.
"""
function show_regime_path(grh::SISOPaths, pth)
    pth_idx = get_idx(grh, pth)
    pth = get_path(grh, pth_idx; return_idx=true)
    vol_is_calc = grh.path_volume_is_calc[pth_idx]
    volume = vol_is_calc ? grh.path_volume[pth_idx] : nothing
    print_path(pth; prefix="#",id = pth_idx,volume=volume)
    return nothing
end


"""
    get_expression_path(grh::SISOPaths, pth; observe_x=nothing) -> (Vector, Vector)

Return expression coefficients and interfaces along a SISO path.
"""
function get_expression_path(grh::SISOPaths, pth; observe_x=nothing)
    
    bn = get_binding_network(grh)
    rgm_pth = get_path(grh, pth; return_idx=true)
    # @show rgm_pth
    rgm_nlt = get_nullities(bn, rgm_pth)
    
    change_qK_idx = grh.change_qK_idx
    observe_x_idx = isnothing(observe_x) ? (1:bn.n) : locate_sym_x.(Ref(bn), observe_x)
    
    rgm_interface = get_interface.(Ref(bn),rgm_pth[1:end-1], rgm_pth[2:end])
    
    H_H0 = Vector{Any}(undef, length(rgm_pth))
    for i in eachindex(rgm_pth)
        rgm = rgm_pth[i]
        nlt = rgm_nlt[i]
        if nlt == 0 # for non-singular regime, we care about the expression, tells by the H[i，：]
            H,H0 = get_H_H0(bn, rgm)
            # @show H,H0, observe_x_idx
            H_H0[i] = (H[observe_x_idx, :], H0[observe_x_idx]) 
        elseif nlt == 1 # for singular regime, we care about the contiuity, tells by the H[i,j]
            H = get_H(bn,rgm)
            H_H0[i] = (H[observe_x_idx, change_qK_idx], nothing)
        else
            error("Nullity > 1 is not supported for expression path.") # should ne change if under constrain.
        end
    end
    return H_H0, rgm_interface
end



#-------------------------------------------------------------------------------------------
# 
"""
    _calc_RO_for_single_path(model, path, change_qK_idx, observe_x_idx) -> Vector

Compute the reaction-order profile along a single path.
"""
function _calc_RO_for_single_path(model, path::AbstractVector{<:Integer}, change_qK_idx, observe_x_idx)::Vector{<:Real}
    r_ord = Vector{Float64}(undef, length(path))
    for i in eachindex(path)
        if !is_singular(model, path[i])
            r_ord[i] = round(Float64(get_H(model, path[i])[observe_x_idx, change_qK_idx]); digits=3)
        else
            ord = get_H(model, path[i])[observe_x_idx, change_qK_idx]
            if abs(ord) < 1e-6
                r_ord[i] = NaN  # We use NaN to denote continuous singular, if reaction order not same before and after, means discontinuity
            else 
                r_ord[i] = Float64(ord) * Inf
            end     
        end
    end
    return r_ord
end
"""
    _dedup(ord_path) -> Vector

Deduplicate consecutive reaction-order values while preserving discontinuities.
"""
function _dedup(ord_path::AbstractVector{T})::Vector{T} where T<:Real
    isempty(ord_path) && return T[]
    out = T[ord_path[1]]
    pending_nan = false
    last_out = out[1]  
    @assert !isnan(last_out) "The first element cannot be NaN for deduplication."

    for x in @view ord_path[2:end]
        if isnan(x)
            pending_nan = true
            continue
        end
        if x != last_out
            if pending_nan
                push!(out, NaN)
                pending_nan = false
            end
            push!(out, x)
            last_out = x
        else
            pending_nan = false
        end
    end
    return out
end





"""
    get_RO_path(model::Bnc, rgm_idx_shift_pth; change_qK, observe_x,
        deduplicate=false, keep_singular=true, keep_nonasymptotic=true) -> Vector

Calculate the reaction-order profile for a single regime path.
"""
function get_RO_path(
    model::Bnc,rgm_idx_shift_pth::AbstractVector; 
    change_qK, observe_x,
    
    deduplicate::Bool=false,
    keep_singular::Bool=true,
    keep_nonasymptotic::Bool=true
    )::Vector{<:Real}

    
    # get reaction order along the path
    rgm_idx_shift_pth = get_idx.(Ref(model), rgm_idx_shift_pth)

    ord_path = let 
        change_qK_idx = locate_sym_qK(model, change_qK)
        observe_x_idx = locate_sym_x(model, observe_x)
        _calc_RO_for_single_path(model, rgm_idx_shift_pth, change_qK_idx, observe_x_idx)
    end
    

    # apply the regime filter
    mask = _get_mask(model, rgm_idx_shift_pth;
        singular=keep_singular ? nothing : false,
        asymptotic=keep_nonasymptotic ? nothing : true)
    
    ord_path = ord_path[mask]

    # remove redundency
    if deduplicate
        ord_path = _dedup(ord_path)
    end

    return ord_path
end




function _ensure_ro_regimes_materialized!(
    model::Bnc,
    rgm_idx_for_each_paths::AbstractVector{<:AbstractVector{<:Integer}},
)
    seen = Set{Int}()
    ordered_idxs = Int[]

    for path in rgm_idx_for_each_paths
        for idx in path
            idx = Int(idx)
            if !(idx in seen)
                push!(ordered_idxs, idx)
                push!(seen, idx)
            end
        end
    end

    for idx in ordered_idxs
        get_regime(model, idx; inv_info=true)
    end

    return nothing
end



"""
    get_RO_paths(model::Bnc, rgm_paths, args...; kwargs...) -> Vector{Vector}

Calculate reaction-order profiles for multiple regime paths.
"""
function get_RO_paths(model::Bnc, rgm_paths::AbstractVector{<:AbstractVector}, args...; kwargs...)::Vector{Vector{<:Real}}
    
    rgm_idx_for_each_paths = rgm_paths .|> x -> get_idx.(Ref(model), x)
    # Different paths may share the same regime. Pre-materialize once so the
    # threaded loop below only reads cached affine/qK data.
    _ensure_ro_regimes_materialized!(model, rgm_idx_for_each_paths)

    ord_for_each_paths = Vector{Vector{<:Real}}(undef, length(rgm_idx_for_each_paths))
    Threads.@threads for i in eachindex(rgm_idx_for_each_paths)
        ord_for_each_paths[i] = get_RO_path(model, rgm_idx_for_each_paths[i], args...; kwargs...)
    end
    return ord_for_each_paths
end
"""
    get_RO_paths(model::SISOPaths, pth_idx=nothing; observe_x, kwargs...) -> Vector{Vector}

Calculate reaction-order profiles for paths in a `SISOPaths` object.
"""
function get_RO_paths(model::SISOPaths, pth_idx::Union{Nothing, AbstractVector}=nothing ; observe_x, kwargs...)
    rgm_paths = isnothing(pth_idx) ? model.rgm_paths : get_path.(Ref(model), pth_idx; return_idx=true)
    observe_x_idx = locate_sym_x(model.bn, observe_x)
    return get_RO_paths(model.bn, rgm_paths; 
        change_qK=model.change_qK_idx, observe_x=observe_x_idx, kwargs...)
end
"""
    get_RO_path(model::SISOPaths, pth_idx, args...; kwargs...) -> Vector

Single-path wrapper for `get_RO_paths`.
"""
get_RO_path(model::SISOPaths, pth_idx, args...; kwargs...) = get_RO_paths(model, [get_idx(model,pth_idx)], args... ; kwargs...)[1]



"""
    summary(grh::SISOPaths; show_volume=true, prefix="#", kwargs...) -> nothing

Print the paths stored in `SISOPaths`, optionally with volumes.
"""
function summary(grh::SISOPaths; show_volume::Bool=true, prefix::AbstractString="#", kwargs...)
    paths = grh.rgm_paths
    if show_volume
        vols = get_volumes(grh; kwargs...)
        print_paths(paths; prefix=prefix, volumes = vols, ids = 1:length(paths))
    else
        print_paths(paths; prefix=prefix, ids = 1:length(paths))
    end
    return nothing
end



"""
    summary_RO_path(grh::SISOPaths; observe_x, show_volume=true, deduplicate=true,
        keep_singular=true, keep_nonasymptotic=true, kwargs...) -> nothing

Summarize reaction-order paths grouped by profile.
"""
function summary_RO_path(grh::SISOPaths;observe_x, show_volume::Bool=true,

    deduplicate::Bool=true,keep_singular::Bool=true,keep_nonasymptotic::Bool=true,kwargs...)

    ord_pth = get_RO_paths(grh; observe_x=observe_x, 
        deduplicate=deduplicate,
        keep_singular=keep_singular,
        keep_nonasymptotic=keep_nonasymptotic)

    volumes = if show_volume
        get_volumes(grh; kwargs...)
    else
        fill(nothing, length(grh.rgm_paths))
    end



    rsts = group_sum(ord_pth, volumes)
    # for (id, pth, volume) in rsts
    #      print_path(pth; prefix="",id=id, volume=volume)
    # end

    # print 
    ids = getindex.(rsts, 1)
    ords = getindex.(rsts, 2)
    vols = getindex.(rsts, 3)
    print_paths(ords; prefix="", ids=ids, volumes=vols)
    return nothing
end

function Base.display(grh::SISOPaths)
    println("SISOPaths object with $(length(grh.rgm_paths)) paths for qK coordinate index $(grh.change_qK_idx)")
    return nothing
end
Base.show(io::IO, grh::SISOPaths) = print(io, "SISOPaths object with $(length(grh.rgm_paths)) paths for qK coordinate index $(grh.change_qK_idx)")
