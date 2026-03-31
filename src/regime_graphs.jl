export get_regimes_graph!, SISOPaths, get_polyhedra, get_polyhedron, get_SISO_graph
export get_path, get_edge, get_intersect
export get_neighbor_graph_x, get_neighbor_graph_qK, get_neighbor_graph
export get_sources, get_sinks, get_sources_sinks
export get_RO_path, group_sum, get_RO_paths, summary_RO_path
export get_volume

#---------------------------------------------------------------------------------------------------
#             Helper functions: Functions for construct the regime graph paths
#----------------------------------------------------------------------------------------------------


"""
    get_sources(g::AbstractGraph) -> Set{Int}

Return source vertices with zero indegree.
"""
get_sources(g::AbstractGraph) = Set(v for v in vertices(g) if indegree(g, v) == 0)
"""
    get_sinks(g::AbstractGraph) -> Set{Int}

Return sink vertices with zero outdegree.
"""
get_sinks(g::AbstractGraph)   = Set(v for v in vertices(g) if outdegree(g, v) == 0)
"""
    get_sources_sinks(g::AbstractGraph) -> (Set{Int}, Set{Int})

Return sources and sinks for a graph.
"""
get_sources_sinks(g::AbstractGraph) = (get_sources(g), get_sinks(g))

"""
    get_sources_sinks(model::Bnc, g::AbstractGraph) -> (Vector{Int}, Vector{Int})

Return sources and sinks while excluding singular regimes.
"""
function get_sources_sinks(model::Bnc, g::AbstractGraph)
    sources_all = get_sources(g) 
    sinks_all   = get_sinks(g) 
    common_vs = intersect(sources_all, sinks_all)
    filter!(common_vs) do v
        get_nullity(model, v) > 0
    end
    sources = setdiff(sources_all, common_vs)
    sinks = setdiff(sinks_all, common_vs)
    return (collect(sources), collect(sinks))
end

# 只遍历子图：sources 可达 & 能到 sinks
"""
    _reachable_from_sources(g::AbstractGraph, sources) -> Vector{Bool}

Return a boolean mask of vertices reachable from sources.
"""
function _reachable_from_sources(g::AbstractGraph, sources::AbstractVector{Int})
    n = nv(g)
    seen = falses(n)
    stack = Int[]
    for s in sources
        if !seen[s]
            seen[s] = true
            push!(stack, s)
            while !isempty(stack)
                v = pop!(stack)
                for nb in outneighbors(g, v)
                    if !seen[nb]
                        seen[nb] = true
                        push!(stack, nb)
                    end
                end
            end
        end
    end
    return seen
end

"""
    _can_reach_sinks(g::AbstractGraph, sinks) -> Vector{Bool}

Return a boolean mask of vertices that can reach sinks.
"""
function _can_reach_sinks(g::AbstractGraph, sinks::AbstractVector{Int})
    n = nv(g)
    seen = falses(n)
    stack = Int[]
    for t in sinks
        if !seen[t]
            seen[t] = true
            push!(stack, t)
            while !isempty(stack)
                v = pop!(stack)
                for nb in inneighbors(g, v)   # 反向走
                    if !seen[nb]
                        seen[nb] = true
                        push!(stack, nb)
                    end
                end
            end
        end
    end
    return seen
end

"""
    _enumerate_paths(g; sources, sinks) -> Vector{Vector{Int}}

Enumerate all paths in a DAG from `sources` to `sinks`.
"""
function _enumerate_paths(
    g::AbstractGraph;
    sources::AbstractVector{Int},
    sinks::AbstractVector{Int},
)::Vector{Vector{Int}}

    @info "sources: $sources"
    @info "sinks: $sinks"
    n = nv(g)

    # 剪枝：只处理相关子图
    fromS = _reachable_from_sources(g, sources)
    toT   = _can_reach_sinks(g, sinks)
    active = fromS .& toT

    is_sink = falses(n)
    @inbounds for t in sinks
        is_sink[t] = true
    end

    # 拓扑排序（DAG）
    topo = topological_sort_by_dfs(g)   # Graphs.jl
    # memo[v] = Vector{Vector{Int}} 或 nothing
    memo = Vector{Union{Nothing, Vector{Vector{Int}}}}(undef, n)
    fill!(memo, nothing)

    @info "Start enumerating paths from sources to sinks. This may take a while if there are many paths."
    # 逆拓扑：先算子节点，再算父节点

    @info "Total vertices to process in topological order: $(length(topo))"
    @showprogress for v in Iterators.reverse(topo)
        active[v] || continue

        if is_sink[v]
            memo[v] = Vector{Vector{Int}}(undef, 1)
            memo[v][1] = [v]
            continue
        end

        # 收集所有 nb 的路径，并在前面加 v
        acc = Vector{Vector{Int}}()
        # 你也可以在这里做 sizehint!（需要先统计 path 数量，会多一次循环；看你取舍）
        for nb in outneighbors(g, v)
            active[nb] || continue
            paths_nb = memo[nb]
            paths_nb === nothing && continue
            for p in paths_nb
                L = length(p)
                np = Vector{Int}(undef, L + 1)
                np[1] = v
                @inbounds copyto!(np, 2, p, 1, L)
                push!(acc, np)
            end
        end

        memo[v] = isempty(acc) ? nothing : acc
    end

    # 汇总 sources 的结果
    @info "Finished enumerating paths. Now collecting paths from sources. Total sources: $(length(sources))"
    out = Vector{Vector{Int}}()
    @showprogress for s in sources
        active[s] || continue
        ps = memo[s]
        ps === nothing && continue
        append!(out, ps)
    end

    sort!(out)
    return out
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

    clean!(p::Polyhedron) = (detecthlinearity!(p); removehredundancy!(p); p)
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
    (edges, edge_dict) = let
        edges = Tuple{Int,Int}[]
        edge_dict = Dict{Tuple{Int,Int},Int}()
        for path in paths
            n = length(path)
            @inbounds for i in 1:(n-1)
                u = Int(path[i]); v = Int(path[i+1])
                a, b = u < v ? (u, v) : (v, u)
                k = (a, b)
                if !haskey(edge_dict, k)
                    push!(edges, k)
                    edge_dict[k] = length(edges)
                end
            end
        end
        (edges, edge_dict)
    end

    # -------------------------
    # 3) Compute poly for each edge = intersect(poly_of[u], poly_of[v])
    # -------------------------

    edge_poly = let 
        edge_poly = Vector{Polyhedron}(undef, length(edge_dict))
        @info "Start building polyhedra for edges (total: $(length(edge_dict)))"
        @showprogress Threads.@threads  for i in eachindex(edges)
            (u, v) = edges[i]
            p = intersect(node_polyhedra[u], node_polyhedra[v])
            edge_poly[i] = eliminate(p, el_dim)
        end
        edge_poly
    end


    edge_paths = let 
        function path_to_edge_idxs(path)
            n = length(path)
            idxs = Vector{Int}(undef, n-1)
            @inbounds for i in 1:(n-1)
                u = Int(path[i]); v = Int(path[i+1])
                a, b = u < v ? (u, v) : (v, u)
                idxs[i] = edge_dict[(a, b)]
            end
            return idxs
        end
        path_to_edge_idxs.(paths)
    end 

    

    out = Vector{Polyhedron}(undef, length(edge_paths))
    @info "Start building polyhedra for paths (total: $(length(edge_paths)))"
    @showprogress Threads.@threads for i in eachindex(edge_paths)
        out[i] = intersect(edge_poly[edge_paths[i]]...) |> clean!
    end
    return out
end
"""
    Polyhedra.intersect(p::Polyhedron) -> Polyhedron

Identity overload for single-polyhedron intersections.
"""
Polyhedra.intersect(p::Polyhedron)= p # a fix for above function for if only one edge, no need to intersect


"""
    _ensure_full_regimes_graph!(grh::VertexGraph) -> nothing

Ensure qK change directions are computed for a vertex graph.
"""
function _ensure_full_regimes_graph!(grh::VertexGraph)
    if !grh.change_dir_qK_computed
        @info "Calculating vertices neighbor graph with qK change dir"
        _fulfill_regimes_graph!(grh)
        grh.change_dir_qK_computed = true
    end
    return nothing
end

_ensure_full_regimes_graph!(model::Bnc) = _ensure_full_regimes_graph!(get_regimes_graph!(model; full=false))




#---------------------------------------------------------------------------
#              Binding Network Graph
#-------------------------------------------------------------------------
"""
    get_binding_network_grh(bnc::Bnc) -> SimpleGraph

Build the bipartite binding network graph between q and x symbols.
"""
function get_binding_network_grh(Bnc::Bnc)::SimpleGraph
    g = SimpleGraph(Bnc.d + Bnc.n)
    for vi in eachindex(Bnc._valid_L_idx)
        for vj in Bnc._valid_L_idx[vi]
            add_edge!(g, vi, vj+Bnc.d)
        end
    end
    return g # get first d nodes as total, last n nodes as x
end




#------------------------------------------------------------------------------
#                  Getting the Graph of of regimes
#----------------------------------------------------------------------------
"""
    get_regimes_graph!(bnc::Bnc; full=false) -> VertexGraph

Ensure the vertex graph is built; when `full=true`, also compute qK change directions.
"""
function get_regimes_graph!(Bnc::Bnc; full::Bool=false)::VertexGraph

    if full
        vtx_graph = get_regimes_graph!(Bnc; full=false)
        _ensure_full_regimes_graph!(vtx_graph)
    else
        if isnothing(Bnc.vertices_graph)
            find_all_regimes!(Bnc)
        end
    end

    return Bnc.vertices_graph
end


"""
    get_edge(grh::VertexGraph, from, to; full=false) -> Union{Nothing, VertexEdge}

Return the edge between two vertices, optionally computing qK directions.
"""
function get_edge(grh::VertexGraph, from, to; kwargs...)::Union{Nothing, VertexEdge}
    
    from = get_idx(get_binding_network(grh), from)
    to = get_idx(get_binding_network(grh), to)
    
    # if full
    #     _ensure_full_regimes_graph!(grh)
    # end
    pos = get(grh.edge_pos[from], to, nothing)
    if pos === nothing
        return nothing
    end
    edge = grh.neighbors[from][pos]
    # full && _materialize_edge_qK_interface!(grh, edge)
    return edge
end


"""
    get_edge(bnc, from, to; kwargs...) -> Union{Nothing, VertexEdge}

Convenience wrapper to fetch an edge from a model.
"""
get_edge(Bnc, from, to; kwargs...)= let
    vtx_grh = get_regimes_graph!(Bnc; full=false)
    bn = get_binding_network(Bnc)
    from = get_idx(Bnc, from)
    to = get_idx(Bnc, to)
    get_edge(vtx_grh, from, to; kwargs...)
end

"""
    get_binding_network(grh::VertexGraph, args...) -> Bnc

Return the model backing a vertex graph.
"""
get_binding_network(grh::VertexGraph,args...) = grh.bn
# get_regimes_graph!(grh::VertexGraph,args...; kwargs...) = grh






#-----------------------------------------------------------------------------------
"""
    get_neighbor_graph_x(grh::VertexGraph) -> SimpleGraph

Return the x-space neighbor graph for a vertex graph.
"""
get_neighbor_graph_x(grh::VertexGraph) = grh.x_grh
"""
    get_neighbor_graph_x(bnc::Bnc) -> SimpleGraph

Return the x-space neighbor graph for a model.
"""
get_neighbor_graph_x(Bnc::Bnc) = get_neighbor_graph_x(get_regimes_graph!(Bnc; full=false))

"""
    get_neighbor_graph_qK(grh::VertexGraph; both_side=false) -> SimpleDiGraph

Return the qK-space neighbor graph for a vertex graph.
"""
get_neighbor_graph_qK(grh::VertexGraph; both_side::Bool=false)::SimpleDiGraph = let
    _ensure_full_regimes_graph!(grh)

    qK_grh = let # construct the qK_graph
        Bnc = get_binding_network(grh)
        n = length(grh.neighbors)
        g = SimpleDiGraph(n)
        for (i, edges) in enumerate(grh.neighbors)
            if get_nullity(Bnc,i) >1
                continue
            end
            for e in edges
                if !_edge_has_qK_interface(e) || (!both_side && e.to < i)
                    continue
                end
                add_edge!(g, i, e.to)
            end
        end
        g
    end

    return qK_grh
end
"""
    get_neighbor_graph_qK(bnc::Bnc; kwargs...) -> SimpleDiGraph

Return the qK neighbor graph for a model.
"""
get_neighbor_graph_qK(Bnc::Bnc; kwargs...) = get_neighbor_graph_qK(get_regimes_graph!(Bnc; full=true); kwargs...)
"""
    get_neighbor_graph_qK(grh::SISOPaths; kwargs...) -> SimpleDiGraph

Return the qK neighbor graph for a SISO path object.
"""
get_neighbor_graph_qK(grh::SISOPaths; kwargs...) = grh.qK_grh
"""
    get_neighbor_graph(args...; kwargs...) -> SimpleDiGraph

Alias for `get_neighbor_graph_qK`.
"""
get_neighbor_graph(args...; kwargs...) = get_neighbor_graph_qK(args...; kwargs...)



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
    grh.paths_dict[idxs] 
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
        polys = _calc_polyhedra_for_path(get_binding_network(grh), grh.rgm_paths[pth_poly_to_calc], grh.change_qK_idx)
        grh.path_polys[pth_poly_to_calc] .= polys
        grh.path_polys_is_calc[pth_poly_to_calc] .= true
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

"""
    get_RO_paths(model::Bnc, rgm_paths, args...; kwargs...) -> Vector{Vector}

Calculate reaction-order profiles for multiple regime paths.
"""
function get_RO_paths(model::Bnc, rgm_paths::AbstractVector{<:AbstractVector}, args...; kwargs...)::Vector{Vector{<:Real}}
    
    rgm_idx_for_each_paths = rgm_paths .|> x -> get_idx.(Ref(model), x)

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
    group_sum(keys, vals; sort_values=true) -> Vector{Tuple}

Group values by keys, returning indices, key, and summed values.
"""
function group_sum(keys::AbstractVector{I}, vals::AbstractVector{J}; 
    sort_values::Bool=true
    ) :: Vector{Tuple{Vector{Int}, I, J}} where {I,J}

    @assert length(keys) == length(vals)
    # Dictionary to accumulate sum of values for each key
    dict = Dict{I,J}()
    # Store indices of keys for later reference
    index_dict = Dict{I, Vector{Int}}()
    
    @inbounds for (i, (k, v)) in enumerate(zip(keys, vals))
        dict[k] = get(dict, k, zero(v)) + v
        push!(get!(index_dict, k, Int[]), i)  # Store the index
    end
    
    # Collect and sort if needed
    dict_vec = collect(dict)
    
    if sort_values
        # Sort by values (sum of vals)
        sort!(dict_vec, by=x->x[2], rev=true)
    end
    
    # Create a Vector of Tuples with (index, key, summed value)
    result = Vector{Tuple{Vector{Int}, I, J}}(undef, length(dict))
    
    # @show dict, index_dict
    for i in eachindex(dict_vec)
        key, sum_val = dict_vec[i]
        group = index_dict[key]
        result[i] = (group, key, sum_val)
    end
    
    return result
end

function group_sum(
    keys::AbstractVector{I},
    vals::AbstractVector{Nothing};
    sort_values::Bool=true,
)::Vector{Tuple{Vector{Int}, I, Nothing}} where {I}

    @assert length(keys) == length(vals)

    index_dict = Dict{I, Vector{Int}}()
    order = I[]

    @inbounds for (i, k) in enumerate(keys)
        if !haskey(index_dict, k)
            push!(order, k)
        end
        push!(get!(index_dict, k, Int[]), i)
    end

    if sort_values
        sort!(order, by = k -> length(index_dict[k]), rev = true)
    end

    return [(index_dict[k], k, nothing) for k in order]
end



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
