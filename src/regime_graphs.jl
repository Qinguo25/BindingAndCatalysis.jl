#-----------------------------------------------------------------------------------------------
#This is graph associated functions for Bnc models and archetyple behaviors associated code
#-----------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------
#              Binding Newtork Graph
#-------------------------------------------------------------------------
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
#              Functions try makeing Graphs.jl functions on VertexGraph
#------------------------------------------------------------------------------

# We need these whole set of functions to make VertexGraph compatible with Graphs.jl,
# But I'm tired, curretly recreate graph is accessable, 
# edgetype
# has_edge
# inneighbors
# ne
# nv
# outneighbors
# vertices
# is_directed

# function nv(g::VertexGraph)
#     return length(g.neighbors)
# end
# function ne(g::VertexGraph)
#     return g.ne_for_current_change_idx
# end

# function edges(g::VertexGraph)
#     if g.current_change_idx == -1
#         return Iterators.flatten(
#             ( (Edge(i, e.to) for e in es if !isnothing(e.change_dir_qK))
#               for (i, es) in enumerate(g.neighbors) )
#         )
#     else
#         idx = g.current_change_idx
#         return Iterators.flatten(
#             ( (Edge(i, e.to) for e in es
#                 if !isnothing(e.change_dir_qK) && e.change_dir_qK[idx] ≥ 1e-6)
#               for (i, es) in enumerate(g.neighbors) )
#         )
#     end
# end

# function vertices(g::VertexGraph)
#     return 1:nv(g)
# end



#------------------------------------------------------------------------------
#                  Getting the whole graph functions. 
#----------------------------------------------------------------------------
"""
    get_vertices_graph!(Bnc; full=false) -> VertexGraph

    full::Bool=false: whether to compute qK change directions on edges.

Ensure vertex graph is built; if full=true, also compute qK change directions on edges.
Returns the cached VertexGraph.
"""
function get_vertices_graph!(Bnc::Bnc; full::Bool=false)::VertexGraph
    """
    get the neighbor of vertices formed graph.
    """
    if full
        vtx_graph = get_vertices_graph!(Bnc; full=false)
        if !vtx_graph.change_dir_qK_computed
            println("-------Start calculating vertices neighbor graph with qK change dir, It may takes a while.------------")
            _fulfill_vertices_graph!(Bnc, vtx_graph)
            vtx_graph.change_dir_qK_computed = true
            println("Done.\n")
        end
    else
        if isnothing(Bnc.vertices_graph)
            find_all_vertices!(Bnc)# Ensure vertices are calculated
            println("----------------Start calculating vertices neighbor graph, It may takes a while.----------------")
            Bnc.vertices_graph =  _calc_vertices_graph_from_perms(Bnc.vertices_perm,Bnc.n)
            println("Done.\n")
        end
    end
    return Bnc.vertices_graph
end

#-----------------------------------------------------------------------------------
function get_neighbor_graph_x(Bnc::Bnc)::SimpleGraph
    vg = get_vertices_graph!(Bnc;full=false)
    return vg.x_grh
end

function get_neighbor_graph_qK(Bnc::Bnc; half::Bool=true)::SimpleGraph
    vg = get_vertices_graph!(Bnc;full=true)
    n = length(vg.neighbors)
    g = SimpleDiGraph(n)
    for (i, edges) in enumerate(vg.neighbors)
        if get_nullity(Bnc,i) >1
            continue
        end
        for e in edges
            if isnothing(e.change_dir_qK) || (half && e.to < i)
                continue
            end
            add_edge!(g, i, e.to)
        end
    end
    return g
end

get_neighbor_graph(args...; kwargs...) = get_neighbor_graph_qK(args...; kwargs...)

"""
Get qK neighbor graph with denoted idx
"""

get_SISO_graph(grh::SISOPaths) = grh.qK_grh
function get_SISO_graph(Bnc::Bnc,change_qK;)::SimpleDiGraph
    change_qK_idx = locate_sym_qK(Bnc, change_qK)
    vg = get_vertices_graph!(Bnc;full=true)
    n = length(vg.neighbors)
    g = SimpleDiGraph(n)
    for (i, edges) in enumerate(vg.neighbors)
        nlt = get_nullity(Bnc,i)
        if nlt >1
            continue
        end
        for e in edges
            if isnothing(e.change_dir_qK) || e.to < i
                continue
            end 
            val = e.change_dir_qK[change_qK_idx]
            if val > 1e-6
                add_edge!(g, i, e.to)
            elseif val < -1e-6
                add_edge!(g, e.to, i)
            end 
        end
    end
    return g
end


#------------------------------------------------------------------------------------------

function SISOPaths(model::Bnc{T}, change_qK; rgm_paths=nothing) where {T}
    change_qK_idx = locate_sym_qK(model, change_qK)

    if rgm_paths === nothing
        qK_grh = get_SISO_graph(model, change_qK)
        sources, sinks = get_sources_sinks(model, qK_grh)
        rgm_paths = _enumerate_paths(qK_grh; sources, sinks)
    else
        qK_grh = graph_from_paths(rgm_paths, length(model.vertices_perm))
        sources, sinks = get_sources_sinks(qK_grh)
    end

    return SISOPaths(model, qK_grh, change_qK_idx, sources, sinks, rgm_paths)
end

get_binding_network(grh::SISOPaths,args...)= grh.bn
get_C_C0_nullity_qK(grh::SISOPaths, pth_idx) = get_polyhedron(grh, pth_idx) |> get_C_C0_nullity

function get_polyhedra(grh::SISOPaths, pth_idx::Union{AbstractVector{<:Integer},Nothing,Integer} = nothing)::Vector{Polyhedron}
    pth_idx = let 
            if isnothing(pth_idx)
                1:length(grh.rgm_paths)
            elseif isa(pth_idx, Integer)
                [pth_idx]
            else
                pth_idx
            end
        end
    
    pth_poly_to_calc = filter(x -> !grh.path_polys_is_calc[x], pth_idx)
    
    if !isempty(pth_poly_to_calc)
        polys = _calc_polyhedra_for_path(grh.bn, grh.rgm_paths[pth_poly_to_calc], grh.change_qK_idx)
        grh.path_polys[pth_poly_to_calc] .= polys
        grh.path_polys_is_calc[pth_poly_to_calc] .= true
    end

    return grh.path_polys[pth_idx]
end
get_polyhedron(grh::SISOPaths, pth_idx::Integer)= get_polyhedra(grh, pth_idx)[1]

function get_volume(grh::SISOPaths, pth_idx::Union{AbstractVector{<:Integer},Nothing,Integer}=nothing; asymptotic=true,recalculate=false, kwargs...)::Vector{Tuple{Float64,Float64}}
    pth_idx = let 
            if isnothing(pth_idx)
                1:length(grh.rgm_paths)
            elseif isa(pth_idx, Integer)
                [pth_idx]
            else
                pth_idx
            end
        end
    
    idxes_to_calculate = recalculate ? pth_idx : filter(x -> !grh.path_volume_is_calc[x], pth_idx)
    
    if !isempty(idxes_to_calculate)
        polys = get_polyhedra(grh, idxes_to_calculate)
        rlts = calc_volume(polys; asymptotic=asymptotic, kwargs...)
        for (i, idx) in enumerate(idxes_to_calculate)
            grh.path_volume[idx] = rlts[i][1]
            grh.path_volume_err[idx] = rlts[i][2]
            grh.path_volume_is_calc[idx] = true
        end
    end

    vol = grh.path_volume[pth_idx]
    err = grh.path_volume_err[pth_idx]
    return [(vol, err) for (vol, err) in zip(vol, err)]
end


#---------------------------------------------------------------------------------------------------
#             Functions for analyzing each individual path in the graph
#----------------------------------------------------------------------------------------------------


get_sources(g::AbstractGraph) = Set(v for v in vertices(g) if indegree(g, v) == 0)
get_sinks(g::AbstractGraph)   = Set(v for v in vertices(g) if outdegree(g, v) == 0)
get_sources_sinks(g::AbstractGraph) = (get_sources(g), get_sinks(g))

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
DAG 专用：自底向上缓存后缀路径
返回所有从 sources 到 sinks 的简单路径（DAG 中天然无环）
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




function _calc_reaction_order_for_single_path(model, path::Vector{Int}, change_qK_idx, observe_x_idx)::Vector{<:Real}
    r_ord = Vector{Float64}(undef, length(path))
    for i in eachindex(path)
        if !is_singular(model, path[i])
            r_ord[i] = get_H(model, path[i])[observe_x_idx, change_qK_idx] |> x->round(x;digits=3)
        else
            ord = get_H(model, path[i])[observe_x_idx, change_qK_idx]
            if abs(ord) < 1e-6
                r_ord[i] = NaN  # We use NaN to denote singular
            else 
                r_ord[i] = ord  * Inf
            end     
        end
    end
    return r_ord
end

function _dedup(ord_path::Vector{T})::Vector{T} where T<:Real
    isempty(ord_path) && return ord_path
    result = [ord_path[1]]
    for x in Iterators.drop(ord_path, 1)
        if x != last(result) && !isnan(x)
            push!(result, x)
        end
    end
    return result
end

function find_reaction_order_for_path(model::Bnc, rgm_path::Vector{<:Integer}, change_qK, observe_x; deduplicate::Bool=false,keep_singular::Bool=true,keep_nonasymptotic::Bool=true)::Vector{<:Real}
    change_qK_idx = locate_sym_qK(model, change_qK)
    observe_x_idx = locate_sym_x(model, observe_x)
    ord_path = _calc_reaction_order_for_single_path(model, rgm_path, change_qK_idx, observe_x_idx)
    
    mask = _get_vertices_mask(model, rgm_path;
        singular=keep_singular ? nothing : false,
        asymptotic=keep_nonasymptotic ? nothing : true)
    
    ord_path = ord_path[mask]

    if deduplicate
        ord_path = _dedup(ord_path)
    end
    return ord_path
end

function find_reaction_order_for_path(model, rgm_paths::Vector{Vector{Int}}, args...; kwargs...)::Vector{Vector{<:Real}}
    ord_pths = Vector{Vector{<:Real}}(undef, length(rgm_paths))
    Threads.@threads for i in eachindex(rgm_paths)
        ord_pths[i] = find_reaction_order_for_path(model, rgm_paths[i], args...; kwargs...)
    end
    return ord_pths
end

function find_reaction_order_for_path(model::SISOPaths,observe_x;kwargs...)
    observe_x_idx = locate_sym_x(model.bn, observe_x)
    return find_reaction_order_for_path(model.bn, model.rgm_paths, model.change_qK_idx, observe_x_idx; kwargs...)
end


# function group_sum(keys::AbstractVector, vals::AbstractVector;sort_values::Bool=true)::Vector{Pair}
#     @assert length(keys) == length(vals)
#     dict = Dict{eltype(keys), eltype(vals)}()
#     @inbounds for (k, v) in zip(keys, vals)
#         dict[k] = get(dict, k, zero(v)) + v
#     end
#     dict_vec = collect(dict)
#     if sort_values
#         sort!(dict_vec, by=x->x[2], rev=true)
#     end
#     return dict_vec
# end

function group_sum(keys::AbstractVector, vals::AbstractVector; sort_values::Bool=true) :: Vector{Tuple{Vector{Int}, eltype(keys), eltype(vals)}}
    @assert length(keys) == length(vals)
    # Dictionary to accumulate sum of values for each key
    dict = Dict{eltype(keys), eltype(vals)}()
    # Store indices of keys for later reference
    index_dict = Dict{eltype(keys), Vector{Int}}()
    
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
    result = Vector{Tuple{Vector{Int}, eltype(keys), eltype(vals)}}(undef, length(dict))
    
    # @show dict, index_dict
    for i in eachindex(dict_vec)
        key, sum_val = dict_vec[i]
        group = index_dict[key]
        result[i] = (group, key, sum_val)
    end
    
    return result
end


function summary_path(grh::SISOPaths,observe_x; 
    deduplicate::Bool=false,keep_singular::Bool=true,keep_nonasymptotic::Bool=true,kwargs...)
    
    observe_x_idx = locate_sym_x(grh.bn, observe_x)
    ord_pth = find_reaction_order_for_path(grh, observe_x_idx; 
        deduplicate=deduplicate,
        keep_singular=keep_singular,
        keep_nonasymptotic=keep_nonasymptotic)
    volumes = get_volume(grh; kwargs...)
    return group_sum(ord_pth, collect.(volumes))
end

function summary_path(grh::SISOPaths; kwargs...)
    get_polyhedra(grh)
    get_volume(grh; kwargs...)
    return map(zip(grh.rgm_paths, grh.path_volume, grh.path_volume_err)) do (pth, vol, err)
        return (pth, [vol, err])
    end
end

summary(grh::SISOPaths,args...;kwargs...) = summary_path(grh,args...;kwargs...)


