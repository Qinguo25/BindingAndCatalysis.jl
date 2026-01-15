#-----------------------------------------------------------------------------------------------
#This is graph associated functions for Bnc models and archetyple behaviors associated code
#-----------------------------------------------------------------------------------------------
"""
    _calc_vertices_graph(data::Vector{<:AbstractVector{T}}, n::Int) where T

Build VertexGraph from a list of vertex permutations.
- Groups vertices differing in exactly one row, creates bidirectional edges with change_dir_x.
"""
function  _calc_vertices_graph(Bnc::Bnc{T}) where {T} # optimized by GPT-5, not fullly understood yet.
    perms = Bnc.vertices_perm
    n = Bnc.n    
    L = Bnc.L

    n_vtxs = length(perms)
    d=length(perms[1])
    thread_edges = [Vector{Tuple{Int, VertexEdge{T}}}() for _ in 1:Threads.maxthreadid()]

    # 按行分桶：key 为去掉该行后的签名（Tuple），值为该签名下的 (顶点索引, 该行取值)
    for i in 1:d
        buckets = Dict{Tuple{Vararg{T}}, Vector{Tuple{Int,T}}}()

        # 构建桶
        @inbounds for i in 1:n_vtxs
            v = perms[i]
            sig = if i == 1
                    Tuple(v[2:end])
                elseif i == d
                    Tuple(v[1:end-1])
                else
                    Tuple((v[1:i-1]..., v[i+1:end]...))
                end
            push!(get!(buckets, sig) do
                Vector{Tuple{Int,T}}()
            end, (i, v[i]))
        end

        groups = collect(values(buckets))

        # 并行生成边：同桶内所有不同取值的顶点两两相连
        Threads.@threads for gi in 1:length(groups)
            tid = Threads.threadid()
            local_edges = thread_edges[tid]
            group = groups[gi]  # ::Vector{Tuple{Int,T}}
            m = length(group)
            m <= 1 && continue

            @inbounds for a in 1:m-1
                p1, j1 = group[a]
                for b in a+1:m
                    p2, j2 = group[b]
                    j1 == j2 && continue

                    dx = if j1 < j2   # go from p2 to p1, decrease x_{j2}, increase x_{j1}
                        SparseVector(n, [j1, j2], Int8[1, -1]) 
                    else
                         SparseVector(n, [j2, j1], Int8[-1, 1])
                    end

                    ins_x = log10(L[i, j1]) - log10(L[i, j2]) # go from p2 to p1

                    push!(local_edges, (p2, VertexEdge(p1, i, dx, ins_x))) # p2 to p1,
                    push!(local_edges, (p1, VertexEdge(p2, i, -dx, -ins_x)))  # p1 to p2
                end
            end
        end
    end

    # 归并线程本地边
    all_edges = reduce(vcat, thread_edges; init=Tuple{Int, VertexEdge{T}}[])
    neighbors = [Vector{VertexEdge{T}}() for _ in 1:n_vtxs]
    for (from, e) in all_edges
        push!(neighbors[from], e)
    end
    return VertexGraph(Bnc, neighbors)
end


"""
    _fulfill_vertices_graph!(Bnc, vtx_graph)

Ensure vertices are discovered and vertex graph is built and cached in Bnc.
Returns nothing.
"""

function _fulfill_vertices_graph!(vtx_graph::VertexGraph)
    Bnc = vtx_graph.bn
    """
    fill the qK space change dir matrix for all vertices in Bnc.
    """
    function _calc_change_dir_qK(Bnc, p1, p2, i, j1, j2, ins_x)
        # calculate the interface and norm points to p2
        n1 = get_nullity(Bnc, p1)
        n2 = get_nullity(Bnc, p2)
        if n1 > 1 || n2 > 1
            return nothing
        end

        if n1 == 0
            H,H0 = get_H_H0(Bnc, p1)
            dir = H[j2, :] - H[j1, :]
            ins_qK = H0[j2] - H0[j1] + ins_x
        elseif n2 == 0
            H,H0 = get_H_H0(Bnc, p1)
            dir = H[j1, :] - H[j2, :] 
            ins_qK = H0[j1] - H0[j2] - ins_x
        else
            H = get_H(Bnc, p1)
            M0 = get_M0(Bnc, p1)
            dir = H[j2, :] - H[j1, :]
            ins_qK = -dot(dir, M0)
        end
        droptol!(dir, 1e-10)
        return nnz(dir)==0 ? nothing : (dir, ins_qK)
    end

    # pre compute H for all vertices with nullity 0 or 1
    Threads.@threads for idx in eachindex(vtx_graph.neighbors)
        if Bnc.vertices_nullity[idx] <= 1
            get_H(Bnc, idx)
        end
    end

    Threads.@threads for p1 in eachindex(vtx_graph.neighbors)
        edges = vtx_graph.neighbors[p1]
        if Bnc.vertices_nullity[p1] > 1 # jump off those regimes with nullity >1
            continue
        end
        for e in edges
            if !isnothing(e.change_dir_qK) # pass if have been computed
                continue
            end
            # from p1 to p2, and change happens on ith row that "1" goes from j1 position to j2 position.
            p2 = e.to # target 
            ins_x = e.intersect_x
            i = e.diff_r # different row
            (j1,j2) = let 
                I,V = findnz(e.change_dir_x) # should be two elements
                V[1] > V[2] ? (I[2], I[1]) : (I[1], I[2])
            end
            # calculate their direction based on formula
            (e.change_dir_qK, e.intersect_qK) = _calc_change_dir_qK(Bnc, p1, p2, i, j1,j2, ins_x)
        end
    end
    return nothing
end

#---------------------------------------------------------------------------------------------------
#             Helper functions: Functions for construct the regime graph paths
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




#---------------------------------------------------------------------------
#              Binding Network Graph
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
#                  Getting the Graph of of regimes
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
            _fulfill_vertices_graph!(vtx_graph)
            vtx_graph.change_dir_qK_computed = true
            println("Done.\n")
        end
    else
        if isnothing(Bnc.vertices_graph)
            find_all_vertices!(Bnc)# Ensure vertices are calculated
            println("----------------Start calculating vertices neighbor graph, It may takes a while.----------------")
            Bnc.vertices_graph =  _calc_vertices_graph(Bnc)
            println("Done.\n")
        end
    end
    return Bnc.vertices_graph
end


function get_edge(grh::VertexGraph, from::Integer, to::Integer; full=false)::Union{Nothing, VertexEdge}
    if full
        if isnothing(Bnc.vertices_graph)
            find_all_vertices!(Bnc)# Ensure vertices are calculated
            println("----------------Start calculating vertices neighbor graph, It may takes a while.----------------")
            Bnc.vertices_graph =  _calc_vertices_graph(Bnc)
            println("Done.\n")
        end
    end
    pos = get(grh.edge_pos[from], to, nothing)
    return pos === nothing ? nothing : grh.neighbors[from][pos]
end


get_edge(Bnc, from, to; kwargs...)= let
    vtx_grh = get_vertices_graph!(Bnc; full=false)
    bn = get_binding_network(Bnc)
    from = get_idx(Bnc, from)
    to = get_idx(Bnc, to)
    get_edge(vtx_grh, from, to; kwargs...)
end

get_binding_network(grh::VertexGraph,args...) = grh.bn







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


#------------------------------------------------------------------------------
# Higher wrapper for regime graph paths
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

function get_path(grh::SISOPaths, pth_idx::Integer; return_idx::Bool=false)
    if return_idx
        return grh.rgm_paths[pth_idx]
    end
    bn = get_binding_network(grh)
    perms = get_perm(bn, pth_idx)
    return perms
end

get_idx(grh::SISOPaths, pth::AbstractVector{<:Integer}) = findfirst(x -> x == pth, grh.rgm_paths)
get_binding_network(grh::SISOPaths,args...)= grh.bn
get_C_C0_nullity_qK(grh::SISOPaths, pth_idx) = get_polyhedron(grh, pth_idx) |> get_C_C0_nullity

function get_polyhedra(grh::SISOPaths, pth_idx::Union{AbstractVector{<:Integer},Nothing} = nothing)::Vector{Polyhedron}
    pth_idx = let 
            if isnothing(pth_idx)
                1:length(grh.rgm_paths)
            else
                pth_idx
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
get_polyhedron(grh::SISOPaths, pth_idx::Integer)= get_polyhedra(grh, [pth_idx])[1]



function get_volume(grh::SISOPaths, pth_idx::Union{AbstractVector{<:Integer},Nothing,Integer}=nothing; 
    asymptotic=true,recalculate=false, kwargs...)

    pth_idx = let 
            if isnothing(pth_idx)
                1:length(grh.rgm_paths)
            else
                pth_idx
            end
        end
    
    idxes_to_calculate = recalculate ? pth_idx : filter(x -> !grh.path_volume_is_calc[x], pth_idx)
    
    if !isempty(idxes_to_calculate)
        polys = get_polyhedra(grh, idxes_to_calculate)
        rlts = calc_volume(polys; asymptotic=asymptotic, kwargs...)
        for (i, idx) in enumerate(idxes_to_calculate)
            grh.path_volume[idx] = rlts[i]
            grh.path_volume_is_calc[idx] = true
        end
    end
    return grh.path_volume
end
get_volume(grh::SISOPaths, pth_idx::Integer; kwargs...) = get_volume(grh, [pth_idx]; kwargs...)[1]



#-------------------------------------------------------------------------------------
# Regime shifting associated functions
#-------------------------------------------------------------------------------------

function show_regime_path(grh::SISOPaths, pth_idx::Integer)
    pth = get_path(grh, pth_idx; return_idx=true)
    print_path(pth; prefix="#")
    return nothing
end

function show_expression_path(grh::SISOPaths, pth_idx::Integer; observe_x=nothing)
    bn = get_binding_network(grh)
    observe_x_idx = isnothing(observe_x) ? bn.n : locate_sym_x.(Ref(bn), observe_x)
    rgm_pth = get_path(grh, pth_idx; return_idx=true)
    pth = map(rgm_pth) do r
        is_singular(bn, r) ? fill(NaN, length(observe_x_idx)) : get_expression(bn, r)[observe_x_idx]
    end
    pth = get_path(grh, pth_idx; return_idx=false)
    print_expression_path(get_binding_network(grh), pth; prefix="#")
    return nothing
end

#-------------------------------------------------------------------------------------------
# 

"""
Given a path in regime graph, find the reaction order profile along the path.
"""
function _calc_reaction_order_for_single_path(model, path::AbstractVector{<:Integer}, change_qK_idx, observe_x_idx)::Vector{<:Real}
    r_ord = Vector{Float64}(undef, length(path))
    for i in eachindex(path)
        if !is_singular(model, path[i])
            r_ord[i] = get_H(model, path[i])[observe_x_idx, change_qK_idx] |> x->round(x;digits=3)
        else
            ord = get_H(model, path[i])[observe_x_idx, change_qK_idx]
            if abs(ord) < 1e-6
                r_ord[i] = NaN  # We use NaN to denote continuous singular, if reaction order not same before and after, means discontinuity
            else 
                r_ord[i] = ord  * Inf
            end     
        end
    end
    return r_ord
end
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
Calc reaction order profile for single path in regime graph.
- `model::Bnc`: Binding network model.
- `rgm_path::AbstractVector{<:Integer}`: Regime path (vector of regime indices).
- `change_qK`: Symbol or index of the changing qK.
- `observe_x`: Symbol or index of the observed x.
- `deduplicate::Bool=false`: Whether to deduplicate the reaction order profile.
- `keep_singular::Bool=true`: Whether to keep singular regimes in the profile.
- `keep_nonasymptotic::Bool=true`: Whether to keep non-asymptotic regimes in the profile.
"""
function get_reaction_order_path(
    model::Bnc, rgm_path::AbstractVector{<:Integer}; 
    change_qK, observe_x,
    deduplicate::Bool=false,
    keep_singular::Bool=true,
    keep_nonasymptotic::Bool=true
    )::Vector{<:Real}

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
"""
Multiple paths version of `get_reaction_order_path`.
"""
function get_reaction_order_path(model, rgm_paths::AbstractVector{<:AbstractVector{<:Integer}}, args...; kwargs...)::Vector{Vector{<:Real}}
    ord_pths = Vector{Vector{<:Real}}(undef, length(rgm_paths))
    Threads.@threads for i in eachindex(rgm_paths)
        ord_pths[i] = get_reaction_order_path(model, rgm_paths[i], args...; kwargs...)
    end
    return ord_pths
end

function get_reaction_order_path(model::SISOPaths, pth_idx = nothing ; observe_x, kwargs...)
    pth_idx = isnothing(pth_idx) ? (1:length(model.rgm_paths)) : pth_idx
    rgm_paths = @view model.rgm_paths[pth_idx]
    observe_x_idx = locate_sym_x(model.bn, observe_x)
    return get_reaction_order_path(model.bn, rgm_paths; 
        change_qK=model.change_qK_idx, observe_x=observe_x_idx, kwargs...)
end
get_reaction_order_path(model::SISOPaths, pth_idx::Integer, args...; kwargs...) = get_reaction_order_path(model, [pth_idx], args... ; kwargs...)



"""
    Group sum values by keys.

    # Arguments
    - `keys::AbstractVector`: Vector of keys.
    - `vals::AbstractVector`: Vector of values to be summed.
    - `sort_values::Bool=true`: Whether to sort the output by summed values in descending order.

    # Returns
    - `Vector{Tuple{Vector{Int}, eltype(keys), eltype(vals)}}`: A vector of tuples, each containing:
        - A vector of indices corresponding to the original positions of the key.
        - The key itself.
        - The summed value for that key.
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



"""
Show (print) paths stored in SISOPaths.

- Reads `grh.rgm_paths`
- Optionally computes volumes via `get_volume(grh; kwargs...)`
"""
function summary(grh::SISOPaths; show_volume::Bool=true, prefix::AbstractString="#", kwargs...)
    paths = grh.rgm_paths
    if show_volume
        vols = get_volume(grh; kwargs...)
        # rows = _normalize_rows(paths; volumes=vols)  # ids 默认 1:N
        print_paths(paths, vols; prefix=prefix)
    else
        print_paths(paths; prefix=prefix)
    end
    return nothing
end



function summary_reaction_order_path(grh::SISOPaths,observe_x; 
    deduplicate::Bool=false,keep_singular::Bool=true,keep_nonasymptotic::Bool=true,kwargs...)
    
    observe_x_idx = locate_sym_x(get_binding_network(grh), observe_x)
    ord_pth = find_reaction_order_for_path(grh, observe_x_idx; 
        deduplicate=deduplicate,
        keep_singular=keep_singular,
        keep_nonasymptotic=keep_nonasymptotic)
    volumes = get_volume(grh; kwargs...)

    rlts = group_sum(ord_pth, collect.(volumes))
    show_paths(rlts; prefix="")
    return nothing
end


