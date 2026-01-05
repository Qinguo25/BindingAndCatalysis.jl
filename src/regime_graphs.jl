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




# struct SISO_graph{T}
#     bn::Bnc{T}
#     qK_grh::SimpleDiGraph
#     change_qK_idx::T
#     sources::Vector{Int}
#     sinks::Vector{Int}
#     rgm_paths::Vector{Vector{Int}}
#     rgm_volume_is_calc::BitVector
#   rgm_polys_is_calc::BitVector
#     rgm_polys::Vector{Polyhedron}
#     rgm_volume::Vector{Float64}
#     rgm_volume_err::Vector{Float64}
# end

function SISO_graph(model::Bnc{T}, change_qK)::SISO_graph{T} where T
    change_qK_idx = locate_sym_qK(model, change_qK)
    qK_grh = get_qK_neighbor_grh(model, change_qK;)
    sources, sinks = get_sources_sinks(model, qK_grh)
    rgm_paths = _enumerate_paths(qK_grh; sources=sources, sinks=sinks)
    # volume_data = calc_volume(rgm_polys;asymptotic=true)
    # rgm_volume = map(x->x[1], volume_data)
    # rgm_volume_err = map(x->x[2], volume_data)
    return SISO_graph(model, qK_grh, change_qK_idx, sources, sinks, rgm_paths)
end


# Construct the finite graph? why this function exist?
function SISO_graph(model::Bnc{T}, change_qK, rgm_paths::AbstractVector{AbstractVector{Integer}})::SISO_graph{T} where T
    grh = SimpleDiGraph(length(model.vertices_perm))
    for p in rgm_paths
        n = length(p)
        for i in 1:(n-1)
            add_edge!(grh, p[i], p[i+1])
        end
    end
    qK_grh = grh
    change_qK_idx = locate_sym_qK(model, change_qK)
    sources, sinks = unique(rgm_paths .|> x->x[1]), unique(rgm_paths .|> x->x[end])
    # volume_data = calc_volume(rgm_polys;asymptotic=true)
    # rgm_volume = map(x->x[1], volume_data)
    # rgm_volume_err = map(x->x[2], volume_data)
    return SISO_graph(model, qK_grh, change_qK_idx, sources, sinks, rgm_paths)
end



function get_volume(grh::SISO_graph, pth_idx::Union{Vector{<:Integer},Nothing}=nothing; asymptotic=true,recalculate=false, kwargs...)::Vector{Tuple{Float64,Float64}}
    pth_idx === nothing && (pth_idx = 1:length(grh.rgm_paths))
    # isa(pth_idx, Integer) && (pth_idx = [pth_idx]) # make sure pth_idx is a vector
    # calculate volumes if not calculated before
    idxes_to_calculate = recalculate ? pth_idx : filter(x -> !grh.rgm_volume_is_calc[x], pth_idx)
    
    if !isempty(idxes_to_calculate)
        polys = get_polyhedra(grh, idxes_to_calculate)
        rlts = calc_volume(polys; asymptotic=asymptotic, kwargs...)
        for (i, idx) in enumerate(idxes_to_calculate)
            grh.rgm_volume[idx] = rlts[i][1]
            grh.rgm_volume_err[idx] = rlts[i][2]
            grh.rgm_volume_is_calc[idx] = true
        end
    end

    vol = grh.rgm_volume[pth_idx]
    err = grh.rgm_volume_err[pth_idx]
    return [(vol, err) for (vol, err) in zip(vol, err)]
end
get_volume(grh::SISO_graph, pth_idx::Integer; kwargs...) = get_volume(grh, [pth_idx]; kwargs...)[1]




function get_polyhedra(grh::SISO_graph, pth_idx = nothing)::Vector{Polyhedron}
    pth_idx === nothing && (pth_idx = 1:length(grh.rgm_paths))
    isa(pth_idx, Integer) && (pth_idx = [pth_idx]) # make sure pth_idx is a vector
    
    let # calculate polyhedra if not initialized before
        idx_to_calculate = filter(x -> !grh.rgm_polys_is_calc[x], pth_idx) # filter those not calculated
        if !isempty(idx_to_calculate)
            polys = _calc_polyhedra_for_path(grh.bn, grh.rgm_paths[idx_to_calculate], grh.change_qK_idx)
            grh.rgm_polys[idx_to_calculate] .= polys
            grh.rgm_polys_is_calc[idx_to_calculate] .= true
        end
    end
    return grh.rgm_polys[pth_idx]
end
get_polyhedron(grh::SISO_graph, pth_idx)= get_polyhedra(grh, pth_idx)[1]


#
get_binding_network(grh::SISO_graph,args...)= grh.bn
get_C_C0_nullity_qK(grh::SISO_graph, pth_idx) = get_polyhedron(grh, pth_idx) |> get_C_C0_nullity
# get_C_C0!, 
#get_C!, 
#get_C0! are already defined by julia multipy dispatch







#-----------------------------------------------------------------------------------
function get_x_neighbor_grh(Bnc::Bnc)::SimpleGraph
    vg = get_vertices_graph!(Bnc;full=false)
    return vg.x_grh
end

function get_qK_neighbor_grh(Bnc::Bnc; half::Bool=true)::SimpleDiGraph
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

"""
Get qK neighbor graph with denoted idx
"""
get_qK_neighbor_grh(grh::SISO_graph) = grh.qK_grh
function get_qK_neighbor_grh(Bnc::Bnc,change_qK;)::SimpleDiGraph
    change_qK_idx = locate_sym_qK(model, change_qK)
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

#---------------------------------------------------------------------------------------------------
#             Functions for analyzing each individual path in the graph
#----------------------------------------------------------------------------------------------------


get_sources(g::AbstractGraph) = Set(v for v in vertices(g) if indegree(g, v) == 0)
get_sinks(g::AbstractGraph)   = Set(v for v in vertices(g) if outdegree(g, v) == 0)

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

function _enumerate_paths(g::AbstractGraph; 
    sources::AbstractVector{Int}, 
    sinks::AbstractVector{Int})::Vector{Vector{Int}}

    @info "sources: $sources"
    @info "sinks: $sinks"

    function dfs(path, local_paths)
        lastv = path[end]
        if lastv in sinks
            push!(local_paths, copy(path))
            return
        end
        for nb in outneighbors(g, lastv)
            if nb ∉ path
                dfs([path...; nb], local_paths)
            end
        end
    end

    paths_per_thread = [Vector{Vector{Int}}() for _ in 1:Threads.maxthreadid()]
    Threads.@threads for s in collect(sources)
        local_paths = paths_per_thread[Threads.threadid()]
        dfs([s], local_paths)
    end

    paths = reduce(vcat, paths_per_thread)
    return sort!(paths)#filter!(paths) do x length(x) > 1 end
end


# function _calc_polyhedra_for_path(model::Bnc, paths::AbstractArray{<:AbstractVector{Ty}},change_qK; cachelevel=2)::Vector{Polyhedron} where Ty<:Integer
#     # Find the dimension to eliminate
#     change_qK_idx = locate_sym_qK(model, change_qK)
#     el_dim = BitSet(change_qK_idx) # dimension to eliminate

#     clean(p) = begin
#         detecthlinearity!(p)
#         removehredundancy!(p)
#     end
#      # build the initial edges cache after eliminate dimension
#     begin
#         # keys = map(Set(a.src, a.dst), edges(g))
#         keys = Set{Set{Int}}()
#         for p in paths
#             n = length(p)
#             for i in 1:(n-1)
#                 u = p[i]
#                 v = p[i+1]
#                 push!(keys, Set([u,v]))
#             end
#         end
#         keys = collect(keys) 

#         idx_2_map = Dict(keys[i] => i for i in eachindex(keys))
#         polys_2 = Vector{Polyhedron}(undef, length(keys))
#         Threads.@threads for i in eachindex(keys)
#             u,v = collect(keys[i])
#             ins = get_polyhedron_intersect(model, u, v)
#             e = eliminate(ins,el_dim)
#             clean(e)
#             polys_2[i] = e
#         end
#     end

#     # turn paths into edge idxs
#     turn_edge(path) = begin
#         n = length(path)
#         edge_idxs = Vector{Int}(undef, n-1)
#         for i in 1:(n-1)
#             u = path[i]
#             v = path[i+1]
#             key = Set([u,v])
#             edge_idxs[i] = idx_2_map[key]
#         end
#         return edge_idxs
#     end

#     path_edge_idxs = map(turn_edge, paths)

#     # build higher level cache
#     begin
#         path_iters = Iterators.partition.(path_edge_idxs, cachelevel)
#         keys = reduce(vcat,collect.(path_iters))
#         idx_cache_map = Dict(Vector(keys[i]) => i for i in eachindex(keys))
#         edge_poly_cache = Vector{Polyhedron}(undef, length(keys))
#         Threads.@threads for i in eachindex(keys)
#             edge_idxs = keys[i]
#             if length(edge_idxs) == 1
#                 edge_poly_cache[i] = polys_2[edge_idxs[1]]
#             else
#                 edge_poly_cache[i] = reduce((a,b)->intersect(a,b), (polys_2[e] for e in edge_idxs))
#             end
#             clean(edge_poly_cache[i])
#         end
#     end

#     # @warn "Problematic for now"
#     polys = Vector{Polyhedron}(undef, length(paths))
#     Threads.@threads for i in eachindex(paths)
#         iter = path_iters[i]
#         keys = Vector.(iter)
#         poly_idxs = map(k->idx_cache_map[k], keys)
#         poly_edges = @view edge_poly_cache[poly_idxs]
#         polys[i] = reduce((a,b)->intersect(a,b), poly_edges)
#         clean(polys[i])
#     end
#     return polys
# end


function _calc_polyhedra_for_path(model::Bnc, rgm_path::AbstractVector{<:Integer}, change_qK)::Polyhedron # Can be extremely slow for long paths
    # Find the dimension to eliminate
    change_qK_idx = locate_sym_qK(model, change_qK)
    el_dim = BitSet(change_qK_idx) # dimension to eliminate

    f(p) = begin
        detecthlinearity!(p)
        removehredundancy!(p)
    end

    if length(rgm_path) ==1
        poly = get_polyhedra(model, rgm_path[1])
        e = eliminate(poly,el_dim)
        f(e)
        return e
    end

    poly_ins = Vector{Polyhedron{Float64}}(undef,length(rgm_path)-1)
    Threads.@threads for i in 1:(length(rgm_path)-1)
        u = rgm_path[i]
        v = rgm_path[i+1]
        ins = get_polyhedron_intersect(model, u, v)
        e = eliminate(ins,el_dim)
        poly_ins[i] = e
    end
    p = reduce((a,b)->intersect(a,b), poly_ins)
    f(p)
    return p
end

function _calc_polyhedra_for_path(
    model::Bnc,
    paths::AbstractVector{<:AbstractVector{Ty}},
    change_qK;
    cachelevel::Int = 2,
)::Vector{Polyhedron} where {Ty<:Integer}

    # dimension to eliminate
    change_qK_idx = locate_sym_qK(model, change_qK)
    el_dim = BitSet((change_qK_idx,))  # or BitSet([change_qK_idx])

    clean!(p) = (detecthlinearity!(p); removehredundancy!(p); p)

    # -------------------------
    # 1) Build regime -> poly map (after eliminate)
    # -------------------------
    unique_rgms = unique(vcat(paths...))

    poly_of = Dict{Int,Polyhedron}()
    # (serial is fine; you can thread if get_polyhedra is heavy and thread-safe)
    for r in unique_rgms
        pr = get_polyhedra(model, r)
        pr = eliminate(pr, el_dim)
        clean!(pr)
        poly_of[r] = pr
    end

    # -------------------------
    # 2) Build unique undirected edges and edge index map
    # key = (min(u,v), max(u,v))
    # -------------------------
    edges = Tuple{Int,Int}[]
    edge_idx = Dict{Tuple{Int,Int},Int}()

    for path in paths
        n = length(path)
        @inbounds for i in 1:(n-1)
            u = Int(path[i]); v = Int(path[i+1])
            a, b = u < v ? (u, v) : (v, u)
            k = (a, b)
            if !haskey(edge_idx, k)
                push!(edges, k)
                edge_idx[k] = length(edges)
            end
        end
    end

    # -------------------------
    # 3) Compute poly for each edge (intersection of endpoint polys)
    # -------------------------
    edge_poly = Vector{Polyhedron}(undef, length(edges))
    Threads.@threads for i in eachindex(edges)
        u, v = edges[i]
        e = intersect(poly_of[u], poly_of[v])
        clean!(e)
        edge_poly[i] = e
    end

    # -------------------------
    # 4) Convert each path into edge index list
    # -------------------------
    function path_to_edge_idxs(path)
        n = length(path)
        idxs = Vector{Int}(undef, n-1)
        @inbounds for i in 1:(n-1)
            u = Int(path[i]); v = Int(path[i+1])
            a, b = u < v ? (u, v) : (v, u)
            idxs[i] = edge_idx[(a, b)]
        end
        return idxs
    end

    path_edge_idxs = map(path_to_edge_idxs, paths)

    # -------------------------
    # 5) Higher-level cache over chunks of edges
    # key = Tuple(edge_idxs_chunk...)  (immutable, content-hashed)
    # -------------------------
    chunk_key_to_idx = Dict{Tuple{Vararg{Int}},Int}()
    chunk_polys = Polyhedron[]

    function get_chunk_poly(edge_idxs_chunk::AbstractVector{Int})
        k = Tuple(edge_idxs_chunk)  # content-hashable
        if haskey(chunk_key_to_idx, k)
            return chunk_polys[chunk_key_to_idx[k]]
        end
        # build new
        p = if length(edge_idxs_chunk) == 1
            edge_poly[edge_idxs_chunk[1]]
        else
            reduce((a,b)->intersect(a,b), (edge_poly[e] for e in edge_idxs_chunk))
        end
        clean!(p)
        push!(chunk_polys, p)
        chunk_key_to_idx[k] = length(chunk_polys)
        return p
    end

    # -------------------------
    # 6) Final: for each path, intersect its chunk-polys
    # -------------------------
    out = Vector{Polyhedron}(undef, length(paths))
    Threads.@threads for i in eachindex(paths)
        idxs = path_edge_idxs[i]
        it = Iterators.partition(idxs, cachelevel)
        # accumulate
        first = true
        acc = Polyhedron()  # placeholder; will be overwritten
        for chunk in it
            pchunk = get_chunk_poly(chunk)
            if first
                acc = pchunk
                first = false
            else
                acc = intersect(acc, pchunk)
            end
        end
        clean!(acc)
        out[i] = acc
    end

    return out
end



# function _calc_polyhedra_for_path(model::Bnc, paths::AbstractArray{<:AbstractVector{Ty}},change_qK; cachelevel=2)::Vector{Polyhedron} where Ty<:Integer
#     # Find the dimension to eliminate
#     change_qK_idx = locate_sym_qK(model, change_qK)
#     el_dim = BitSet(change_qK_idx) # dimension to eliminate

#     clean(p) = begin
#         detecthlinearity!(p)
#         removehredundancy!(p)
#     end
#      # build the initial edges cache after eliminate dimension

#     # project all the polyhedra to its eliminate dimension=0 hyperplane
#     begin
#         keys = vcat(paths...) |> unique
#         ps = keys .|> p -> get_polyhedra(model, p) |> p -> eliminate(p, el_dim) |> clean
#     end
# end


# _calc_polyhedra_for_path(args...;kwargs...) = _calc_polyhedra_for_path_direct(args...;kwargs...)
# function _calc_polyhedra_for_path(model::Bnc,path,change_qK_idx)::Polyhedron
#     @warn "This function is buggy for now, use _calc_polyhedra_for_path_direct instead"
#     # Buggy, not working for now.

#     # Handle invertible regimes first

#     # Firstly let's try assuming regiems with nullity 1 have no contribution()
#     nlts = [get_nullity(model, p) for p in path]
#     idxs = findall(x -> x == 0, nlts)
#     C = Vector{Matrix{Float64}}(undef, length(idxs))
#     C0 = Vector{Vector{Float64}}(undef, length(idxs))

#     function get_empty_row_idxs(L::SparseMatrixCSC,i)
#         m = size(L,1)
#         rows_i = L.rowval[L.colptr[i]:(L.colptr[i+1]-1)]
#         zero_rows = setdiff(1:m, rows_i)
#         return zero_rows
#     end

#     Threads.@threads for i in eachindex(idxs)
#         idx = idxs[i]
#         perm = path[idx]
#         C_tmp, C0_tmp = get_C_C0_qK(model, perm)
#         rows = get_empty_row_idxs(C_tmp, change_qK_idx)
#         cols = setdiff(1:model.n, change_qK_idx)
#         C[i] = C_tmp[rows, cols]
#         C0[i] = C0_tmp[rows]
#     end
#     C_all = reduce(vcat, C)
#     C0_all = reduce(vcat, C0)


#     # # Now handle regimes with nullity 1
#     # idxs_nlt = findall(x -> x ==1, nlts)
#     # C_nlt = Vector{Matrix{Float64}}(undef, length(idxs_nlt))
#     # C0_nlt = Vector{Vector{Float64}}(undef, length(idxs_nlt))
#     # Threads.@threads for i in eachindex(idxs_nlt)
#     #     poly = get_polyhedra

#     p = get_polyhedra(C_all, C0_all, 0)
#     detecthlinearity!(p)
#     removehredundancy!(p)
#     return p
# end




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
                r_ord[i] = ord * model.direction * Inf
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

function find_reaction_order_for_path(model::SISO_graph,observe_x;kwargs...)
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


function summary_path(grh::SISO_graph,observe_x; 
    deduplicate::Bool=false,keep_singular::Bool=true,keep_nonasymptotic::Bool=true,kwargs...)
    
    observe_x_idx = locate_sym_x(grh.bn, observe_x)
    ord_pth = find_reaction_order_for_path(grh, observe_x_idx; 
        deduplicate=deduplicate,
        keep_singular=keep_singular,
        keep_nonasymptotic=keep_nonasymptotic)
    volumes = get_volume(grh; kwargs...)
    return group_sum(ord_pth, collect.(volumes))
end

function summary_path(grh::SISO_graph; kwargs...)
    get_polyhedra(grh)
    get_volume(grh; kwargs...)
    return map(zip(grh.rgm_paths, grh.rgm_volume, grh.rgm_volume_err)) do (pth, vol, err)
        return (pth, [vol, err])
    end
end

summary(grh::SISO_graph,args...;kwargs...) = summary_path(grh,args...;kwargs...)


