#-----------------------------------------------------------------------------------------------
#This is graph associated functions for Bnc models and archetyple behaviors associated code
#-----------------------------------------------------------------------------------------------

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
#                  Things involving with drawing plot 
#----------------------------------------------------------------------------
function get_x_neighbor_grh(Bnc::Bnc)::SimpleGraph
    vg = get_vertices_graph!(Bnc)
    n = length(vg.neighbors)
    g = SimpleGraph(n)
    for (i, edges) in enumerate(vg.neighbors)
        for e in edges
            add_edge!(g, i, e.to)
        end
    end
    return g
end

function get_qK_neighbor_grh(Bnc::Bnc; half::Bool=true)::SimpleDiGraph
    vg = get_vertices_graph!(Bnc;full=true)
    n = length(vg.neighbors)
    g = SimpleDiGraph(n)
    for (i, edges) in enumerate(vg.neighbors)
        if get_nullity!(Bnc,i) >1
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
function get_qK_neighbor_grh(Bnc::Bnc,change_qK_idx;)::SimpleDiGraph
    vg = get_vertices_graph!(Bnc;full=true)
    n = length(vg.neighbors)
    g = SimpleDiGraph(n)
    for (i, edges) in enumerate(vg.neighbors)
        nlt = get_nullity!(Bnc,i)
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




get_sources(g::DiGraph) = Set(v for v in vertices(g) if indegree(g, v) == 0)
get_sinks(g::DiGraph)   = Set(v for v in vertices(g) if outdegree(g, v) == 0)

function find_all_complete_paths(model::Bnc, g::DiGraph)
    sources_all = get_sources(g)
    sinks_all   = get_sinks(g)
    common_vs = intersect(sources_all, sinks_all)
    filter!(common_vs) do v
        get_nullity!(model, v) > 0
    end
    sources = setdiff(sources_all, common_vs)
    sinks = setdiff(sinks_all, common_vs)

    @show sources
    @show sinks
    

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

    paths_per_thread = [Vector{Vector{Int}}() for _ in 1:Threads.nthreads()]
    Threads.@threads for s in collect(sources)
        local_paths = paths_per_thread[Threads.threadid()]
        dfs([s], local_paths)
    end

    paths = reduce(vcat, paths_per_thread)
    return sort!(paths)#filter!(paths) do x length(x) > 1 end
end

function find_conditions_for_path_direct(model::Bnc, path, change_qK_idx)::Polyhedron # Can be extremely slow for long paths
    el_dim = BitSet(change_qK_idx)

    if length(path) ==1
        poly = get_polyhedra(model, path[1])
        e = eliminate(poly,el_dim)
        detecthlinearity!(e)
        return e
    end

    poly_ins = Vector{Polyhedron{Float64}}(undef,length(path)-1)
    Threads.@threads for i in 1:(length(path)-1)
        u = path[i]
        v = path[i+1]
        poly1 = get_polyhedra(model,u)
        poly2 = get_polyhedra(model,v)
        ins = intersect(poly1, poly2)
        e = eliminate(ins,el_dim)
        poly_ins[i] = e
    end
    p = reduce((a,b)->intersect(a,b), poly_ins)
    detecthlinearity!(p)
    removehredundancy!(p)
    return p
end



function find_conditions_for_path(model::Bnc,path,change_qK_idx)::Polyhedron
    @warn "This function is buggy for now, use find_conditions_for_path_direct instead"
    # Buggy, not working for now.

    # Handle invertible regimes first

    # Firstly let's try assuming regiems with nullity 1 have no contribution()
    nlts = [get_nullity!(model, p) for p in path]
    idxs = findall(x -> x == 0, nlts)
    C = Vector{Matrix{Float64}}(undef, length(idxs))
    C0 = Vector{Vector{Float64}}(undef, length(idxs))

    function get_empty_row_idxs(L::SparseMatrixCSC,i)
        m = size(L,1)
        rows_i = L.rowval[L.colptr[i]:(L.colptr[i+1]-1)]
        zero_rows = setdiff(1:m, rows_i)
        return zero_rows
    end

    Threads.@threads for i in eachindex(idxs)
        idx = idxs[i]
        perm = path[idx]
        C_tmp, C0_tmp = get_C_C0_qK!(model, perm)
        rows = get_empty_row_idxs(C_tmp, change_qK_idx)
        cols = setdiff(1:model.n, change_qK_idx)
        C[i] = C_tmp[rows, cols]
        C0[i] = C0_tmp[rows]
    end
    C_all = reduce(vcat, C)
    C0_all = reduce(vcat, C0)


    # # Now handle regimes with nullity 1
    # idxs_nlt = findall(x -> x ==1, nlts)
    # C_nlt = Vector{Matrix{Float64}}(undef, length(idxs_nlt))
    # C0_nlt = Vector{Vector{Float64}}(undef, length(idxs_nlt))
    # Threads.@threads for i in eachindex(idxs_nlt)
    #     poly = get_polyhedra

    p = get_polyhedra(C_all, C0_all, 0)
    detecthlinearity!(p)
    removehredundancy!(p)
    return p
end


function find_conditions_for_pathes(model::Bnc, paths, change_qK_idx)::Vector{Polyhedron}
    # @warn "Problematic for now"
    polys = Vector{Polyhedron}(undef, length(paths))
    Threads.@threads for i in eachindex(paths)
        # polys[i] = find_conditions_for_path(model, paths[i], change_qK_idx)
        polys[i] = find_conditions_for_path_direct(model, paths[i], change_qK_idx)
    end
    return polys
end


function find_reaction_order_for_single_path(model, path::Vector{Int}, change_qK_idx, observe_x_idx; deduplicate::Bool=false,keep_singular::Bool=true)::Vector{<:Real}
    r_ord = Vector{Float64}(undef, length(path))
    for i in eachindex(path)
        null = get_nullity!(model, path[i])
        if null == 0
            r_ord[i] = get_H!(model, path[i])[observe_x_idx, change_qK_idx] |> x->round(x;digits=3)
        else
            ord = get_H!(model, path[i])[observe_x_idx, change_qK_idx]
            if abs(ord) < 1e-6
                r_ord[i] = 0.0
            else 
                r_ord[i] = ord * model.direction * Inf
            end     
        end
    end
    if deduplicate
        r_ord = dedup(r_ord,keep_singular)
    end
    return r_ord
end

function find_reaction_order_for_pathes(model, paths::Vector{Vector{Int}}, args...; kwargs...)::Vector{Vector{<:Real}}
    r_ords = Vector{Vector{<:Real}}(undef, length(paths))
    Threads.@threads for i in eachindex(paths)
        r_ords[i] = find_reaction_order_for_single_path(model, paths[i], args...; kwargs...)
    end
    return r_ords
end

function dedup(v,keep_singular::Bool=true)
    isempty(v) && return v
    i_fst = findfirst(x->!isinf(x), v)
    result = [v[i_fst]]
    for x in Iterators.drop(v, i_fst)
        if x != last(result)
            if isinf(x) && !keep_singular
                continue
            end
            push!(result, x)
        end
    end
    return result
end

function group_sum(keys::AbstractVector, vals::AbstractVector;sort_values::Bool=true)
    @assert length(keys) == length(vals)
    dict = Dict{eltype(keys), eltype(vals)}()
    @inbounds for (k, v) in zip(keys, vals)
        dict[k] = get(dict, k, zero(v)) + v
    end
    dict = collect(dict)
    if sort_values
        sort!(dict, by=x->x[2], rev=true)
    end
    return dict
end


function get_binding_network_grh(Bnc::Bnc)::SimpleGraph
    g = SimpleGraph(Bnc.d + Bnc.n)
    for vi in eachindex(Bnc._valid_L_idx)
        for vj in Bnc._valid_L_idx[vi]
            add_edge!(g, vi, vj+Bnc.d)
        end
    end
    return g # get first d nodes as total, last n nodes as x
end




function get_node_size(model::Bnc, default=1; asymptotic=true, kwargs...)
    # seems properly handel non-asyntotic nodes
    vals = calc_volume(model;asymptotic=asymptotic, kwargs...) .|> x->x[1]
    idx = if asymptotic
        non_asym_idx = get_vertices(model, singular=nothing, asymptotic=false, return_idx=true) # non-asymptotic
        singular_asym_idx = get_vertices(model, singular=true, asymptotic=true, return_idx=true)# singular asymptotic
        vcat(non_asym_idx, singular_asym_idx)
    else
        get_vertices(model, singular=true, asymptotic=nothing, return_idx=true) # only care about singular
    end
    n_data = length(vals)-length(idx)

    Volume = vals .* n_data .* default^2
    Volume[idx] .= default^2
    return Dict(i=>sqrt(Volume[i]) for i in eachindex(Volume))
end



