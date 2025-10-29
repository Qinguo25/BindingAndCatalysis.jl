#-----------------------------------------------------------------------------------------------
#This is graph associated functions for Bnc models and archetyple behaviors associated code
#-----------------------------------------------------------------------------------------------


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
    vg = get_vertices_graph!(Bnc;fulfill=true)
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
    vg = get_vertices_graph!(Bnc;fulfill=true)
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

"""
    get_edge_labels(Bnc; sym=false) -> Dict{Edge,String}
Get edge labels for qK-space edges. IF half, only label one direction.
"""
function get_edge_labels(Bnc::Bnc; sym::Bool=false, half::Bool=true)::Dict{Edge,String}
    vg = get_vertices_graph!(Bnc;fulfill=true)
    labels = Dict{Edge,String}()
    for (i, edges) in enumerate(vg.neighbors)
        if get_nullity!(Bnc,i) >1
            continue
        end
        for e in edges
            if isnothing(e.change_dir_qK) || (half && e.to < i)    # only label one direction
                continue
            end 
            ch_dir = round.(e.change_dir_qK;digits=1)
            labels[Edge(i,e.to)] = sym ? _chdir_to_sym(Bnc,ch_dir) : repr(Vector(ch_dir))
        end
    end
    return labels
end

function _chdir_to_sym(Bnc::Bnc,dir)
    rst = ""
    for i in 1:Bnc.d
        if dir[i] > 0
            rst *= "+"*repr(Bnc.q_sym[i])*" "
        elseif dir[i] < 0
            rst *= "-"*repr(Bnc.q_sym[i])*" "
        end
    end
    rst*="; "
    for j in 1:Bnc.r
        if dir[j+Bnc.d] > 0
            rst *= "+"*repr(Bnc.K_sym[j])*" "
        elseif dir[j+Bnc.d] < 0
            rst *= "-"*repr(Bnc.K_sym[j])*" "
        end
    end
    return rst
end

"""
    get_edge_weight_vec(Bnc, change_qK_idx) -> Vector{Tuple{Edge,Dict{Symbol,Any}}}
Given the change_qK_idx, return the edges that contains the change in that direction with weight magnitude.
"""
function get_edge_weight_vec(Bnc::Bnc,change_qK_idx)::Vector{Tuple{Edge,Dict{Symbol,Any}}}
    vg = get_vertices_graph!(Bnc;fulfill=true)
    n = length(vg.neighbors)
    weight_vec = Vector{Tuple{Edge,Dict{Symbol,Any}}}()
    for (i, edges) in enumerate(vg.neighbors)
        nlt = get_nullity!(Bnc,i)
        if nlt >1
            continue
        end
        for e in edges
            if isnothing(e.change_dir_qK)
                continue
            end 
            # if nlt ==0 
            #     val = e.change_dir_qK[change_qK_idx]
            #     if val > 1e-6
            #         push!(weight_vec, (Edge(i,e.to), Dict(:magnitude=>1.0)))
            #     end
            # else
                val = e.change_dir_qK[change_qK_idx]
                if val > 1e-6
                    push!(weight_vec, (Edge(i,e.to), Dict(:magnitude=>val)))
                end
            # end 
        end
    end
    return weight_vec
end
"""
    Add arrows on an existing graph plot based on edge weights from change_qK_idx.
"""
function add_arrows!(ax,p, model,change_qK_idx; kwargs...)
    edge_dir = get_edge_weight_vec(model,change_qK_idx)
    arws1 = map(edge_dir) do (edge, meta)
        u,v = edge.src, edge.dst
        mag = meta[:magnitude]
            p1 = p.node_pos[][u]
            p2 = p.node_pos[][v]
            Δp = p2.-p1
            norm_Δp = norm(Δp)
            p1 = p1 .+ Δp/norm_Δp .*0.1
            p2 = p2 .- Δp/norm_Δp .*0.1
            shaftwidth = mag *8
            tipwidth = mag *15
            return [p1,p2], shaftwidth, tipwidth
        end
    for (points, shaftwidth, tipwidth) in arws1
        arrows2d!(ax, points...; shaftwidth=shaftwidth, tipwidth=tipwidth,tiplength=20,argmode=:endpoint,kwargs...)
    end
    return nothing
end

get_sources(g::DiGraph) = [v for v in vertices(g) if indegree(g, v) == 0]
get_sinks(g::DiGraph)   = [v for v in vertices(g) if outdegree(g, v) == 0]

function find_all_complete_paths(g::DiGraph)
    sources = get_sources(g)
    sinks   = get_sinks(g)
    paths = Vector{Vector{Int}}()

    function dfs(path)
        lastv = path[end]
        if lastv in sinks
            push!(paths, copy(path))
            return
        end
        for nb in outneighbors(g, lastv)
            if nb ∉ path  # 避免循环
                dfs([path...; nb])
            end
        end
    end

    for s in sources
        dfs([s])
    end

    return paths
end


function plot_model_graph(model::Bnc, grh=nothing; arrow_idx=nothing, figsize=(1000,1000), arrow_color=(:green, 0.5))
    # use provided grh or compute a default neighbor graph
    grh = isnothing(grh) ? get_qK_neighbor_grh(model) : grh

    # prepare labels / colors
    node_labels = model.vertices_perm .|> (x-> Int.(x) |> repr)
    node_colors = [model.vertices_nullity[i] > 0 ? "#CCCCFF" : "#FFCCCC" for i in eachindex(model.vertices_nullity)]
    edge_labels = get_edge_labels(model, sym=true, half=false)

    # build figure and plot
    f = Figure(size = figsize)
    ax = Axis(f[1, 1])

    p = graphplot!(ax, SimpleDiGraph(grh),
                    node_color = node_colors,
                    elabels = edge_labels,
                    edge_color = (:black, 0.7),
                    ilabels = node_labels,
                    arrow_size = 20,
                    arrow_shift = 0.8,
                    layout = Spring(; dim = 2))

    # add arrows from your helper (keeps behavior from original cell)
    if !isnothing(arrow_idx)
        add_arrows!(ax, p, model, arrow_idx; color = arrow_color)
    end

    hidedecorations!(ax); hidespines!(ax)
    return f, ax, p
end

function find_conditions_for_path(model, path, change_qK_idx)::Polyhedron
    el_dim = BitSet(change_qK_idx)
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
    ins_fianl = reduce((a,b)->intersect(a,b), poly_ins)
    detecthlinearity!(ins_fianl)
    return ins_fianl
end

function find_reaction_order_for_path(model, path, change_qK_idx, observe_x_idx; deduplicate::Bool=false)::Vector{<:Real}
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
        r_ord = dedup(r_ord)
    end
    return r_ord
end

function dedup(v)
    isempty(v) && return v
    result = [first(v)]
    for x in Iterators.drop(v, 1)
        if x != last(result)
            push!(result, x)
        end
    end
    return result
end


