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

"""
    get_edge_labels(Bnc; sym=false) -> Dict{Edge,String}
Get edge labels for qK-space edges. IF half, only label one direction.
"""
function get_edge_labels(Bnc::Bnc; sym::Bool=false, half::Bool=true)::Dict{Edge,String}
    vg = get_vertices_graph!(Bnc;full=true)
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
            labels[Edge(i,e.to)] = sym ? sym_direction(Bnc,ch_dir) : repr(Vector(ch_dir))
        end
    end
    return labels
end



"""
    get_edge_weight_vec(Bnc, change_qK_idx) -> Vector{Tuple{Edge,Dict{Symbol,Any}}}
Given the change_qK_idx, return the edges that contains the change in that direction with weight magnitude.
"""
function get_edge_weight_vec(Bnc::Bnc,change_qK_idx)::Vector{Tuple{Edge,Dict{Symbol,Any}}}
    vg = get_vertices_graph!(Bnc;full=true)
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


function find_conditions_for_pathes_direct(model::Bnc, paths, change_qK_idx)::Vector{Polyhedron}
    polys = Vector{Polyhedron}(undef, length(paths))
    Threads.@threads for i in eachindex(paths)
        polys[i] = find_conditions_for_path_direct(model, paths[i], change_qK_idx)
    end
    return polys
end


function find_conditions_for_pathes(model::Bnc, paths, change_qK_idx)::Vector{Polyhedron}
    polys = Vector{Polyhedron}(undef, length(paths))
    Threads.@threads for i in eachindex(paths)
        polys[i] = find_conditions_for_path(model, paths[i], change_qK_idx)
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

function group_sum(keys::AbstractVector, vals::AbstractVector)
    @assert length(keys) == length(vals)
    dict = Dict{eltype(keys), eltype(vals)}()
    @inbounds for (k, v) in zip(keys, vals)
        dict[k] = get(dict, k, zero(v)) + v
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



function draw_binding_network_grh(Bnc::Bnc,grh::AbstractGraph; figsize=(800,800))
    f = Figure(size = figsize)
    ax = Axis(f[1, 1])
    node_labels = [i <= Bnc.d ? repr(Bnc.q_sym[i]) : repr(Bnc.x_sym[i-Bnc.d]) for i in 1:(Bnc.d + Bnc.n)]
    node_colors = [i <= Bnc.d ? "#FFCCCC" : "#CCCCFF" for i in 1:(Bnc.d + Bnc.n)]
    p = graphplot!(ax, grh,
                    node_color = node_colors,
                    edge_color = (:black, 0.7),
                    ilabels = node_labels,
                    arrow_size = 20,
                    arrow_shift = 0.8,
                    layout = Spring(; dim = 2))
    hidedecorations!(ax); hidespines!(ax)
    return f, ax, p
end


draw_qK_neighbor_grh(args...;kwargs...) = draw_vertices_neighbor_graph(args...; kwargs...)

function draw_vertices_neighbor_graph(model::Bnc, grh=nothing; 
    arrow_idx=nothing, 
    edge_labels=nothing, 
    figsize=(1000,1000), 
    arrow_color=(:green, 0.5),
    kwargs...)
    # use provided grh or compute a default neighbor graph
    grh = isnothing(grh) ? get_qK_neighbor_grh(model) : grh
    # prepare labels / colors
    # node_labels = model.vertices_perm .|> (x-> Int.(x) |> repr)
    node_labels = model.vertices_perm .|> x->model.x_sym[x] |> x->repr(x)[4:end] # remove the "Num"

    # assign node_colors based on nullity and asymptoticity
    node_colors = Vector{String}(undef, length(model.vertices_perm))
    for i in eachindex(model.vertices_perm)
        is_sin = is_singular(model, i)
        is_asym = is_asymptotic(model, i)
        if is_sin
            node_colors[i] = "#CCCCFF"  # light blue for singular regimes
        else
            if is_asym
                node_colors[i] = "#FFCCCC"  # light green for asymptotic regimes
            else
                node_colors[i] = "#CCFFCC"  # light red for regular regimes
            end
        end
    end

    # node_colors = [model.vertices_nullity[i] > 0 ? "#CCCCFF" : "#FFCCCC" for i in eachindex(model.vertices_nullity)]

    f = Figure(size = figsize)
    ax = Axis(f[1, 1],title = "Dominant mode of "*repr(model.q_sym)[4:end], titlealign = :right,titlegap =2)

    if !isnothing(edge_labels) && isa(edge_labels, String)
        edge_labels = repeat([edge_labels], ne(grh))
    end
    p = graphplot!(ax, grh;
                    node_color = node_colors,
                    elabels = edge_labels,
                    edge_color = (:black, 0.7),
                    ilabels = node_labels,
                    arrow_size = 20,
                    arrow_shift = 0.8,
                    kwargs...,
                    layout = Spring(; dim = 2))
    # p.node_pos[] = posi
    # add arrows from your helper (keeps behavior from original cell)
    if !isnothing(arrow_idx)
        add_arrows!(ax, p, model, arrow_idx; color = arrow_color)
    end


    hidedecorations!(ax); hidespines!(ax)
    return f, ax, p
end

function add_vertices_idx!(ax,p)
    posi = p.node_pos[]
    text!(ax, posi; text = "#".*string.(1:length(posi)),align = (:center, :bottom), color = :black,offset = (0,5))
    return nothing
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

function find_lim_of_p(p,x_prochunge=1.2, y_prochunge=1.2)
    ps = p.node_pos[]
    xs = [ps[i][1] for i in 1:length(ps)]
    ys = [ps[i][2] for i in 1:length(ps)]
    xmin = minimum(xs)
    xmax = maximum(xs)
    ymin = minimum(ys)
    ymax = maximum(ys)

    xmid = (xmin + xmax)/2
    ymid = (ymin + ymax)/2
    xmin = xmid + (xmin - xmid)*x_prochunge
    xmax = xmid + (xmax - xmid)*x_prochunge
    ymin = ymid + (ymin - ymid)*y_prochunge
    ymax = ymid + (ymax - ymid)*y_prochunge

    return (xmin, xmax, ymin, ymax)
end

function SISO_plot(model, cond_s, change_idx; npoints=1000,start=-6, stop=6,cmap=:rainbow, size = (800,600),draw_idx=nothing)
    change_sym = "log"*repr([model.q_sym;model.K_sym][change_idx])
    change_S = range(start, stop, npoints)
    start_logqK = copy(cond_s)|> x-> insert!(x, change_idx, start)
    end_logqK = copy(cond_s)|> x-> insert!(x, change_idx, stop)
    logx = x_traj_with_qK_change(model, start_logqK, end_logqK;input_logspace=true, output_logspace=true, 
                                    tstops = range(0,1,npoints), saveat = range(0,1,npoints))

    #assign color
    rgm = logx[2] .|> x-> assign_vertex_x(model, x;input_logspace=true) |> x->get_idx(model,x)

    unique_rgm = unique(rgm)
    col_map_dict = Dict(unique_rgm[i]=>i for i in eachindex(unique_rgm))

    crange =(1, length(unique_rgm))
    nlevels = crange[2]-crange[1] + 1
    cmap_disc = cgrad(cmap, nlevels, categorical=true)

    @show change_sym
        
    draw_idx = isnothing(draw_idx) ? (1:model.n) : draw_idx
    F = Figure(size = size)
    for (i, j) in enumerate(draw_idx)
        target_sym = "log"*repr(model.x_sym[j])
        @show target_sym
        ax = Axis(F[i,1]; xlabel = change_sym, ylabel = target_sym)
        lines!(ax, change_S, logx[2] .|> x-> x[j]; color = map(r->col_map_dict[r], rgm), colorrange = crange, colormap = cmap)
    end
    Colorbar(F[:,end+1], colorrange = crange, colormap = cmap_disc,ticks=[0])

    # add perm label
    ax = Axis(F[:,end+1])
    hidexdecorations!(ax)
    hideydecorations!(ax)
    hidespines!(ax)
    colsize!(F.layout,3,Fixed(30))
    colsize!(F.layout,2,Fixed(0))

    for i in eachindex(unique_rgm)
        y_pos = (i - 0.5)*(1/length(unique_rgm))
        text!(ax, Point2f(0.5,y_pos); text = "#"*string(unique_rgm[i]), align = (:center, :center), color = :black)
    end
    ylims!(ax, (0,1))
    return F
end