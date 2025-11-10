using CairoMakie, GraphMakie
using GraphMakie.NetworkLayout
using Latexify


#-------------------------------------------------------------
# Key visualizing functions
#----------------------------------------------------------------
function SISO_plot(model, cond_s, change_idx; 
        npoints=1000,start=-6, stop=6,cmap=:rainbow, size = (800,600),draw_idx=nothing,
        asymptotic=false)

    change_sym = "log"*repr([model.q_sym;model.K_sym][change_idx])
    change_S = range(start, stop, npoints)
    start_logqK = copy(cond_s)|> x-> insert!(x, change_idx, start)
    end_logqK = copy(cond_s)|> x-> insert!(x, change_idx, stop)
    logx = x_traj_with_qK_change(model, start_logqK, end_logqK;input_logspace=true, output_logspace=true, 
                                    tstops = range(0,1,npoints), saveat = range(0,1,npoints))

    #assign color
    rgm = logx[2] .|> x-> assign_vertex_x(model, x;input_logspace=true,asymptotic=asymptotic) |> x->get_idx(model,x)

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


draw_qK_neighbor_grh(args...;kwargs...) = draw_vertices_neighbor_graph(args...; kwargs...)





function draw_binding_network_grh(Bnc::Bnc,grh::Union{AbstractGraph, Nothing}=nothing; figsize=(800,800),q_color="#FFCCCC", x_color="#CCCCFF")
    f = Figure(size = figsize)
    grh = isnothing(grh) ? get_binding_network_grh(Bnc) : grh
    ax = Axis(f[1, 1])
    node_labels = [i <= Bnc.d ? repr(Bnc.q_sym[i]) : repr(Bnc.x_sym[i-Bnc.d]) for i in 1:(Bnc.d + Bnc.n)]
    node_colors = [i <= Bnc.d ? q_color : x_color for i in 1:(Bnc.d + Bnc.n)]
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








#-------------------------------------------------------------
#Helper functions for plotting
#-------------------------------------------------------------
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




function find_proper_bounds_for_graph_plot(p,x_prochunge=1.2, y_prochunge=1.2)
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