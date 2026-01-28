#-------------------------------------------------------------
# Key visualizing functions
#----------------------------------------------------------------

"""
    SISO_plot(pths::SISOPaths, pth_idx; rand_line=false, rand_ray=false, extend=4, kwargs...) -> Figure

Plot a SISO path trajectory in x space colored by dominant regime.
"""
function SISO_plot(SISOPaths::SISOPaths,pth_idx;rand_line=false, rand_ray=false, extend=4, kwargs...)
    pth_idx = get_idx(SISOPaths, pth_idx)
    parameters = get_one_inner_point(SISOPaths.path_polys[pth_idx], rand_line=rand_line, rand_ray=rand_ray, extend=extend)
    @show parameters
    return SISO_plot(SISOPaths.bn, parameters, SISOPaths.change_qK_idx; kwargs...)
end
"""
    SISO_plot(model::Bnc, parameters, change_idx; npoints=1000, start=-6, stop=6, colormap=:rainbow,
        size=(800, 600), observe_x=nothing, add_archeatype_lines=false, asymptotic_only=false) -> Figure

Plot x trajectories for a single changing qK coordinate.
"""
function SISO_plot(model::Bnc, parameters, change_idx; 
        npoints=1000,start=-6, stop=6,colormap=:rainbow, size = (800,600),observe_x=nothing,
        add_archeatype_lines::Bool=false,
        asymptotic_only::Bool=false)


    change_idx = locate_sym_qK(model, change_idx)
    change_sym = "log"*repr(qK_sym(model)[change_idx])
    change_S = range(start, stop, npoints)


    # compute trajectory with change in logqK
    begin
        start_logqK = copy(parameters)|> x-> insert!(x, change_idx, start)
        end_logqK = copy(parameters)|> x-> insert!(x, change_idx, stop)
        logx =  x_traj_with_qK_change(model, start_logqK, end_logqK;
                                input_logspace=true, output_logspace=true, 
                                npoints=npoints,ensure_manifold=true)[2]

        logx_arch = if add_archeatype_lines
                    [qK2x(model, logqK;input_logspace=true, use_vtx=true,output_logspace=true) for logqK in range(start_logqK, end_logqK, npoints)]  # precompute x for archetype lines
                    else 
                        nothing
                    end
    end
    

    #assign color
    rgms = logx .|> x-> assign_vertex_x(model, x;input_logspace=true,asymptotic_only=asymptotic_only, return_idx=true)
    cmap = get_color_map(rgms; colormap=colormap)
    colors = getindex.(Ref(cmap), rgms)


    @info "Change in $(change_sym)"
    @info "parameters: $([i=>j for (i,j) in zip([model.q_sym;model.K_sym] |> x->deleteat!(x,change_idx), parameters)])"
    
    # draw plots
    draw_idx = isnothing(observe_x) ? (1:model.n) : locate_sym_x(model, observe_x)
    F = Figure(size = size)
    axes = Axis[]
    for (i, j) in enumerate(draw_idx)
        target_sym = "log"*repr(model.x_sym[j])
        @info "Target syms contains: $(target_sym) "
        ax = Axis(F[i,1]; xlabel = change_sym, ylabel = target_sym)
        push!(axes, ax)
        
        y = getindex.(logx, j)
        lines!(ax, change_S, y; color = colors)
        if add_archeatype_lines
            yarch = getindex.(logx_arch, j)
            lines!(ax, change_S, yarch; color = :black, linestyle = :dash)
        end
    end
    linkxaxes!(axes...)

    add_rgm_colorbar!(F, cmap)
    return F
end


struct RegimeColorMap{K,C,R}
    keys::Vector{K}          # 有序的 regime 列表（顺序即 colorbar 顺序）
    index::Dict{K,Int}       # regime => 颜色等级（1..n）
    cmap::C                  # categorical colormap (cgrad(..., categorical=true))
    render::R                # regime -> String 的渲染函数（可选）
end

Base.getindex(rcm::RegimeColorMap, key) = rcm.cmap[rcm.index[key]]

# Makie.to_colormap(rcm::RegimeColorMap) = let  # try to support using RegimeColorMap as colormap directly
#    cmap = copy(rcm.cmap)
#    cmap.values = rcm.keys 
# end


"""
    add_rgm_colorbar!(F, cmap::RegimeColorMap) -> nothing

Add a regime colorbar and labels to a Makie figure. 
"""
function add_rgm_colorbar!(F, cmap::RegimeColorMap)::Nothing

    text = cmap.render.(cmap.keys)
    txt_length = length(text[1])*26

    ncol = size(F.layout)[2]              # 当前已有列数
    cb_col   = ncol + 1                   # colorbar col
    text_col = ncol + 2
    
    # add colorbar
    Colorbar(F[:,end+1], colormap = cmap.cmap,ticks=[-1]) # DO NOT ADD COLORRANGE, by defining ticts= [-1] may other more elegant way?
    
    # add perm label
    ## initialize axis for text
    ax = let 
        ax = Axis(F[:,end+1])
        hidexdecorations!(ax)
        hideydecorations!(ax)
        hidespines!(ax)
        ylims!(ax, (0,1))
        ax
    end
    
    ## add text labels
    for i in eachindex(cmap.keys)
        y_pos = (i - 0.5)*(1/length(text))
        text!(ax, Point2f(0.5,y_pos); text = text[i], align = (:center, :center), color = :black)
    end
    
    ## adjust layout
    colsize!(F.layout, cb_col,   Fixed(0))
    colsize!(F.layout, text_col, Fixed(txt_length))

    return nothing
end

"""
    get_color_map(vec; colormap=:rainbow) -> (Dict, Any)

Return a mapping from values to color indices and a categorical colormap.
"""
function get_color_map(vec::AbstractArray; colormap=:rainbow,render_func=nothing,appendix = "#")::RegimeColorMap
    keys = sort!(unique(vec))

    col_map_dict = Dict(keys[i]=>i for i in eachindex(keys))
    
    cmap_disc = let
            crange =(1, length(keys))
            nlevels = crange[2]-crange[1] + 1
            cgrad(colormap, nlevels, categorical=true)
        end

    # funcion of how to render regime key
    render(rgm) = if !isnothing(render_func)
            render_func(rgm)
        elseif typeof(vec[1])<: AbstractArray  
            repr(rgm) |> strip_before_bracket
        else 
            appendix*string(rgm)
        end

    return RegimeColorMap(keys, col_map_dict, cmap_disc, render)
end

"""
    get_color_map(model::Bnc, args...; colormap=:rainbow, kwargs...) -> (Dict, Any)

Return a color map for vertices in a model.
"""
get_color_map(model::Bnc, args...;colormap=:rainbow, kwargs...) = get_color_map(get_vertices(model,args...;kwargs...), colormap=colormap)








#-------------------------------------------------------------
#Helper functions for plotting graphs
#-------------------------------------------------------------

"""
    get_edge_weight_vec(bnc::Bnc, change_qK_idx) -> Vector{Tuple{Edge, Dict{Symbol,Any}}}

Return edges with weights for a specified qK change direction.
"""
function get_edge_weight_vec(Bnc::Bnc,change_qK_idx)::Vector{Tuple{Edge,Dict{Symbol,Any}}}
    vg = get_vertices_graph!(Bnc;full=true)
    n = length(vg.neighbors)
    weight_vec = Vector{Tuple{Edge,Dict{Symbol,Any}}}()
    for (i, edges) in enumerate(vg.neighbors)
        nlt = get_nullity(Bnc,i)
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
    find_proper_bounds_for_graph_plot(p; x_margin=0.1, y_margin=0.1) -> Tuple

Return axis bounds with margins for a graph plot.
"""
function find_proper_bounds_for_graph_plot(p; x_margin=0.1, y_margin=0.1)
    # 支持 p.node_pos[] (Observable) 或直接 Vector / Dict
    # ps = node_pos isa Observable ? node_pos[] : node_pos
    coords = p.node_pos[]
    xs = first.(coords)
    ys = last.(coords)

    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)

    xspan = xmax - xmin
    yspan = ymax - ymin

    xmin -= x_margin * xspan
    xmax += x_margin * xspan
    ymin -= y_margin * yspan
    ymax += y_margin * yspan

    return (xmin, xmax, ymin, ymax)
end

"""
    set_proper_bounds_for_graph_plot!(ax, p; kwargs...) -> nothing

Set axis limits using graph plot bounds.
"""
set_proper_bounds_for_graph_plot!(ax, p; kwargs...) = let
    bounds = find_proper_bounds_for_graph_plot(p; kwargs...)
    limits!(ax, bounds...)
end 

"""
    get_edge_labels(bnc::Bnc; half=false, f=nothing) -> Dict{Edge,String}

Return edge labels for qK-space edges, optionally only one direction.
"""
function get_edge_labels(Bnc::Bnc; half::Bool=false,f=nothing)::Dict{Edge,String}
    vg = get_vertices_graph!(Bnc;full=true)
    labels = Dict{Edge,String}()
    for (i, edges) in enumerate(vg.neighbors)
        if get_nullity(Bnc,i) >1 # skip higher nullity
            continue
        end

        f = isnothing(f) ? (from, to) -> get_change_dir_qK(Bnc, from, to)|> x-> sym_direction(Bnc,x) : f

        for e in edges
            if isnothing(e.change_dir_qK) || (half && e.to < i)    # only label one direction
                continue
            end 
            labels[Edge(i, e.to)] = f(i, e.to)
        end
    end
    return labels
end




"""
    get_node_positions(model::Bnc; kwargs...) -> Vector{Point2f}

Return node positions derived from the x-neighbor graph layout.
"""
function get_node_positions(model::Bnc; kwargs...)
    grh = get_neighbor_graph_x(model)
    f,ax,p = graphplot(grh; kwargs...)
    posi = p.node_pos[]
    return posi
end

"""
    get_node_positions(p) -> Vector{Point2f}

Return node positions from a graphplot object.
"""
get_node_positions(p) = p.node_pos[] # support directly passing 
"""
    set_node_positions(p, new_pos) -> nothing

Set node positions on a graphplot object.
"""
set_node_positions(p, new_pos)= let
 new_posi = Point2f.(new_pos)
 p.node_pos[] = new_posi # support directly setting positions
end

"""
    get_node_colors(model; singular_color="#CCCCFF", asymptotic_color="#FFCCCC", regular_color="#CCFFCC") -> Vector{String}

Return node colors based on regime types.
"""
function get_node_colors(model; singular_color="#CCCCFF", asymptotic_color="#FFCCCC", regular_color="#CCFFCC")::Vector{String}
    node_colors = Vector{String}(undef, length(model.vertices_perm))
    for i in eachindex(model.vertices_perm)
        is_sin = is_singular(model, i)
        is_asym = is_asymptotic(model, i)
        if is_sin
            node_colors[i] = singular_color  # light blue for singular regimes
        else
            if is_asym
                node_colors[i] = asymptotic_color  # light green for asymptotic regimes
            else
                node_colors[i] = regular_color  # light red for regular regimes
            end
        end
    end
    return node_colors
end

"""
    get_node_labels(model::Bnc) -> Vector{String}

Return labels for nodes based on dominant species symbols.
"""
function get_node_labels(model::Bnc)
    model.vertices_perm .|>
        x -> model.x_sym[x] |>
        repr |> strip_before_bracket
end

"""
    get_node_size(model::Bnc; default_node_size=50, asymptotic=true, kwargs...) -> Dict

Return node sizes scaled by regime volumes.
"""
function get_node_size(model::Bnc; default_node_size=50, asymptotic=true, kwargs...)
    # seems properly handel non-asyntotic nodes
    vals = get_volumes(model; asymptotic=asymptotic, kwargs...) .|> x->x.mean
    
    zero_volume_idx = if asymptotic # both non-asymptotic and singular
        non_asym_idx = get_vertices(model, singular=nothing, asymptotic=false, return_idx=true) # non-asymptotic
        singular_asym_idx = get_vertices(model, singular=true, asymptotic=true, return_idx=true)# singular asymptotic
        vcat(non_asym_idx, singular_asym_idx)
    else # only singular
        get_vertices(model, singular=true, asymptotic=nothing, return_idx=true) # only care about singular
    end

    n_data = length(vals)-length(zero_volume_idx)

    Volume = vals .* n_data .* default_node_size^2
    Volume[zero_volume_idx] .= default_node_size^2
    return Dict(i=>sqrt(Volume[i]) for i in eachindex(Volume))
end






"""
    draw_graph(model; kwargs...) -> (Figure, Axis, Plot)

Draw the qK-neighbor graph of the model.
"""
draw_graph(model; kwargs...) = draw_graph(get_binding_network(model), get_neighbor_graph_qK(model); kwargs...)
"""
    draw_graph(grh::SISOPaths; kwargs...) -> (Figure, Axis, Plot)

Draw a SISO path graph with direction labels.
"""
function draw_graph(grh::SISOPaths;kwargs...)
    bn = get_binding_network(grh)
    change_sym = qK_sym(bn)[grh.change_qK_idx]
    grh = get_neighbor_graph_qK(grh)
    edge_labels = ["+"* repr(change_sym) for _ in 1:ne(grh)]
    f,ax,p = draw_graph(bn, grh; edge_labels = edge_labels, kwargs...)
    return f,ax,p
end

"""
    draw_graph(model::Bnc, grh=nothing; default_node_size=50, node_posi=nothing, edge_labels=nothing,
        node_labels=nothing, node_colors=nothing, add_rgm_idx=true, figsize=(1000,1000), kwargs...) -> (Figure, Axis, Plot)

Draw a graph with customizable node/edge annotations.
"""
function draw_graph(model::Bnc, grh=nothing; 
    default_node_size=50,
    node_posi =nothing,
    edge_labels=nothing,
    node_labels=nothing,
    node_colors=nothing,
    add_rgm_idx::Bool=true, 
    figsize=(1000,1000), 
    kwargs...)

    # use provided grh or compute a default neighbor graph
    grh = isnothing(grh) ? get_neighbor_graph_qK(model) : grh

    edge_labels =  isnothing(edge_labels) ? get_edge_labels(model) : edge_labels
    posi = isnothing(node_posi) ? get_node_positions(model) : Point2f.(node_posi)
    node_labels = isnothing(node_labels) ? get_node_labels(model) : node_labels
    node_colors = isnothing(node_colors) ? get_node_colors(model) : node_colors
    node_size = get_node_size(model; default_node_size=default_node_size)


    f = Figure(size = figsize)
    ax = Axis(f[1, 1],title = "Dominant mode of "*strip_before_bracket(repr(model.q_sym)), titlealign = :right,titlegap =2)

    
    p = graphplot!(ax, grh;
                    node_color = node_colors,
                    elabels = edge_labels,
                    node_size = node_size,
                    ilabels = node_labels,
                    layout = posi,
                    arrow_size = 20,
                    arrow_shift = 0.8,
                    edge_color = (:black, 0.7),
                    kwargs...,
                    )
    hidedecorations!(ax); hidespines!(ax)

    
    set_proper_bounds_for_graph_plot!(ax, p)

    if add_rgm_idx
        add_nodes_text!(ax,p)
    end
    
    return f, ax, p
end





"""
    add_nodes_text!(ax, p; texts=nothing, align=(:center,:bottom), color=:black, offset=(0,0), kwargs...) -> nothing

Add custom text labels to graph nodes.
"""
function add_nodes_text!(ax,p, texts=nothing; 
    align = (:center, :bottom), 
    color = :black,
    offset = (0,5), kwargs...)

    posi = p.node_pos

    texts = isnothing(texts) ? "#".*string.(1:length(posi[])) : texts

    text!(ax, posi; text = texts,align = align, color = color,offset = offset, kwargs...)
    return nothing
end



"""
    add_arrows!(ax, p, model, change_qK_idx; color=(:green, 0.5), kwargs...) -> nothing

Add arrows on an existing graph plot based on edge weights for a qK index.
"""
function add_arrows!(ax,p, model,change_qK_idx;color = (:green, 0.5), kwargs...)
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
        arrows2d!(ax, points...; shaftwidth=shaftwidth, tipwidth=tipwidth,tiplength=20,argmode=:endpoint,color=color,kwargs...)
    end
    return nothing
end


"""
    draw_qK_neighbor_grh(args...; kwargs...)

Alias for `draw_vertices_neighbor_graph` (legacy name).
"""
draw_qK_neighbor_grh(args...;kwargs...) = draw_vertices_neighbor_graph(args...; kwargs...)





"""
    draw_binding_network_grh(bnc::Bnc, grh=nothing; figsize=(800,800), q_color="#A2A544", x_color="#DBCC8C") -> (Figure, Axis, Plot)

Draw the bipartite binding network graph with q and x nodes.
"""
function draw_binding_network_grh(Bnc::Bnc,grh::Union{AbstractGraph, Nothing}=nothing; figsize=(800,800),q_color="#A2A544", x_color="#DBCC8C")
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


#-----------------------------------
# Draw plot helper functions
#--------------------------------------

"""
    find_bounds(lattice) -> BitMatrix

Compute regime boundaries using a Laplacian filter.
"""
function find_bounds(lattice)
    col_asym_x_bounds = imfilter(lattice, Kernel.Laplacian(), "replicate") # findboundary
    edge_map = col_asym_x_bounds .!= 0
    return edge_map
end





