#-------------------------------------------------------------
# Key visualizing functions
#----------------------------------------------------------------

"""
plot function, given a change in qK, plot the trajectory of x and color by dominant regime. 
"""
function SISO_plot(SISOPaths::SISOPaths,pth_idx;rand_line=false, rand_ray=false, extend=4, kwargs...)
    parameters = get_one_inner_point(SISOPaths.path_polys[pth_idx], rand_line=rand_line, rand_ray=rand_ray, extend=extend)
    @show parameters
    return SISO_plot(SISOPaths.bn, parameters, SISOPaths.change_qK_idx; kwargs...)
end
function SISO_plot(model::Bnc, parameters, change_idx; 
        npoints=1000,start=-6, stop=6,colormap=:rainbow, size = (800,600),draw_idx=nothing,
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
    unique_rgm =  sort!(unique(rgms))
    col_map_dict, colormap_disc = get_color_map(unique_rgm; colormap=colormap)
    colors = getindex.(Ref(col_map_dict), rgms)


    @info "Change in $(change_sym)"
    @info "parameters: $([i=>j for (i,j) in zip([model.q_sym;model.K_sym] |> x->deleteat!(x,change_idx), parameters)])"
    
    # draw plots
    draw_idx = isnothing(draw_idx) ? (1:model.n) : draw_idx
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

    add_rgm_colorbar!(F, unique_rgm; colormap=colormap_disc)
    return F
end


"""
Add the colorbar of the regimes 
- `F`: the figure to add colorbar to
- `unique_rgm`: the unique regimes to label on the colorbar, shall be sorted in the same order as the colormap, eg. `unique_rgm = sort!(unique(rgms))`
- `colormap`: the colormap to use for the colorbar, shall be striped with the same number of colors as unique_rgm,
eg. `add_rgm_colorbar!(F, unique_rgm; colormap=cmap_disc)`
"""
function add_rgm_colorbar!(F, unique_rgm;colormap)
    txt_length = length(string.(unique_rgm[1]))*26

    render(rgm) = if typeof(unique_rgm[1])<: AbstractArray
        repr(rgm)
    else
        "#"*string(rgm)
    end

    ncol = size(F.layout)[2]              # 当前已有列数
    cb_col   = ncol + 1                   # colorbar col
    text_col = ncol + 2
    
    # add colorbar
    Colorbar(F[:,end+1], colormap = colormap,ticks=[-1]) # DO NOT ADD COLORRANGE, 
    #add perm label
    ax = Axis(F[:,end+1])
    hidexdecorations!(ax)
    hideydecorations!(ax)
    hidespines!(ax)
    for i in eachindex(unique_rgm)
        y_pos = (i - 0.5)*(1/length(unique_rgm))
        text!(ax, Point2f(0.5,y_pos); text = render(unique_rgm[i]), align = (:center, :center), color = :black)
    end
    ylims!(ax, (0,1))
    colsize!(F.layout, cb_col,   Fixed(0))
    colsize!(F.layout, text_col, Fixed(txt_length))
    return nothing
end

"""
"""
function get_color_map(vec::AbstractArray; colormap=:rainbow)
    keys = sort!(unique(vec))
    col_map_dict = Dict(keys[i]=>i for i in eachindex(keys))
    crange =(1, length(keys))
    nlevels = crange[2]-crange[1] + 1
    cmap_disc = cgrad(colormap, nlevels, categorical=true)
    return col_map_dict, cmap_disc
end
get_color_map(model::Bnc, args...;colormap=:rainbow, kwargs...) = get_color_map(get_vertices(model,args...;kwargs...), colormap=colormap)








#-------------------------------------------------------------
#Helper functions for plotting graphs
#-------------------------------------------------------------

"""
    get_edge_weight_vec(Bnc, change_qK_idx) -> Vector{Tuple{Edge,Dict{Symbol,Any}}}
Given the change_qK_idx, return the edges that contains the change in that direction with weight magnitude.
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
    get_edge_labels(Bnc; sym=false) -> Dict{Edge,String}
Get edge labels for qK-space edges. IF half, only label one direction.
"""
function get_edge_labels(Bnc::Bnc; sym::Bool=false, half::Bool=true)::Dict{Edge,String}
    vg = get_vertices_graph!(Bnc;full=true)
    labels = Dict{Edge,String}()
    for (i, edges) in enumerate(vg.neighbors)
        if get_nullity(Bnc,i) >1 # skip higher nullity
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
    Get fixed node positions for the qK-neighbor from x-neighbor graph of the model.
"""
function get_node_positions(model::Bnc; kwargs...)
    grh = get_neighbor_graph_x(model)
    f,ax,p = graphplot(grh; kwargs...)
    posi = p.node_pos[]
    return posi
end

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

function get_node_labels(model::Bnc)
    model.vertices_perm .|>
        x -> model.x_sym[x] |>
        repr |> strip_before_bracket
end

function get_node_size(model::Bnc; default_node_size=50, asymptotic=true, kwargs...)
    # seems properly handel non-asyntotic nodes
    vals = (asymptotic ? get_volumes(model) : calc_volume(model;asymptotic=asymptotic, kwargs...)) .|> x->x[1]
    
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
    Draw the qK-neighbor graph of the model, with optional edge labels, node colors, and node sizes.
"""
function draw_vertices_neighbor_graph(model::Bnc, grh=nothing; 
    default_node_size=50,
    edge_labels=nothing, 
    figsize=(1000,1000), 
    kwargs...)

    # use provided grh or compute a default neighbor graph
    grh = isnothing(grh) ? get_neighbor_graph_qK(model) : grh

    edge_labels =   if isnothing(edge_labels) 
                        get_edge_labels(model, sym=true) 
                    elseif isa(edge_labels, String) 
                        repeat([edge_labels], ne(grh)) 
                    else 
                        edge_labels
                    end
    
    posi = get_node_positions(model)
    node_labels = get_node_labels(model)
    node_colors = get_node_colors(model)
    node_size = get_node_size(model; default_node_size=default_node_size)


    f = Figure(size = figsize)
    ax = Axis(f[1, 1],title = "Dominant mode of "*repr(model.q_sym)[4:end], titlealign = :right,titlegap =2)

    
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

    bounds = find_proper_bounds_for_graph_plot(p)
    limits!(ax, bounds...)

    return f, ax, p
end

function draw_vertices_neighbor_graph(grh::SISOPaths,args...;kwargs...)
    edge_labels = "+"* repr([get_binding_network(grh).q_sym; get_binding_network(grh).K_sym][grh.change_qK_idx])
    f,ax,p = draw_vertices_neighbor_graph(get_binding_network(grh), grh.qK_grh, args...; edge_labels = edge_labels, kwargs...)
    return f,ax,p
end

function add_vertices_idx!(ax,p)
    posi = p.node_pos[]
    text!(ax, posi; text = "#".*string.(1:length(posi)),align = (:center, :bottom), color = :black,offset = (0,5))
    return nothing
end

"""
    Add arrows on an existing graph plot based on edge weights from change_qK_idx.
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


draw_qK_neighbor_grh(args...;kwargs...) = draw_vertices_neighbor_graph(args...; kwargs...)





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

# find boundary between different regimes for regime map, to draw boundary for different regimes.
function find_bounds(lattice)
    col_asym_x_bounds = imfilter(lattice, Kernel.Laplacian(), "replicate") # findboundary
    edge_map = col_asym_x_bounds .!= 0
    return edge_map
end








