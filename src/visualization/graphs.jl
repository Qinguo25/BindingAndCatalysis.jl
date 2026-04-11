struct RegimeColorMap{K,C,R}
    keys::Vector{K}
    index::Dict{K,Int}
    cmap::C
    render::R
end

Base.getindex(rcm::RegimeColorMap, key) = rcm.cmap[rcm.index[key]]

function add_rgm_colorbar!(F, cmap::RegimeColorMap)::Nothing
    text = cmap.render.(cmap.keys)
    txt_length = length(text[1]) * 26

    ncol = size(F.layout)[2]
    cb_col = ncol + 1
    text_col = ncol + 2

    Colorbar(F[:, end + 1], colormap=cmap.cmap, ticks=[-1])

    ax = let
        ax = Axis(F[:, end + 1])
        hidexdecorations!(ax)
        hideydecorations!(ax)
        hidespines!(ax)
        ylims!(ax, (0, 1))
        ax
    end

    for i in eachindex(cmap.keys)
        y_pos = (i - 0.5) * (1 / length(text))
        text!(ax, Point2f(0.5, y_pos); text=text[i], align=(:center, :center), color=:black)
    end

    colsize!(F.layout, cb_col, Fixed(0))
    colsize!(F.layout, text_col, Fixed(txt_length))
    return nothing
end

function get_color_map(vec::AbstractArray; colormap=:rainbow, render_func=nothing, appendix="#")::RegimeColorMap
    keys = sort!(unique(vec))
    col_map_dict = Dict(keys[i] => i for i in eachindex(keys))
    cmap_disc = let
        crange = (1, length(keys))
        nlevels = crange[2] - crange[1] + 1
        cgrad(colormap, nlevels, categorical=true)
    end

    render(rgm) = if !isnothing(render_func)
        render_func(rgm)
    elseif typeof(vec[1]) <: AbstractArray
        repr(rgm) |> strip_before_bracket
    else
        appendix * string(rgm)
    end

    return RegimeColorMap(keys, col_map_dict, cmap_disc, render)
end

get_color_map(model::Bnc, args...; colormap=:rainbow, kwargs...) = get_color_map(get_perms(model, args...; kwargs...), colormap=colormap)

function get_edge_weight_vec(Bnc::Bnc, change_qK_idx)::Vector{Tuple{Edge,Dict{Symbol,Any}}}
    vg = get_regimes_graph!(Bnc; full=true)
    weight_vec = Vector{Tuple{Edge,Dict{Symbol,Any}}}()
    for (i, edges) in enumerate(vg.neighbors)
        get_nullity(Bnc, i) > 1 && continue
        for e in edges
            !_edge_has_qK_interface(e) && continue
            iface = _edge_qK_interface(vg, e)
            iface === nothing && continue
            val = iface[1][change_qK_idx]
            if val > 1e-6
                push!(weight_vec, (Edge(i, e.to), Dict(:magnitude => val)))
            end
        end
    end
    return weight_vec
end

function find_proper_bounds_for_graph_plot(p; x_margin=0.1, y_margin=0.1)
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

set_proper_bounds_for_graph_plot!(ax, p; kwargs...) = limits!(ax, find_proper_bounds_for_graph_plot(p; kwargs...)...)

function get_edge_labels(Bnc::Bnc; half::Bool=false, f=nothing)::Dict{Edge,String}
    vg = get_regimes_graph!(Bnc; full=true)
    labels = Dict{Edge,String}()
    render = isnothing(f) ? (from, to) -> get_change_dir_qK(Bnc, from, to) |> x -> sym_direction(Bnc, x) : f
    for (i, edges) in enumerate(vg.neighbors)
        get_nullity(Bnc, i) > 1 && continue
        for e in edges
            if !_edge_has_qK_interface(e) || (half && e.to < i)
                continue
            end
            labels[Edge(i, e.to)] = render(i, e.to)
        end
    end
    return labels
end

@inline _resolve_graph_layout(grh::AbstractGraph, layout) = layout isa AbstractVector ? Point2f.(layout) : Point2f.(layout(grh))

function get_node_positions(model::Bnc; layout=Spring(; dim=2), kwargs...)
    return _resolve_graph_layout(get_neighbor_graph_x(model), layout)
end

function get_node_positions(grh::AbstractGraph; layout=Spring(; dim=2), kwargs...)
    return _resolve_graph_layout(grh, layout)
end

get_node_positions(p) = p.node_pos[]
set_node_positions(p, new_pos) = (p.node_pos[] = Point2f.(new_pos))

function get_node_colors(model, regimes=nothing; singular_color="#CCCCFF", asymptotic_color="#FFCCCC", regular_color="#CCFFCC")::Vector{String}
    all_regimes = isnothing(regimes) ? get_regimes(model; return_idx=true) : regimes
    node_colors = Vector{String}(undef, length(all_regimes))
    for (i, j) in enumerate(all_regimes)
        if is_singular(model, j)
            node_colors[i] = singular_color
        elseif is_asymptotic(model, j)
            node_colors[i] = asymptotic_color
        else
            node_colors[i] = regular_color
        end
    end
    return node_colors
end

function get_node_labels(model::Bnc)
    getfield.(_bind_regimes_data(model), :perm) .|> x -> model.x_sym[x] |> repr |> strip_before_bracket
end

function get_node_size(model::Bnc; default_node_size=50, asymptotic=true, kwargs...)
    vals = get_volumes(model; asymptotic=asymptotic, kwargs...) .|> x -> x.mean
    zero_volume_idx = if asymptotic
        non_asym_idx = get_regimes(model, singular=nothing, asymptotic=false, return_idx=true)
        singular_asym_idx = get_regimes(model, singular=true, asymptotic=true, return_idx=true)
        vcat(non_asym_idx, singular_asym_idx)
    else
        get_regimes(model, singular=true, asymptotic=nothing, return_idx=true)
    end

    n_data = length(vals) - length(zero_volume_idx)
    volume = vals .* n_data .* default_node_size^2
    volume[zero_volume_idx] .= default_node_size^2
    return Dict(i => sqrt(volume[i]) for i in eachindex(volume))
end

@inline function _node_subset_by_nullity(model::Bnc; hide_nullity_ge_2::Bool=false)
    return hide_nullity_ge_2 ? [i for i in 1:n_regimes(model) if get_nullity(model, i) <= 1] : collect(1:n_regimes(model))
end

function _filter_edge_labels_for_nodes(edge_labels, grh::AbstractGraph, keep_nodes::Vector{Int})
    keep_set = Set(keep_nodes)
    old_to_new = Dict(keep_nodes[i] => i for i in eachindex(keep_nodes))

    if edge_labels isa Dict
        labels = Dict{Edge,Any}()
        for (e, lbl) in edge_labels
            (e.src in keep_set && e.dst in keep_set) || continue
            labels[Edge(old_to_new[e.src], old_to_new[e.dst])] = lbl
        end
        return labels
    elseif edge_labels isa AbstractVector
        labels = Any[]
        for (e, lbl) in zip(edges(grh), edge_labels)
            (src(e) in keep_set && dst(e) in keep_set) || continue
            push!(labels, lbl)
        end
        return labels
    else
        return edge_labels
    end
end

function _materialize_node_sizes(raw_node_size, node_indices::AbstractVector{<:Integer})
    if raw_node_size isa AbstractDict
        return [raw_node_size[i] for i in node_indices]
    else
        vals = collect(raw_node_size)
        return vals[node_indices]
    end
end

draw_graph(model; kwargs...) = draw_graph(get_binding_network(model), get_neighbor_graph_qK(model); kwargs...)

function draw_graph(grh::SISOPaths; layout=Spring(; dim=2), kwargs...)
    bn = get_binding_network(grh)
    change_sym = qK_sym(bn)[grh.change_qK_idx]
    qk_grh = get_neighbor_graph_qK(grh)
    edge_labels = ["+" * repr(change_sym) for _ in 1:ne(qk_grh)]
    return draw_graph(bn, qk_grh; edge_labels=edge_labels, layout=layout, kwargs...)
end

function draw_graph(
    model::Bnc,
    grh=nothing;
    default_node_size=50,
    node_posi=nothing,
    node_size=nothing,
    edge_labels=nothing,
    node_labels=nothing,
    node_colors=nothing,
    add_rgm_idx::Bool=true,
    hide_nullity_ge_2::Bool=false,
    figsize=(1000, 1000),
    layout=Spring(; dim=2),
    kwargs...,
)
    grh = isnothing(grh) ? get_neighbor_graph_qK(model) : grh
    full_grh = grh

    edge_labels = isnothing(edge_labels) ? get_edge_labels(model) : edge_labels
    posi = isnothing(node_posi) ? get_node_positions(grh; layout=layout) : Point2f.(node_posi)
    node_labels = isnothing(node_labels) ? get_node_labels(model) : collect(node_labels)
    node_colors = isnothing(node_colors) ? get_node_colors(model) : collect(node_colors)
    raw_node_size = isnothing(node_size) ? get_node_size(model; default_node_size=default_node_size) : node_size

    keep_nodes = _node_subset_by_nullity(model; hide_nullity_ge_2=hide_nullity_ge_2)
    node_indices = if length(keep_nodes) < nv(grh)
        grh, _ = induced_subgraph(grh, keep_nodes)
        edge_labels = _filter_edge_labels_for_nodes(edge_labels, full_grh, keep_nodes)
        posi = posi[keep_nodes]
        node_labels = node_labels[keep_nodes]
        node_colors = node_colors[keep_nodes]
        keep_nodes
    else
        collect(1:length(node_labels))
    end
    node_size_vec = _materialize_node_sizes(raw_node_size, node_indices)

    f = Figure(size=figsize)
    ax = Axis(
        f[1, 1],
        title="Dominant mode of " * strip_before_bracket(repr(model.q_sym)),
        titlealign=:right,
        titlegap=2,
    )

    p = graphplot!(
        ax,
        grh;
        node_color=node_colors,
        elabels=edge_labels,
        node_size=node_size_vec,
        ilabels=node_labels,
        layout=posi,
        arrow_size=20,
        arrow_shift=0.8,
        edge_color=(:black, 0.7),
        kwargs...,
    )
    hidedecorations!(ax)
    hidespines!(ax)
    set_proper_bounds_for_graph_plot!(ax, p)

    add_rgm_idx && add_nodes_text!(ax, p)
    return f, ax, p
end

function add_nodes_text!(
    ax,
    p,
    texts=nothing;
    align=(:center, :bottom),
    color=:black,
    offset=(0, 5),
    kwargs...,
)
    posi = p.node_pos
    texts = isnothing(texts) ? "#" .* string.(1:length(posi[])) : texts
    text!(ax, posi; text=texts, align=align, color=color, offset=offset, kwargs...)
    return nothing
end

add_vertices_idx!(args...; kwargs...) = add_nodes_text!(args...; kwargs...)

function add_arrows!(ax, p, model, change_qK_idx; color=(:green, 0.5), kwargs...)
    edge_dir = get_edge_weight_vec(model, change_qK_idx)
    arws1 = map(edge_dir) do (edge, meta)
        u, v = edge.src, edge.dst
        mag = meta[:magnitude]
        p1 = p.node_pos[][u]
        p2 = p.node_pos[][v]
        Δp = p2 .- p1
        norm_Δp = norm(Δp)
        p1 = p1 .+ Δp / norm_Δp .* 0.1
        p2 = p2 .- Δp / norm_Δp .* 0.1
        shaftwidth = mag * 8
        tipwidth = mag * 15
        return [p1, p2], shaftwidth, tipwidth
    end
    for (points, shaftwidth, tipwidth) in arws1
        arrows2d!(ax, points...; shaftwidth=shaftwidth, tipwidth=tipwidth, tiplength=20, argmode=:endpoint, color=color, kwargs...)
    end
    return nothing
end

draw_vertices_neighbor_graph(args...; kwargs...) = draw_graph(args...; kwargs...)
draw_qK_neighbor_grh(args...; kwargs...) = draw_graph(args...; kwargs...)

function draw_binding_network_grh(Bnc::Bnc, grh::Union{AbstractGraph,Nothing}=nothing; figsize=(800, 800), q_color="#A2A544", x_color="#DBCC8C")
    f = Figure(size=figsize)
    grh = isnothing(grh) ? get_binding_network_grh(Bnc) : grh
    ax = Axis(f[1, 1])
    node_labels = [i <= Bnc.d ? repr(Bnc.q_sym[i]) : repr(Bnc.x_sym[i - Bnc.d]) for i in 1:(Bnc.d + Bnc.n)]
    node_colors = [i <= Bnc.d ? q_color : x_color for i in 1:(Bnc.d + Bnc.n)]
    p = graphplot!(
        ax,
        grh;
        node_color=node_colors,
        edge_color=(:black, 0.7),
        ilabels=node_labels,
        arrow_size=20,
        arrow_shift=0.8,
        layout=Spring(; dim=2),
    )
    hidedecorations!(ax)
    hidespines!(ax)
    return f, ax, p
end
