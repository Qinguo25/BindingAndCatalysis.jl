export SIMO_plot, get_edge_labels, set_proper_bounds_for_graph_plot!
export get_node_positions, get_node_colors, get_node_labels, get_node_size
export draw_graph, add_vertices_idx!, add_arrows!, add_nodes_text!, set_node_positions
export draw_qK_neighbor_grh, find_bounds, add_rgm_colorbar!, get_color_map
export draw_ROP
export plot_polyhedron_slices
export plot_binding_regime_partition, plot_bnc_regime_partition, plot_qcat_slice_with_flux

function _visualization_extension_required(name::Symbol)
    throw(
        ArgumentError(
            "`$name` requires optional visualization packages. Add GraphMakie and a Makie backend to your active environment, then load them first, for example `using CairoMakie, GraphMakie`.",
        ),
    )
end

SIMO_plot(args...; kwargs...) = _visualization_extension_required(:SIMO_plot)
get_edge_labels(args...; kwargs...) = _visualization_extension_required(:get_edge_labels)
set_proper_bounds_for_graph_plot!(args...; kwargs...) =
    _visualization_extension_required(:set_proper_bounds_for_graph_plot!)
get_node_positions(args...; kwargs...) = _visualization_extension_required(:get_node_positions)
get_node_colors(args...; kwargs...) = _visualization_extension_required(:get_node_colors)
get_node_labels(args...; kwargs...) = _visualization_extension_required(:get_node_labels)
get_node_size(args...; kwargs...) = _visualization_extension_required(:get_node_size)
draw_graph(args...; kwargs...) = _visualization_extension_required(:draw_graph)
add_vertices_idx!(args...; kwargs...) = _visualization_extension_required(:add_vertices_idx!)
add_arrows!(args...; kwargs...) = _visualization_extension_required(:add_arrows!)
add_nodes_text!(args...; kwargs...) = _visualization_extension_required(:add_nodes_text!)
set_node_positions(args...; kwargs...) = _visualization_extension_required(:set_node_positions)
draw_vertices_neighbor_graph(args...; kwargs...) =
    _visualization_extension_required(:draw_vertices_neighbor_graph)
draw_qK_neighbor_grh(args...; kwargs...) = _visualization_extension_required(:draw_qK_neighbor_grh)
find_bounds(args...; kwargs...) = _visualization_extension_required(:find_bounds)
add_rgm_colorbar!(args...; kwargs...) = _visualization_extension_required(:add_rgm_colorbar!)
get_color_map(args...; kwargs...) = _visualization_extension_required(:get_color_map)
draw_ROP(args...; kwargs...) = _visualization_extension_required(:draw_ROP)
plot_polyhedron_slices(args...; kwargs...) =
    _visualization_extension_required(:plot_polyhedron_slices)
plot_binding_regime_partition(args...; kwargs...) =
    _visualization_extension_required(:plot_binding_regime_partition)
plot_bnc_regime_partition(args...; kwargs...) =
    _visualization_extension_required(:plot_bnc_regime_partition)
plot_qcat_slice_with_flux(args...; kwargs...) =
    _visualization_extension_required(:plot_qcat_slice_with_flux)
