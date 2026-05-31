export SIMO_plot, get_edge_labels, set_proper_bounds_for_graph_plot!
export get_node_positions, get_node_colors, get_node_labels, get_node_size
export draw_graph, add_vertices_idx!, add_arrows!, add_nodes_text!, set_node_positions
export draw_qK_neighbor_grh, find_bounds, add_rgm_colorbar!, get_color_map
export draw_ROP
export plot_polyhedron_slices
export plot_binding_regime_partition, plot_bnc_regime_partition, plot_qcat_slice_with_flux

const _VISUALIZATION_LOADED = Ref(false)
const _VISUALIZATION_LOADING = Ref(false)

function _load_visualization!()
    _VISUALIZATION_LOADED[] && return nothing
    _VISUALIZATION_LOADING[] && return nothing

    _VISUALIZATION_LOADING[] = true
    try
        @eval using Makie
        @eval using GraphMakie
        @eval using GraphMakie.NetworkLayout
        @eval using Latexify
        @eval import ImageFiltering: imfilter, Kernel

        include(joinpath(@__DIR__, "visualization/simo_plot.jl"))
        include(joinpath(@__DIR__, "visualization/graphs.jl"))
        include(joinpath(@__DIR__, "visualization/rop.jl"))
        include(joinpath(@__DIR__, "visualization/poly_slices.jl"))
        include(joinpath(@__DIR__, "visualization/regime_partition.jl"))

        _VISUALIZATION_LOADED[] = true
    finally
        _VISUALIZATION_LOADING[] = false
    end

    return nothing
end

function _ensure_visualization_loaded(name::Symbol)
    try
        _load_visualization!()
    catch err
        throw(
            ArgumentError(
                "`$name` requires visualization dependencies available in the active environment. Failed to load Makie/GraphMakie visualization support: $err",
            ),
        )
    end
    return nothing
end

function _visualization_extension_required(name::Symbol)
    _ensure_visualization_loaded(name)
    return nothing
end

function _call_visualization(name::Symbol, args...; kwargs...)
    _ensure_visualization_loaded(name)
    return Base.invokelatest(getfield(@__MODULE__, name), args...; kwargs...)
end

function _visualization_dependency_error(name::Symbol)
    throw(
        ArgumentError(
            "`$name` requires optional visualization packages. Add them to your active environment and load them first, for example `using Makie, GraphMakie`.",
        ),
    )
end

SIMO_plot(args...; kwargs...) = _call_visualization(:SIMO_plot, args...; kwargs...)
get_edge_labels(args...; kwargs...) = _call_visualization(:get_edge_labels, args...; kwargs...)
set_proper_bounds_for_graph_plot!(args...; kwargs...) =
    _call_visualization(:set_proper_bounds_for_graph_plot!, args...; kwargs...)
get_node_positions(args...; kwargs...) = _call_visualization(:get_node_positions, args...; kwargs...)
get_node_colors(args...; kwargs...) = _call_visualization(:get_node_colors, args...; kwargs...)
get_node_labels(args...; kwargs...) = _call_visualization(:get_node_labels, args...; kwargs...)
get_node_size(args...; kwargs...) = _call_visualization(:get_node_size, args...; kwargs...)
draw_graph(args...; kwargs...) = _call_visualization(:draw_graph, args...; kwargs...)
add_vertices_idx!(args...; kwargs...) = _call_visualization(:add_vertices_idx!, args...; kwargs...)
add_arrows!(args...; kwargs...) = _call_visualization(:add_arrows!, args...; kwargs...)
add_nodes_text!(args...; kwargs...) = _call_visualization(:add_nodes_text!, args...; kwargs...)
set_node_positions(args...; kwargs...) = _call_visualization(:set_node_positions, args...; kwargs...)
draw_vertices_neighbor_graph(args...; kwargs...) =
    _call_visualization(:draw_vertices_neighbor_graph, args...; kwargs...)
draw_qK_neighbor_grh(args...; kwargs...) = _call_visualization(:draw_qK_neighbor_grh, args...; kwargs...)
find_bounds(args...; kwargs...) = _call_visualization(:find_bounds, args...; kwargs...)
add_rgm_colorbar!(args...; kwargs...) = _call_visualization(:add_rgm_colorbar!, args...; kwargs...)
get_color_map(args...; kwargs...) = _call_visualization(:get_color_map, args...; kwargs...)
draw_ROP(args...; kwargs...) = _call_visualization(:draw_ROP, args...; kwargs...)
plot_polyhedron_slices(args...; kwargs...) = _call_visualization(:plot_polyhedron_slices, args...; kwargs...)
plot_binding_regime_partition(args...; kwargs...) =
    _call_visualization(:plot_binding_regime_partition, args...; kwargs...)
plot_bnc_regime_partition(args...; kwargs...) =
    _call_visualization(:plot_bnc_regime_partition, args...; kwargs...)
plot_qcat_slice_with_flux(args...; kwargs...) =
    _call_visualization(:plot_qcat_slice_with_flux, args...; kwargs...)
