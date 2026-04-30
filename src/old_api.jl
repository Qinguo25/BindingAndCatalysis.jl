#------------------------------------------------------------------------------
# Legacy aliases: vertex/vertices -> regime/regimes
#------------------------------------------------------------------------------
const VertexGraph = RegimeGraph
const VertexEdge = RegimeEdge

find_all_vertices(args...; kwargs...) = find_all_regimes(args...; kwargs...)
find_all_vertices!(args...; kwargs...) = find_all_regimes!(args...; kwargs...)

get_vertices_perm_dict(args...; kwargs...) = get_regimes_perm_dict(args...; kwargs...)

assign_vertex_x(args...; kwargs...) = assign_regime_x(args...; kwargs...)
assign_vertex_qK(args...; kwargs...) = assign_regime_qK(args...; kwargs...)
assign_vertex(args...; kwargs...) = assign_regime(args...; kwargs...)

_calc_vertices_graph(args...; kwargs...) = _calc_regimes_graph(args...; kwargs...)
_fulfill_vertices_graph!(args...; kwargs...) = _fulfill_regimes_graph!(args...; kwargs...)
get_vertices_graph!(args...; kwargs...) = get_regimes_graph!(args...; kwargs...)

_vertex_graph_to_sparse(args...; kwargs...) = _regime_graph_to_sparse(args...; kwargs...)
_create_vertex(args...; kwargs...) = _create_regime(args...; kwargs...)
_is_vertex_graph_neighbor(args...; kwargs...) = _is_regime_graph_neighbor(args...; kwargs...)
_get_vertices_mask(args...; kwargs...) = _get_regimes_mask(args...; kwargs...)

get_vertices_neighbor_mat_x(args...; kwargs...) = get_regimes_neighbor_mat_x(args...; kwargs...)
get_vertices_neighbor_mat_qK(args...; kwargs...) = get_regimes_neighbor_mat_qK(args...; kwargs...)
get_vertices_neighbor_mat(args...; kwargs...) = get_regimes_neighbor_mat(args...; kwargs...)

get_vertex(args...; kwargs...) = get_regime(args...; kwargs...)
get_vertices(args...; kwargs...) = get_regimes(args...; kwargs...)

n_vertices(args...; kwargs...) = n_regimes(args...; kwargs...)
summary_vertex(args...; kwargs...) = summary_regime(args...; kwargs...)
get_regimes_perm_dict(args...; kwargs...) = get_bind_regimes_dict(args...; kwargs...)

get_mixed_regime(args...; kwargs...) = get_bnc_regime(args...; kwargs...)
get_mixed_regimes(args...; kwargs...) = get_bnc_regimes(args...; kwargs...)
show_cat_dynamics(args...; kwargs...) = show_catalysis_dynamics(args...; kwargs...)
show_reduced_cat_dynamics(args...; kwargs...) = show_reduced_catalysis_dynamics(args...; kwargs...)
show_qcat_expression(args...; kwargs...) = show_expression_qcat(args...; kwargs...)
q_para_sym(args...) = begin
    cn = get_catalysis_network(args...)
    w = w_sym(args...)
    w[cn.a_w + 1:end]
end
q_ss_sym(args...) = w_sym(args...)
qssKk_sym(args...) = wKk_sym(args...)
get_C_C0_nullity_qssKk(args...; kwargs...) = get_C_C0_nullity_wKk(args...; kwargs...)
get_C_C0_qssKk(args...; kwargs...) = get_C_C0_wKk(args...; kwargs...)
get_C_qssKk(args...; kwargs...) = get_C_wKk(args...; kwargs...)
get_C0_qssKk(args...; kwargs...) = get_C0_wKk(args...; kwargs...)
show_condition_qssKk(args...; kwargs...) = show_condition_wKk(args...; kwargs...)
show_ss_condition(args...; kwargs...) = show_condition_wKk(args...; kwargs...)
const SISOPaths = SIMOPaths
SISOPaths(args...; kwargs...) = SIMOPaths(args...; kwargs...)
get_SISO_graph(args...; kwargs...) = get_SIMO_graph(args...; kwargs...)
SISO_plot(args...; kwargs...) = SIMO_plot(args...; kwargs...)

export find_all_vertices, find_all_vertices!, get_vertices_perm_dict
export VertexGraph, VertexEdge
export assign_vertex_x, assign_vertex_qK, assign_vertex
export get_vertices_graph!, get_vertices_neighbor_mat_x, get_vertices_neighbor_mat_qK, get_vertices_neighbor_mat
export get_vertex, get_vertices
export n_vertices, summary_vertex
export get_regimes_perm_dict
export get_mixed_regime, get_mixed_regimes
export SISOPaths, get_SISO_graph, SISO_plot
export q_para_sym, q_ss_sym, qssKk_sym
export get_C_C0_nullity_qssKk, get_C_C0_qssKk, get_C_qssKk, get_C0_qssKk
export show_cat_dynamics, show_reduced_cat_dynamics, show_qcat_expression, show_condition_qssKk, show_ss_condition
