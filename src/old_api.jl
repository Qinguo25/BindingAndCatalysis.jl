#------------------------------------------------------------------------------
# Legacy aliases. These remain callable for compatibility, but warn users toward
# the regime/catalysis/SIMO terminology used by the maintained API.
#------------------------------------------------------------------------------
const VertexGraph = RegimeGraph
const VertexEdge = RegimeEdge
const SISOPaths = SIMOPaths

@inline function _legacy_api_depwarn(old::Symbol, new::Symbol)
    Base.depwarn("`$old` is deprecated; use `$new` instead.", old)
    return nothing
end

find_all_vertices(args...; kwargs...) = (_legacy_api_depwarn(:find_all_vertices, :find_all_regimes); find_all_regimes(args...; kwargs...))
find_all_vertices!(args...; kwargs...) = (_legacy_api_depwarn(:find_all_vertices!, :find_all_regimes!); find_all_regimes!(args...; kwargs...))

get_vertices_perm_dict(args...; kwargs...) = (_legacy_api_depwarn(:get_vertices_perm_dict, :get_binding_regimes_dict); get_binding_regimes_dict(args...; kwargs...))

assign_vertex_x(args...; kwargs...) = (_legacy_api_depwarn(:assign_vertex_x, :assign_regime_x); assign_regime_x(args...; kwargs...))
assign_vertex_qK(args...; kwargs...) = (_legacy_api_depwarn(:assign_vertex_qK, :assign_regime_qK); assign_regime_qK(args...; kwargs...))
assign_vertex(args...; kwargs...) = (_legacy_api_depwarn(:assign_vertex, :assign_regime); assign_regime(args...; kwargs...))

_calc_vertices_graph(args...; kwargs...) = (_legacy_api_depwarn(:_calc_vertices_graph, :_calc_regimes_graph); _calc_regimes_graph(args...; kwargs...))
_fulfill_vertices_graph!(args...; kwargs...) = (_legacy_api_depwarn(:_fulfill_vertices_graph!, :_fulfill_regimes_graph!); _fulfill_regimes_graph!(args...; kwargs...))
get_vertices_graph!(args...; kwargs...) = (_legacy_api_depwarn(:get_vertices_graph!, :get_regimes_graph!); get_regimes_graph!(args...; kwargs...))

_vertex_graph_to_sparse(args...; kwargs...) = (_legacy_api_depwarn(:_vertex_graph_to_sparse, :_regime_graph_to_sparse); _regime_graph_to_sparse(args...; kwargs...))
_create_vertex(args...; kwargs...) = (_legacy_api_depwarn(:_create_vertex, :_create_regime); _create_regime(args...; kwargs...))
_is_vertex_graph_neighbor(args...; kwargs...) = (_legacy_api_depwarn(:_is_vertex_graph_neighbor, :_is_regime_graph_neighbor); _is_regime_graph_neighbor(args...; kwargs...))
_get_vertices_mask(args...; kwargs...) = (_legacy_api_depwarn(:_get_vertices_mask, :_get_regimes_mask); _get_regimes_mask(args...; kwargs...))

get_vertices_neighbor_mat_x(args...; kwargs...) = (_legacy_api_depwarn(:get_vertices_neighbor_mat_x, :get_regimes_neighbor_mat_x); get_regimes_neighbor_mat_x(args...; kwargs...))
get_vertices_neighbor_mat_qK(args...; kwargs...) = (_legacy_api_depwarn(:get_vertices_neighbor_mat_qK, :get_regimes_neighbor_mat_qK); get_regimes_neighbor_mat_qK(args...; kwargs...))
get_vertices_neighbor_mat(args...; kwargs...) = (_legacy_api_depwarn(:get_vertices_neighbor_mat, :get_regimes_neighbor_mat); get_regimes_neighbor_mat(args...; kwargs...))

get_vertex(args...; kwargs...) = (_legacy_api_depwarn(:get_vertex, :get_regime); get_regime(args...; kwargs...))
get_vertices(args...; kwargs...) = (_legacy_api_depwarn(:get_vertices, :get_regimes); get_regimes(args...; kwargs...))

n_vertices(args...; kwargs...) = (_legacy_api_depwarn(:n_vertices, :n_regimes); n_regimes(args...; kwargs...))
summary_vertex(args...; kwargs...) = (_legacy_api_depwarn(:summary_vertex, :summary_regime); summary_regime(args...; kwargs...))
get_regimes_perm_dict(args...; kwargs...) = (_legacy_api_depwarn(:get_regimes_perm_dict, :get_binding_regimes_dict); get_binding_regimes_dict(args...; kwargs...))

get_mixed_regime(args...; kwargs...) = (_legacy_api_depwarn(:get_mixed_regime, :get_bnc_regime); get_bnc_regime(args...; kwargs...))
get_mixed_regimes(args...; kwargs...) = (_legacy_api_depwarn(:get_mixed_regimes, :get_bnc_regimes); get_bnc_regimes(args...; kwargs...))
show_cat_dynamics(args...; kwargs...) = (_legacy_api_depwarn(:show_cat_dynamics, :show_catalysis_dynamics); show_catalysis_dynamics(args...; kwargs...))
show_reduced_cat_dynamics(args...; kwargs...) = (_legacy_api_depwarn(:show_reduced_cat_dynamics, :show_reduced_catalysis_dynamics); show_reduced_catalysis_dynamics(args...; kwargs...))
show_qcat_expression(args...; kwargs...) = (_legacy_api_depwarn(:show_qcat_expression, :show_expression_qcat); show_expression_qcat(args...; kwargs...))
q_para_sym(args...) = begin
    _legacy_api_depwarn(:q_para_sym, :w_sym)
    cn = get_catalysis_network(args...)
    w = w_sym(args...)
    w[cn.a_w + 1:end]
end
q_ss_sym(args...) = (_legacy_api_depwarn(:q_ss_sym, :w_sym); w_sym(args...))
qssKk_sym(args...) = (_legacy_api_depwarn(:qssKk_sym, :wKk_sym); wKk_sym(args...))
get_C_C0_nullity_qssKk(args...; kwargs...) = (_legacy_api_depwarn(:get_C_C0_nullity_qssKk, :get_C_C0_nullity_wKk); get_C_C0_nullity_wKk(args...; kwargs...))
get_C_C0_qssKk(args...; kwargs...) = (_legacy_api_depwarn(:get_C_C0_qssKk, :get_C_C0_wKk); get_C_C0_wKk(args...; kwargs...))
get_C_qssKk(args...; kwargs...) = (_legacy_api_depwarn(:get_C_qssKk, :get_C_wKk); get_C_wKk(args...; kwargs...))
get_C0_qssKk(args...; kwargs...) = (_legacy_api_depwarn(:get_C0_qssKk, :get_C0_wKk); get_C0_wKk(args...; kwargs...))
show_condition_qssKk(args...; kwargs...) = (_legacy_api_depwarn(:show_condition_qssKk, :show_condition_wKk); show_condition_wKk(args...; kwargs...))
show_ss_condition(args...; kwargs...) = (_legacy_api_depwarn(:show_ss_condition, :show_condition_wKk); show_condition_wKk(args...; kwargs...))

SISOPaths(args...; kwargs...) = (_legacy_api_depwarn(:SISOPaths, :SIMOPaths); SIMOPaths(args...; kwargs...))
get_SISO_graph(args...; kwargs...) = (_legacy_api_depwarn(:get_SISO_graph, :get_SIMO_graph); get_SIMO_graph(args...; kwargs...))
SISO_plot(args...; kwargs...) = (_legacy_api_depwarn(:SISO_plot, :SIMO_plot); SIMO_plot(args...; kwargs...))


# Default binding-network convenience API. These are still maintained because
# many user workflows use `get_regime(model, idx)` for binding regimes.
get_perm(args...;kwargs...) = get_binding_perm(args...;kwargs...)
get_regime(args...; kwargs...) = get_binding_regime(args...; kwargs...)
get_idx(args...; kwargs...) = get_binding_index(args...; kwargs...)
get_nullity(args...; kwargs...) = get_binding_nullity(args...; kwargs...)
is_singular(args...; kwargs...) = is_binding_singular(args...; kwargs...)
is_asymptotic(args...; kwargs...) = is_binding_asymptotic(args...; kwargs...)
get_regimes(args...; return_idx::Bool=false, kwargs...) =
    return_idx ? get_binding_indices(args...; kwargs...) : get_binding_regimes(args...; kwargs...)
get_perms(args...; kwargs...) = get_binding_perms(args...; kwargs...)
get_indices(args...; kwargs...) = get_binding_indices(args...; kwargs...)
n_binding_regimes(rgms::Regimes) = n_bind_regimes(rgms)
n_regimes(model::Bnc) = n_bind_regimes(model)
n_regimes(model::CatalysisData) = n_catalysis_regimes(model)

export get_binding_regime, get_perm, get_regimes_perm_dict, get_regime, get_idx, get_nullity, is_singular, is_asymptotic
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
