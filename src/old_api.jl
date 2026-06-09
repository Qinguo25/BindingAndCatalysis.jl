#------------------------------------------------------------------------------
# Legacy aliases. These remain callable for compatibility, but warn users toward
# the regime/catalysis/SIMO terminology used by the maintained API.
#
# Deprecation policy:
# - SISO aliases are kept for notebook/report compatibility. New code should use
#   SIMO names, but no removal is currently scheduled.
# - Non-SISO legacy aliases are migration shims. They should stay callable
#   through the 1.x maintenance window and can stop being exported in the next
#   breaking release after documentation and examples no longer use them.
#------------------------------------------------------------------------------
const VertexGraph = RegimeGraph
const VertexEdge = RegimeEdge
const SISOPaths = SIMOPaths

const LEGACY_API_DEPRECATION_POLICY = (;
    siso="kept compatibility alias; prefer SIMO names in new code",
    vertex="1.x migration alias; use regime terminology in new code",
    mixed="1.x migration alias; use BNC terminology in new code",
    catalysis_short="1.x migration alias; use full catalysis names in new code",
    qssKk="1.x migration alias; use wKk terminology in new code",
)

@inline function _legacy_api_depwarn(old::Symbol, new::Symbol, category::Symbol)
    policy = getproperty(LEGACY_API_DEPRECATION_POLICY, category)
    Base.depwarn("`$old` is deprecated; use `$new` instead. Policy: $policy.", old)
    return nothing
end

function find_all_vertices(args...; kwargs...)
    (
        _legacy_api_depwarn(:find_all_vertices, :find_all_regimes, :vertex); find_all_regimes(
            args...; kwargs...
        )
    )
end
function find_all_vertices!(args...; kwargs...)
    (
        _legacy_api_depwarn(:find_all_vertices!, :find_all_regimes!, :vertex); find_all_regimes!(
            args...; kwargs...
        )
    )
end

function get_vertices_perm_dict(args...; kwargs...)
    (
        _legacy_api_depwarn(:get_vertices_perm_dict, :get_binding_regimes_dict, :vertex); get_binding_regimes_dict(
            args...; kwargs...
        )
    )
end

function assign_vertex_x(args...; kwargs...)
    (
        _legacy_api_depwarn(:assign_vertex_x, :assign_regime_x, :vertex); assign_regime_x(
            args...; kwargs...
        )
    )
end
function assign_vertex_qK(args...; kwargs...)
    (
        _legacy_api_depwarn(:assign_vertex_qK, :assign_regime_qK, :vertex); assign_regime_qK(
            args...; kwargs...
        )
    )
end
function assign_vertex(args...; kwargs...)
    (
        _legacy_api_depwarn(:assign_vertex, :assign_regime, :vertex); assign_regime(
            args...; kwargs...
        )
    )
end

function _calc_vertices_graph(args...; kwargs...)
    (
        _legacy_api_depwarn(:_calc_vertices_graph, :_calc_regimes_graph, :vertex); _calc_regimes_graph(
            args...; kwargs...
        )
    )
end
function _fulfill_vertices_graph!(args...; kwargs...)
    (
        _legacy_api_depwarn(:_fulfill_vertices_graph!, :_fulfill_regimes_graph!, :vertex); _fulfill_regimes_graph!(
            args...; kwargs...
        )
    )
end
function get_vertices_graph!(args...; kwargs...)
    (
        _legacy_api_depwarn(:get_vertices_graph!, :get_regimes_graph!, :vertex); get_regimes_graph!(
            args...; kwargs...
        )
    )
end

function _vertex_graph_to_sparse(args...; kwargs...)
    (
        _legacy_api_depwarn(:_vertex_graph_to_sparse, :_regime_graph_to_sparse, :vertex); _regime_graph_to_sparse(
            args...; kwargs...
        )
    )
end
function _create_vertex(args...; kwargs...)
    (
        _legacy_api_depwarn(:_create_vertex, :_create_regime, :vertex); _create_regime(
            args...; kwargs...
        )
    )
end
function _is_vertex_graph_neighbor(args...; kwargs...)
    (
        _legacy_api_depwarn(
            :_is_vertex_graph_neighbor, :_is_regime_graph_neighbor, :vertex
        ); _is_regime_graph_neighbor(args...; kwargs...)
    )
end
function _get_vertices_mask(args...; kwargs...)
    (
        _legacy_api_depwarn(:_get_vertices_mask, :_get_regimes_mask, :vertex); _get_regimes_mask(
            args...; kwargs...
        )
    )
end

function get_vertices_neighbor_mat_x(args...; kwargs...)
    (
        _legacy_api_depwarn(
            :get_vertices_neighbor_mat_x, :get_regimes_neighbor_mat_x, :vertex
        ); get_regimes_neighbor_mat_x(args...; kwargs...)
    )
end
function get_vertices_neighbor_mat_qK(args...; kwargs...)
    (
        _legacy_api_depwarn(
            :get_vertices_neighbor_mat_qK, :get_regimes_neighbor_mat_qK, :vertex
        ); get_regimes_neighbor_mat_qK(args...; kwargs...)
    )
end
function get_vertices_neighbor_mat(args...; kwargs...)
    (
        _legacy_api_depwarn(:get_vertices_neighbor_mat, :get_regimes_neighbor_mat, :vertex); get_regimes_neighbor_mat(
            args...; kwargs...
        )
    )
end

function get_vertex(args...; kwargs...)
    (_legacy_api_depwarn(:get_vertex, :get_regime, :vertex); get_regime(args...; kwargs...))
end
function get_vertices(args...; kwargs...)
    (
        _legacy_api_depwarn(:get_vertices, :get_regimes, :vertex); get_regimes(
            args...; kwargs...
        )
    )
end

function n_vertices(args...; kwargs...)
    (_legacy_api_depwarn(:n_vertices, :n_regimes, :vertex); n_regimes(args...; kwargs...))
end
function summary_vertex(args...; kwargs...)
    (
        _legacy_api_depwarn(:summary_vertex, :summary_regime, :vertex); summary_regime(
            args...; kwargs...
        )
    )
end
function get_regimes_perm_dict(args...; kwargs...)
    (
        _legacy_api_depwarn(:get_regimes_perm_dict, :get_binding_regimes_dict, :vertex); get_binding_regimes_dict(
            args...; kwargs...
        )
    )
end

function get_mixed_regime(args...; kwargs...)
    (
        _legacy_api_depwarn(:get_mixed_regime, :get_bnc_regime, :mixed); get_bnc_regime(
            args...; kwargs...
        )
    )
end
function get_mixed_regimes(args...; kwargs...)
    (
        _legacy_api_depwarn(:get_mixed_regimes, :get_bnc_regimes, :mixed); get_bnc_regimes(
            args...; kwargs...
        )
    )
end
function show_cat_dynamics(args...; kwargs...)
    (
        _legacy_api_depwarn(:show_cat_dynamics, :show_catalysis_dynamics, :catalysis_short); show_catalysis_dynamics(
            args...; kwargs...
        )
    )
end
function show_reduced_cat_dynamics(args...; kwargs...)
    (
        _legacy_api_depwarn(
            :show_reduced_cat_dynamics, :show_reduced_catalysis_dynamics, :catalysis_short
        ); show_reduced_catalysis_dynamics(args...; kwargs...)
    )
end
function show_qcat_expression(args...; kwargs...)
    (
        _legacy_api_depwarn(:show_qcat_expression, :show_expression_qcat, :catalysis_short); show_expression_qcat(
            args...; kwargs...
        )
    )
end
q_para_sym(args...) = begin
    _legacy_api_depwarn(:q_para_sym, :w_sym, :qssKk)
    cn = get_catalysis_network(args...)
    w = w_sym(args...)
    w[(cn.a_w + 1):end]
end
q_ss_sym(args...) = (_legacy_api_depwarn(:q_ss_sym, :w_sym, :qssKk); w_sym(args...))
qssKk_sym(args...) = (_legacy_api_depwarn(:qssKk_sym, :wKk_sym, :qssKk); wKk_sym(args...))
function get_C_C0_nullity_qssKk(args...; kwargs...)
    (
        _legacy_api_depwarn(:get_C_C0_nullity_qssKk, :get_C_C0_nullity_wKk, :qssKk); get_C_C0_nullity_wKk(
            args...; kwargs...
        )
    )
end
function get_C_C0_qssKk(args...; kwargs...)
    (
        _legacy_api_depwarn(:get_C_C0_qssKk, :get_C_C0_wKk, :qssKk); get_C_C0_wKk(
            args...; kwargs...
        )
    )
end
function get_C_qssKk(args...; kwargs...)
    (_legacy_api_depwarn(:get_C_qssKk, :get_C_wKk, :qssKk); get_C_wKk(args...; kwargs...))
end
function get_C0_qssKk(args...; kwargs...)
    (
        _legacy_api_depwarn(:get_C0_qssKk, :get_C0_wKk, :qssKk); get_C0_wKk(
            args...; kwargs...
        )
    )
end
function show_condition_qssKk(args...; kwargs...)
    (
        _legacy_api_depwarn(:show_condition_qssKk, :show_condition_wKk, :qssKk); show_condition_wKk(
            args...; kwargs...
        )
    )
end
function show_ss_condition(args...; kwargs...)
    (
        _legacy_api_depwarn(:show_ss_condition, :show_condition_wKk, :qssKk); show_condition_wKk(
            args...; kwargs...
        )
    )
end

function SISOPaths(args...; kwargs...)
    (_legacy_api_depwarn(:SISOPaths, :SIMOPaths, :siso); SIMOPaths(args...; kwargs...))
end
function get_SISO_graph(args...; kwargs...)
    (
        _legacy_api_depwarn(:get_SISO_graph, :get_SIMO_graph, :siso); get_SIMO_graph(
            args...; kwargs...
        )
    )
end
function SISO_plot(args...; kwargs...)
    (_legacy_api_depwarn(:SISO_plot, :SIMO_plot, :siso); SIMO_plot(args...; kwargs...))
end

# Default binding-network convenience API. These are still maintained because
# many user workflows use `get_regime(model, idx)` for binding regimes.
get_perm(args...; kwargs...) = get_binding_perm(args...; kwargs...)
get_regime(args...; kwargs...) = get_binding_regime(args...; kwargs...)
get_idx(args...; kwargs...) = get_binding_index(args...; kwargs...)
get_nullity(args...; kwargs...) = get_binding_nullity(args...; kwargs...)
is_singular(args...; kwargs...) = is_binding_singular(args...; kwargs...)
is_asymptotic(args...; kwargs...) = is_binding_asymptotic(args...; kwargs...)
function get_regimes(args...; return_idx::Bool=false, kwargs...)
    return if return_idx
        get_binding_indices(args...; kwargs...)
    else
        get_binding_regimes(args...; kwargs...)
    end
end
get_perms(args...; kwargs...) = get_binding_perms(args...; kwargs...)
get_indices(args...; kwargs...) = get_binding_indices(args...; kwargs...)
n_binding_regimes(rgms::Regimes) = n_bind_regimes(rgms)
n_regimes(model::Bnc) = n_bind_regimes(model)
n_regimes(model::CatalysisData) = n_catalysis_regimes(model)

export get_binding_regime,
    get_perm,
    get_regimes_perm_dict,
    get_regime,
    get_idx,
    get_nullity,
    is_singular,
    is_asymptotic
export find_all_vertices, find_all_vertices!, get_vertices_perm_dict
export VertexGraph, VertexEdge
export assign_vertex_x, assign_vertex_qK, assign_vertex
export get_vertices_graph!,
    get_vertices_neighbor_mat_x, get_vertices_neighbor_mat_qK, get_vertices_neighbor_mat
export get_vertex, get_vertices
export n_vertices, summary_vertex
export get_regimes_perm_dict
export get_mixed_regime, get_mixed_regimes
export SISOPaths, get_SISO_graph, SISO_plot
export q_para_sym, q_ss_sym, qssKk_sym
export get_C_C0_nullity_qssKk, get_C_C0_qssKk, get_C_qssKk, get_C0_qssKk
export show_cat_dynamics,
    show_reduced_cat_dynamics, show_qcat_expression, show_condition_qssKk, show_ss_condition
