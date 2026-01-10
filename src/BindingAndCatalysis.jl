__precompile__(false)
module BindingAndCatalysis

include("initialize.jl")
export Bnc
export update_catalysis!

# include("helperfunctions.jl")
export locate_sym_x, locate_sym_qK, pythonprint, N_generator, L_generator, randomize

# include("qK_x_mapping.jl")
export x2qK, qK2x, x_traj_with_qK_change, x_traj_with_q_change, x_traj_cat, qK_traj_cat 

# include("volume_calc.jl")
export calc_volume

# include("numeric.jl")
export logder_x_qK, logder_qK_x, ∂logx_∂logqK, ∂logqK_∂logx

# include("regime_enumerate.jl")
export find_all_vertices

# include("regimes.jl")
export find_all_vertices!, get_vertices_perm_dict, get_vertices_nullity, get_vertices_volume!, have_perm
export get_vertices
export is_singular, is_asymptotic

export get_idx, get_perm, get_vertex, get_neighbors, get_nullity, get_one_inner_point
export get_P_P0, get_P,get_P0
export get_M_M0, get_M, get_M0
export get_H_H0,get_H,get_H0
export get_C_C0_x,get_C_x, get_C0_x
export get_C_C0_nullity_qK, get_C_C0_qK, get_C_qK, get_C0_qK
export get_C_C0_nullity, get_C_C0, get_C, get_C0

export check_feasibility_with_constraint, feasible_vertieces_with_constraint
export get_polyhedron, get_volume
export is_neighbor, get_interface


# include("regime_assign.jl")
export assign_vertex, assign_vertex_qK, assign_vertex_x
# include("symbolics.jl")
export x_sym, q_sym, K_sym, qK_sym, ∂logqK_∂logx_sym, ∂logx_∂logqK_sym, logder_qK_x_sym, logder_x_qK_sym
export show_condition_poly, show_condition_x, show_condition_qK, show_condition
export show_expression_mapping, show_expression_x, show_expression_qK, show_expression_path
export show_dominant_condition, show_conservation, show_equilibrium, show_interface

# include("regime_graphs.jl")
export get_vertices_graph!, SISO_graph,  get_polyhedra, get_polyhedron
export get_x_neighbor_grh, get_qK_neighbor_grh
export get_sources, get_sinks, get_sources_sinks
export find_reaction_order_for_path, group_sum
# export get_volume, get_C_C0_nullity_qK
# include("visualize.jl")
export SISO_plot, get_edge_labels, get_node_positions, get_node_colors, get_node_labels
export get_node_size, draw_vertices_neighbor_graph, add_vertices_idx!,add_arrows!
export draw_qK_neighbor_grh, find_bounds, add_rgm_colorbar!, get_color_map

end # module
