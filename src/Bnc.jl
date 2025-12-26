module Bnc

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
export is_singular, is_asymptotic
export get_idx, get_perm, get_vertex!, get_neighbors, get_nullity!
export get_P_P0!, get_P!,get_P0!
export get_M_M0!, get_M!, get_M0!
export get_H_H0!,get_H!,get_H0!
export get_C_C0_x!,get_C_x!, get_C0_x!
export get_C_C0_nullity_qK!, get_C_C0_qK!, get_C_qK!, get_C0_qK!
export get_polyhedron, get_volume!

# include("regime_assign.jl")
# include("symbolics.jl")
# include("regime_graphs.jl")
# include("visualize.jl")

end
