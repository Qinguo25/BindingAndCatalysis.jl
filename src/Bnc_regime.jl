export match_regimes!, get_bnc_regime, get_bnc_regimes, n_bnc_regimes
export get_binding_regime, get_binding_perm, get_catalysis_perm, get_steady_state_perm
export get_C_C0_xk, get_C0_xk, get_C_xk
export get_C_C0_qKk, get_C0_qKk, get_C_qKk, get_C_C0_nullity_qKk
export get_C_C0_wKk, get_C0_wKk, get_C_wKk, get_C_C0_nullity_wKk
export get_H_bd, get_qcat_F_F0
export judge_stability!, is_stable

include(joinpath(@__DIR__, "mixed_regime/bnc_core.jl"))
include(joinpath(@__DIR__, "mixed_regime/bnc_conditions.jl"))
include(joinpath(@__DIR__, "mixed_regime/bnc_initialization.jl"))
include(joinpath(@__DIR__, "mixed_regime/bnc_display.jl"))
