show_condition_x(args...; kwargs...) = _render_condition_from(get_C_C0_x(args...), x_sym(args...); kwargs...)
show_condition_qK(args...; kwargs...) = _render_condition_from(get_C_C0_nullity_qK(args...), qK_sym(args...); kwargs...)
show_condition(args...; kwargs...) = show_condition_qK(args...; kwargs...)

function show_condition_path(Bnc::Bnc, path::AbstractVector{<:Integer}, change_qK; kwargs...)
    poly = _calc_polyhedra_for_path(Bnc, path, change_qK)
    syms = copy(qK_sym(Bnc)) |> x -> deleteat!(x, locate_sym_qK(Bnc, change_qK))
    show_condition_poly(poly; syms=syms, kwargs...)
end

function show_condition_path(grh::SISOPaths, pth_idx; kwargs...)
    poly = get_polyhedron(grh, pth_idx)
    show_condition_poly(poly; syms=qK_sym(grh), kwargs...)
end

show_expression_x(args...; kwargs...) = begin
    bn = get_binding_network(args...)
    _render_expression_from(get_H_H0(args...), x_sym(bn), qK_sym(bn); kwargs...)
end
show_expression_x(rgm::BncRegime; kwargs...) = _render_expression_from(get_H_H0(rgm), x_sym(rgm), qssKk_sym(rgm); kwargs...)
show_expression_x(model::Bnc, bind, cat; kwargs...) = show_expression_x(get_bnc_regime(model, bind, cat; check=true); kwargs...)

show_expression_qK(args...; kwargs...) = begin
    bn = get_binding_network(args...)
    _render_expression_from(get_M_M0(args...), qK_sym(bn), x_sym(bn); kwargs...)
end

show_expression_qcat(rgm::BncRegime; kwargs...) = _render_expression_from(get_qcat_F_F0(rgm), q_cat_sym(rgm), qssKk_sym(rgm); kwargs...)
show_expression_qcat(model::Bnc, bind, cat; kwargs...) = show_expression_qcat(get_bnc_regime(model, bind, cat; check=true); kwargs...)

show_dominant_condition(args...; log_space=false, kwargs...) = begin
    bn = get_binding_network(args...)
    _render_expression_from(get_P_P0(args...), q_sym(bn), x_sym(bn); log_space=log_space, kwargs...)
end

show_conservation(Bnc::Bnc) = Bnc.q_sym .~ Bnc.L * Bnc.x_sym
show_equilibrium(Bnc::Bnc; log_space::Bool=true) = show_expression_mapping(Bnc.N, zeros(Int, Bnc.r), Bnc.K_sym, Bnc.x_sym; log_space=log_space)

function show_catalysis_dynamics(args...)
    cn = _require_catalysis_network(args...)
    q_cat_w = [q_cat_sym(args...); w_sym(args...)]
    q_para = q_para_sym(args...)
    v = _flux_sym(args...)
    eqs = Symbolics.Equation[]
    append!(eqs, _d_dt(q_cat_w) .~ (cn.Γ * v))
    append!(eqs, _d_dt(q_para) .~ 0)
    return eqs
end

function show_reduced_catalysis_dynamics(args...)
    cn = _require_catalysis_network(args...)
    v = _flux_sym(args...)
    eqs = Symbolics.Equation[]
    append!(eqs, _d_dt(q_cat_sym(args...)) .~ (cn.S * v))
    append!(eqs, _d_dt(w_sym(args...)) .~ 0)
    append!(eqs, _d_dt(q_para_sym(args...)) .~ 0)
    return eqs
end

function show_condition_xk(rgm::CatalysisRegime; kind::Symbol=:all, kwargs...)
    syms = xk_sym(rgm)
    if kind === :steady_state
        return show_condition_poly(get_P_xk(rgm), get_P0(rgm), size(get_P(rgm), 1); syms=syms, kwargs...)
    elseif kind === :dominance
        return show_condition_poly(get_C_xk(rgm), get_C0(rgm); syms=syms, kwargs...)
    elseif kind === :all || kind === :combined
        return _render_condition_from(get_C_C0_nullity_xk(rgm), syms; kwargs...)
    else
        error("Unsupported kind=$kind. Use :steady_state, :dominance, or :all.")
    end
end
show_condition_xk(model::CatalysisData, perm_or_idx; kwargs...) = show_condition_xk(get_catalysis_regime(model, perm_or_idx; check=true); kwargs...)
show_condition_xk(model::AbstractBnc, perm_or_idx; kwargs...) = show_condition_xk(get_catalysis_regime(model, perm_or_idx; check=true); kwargs...)
show_condition_xk(rgm::BncRegime; kind::Symbol=:combined, kwargs...) = _render_condition_from(get_C_C0_nullity_xk(rgm, kind), xk_sym(rgm); kwargs...)
show_condition_xk(model::Bnc, bind, cat; kwargs...) = show_condition_xk(get_bnc_regime(model, bind, cat; check=true); kwargs...)

show_condition_qKk(rgm::BncRegime; kind::Symbol=:combined, kwargs...) = _render_condition_from(get_C_C0_nullity_qKk(rgm, kind), qKk_sym(rgm); kwargs...)
show_condition_qKk(model::Bnc, bind, cat; kwargs...) = show_condition_qKk(get_bnc_regime(model, bind, cat; check=true); kwargs...)

show_condition_qssKk(rgm::BncRegime; kwargs...) = _render_condition_from(get_C_C0_nullity_qssKk(rgm), qssKk_sym(rgm); kwargs...)
show_condition_qssKk(model::Bnc, bind, cat; kwargs...) = show_condition_qssKk(get_bnc_regime(model, bind, cat; check=true); kwargs...)
show_consistency_condition(args...; kwargs...) = show_condition_qssKk(args...; kwargs...)

function show_interface(Bnc::Bnc, from, to; lhs_idx::Union{Nothing,Integer}=nothing, kwargs...)
    C, C0 = get_interface(Bnc, from, to)
    if isnothing(lhs_idx)
        return show_condition_poly(C, C0, 1; syms=qK_sym(Bnc), kwargs...)
    else
        return solve_sym_expr(C, C0, qK_sym(Bnc), lhs_idx; kwargs...)
    end
end
