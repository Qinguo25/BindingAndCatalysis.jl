show_condition_x(args...; kwargs...) = _render_condition_from(get_C_C0_x(args...), x_sym(args...); kwargs...)
show_condition_qK(args...; kwargs...) = _render_condition_from(get_C_C0_nullity_qK(args...), qK_sym(args...); kwargs...)
show_condition(args...; kwargs...) = show_condition_qK(args...; kwargs...)

function show_condition_path(Bnc::Bnc, path::AbstractVector{<:Integer}, change_qK; kwargs...)
    poly = _calc_polyhedra_for_path(Bnc, path, change_qK)
    syms = copy(qK_sym(Bnc)) |> x -> deleteat!(x, locate_sym_qK(Bnc, change_qK))
    show_condition_poly(poly; syms=syms, kwargs...)
end

function show_condition_path(grh::SIMOPaths, pth_idx; kwargs...)
    poly = get_polyhedron(grh, pth_idx)
    show_condition_poly(poly; syms=qK_sym(grh), kwargs...)
end

show_expression_x(args...; kwargs...) = begin
    bn = get_binding_network(args...)
    _render_expression_from(get_H_H0(args...), x_sym(bn), qK_sym(bn); kwargs...)
end
show_expression_x(rgm::BncRegime; kwargs...) = _render_expression_from(get_H_H0(rgm), x_sym(rgm), wKk_sym(rgm); kwargs...)
show_expression_x(model::Bnc, bind, cat; kwargs...) = show_expression_x(get_bnc_regime(model, bind, cat; check=true); kwargs...)

show_expression_qK(args...; kwargs...) = begin
    bn = get_binding_network(args...)
    _render_expression_from(get_M_M0(args...), qK_sym(bn), x_sym(bn); kwargs...)
end

show_expression_qcat(rgm::BncRegime; kwargs...) = _render_expression_from(get_qcat_F_F0(rgm), q_cat_sym(rgm), wKk_sym(rgm); kwargs...)
show_expression_qcat(model::Bnc, bind, cat; kwargs...) = show_expression_qcat(get_bnc_regime(model, bind, cat; check=true); kwargs...)

show_dominant_condition(args...; kwargs...) = begin
    bn = get_binding_network(args...)
    _render_expression_from(get_P_P0(args...), q_sym(bn), x_sym(bn);  kwargs...)
end

show_conservation(Bnc::Bnc) = Bnc.q_sym .~ Bnc.L * Bnc.x_sym
show_equilibrium(Bnc::Bnc; kwargs...) = show_expression_mapping(Bnc.N, zeros(Int, Bnc.r), Bnc.K_sym, Bnc.x_sym; kwargs...)

function _catalysis_dynamics(args...; reduced::Bool=false)
    cn = _require_catalysis_network(args...)
    v = _flux_sym(args...)
    eqs = Symbolics.Equation[]
    if reduced
        append!(eqs, _d_dt(q_cat_sym(args...)) .~ (cn.S * v))
        append!(eqs, _d_dt(w_sym(args...)) .~ 0)
    else
        w = w_sym(args...)
        a_w = cn.a_w
        q_cat_w = [q_cat_sym(args...); w[1:a_w]]
        append!(eqs, _d_dt(q_cat_w) .~ (cn.Γ * v))
        append!(eqs, _d_dt(w[a_w + 1:end]) .~ 0)
    end
    return eqs
end

@inline function _dominant_flux_terms(args...)
    cat_rgm = get_catalysis_regime(args...)
    cn = _require_catalysis_network(cat_rgm)
    P_pos_neg = get_P_pos_neg(cat_rgm)
    P0_pos_neg = get_P0_pos_neg(cat_rgm)
    flux_terms = handle_log_weighted_sum(hcat(P_pos_neg * cn.Π, P_pos_neg), xk_sym(args...), P0_pos_neg)
    return flux_terms[1:cn.r_v], flux_terms[cn.r_v + 1:end]
end

function _dominant_catalysis_dynamics(args...)
    pos, neg = _dominant_flux_terms(args...)
    eqs = Symbolics.Equation[]
    append!(eqs, _d_dt(q_cat_sym(args...)) .~ (pos .- neg))
    append!(eqs, _d_dt(w_sym(args...)) .~ 0)
    return eqs
end

@inline function _substitute_binding_chart(eqs::AbstractVector{<:Symbolics.Equation}, rgm::Union{BindRegime,BncRegime})
    bind_rgm = rgm isa BncRegime ? get_binding_regime(rgm) : rgm
    subs = Dict(eq.lhs => eq.rhs for eq in show_expression_x(bind_rgm; log_space=false))
    return [eq.lhs ~ Symbolics.substitute(eq.rhs, subs) for eq in eqs]
end

show_catalysis_dynamics(args...; reduced::Bool=true) = _catalysis_dynamics(args...; reduced=reduced)

show_catalysis_dynamics(rgm::BindRegime; reduced::Bool=true) =
    _substitute_binding_chart(_catalysis_dynamics(rgm; reduced=reduced), rgm)

show_catalysis_dynamics(rgm::CatalysisRegime; reduced::Bool=true) = _dominant_catalysis_dynamics(rgm)

show_catalysis_dynamics(rgm::BncRegime; reduced::Bool=true) =
    _substitute_binding_chart(_dominant_catalysis_dynamics(rgm), rgm)

show_catalysis_dynamics(model::Bnc, bind, cat; reduced::Bool=true) =
    show_catalysis_dynamics(get_bnc_regime(model, bind, cat; check=true); reduced=reduced)

show_reduced_catalysis_dynamics(args...; kwargs...) = show_catalysis_dynamics(args...; reduced=true, kwargs...)

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

show_condition_wKk(rgm::BncRegime; kwargs...) = _render_condition_from(get_C_C0_nullity_wKk(rgm), wKk_sym(rgm); kwargs...)
show_condition_wKk(model::Bnc, bind, cat; kwargs...) = show_condition_wKk(get_bnc_regime(model, bind, cat; check=true); kwargs...)
show_consistency_condition(args...; kwargs...) = show_condition_wKk(args...; kwargs...)

function show_interface(Bnc::Bnc, from, to; lhs_idx::Union{Nothing,Integer}=nothing, kwargs...)
    C, C0 = get_interface(Bnc, from, to)
    if isnothing(lhs_idx)
        return show_condition_poly(C, C0, 1; syms=qK_sym(Bnc), kwargs...)
    else
        return solve_sym_expr(C, C0, qK_sym(Bnc), lhs_idx; kwargs...)
    end
end
