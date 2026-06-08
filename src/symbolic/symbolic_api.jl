function show_condition_x(args...; remove_h_redundancy::Bool=false, kwargs...)
    return _render_condition_from(
        get_C_C0_x(args...; remove_h_redundancy=remove_h_redundancy),
        x_sym(args...);
        kwargs...,
    )
end
function show_condition_qK(args...; remove_h_redundancy::Bool=false, kwargs...)
    return _render_condition_from(
        get_C_C0_nullity_qK(args...; remove_h_redundancy=remove_h_redundancy),
        qK_sym(args...);
        kwargs...,
    )
end
show_condition(args...; kwargs...) = show_condition_qK(args...; kwargs...)

function show_condition_path(
    Bnc::Bnc, path::AbstractVector{<:Integer}, change_qK; kwargs...
)
    poly = _calc_polyhedra_for_path(Bnc, path, change_qK)
    syms = (x -> deleteat!(x, locate_sym_qK(Bnc, change_qK)))(copy(qK_sym(Bnc)))
    return show_condition_poly(poly; syms=syms, kwargs...)
end

function show_condition_path(grh::SIMOPaths, pth_idx; kwargs...)
    poly = get_polyhedron(grh, pth_idx)
    return show_condition_poly(poly; syms=qK_sym(grh), kwargs...)
end

function show_expression_x(args...; kwargs...)
    rgm = get_binding_regime(args...; kwargs...)
    bn = get_binding_network(rgm)
    if is_singular(rgm)
        @error "The regime is singular. The expression is not valid."
    end
    _render_expression_from(get_H_H0(rgm), x_sym(bn), qK_sym(bn); kwargs...)
end

function show_expression_x(rgm::BncRegime; kwargs...)
    let
        if is_singular(rgm)
            @error "The regime is singular. The expression is not valid."
        end
        _render_expression_from(get_H_H0(rgm), x_sym(rgm), wKk_sym(rgm); kwargs...)
    end
end

function show_expression_x(model::Bnc, bind, cat; kwargs...)
    return show_expression_x(get_bnc_regime(model, bind, cat; check=true); kwargs...)
end

function show_expression_qK(args...; kwargs...)
    bn = get_binding_network(args...)
    _render_expression_from(get_M_M0(args...), qK_sym(bn), x_sym(bn); kwargs...)
end

function show_expression_qcat(rgm::BncRegime; kwargs...)
    return _render_expression_from(
        get_qcat_F_F0(rgm), q_cat_sym(rgm), wKk_sym(rgm); kwargs...
    )
end
function show_expression_qcat(model::Bnc, bind, cat; kwargs...)
    return show_expression_qcat(get_bnc_regime(model, bind, cat; check=true); kwargs...)
end

function show_dominant_condition(args...; kwargs...)
    bn = get_binding_network(args...)
    _render_expression_from(get_P_P0(args...), q_sym(bn), x_sym(bn); kwargs...)
end

show_conservation(Bnc::Bnc) = Bnc.q_sym .~ Bnc.L * Bnc.x_sym
function show_equilibrium(Bnc::Bnc; kwargs...)
    return show_expression_mapping(
        Bnc.N, zeros(Int, Bnc.r), Bnc.K_sym, Bnc.x_sym; kwargs...
    )
end

function _catalysis_dynamics(args...; reduced::Bool=false)
    cn = get_catalysis_network(args...)
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
        append!(eqs, _d_dt(w[(a_w + 1):end]) .~ 0)
    end
    return eqs
end

@inline function _dominant_flux_terms(args...)
    cat_rgm = get_catalysis_regime(args...)
    cn = get_catalysis_network(cat_rgm)
    P_pos_neg = get_P_pos_neg(cat_rgm)
    P0_pos_neg = get_P0_pos_neg(cat_rgm)
    z, z0 = get_affine_xk2v(cn)
    flux_terms = handle_log_weighted_sum(
        P_pos_neg * z, xk_sym(args...), P_pos_neg * z0 + P0_pos_neg
    )
    return flux_terms[1:(cn.r_v)], flux_terms[(cn.r_v + 1):end]
end

function _dominant_catalysis_dynamics(args...)
    pos, neg = _dominant_flux_terms(args...)
    eqs = Symbolics.Equation[]
    append!(eqs, _d_dt(q_cat_sym(args...)) .~ (pos .- neg))
    append!(eqs, _d_dt(w_sym(args...)) .~ 0)
    return eqs
end

@inline function _substitute_binding_chart(
    eqs::AbstractVector{<:Symbolics.Equation}, rgm::Union{BindRegime, BncRegime}
)
    bind_rgm = rgm isa BncRegime ? get_binding_regime(rgm) : rgm
    subs = Dict(eq.lhs => eq.rhs for eq in show_expression_x(bind_rgm; log_space=false))
    return [eq.lhs ~ Symbolics.substitute(eq.rhs, subs) for eq in eqs]
end

function show_catalysis_dynamics(args...; reduced::Bool=true)
    return _catalysis_dynamics(args...; reduced=reduced)
end

function show_catalysis_dynamics(rgm::BindRegime; reduced::Bool=true)
    return _substitute_binding_chart(_catalysis_dynamics(rgm; reduced=reduced), rgm)
end

function show_catalysis_dynamics(rgm::CatalysisRegime; reduced::Bool=true)
    return _dominant_catalysis_dynamics(rgm)
end

function show_catalysis_dynamics(rgm::BncRegime; reduced::Bool=true)
    return _substitute_binding_chart(_dominant_catalysis_dynamics(rgm), rgm)
end

function show_catalysis_dynamics(model::Bnc, bind, cat; reduced::Bool=true)
    return show_catalysis_dynamics(
        get_bnc_regime(model, bind, cat; check=true); reduced=reduced
    )
end

function show_reduced_catalysis_dynamics(args...; kwargs...)
    return show_catalysis_dynamics(args...; reduced=true, kwargs...)
end

function show_condition_xk(
    rgm::CatalysisRegime; kind::Symbol=:all, remove_h_redundancy::Bool=false, kwargs...
)
    syms = xk_sym(rgm)
    if kind === :steady_state
        data = _maybe_remove_h_redundancy(
            get_P_xk(rgm),
            get_P0(rgm),
            size(get_P(rgm), 1);
            remove_h_redundancy=remove_h_redundancy,
        )
        return show_condition_poly(data...; syms=syms, kwargs...)
    elseif kind === :dominance
        return _render_condition_from(
            get_C_C0_xk(rgm; remove_h_redundancy=remove_h_redundancy), syms; kwargs...
        )
    elseif kind === :all || kind === :combined
        return _render_condition_from(
            get_C_C0_nullity_xk(rgm; remove_h_redundancy=remove_h_redundancy),
            syms;
            kwargs...,
        )
    else
        error("Unsupported kind=$kind. Use :steady_state, :dominance, or :all.")
    end
end
function show_condition_xk(model::CatalysisData, perm_or_idx; kwargs...)
    return show_condition_xk(
        get_catalysis_regime(model, perm_or_idx; check=true); kwargs...
    )
end
function show_condition_xk(model::AbstractBnc, perm_or_idx; kwargs...)
    return show_condition_xk(
        get_catalysis_regime(model, perm_or_idx; check=true); kwargs...
    )
end
function show_condition_xk(
    rgm::BncRegime; kind::Symbol=:combined, remove_h_redundancy::Bool=false, kwargs...
)
    return _render_condition_from(
        get_C_C0_nullity_xk(rgm, kind; remove_h_redundancy=remove_h_redundancy),
        xk_sym(rgm);
        kwargs...,
    )
end
function show_condition_xk(model::Bnc, bind, cat; kwargs...)
    return show_condition_xk(get_bnc_regime(model, bind, cat; check=true); kwargs...)
end

function show_condition_qKk(
    rgm::BncRegime; kind::Symbol=:combined, remove_h_redundancy::Bool=false, kwargs...
)
    return _render_condition_from(
        get_C_C0_nullity_qKk(rgm, kind; remove_h_redundancy=remove_h_redundancy),
        qKk_sym(rgm);
        kwargs...,
    )
end
function show_condition_qKk(model::Bnc, bind, cat; kwargs...)
    return show_condition_qKk(get_bnc_regime(model, bind, cat; check=true); kwargs...)
end

function show_condition_wKk(rgm::BncRegime; remove_h_redundancy::Bool=false, kwargs...)
    return _render_condition_from(
        get_C_C0_nullity_wKk(rgm; remove_h_redundancy=remove_h_redundancy),
        wKk_sym(rgm);
        kwargs...,
    )
end
function show_condition_wKk(model::Bnc, bind, cat; kwargs...)
    return show_condition_wKk(get_bnc_regime(model, bind, cat; check=true); kwargs...)
end
show_consistency_condition(args...; kwargs...) = show_condition_wKk(args...; kwargs...)

function show_interface(
    Bnc::Bnc, from, to; lhs_idx::Union{Nothing, Integer}=nothing, kwargs...
)
    C, C0 = get_interface(Bnc, from, to)
    if isnothing(lhs_idx)
        return show_condition_poly(C, C0, 1; syms=qK_sym(Bnc), kwargs...)
    else
        return solve_sym_expr(C, C0, qK_sym(Bnc), lhs_idx; kwargs...)
    end
end
