function summary_regime(rgm::BncRegime)
    rgm = get_regime(rgm)
    println("bind_idx=$(get_idx(rgm.bind_rgm)), cat_idx=$(get_idx(rgm.catalysis_rgm)), nlt=$(rgm.nlt), stable=$(is_stable(rgm))")
    println("Binding / catalysis conditions in (x, k):")
    display.(show_condition_xk(rgm; kind=:binding, log_space=false))
    display.(show_condition_xk(rgm; kind=:catalysis, log_space=false))
    println("Combined consistency in (w, K, k):")
    display.(show_condition_wKk(rgm; log_space=false))
    return nothing
end

summary(rgm::BncRegime) = summary_regime(rgm)
summary_regime(model::Bnc, bind, cat) = summary_regime(get_bnc_regime(model, bind, cat; check=true))
summary(model::Bnc, bind, cat) = summary_regime(model, bind, cat)

@inline function _is_asymptotic(rgm::BncRegime)
    return is_asymptotic(rgm.bind_rgm) && is_asymptotic(rgm.catalysis_rgm)
end

@inline function _regime_display_dominant_mode(rgm::BncRegime)
    return "bind=$(get_binding_perm(rgm)), cat=$(get_catalysis_perm(rgm)), ss=$(get_perm(rgm))"
end

function Base.show(io::IO, rgm::BncRegime)
    print(
        io,
        "BncRegime(",
        _regime_display_dominant_mode(rgm),
        ", nullity=",
        rgm.nlt,
        ", asymptotic=",
        _is_asymptotic(rgm),
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", rgm::BncRegime)
    println(io, "BncRegime")
    println(io, "  dominant mode: ", _regime_display_dominant_mode(rgm))
    println(io, "  nullity: ", rgm.nlt)
    print(io, "  asymptotic: ", _is_asymptotic(rgm))
end

function match_regimes!(model::Bnc)
    if is_bnc_regimes_built(model)
        return model.BncRegimes
    end

    find_all_regimes!(model)
    find_catalysis_regimes!(model)

    model.BncRegimes = _build_BncRegime(
        model.catalysis.CatalysisRegimes,
        model.BindRegimes,
    )
    _initialize_regime!(model.BncRegimes)

    return model.BncRegimes
end
