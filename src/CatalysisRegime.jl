export find_catalysis_regimes!, get_catalysis_network, get_catalysis_regime, get_catalysis_regimes, get_catalysis_regimes_dict
export get_PΠ, get_CΠ, get_P_pos_neg, get_P0_pos_neg
export get_affine_xk2v, get_affine_v2f, get_affine_xk2f, get_affine_x2k̃, get_affine_k2k̃
export get_C_k, get_C_C0_xk, get_C0_xk, get_C_xk, get_C_C0_nullity_xk



# ------------------------------------------------------------------------------------------------------------------------------------
#                            CORE FUNCTIONS FOR CATALYSIS REGIMES
#-------------------------------------------------------------------------------------------------------------------------------------

find_catalysis_regimes!(args...; kwargs...) = find_catalysis_regimes!(get_catalysis_network(args...; kwargs...))
find_all_regimes!(model::CatalysisData) = find_catalysis_regimes!(model)
function find_catalysis_regimes!(model::CatalysisData)
    if !isnothing(getfield(model, :CatalysisRegimes))
        return nothing
    end

    @info "---------------------Start finding all vertices--------------------"
    all_vertices, is_asymptotic = _enumerate_all_regimes(model._S_helper)

    n_vertices = length(all_vertices)
    n_asym_rgms = sum(is_asymptotic)
    @info "Finished, with $(n_vertices) catalysis vertices found and $(n_asym_rgms) asymptotic vertices."

    @info "3.Building Regimes..."
    model.CatalysisRegimes = let
        regimes = _build_catalysis_regimes(model, all_vertices, is_asymptotic)
        vertices_perm_dict = Dict(perm => idx for (idx, perm) in enumerate(all_vertices))
        Regimes(vertices_perm_dict, regimes)
    end
    return nothing
end


@inline function _build_catalysis_regimes(model::CatalysisData, all_vertices, is_asymptotic)
    n_vertices = length(all_vertices)
    regimes = Vector{CatalysisRegime}(undef, n_vertices)
    for i in 1:n_vertices
        regimes[i] = CatalysisRegime(
            network = model,
            perm = all_vertices[i],
            idx = i,
            is_asymptotic = is_asymptotic[i],
        )
    end
    return regimes
end


function _initialize_regime!(vtx::CatalysisRegime)
    if !isnothing(vtx.P_pos_neg)
        return vtx
    end

    model = get_catalysis_network(vtx)
    isnothing(model) && error("Catalysis network not found in the model.")
    perm = vtx.perm

    P_pos_neg, P0_pos_neg = _calc_P_P0(perm, model._S_helper)

    P = P_pos_neg[1:model.r_v, :] - P_pos_neg[model.r_v+1:end, :]
    P0 = P0_pos_neg[1:model.r_v] - P0_pos_neg[model.r_v+1:end]

    C, C0 = _calc_C_C0(perm, model._S_helper) # C_pos and C_neg for v

    vtx.P_pos_neg = P_pos_neg
    vtx.P0_pos_neg = P0_pos_neg
    
    vtx.P = P
    vtx.P0 = P0

    vtx.C = C
    vtx.C0 = C0

    vtx.CΠ = C * model.Π
    vtx.PΠ = P * model.Π
    
    return vtx
end








get_affine_xk2v(cn::CatalysisData) = let
    return hcat(cn.Π, cn.F), cn.F0
end

function _has_nontrivial_k_constraints(cn::CatalysisData; atol::Float64=1e-12)
    cn.n_k == cn.n_v || return true
    nnz(cn.F - spdiagm(0 => ones(Rational{Int}, cn.n_v))) == 0 || return true
    return any(abs.(Float64.(cn.F0)) .> atol)
end




function summary(rgm::CatalysisRegime)
    rgm = get_catalysis_regime(rgm)
    println("idx=$(rgm.idx), perm=$(rgm.perm), asymptotic=$(rgm.is_asymptotic)")
    println("Steady-state equalities and dominance inequalities in (x, k):")
    display.(show_condition_xk(rgm; log_space=false))
    return nothing
end

function Base.show(io::IO, rgm::CatalysisRegime)
    print(
        io,
        "CatalysisRegime(",
        "perm=$(get_perm(rgm))",
        ", nullity=",
        get_nullity(rgm),
        ", asymptotic=",
        is_asymptotic(rgm),
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", rgm::CatalysisRegime)
    println(io, "CatalysisRegime")
    println(io, "  dominant mode: ", "perm=$(get_perm(rgm))")
    println(io, "  nullity: ", get_nullity(rgm))
    print(io, "  asymptotic: ", is_asymptotic(rgm))
end
