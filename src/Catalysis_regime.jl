function get_binding_network(model::CatalysisData)
        if isnothing(model.network)
            @warn "Binding Network not found in the model"
            return nothing
        end
    return model.network
end


function get_catalysis_network(model::Bnc)
        if isnothing(model.catalysis)
            @warn "Catalysis Network not found in the model"
            return nothing
        end
    return model.catalysis
end
get_catalysis_network(model::CatalysisData) = model




function find_catalysis_regimes!(model::Bnc)
    find_catalysis_regimes!(get_catalysis_network(model))
end



function find_catalysis_regimes!(model::CatalysisData;)
    if !isnothing(model.BindRegimes)
        return nothing
    end

    @info "---------------------Start finding all vertices--------------------"
    all_vertices, is_asymptotic =  _enumerate_all_regimes(model._S_helper)
    # all_vertices = [Vector{T}(v) for v in all_vertices]

    n_vertices = length(all_vertices)
    n_asym_rgms = sum(is_asymptotic)
    @info "Finished, with $(n_vertices) catalysis vertices found and $(n_asym_rgms) asymptotic vertices."
    
    @info "3.Building Regimes..."
    model.CatalysisRegimes = let
        regimes = _build_catalysis_regimes(model, all_vertices, is_asymptotic, nullity)    
        vertices_perm_dict = Dict(perm => idx for (idx, perm) in enumerate(all_vertices))
        CatalysisRegimes(vertices_perm_dict, regimes)
    end
    return nothing
end

@inline function _build_catalysis_regimes(model::CatalysisData, all_vertices, is_asymptotic, nullity) where T
    n_vertices = length(all_vertices)
    regimes = Vector{CatalysisRegime}(undef, n_vertices)
    for i in 1:n_vertices
        regimes[i] = CatalysisRegime(
            network = model, # network
            perm = all_vertices[i], #perm
            idx = i, # idx
            is_asymptotic = is_asymptotic[i]
        )
    end
    return regimes
end



function _initialize_regime!(vtx::CatalysisRegime)
    if !isnothing(vtx.P_pos_neg)
        return vtx
    end

    model = vtx.network
    perm = vtx.perm
    idx = vtx.idx

    P_pos_neg, C = _calc_P_P0(perm, model._S_helper)
    P = P_pos_neg[1:model.r_v, :] - P_pos_neg[model.r_v+1:end, :]

    CΠ = C * model.Π
    PΠ = P * model.Π

    vtx.P_pos_neg = P_pos_neg
    vtx.C_pos_neg = C_pos_neg
    vtx.P = P
    vtx.C = C
    vtx.CΠ = CΠ
    vtx.PΠ = PΠ
    return vtx
end

get_catalysis_regimes_dict(model::CatalysisData) = let
    find_catalysis_regimes!(model)
    get_regimes_dict(model.CatalysisRegimes)
end




























function get_catalysis_regime(model::AbstractBnc, perm::AbstractVector{<:Integer};)
    cn = get_catalysis_network(model)
    return get_catalysis_regimes_dict(cn)[perm]
end

