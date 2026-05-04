# Network fetch
"""
    get_binding_network(bnc_or_vertex, args...) -> Bnc

Return the binding network associated with a vertex or the model itself.
"""
get_binding_network(model::CatalysisData) = model.bn
get_binding_network(model::Bnc,args...)=model

get_binding_network(rgm::CatalysisRegime) = get_binding_network(rgm.network)
get_binding_network(rgm::BindRegime,args...)=get_binding_network(rgm.network)
get_binding_network(rgm::BncRegime,args...)=get_binding_network(rgm.bind_rgm)


"""    
get_catalysis_network(model::CatalysisData) -> CatalysisData
"""
get_catalysis_network(model::CatalysisData) = model
get_catalysis_network(model::Bnc,args...) = let
    if isnothing(model.catalysis)
        error("Model does not contain a catalysis network. Please provide a Bnc model with a catalysis network.")
    end
   return model.catalysis 
end


get_catalysis_network(rgm::CatalysisRegime) = get_catalysis_network(rgm.network)
get_catalysis_network(rgm::BindRegime,args...) = get_catalysis_network(rgm.network)
get_catalysis_network(rgm::BncRegime,args...) = get_catalysis_network(rgm.catalysis_rgm)













# Regimes fetch
@inline _bind_regimes(model::Bnc) = let 
    if isnothing(model.BindRegimes)
        error("Model does not contain binding regimes. Please run \"find_all_regimes!($model)\" to compute the binding regimes.")
    end
    return model.BindRegimes
end
@inline _catalysis_regimes(model::CatalysisData) = let
    if isnothing(model.CatalysisRegimes)
        error("Model does not contain catalysis regimes. Please run \"find_catalysis_regimes!($model)\" to compute the catalysis regimes.")
    end
    return model.CatalysisRegimes
end

@inline _bnc_regimes(model::Bnc) = let
    if isnothing(model.BncRegimes)
        error("Model does not contain mixed regimes. Please run \"match_regimes!($model)\" to compute the mixed regimes.")
    end
    return model.BncRegimes
end

@inline _bind_regimes(args...; kwargs...) = _bind_regimes(get_binding_network(args...; kwargs...))
@inline _catalysis_regimes(args...; kwargs...) = _catalysis_regimes(get_catalysis_network(args...; kwargs...))
@inline _bnc_regimes(args...; kwargs...) = _bnc_regimes(get_binding_network(args...; kwargs...))


# Is the following three functions really necessary?
@inline _bind_regimes_data(args...; kwargs...) = _bind_regimes(args...; kwargs...).vertices_data
@inline _catalysis_regimes_data(args...; kwargs...) = _catalysis_regimes(args...; kwargs...).vertices_data
@inline _bnc_regimes_data(args...; kwargs...) = _bnc_regimes(args...; kwargs...).vertices_data

n_bind_regimes = length ∘ _bind_regimes_data
n_catalysis_regimes = length ∘ _catalysis_regimes_data
n_bnc_regimes = length ∘ _bnc_regimes_data


@inline _bind_regimes_perm_dict(args...; kwargs...) = _bind_regimes(args...; kwargs...).vertices_perm_dict
@inline _catalysis_regimes_perm_dict(args...; kwargs...) = _catalysis_regimes(args...; kwargs...).vertices_perm_dict
@inline _bnc_regimes_perm_dict(args...; kwargs...) = _bnc_regimes(args...; kwargs...).vertices_perm_dict








# Properties involving inner struct fields
@inline _bind_regimes_perms(args...; kwargs...) = getfield.(_bind_regimes_data(args...; kwargs...), :perm)
@inline _catalysis_regimes_perms(args...; kwargs...) = getfield.(_catalysis_regimes_data(args...; kwargs...), :perm)
@inline _bnc_regimes_perms(args...; kwargs...) = getfield.(_bnc_regimes_data(args...; kwargs...), :perm) #This is needed to be fixed as the current implementation could problematic

@inline _bind_regimes_asymptotic_flag(args...; kwargs...) = getfield.(_bind_regimes_data(args...; kwargs...), :is_asymptotic)
@inline _catalysis_regimes_asymptotic_flag(args...; kwargs...) = getfield.(_catalysis_regimes_data(args...; kwargs...), :is_asymptotic)
@inline _bnc_regimes_asymptotic_flag(args...; kwargs...) = getfield.(_bnc_regimes_data(args...; kwargs...), :is_asymptotic)



function get_bind_perm_dict(args...;kwargs...)
    bn = get_binding_network(args...; kwargs...)
    find_all_regimes!(bn; kwargs...)
    _bind_regimes_perm_dict(bn)
end

function get_catalysis_perm_dict(args...;kwargs...)
    cn = get_catalysis_network(args...; kwargs...)
    find_catalysis_regimes!(cn; kwargs...)
    _catalysis_regimes_perm_dict(cn)
end

function get_bnc_perm_dict(args...;kwargs...)
    bn = get_binding_network(args...; kwargs...)
    cn = get_catalysis_network(args...; kwargs...)
    find_all_regimes!(bn; kwargs...)
    find_catalysis_regimes!(cn; kwargs...)
    return _bind_regimes_perm_dict(bn), _catalysis_regimes_perm_dict(cn)
end

# alias
get_regimes_perm_dict(args...; kwargs...) = get_bind_perm_dict(args...; kwargs...)

function get_Lcat(model)
    bn = get_binding_network(model)
    cn = model.catalysis
    if isnothing(cn)
        @warn "Model does not contain a catalysis network. Returning an empty sparse matrix for Lcat."
        return spzeros(Int,0, n)
    else
        r_v = cn.r_v
        return bn.L[r_v + 1:end, :]
    end
end







function Base.getproperty(model::CatalysisData, sym::Symbol)
    if sym === :d_para
        return getfield(model, :d_w) - getfield(model, :a_w)
    end
    return getfield(model, sym)
end
function Base.propertynames(model::CatalysisData, private::Bool=false)
    names = Symbol[fieldnames(typeof(model))...,
        :d_para,
    ]
    return private ? Tuple(unique(names)) : Tuple(sym for sym in unique(names) if !startswith(String(sym), "_"))
end




get_bind_regimes_dict(args...; kwargs...) =let 
    bn = get_binding_network(args...; kwargs...)
    find_all_regimes!(bn; kwargs...)
    _bind_regimes_perm_dict(bn)
end








