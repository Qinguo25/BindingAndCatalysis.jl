export n_bind_regimes, n_catalysis_regimes, n_bnc_regimes, get_binding_network, get_catalysis_network, is_feasible
export ensure_binding_regimes!, ensure_catalysis_regimes!, ensure_bnc_regimes!, ensure_regime_data!
export get_binding_regime, get_binding_regimes, get_binding_perms, get_binding_indices
export get_binding_perm_dict, get_binding_regimes_dict
export get_binding_index, get_catalysis_index, get_bnc_index
export filter_regimes, filter_regimes_mask, filter_regimes_with_mask

#========================================================================================#
# Index helpers
#========================================================================================#

@inline _bnc_linear_index(n_bind::Int, bind_idx::Int, cat_idx::Int) = bind_idx + (cat_idx - 1) * n_bind
@inline _bnc_cart_index(n_bind::Int, idx::Int) = ((idx - 1) % n_bind + 1, (idx - 1) ÷ n_bind + 1)


#========================================================================================#
# Network accessors
#========================================================================================#

"""
    get_binding_network(bnc_or_regime, args...) -> Bnc

Return the binding network associated with a regime or the model itself.
"""
get_binding_network(model::CatalysisData) = model.bn
get_binding_network(model::Bnc, args...) = model

get_binding_network(rgm::CatalysisRegime) = get_binding_network(rgm.network)
get_binding_network(rgm::BindRegime, args...) = get_binding_network(rgm.network)
get_binding_network(rgm::BncRegime, args...) = get_binding_network(rgm.bind_rgm)


"""
    get_catalysis_network(model::CatalysisData) -> CatalysisData
"""
get_catalysis_network(model::CatalysisData) = model
get_catalysis_network(model::Bnc, args...) = let
    if isnothing(model.catalysis)
        error("Model does not contain a catalysis network. Please provide a Bnc model with a catalysis network.")
    end
    return model.catalysis
end


get_catalysis_network(rgm::CatalysisRegime) = get_catalysis_network(rgm.network)
get_catalysis_network(rgm::BindRegime, args...) = get_catalysis_network(rgm.network)
get_catalysis_network(rgm::BncRegime, args...) = get_catalysis_network(rgm.catalysis_rgm)






#========================================================================================#
# Regime cache accessors
#========================================================================================#

@inline is_bind_regimes_built(model::Bnc) = !isnothing(model.BindRegimes)
@inline is_catalysis_regimes_built(model::CatalysisData) = !isnothing(model.CatalysisRegimes)
@inline is_bnc_regimes_built(model::Bnc) = !isnothing(model.BncRegimes)

"""
    ensure_binding_regimes!(model) -> nothing

Compute and cache binding regimes for `model` if they are not already built.
This is the explicit cache-building entry point behind binding regime getters.
"""
function ensure_binding_regimes!(model::AbstractBnc)
    find_all_regimes!(get_binding_network(model))
    return nothing
end

"""
    ensure_catalysis_regimes!(model) -> nothing

Compute and cache catalysis regimes for `model` if they are not already built.
"""
function ensure_catalysis_regimes!(model::AbstractBnc)
    find_catalysis_regimes!(get_catalysis_network(model))
    return nothing
end

"""
    ensure_bnc_regimes!(model) -> nothing

Compute and cache matched binding-catalysis regimes for `model` if needed.
"""
function ensure_bnc_regimes!(model::AbstractBnc)
    match_regimes!(get_binding_network(model))
    return nothing
end

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
        error("Model does not contain Bnc regimes. Please run \"match_regimes!($model)\" to compute the Bnc regimes.")
    end
    return model.BncRegimes
end

@inline _bind_regimes(args...; kwargs...) = _bind_regimes(get_binding_network(args...; kwargs...))
@inline _catalysis_regimes(args...; kwargs...) = _catalysis_regimes(get_catalysis_network(args...; kwargs...))
@inline _bnc_regimes(args...; kwargs...) = _bnc_regimes(get_binding_network(args...; kwargs...))


@inline _bind_regimes_data(args...; kwargs...) = _bind_regimes(args...; kwargs...).regimes_data
@inline _catalysis_regimes_data(args...; kwargs...) = _catalysis_regimes(args...; kwargs...).regimes_data
@inline _bnc_regimes_data(args...; kwargs...) = _bnc_regimes(args...; kwargs...)

n_bind_regimes(args...; kwargs...) = length(_bind_regimes_data(args...; kwargs...))
n_catalysis_regimes(args...; kwargs...) = length(_catalysis_regimes_data(args...; kwargs...))
n_bnc_regimes(args...; kwargs...) = length(_bnc_regimes_data(args...; kwargs...))

n_regimes(rgms::Regimes) = length(rgms.regimes_data)
n_bind_regimes(rgms::Regimes) = n_regimes(rgms)
n_catalysis_regimes(rgms::Regimes) = n_regimes(rgms)

n_bnc_regimes(rgms::AbstractArray{<:BncRegime}; feasible::Union{Bool,Nothing}=true, kwargs...) =
    count(rgm -> isnothing(feasible) || is_feasible(rgm) == feasible, vec(rgms))




@inline _bind_regimes_perm_dict(args...; kwargs...) = _bind_regimes(args...; kwargs...).regimes_perm_dict
@inline _catalysis_regimes_perm_dict(args...; kwargs...) = _catalysis_regimes(args...; kwargs...).regimes_perm_dict
@inline _bnc_regimes_perm_dict(args...; kwargs...) = _bnc_regimes(args...; kwargs...).regimes_perm_dict








# Properties involving inner struct fields
@inline _bind_regimes_perms(args...; kwargs...) = getfield.(_bind_regimes_data(args...; kwargs...), :perm)
@inline _catalysis_regimes_perms(args...; kwargs...) = getfield.(_catalysis_regimes_data(args...; kwargs...), :perm)
@inline _bnc_regimes_perms(args...; kwargs...) = getfield.(_bnc_regimes_data(args...; kwargs...), :perm)

@inline _bind_regimes_asymptotic_flag(args...; kwargs...) = getfield.(_bind_regimes_data(args...; kwargs...), :is_asymptotic)
@inline _catalysis_regimes_asymptotic_flag(args...; kwargs...) = getfield.(_catalysis_regimes_data(args...; kwargs...), :is_asymptotic)
@inline _bnc_regimes_asymptotic_flag(args...; kwargs...) = getfield.(_bnc_regimes_data(args...; kwargs...), :is_asymptotic)



function get_bind_perm_dict(args...; kwargs...)
    bn = get_binding_network(args...; kwargs...)
    ensure_binding_regimes!(bn)
    _bind_regimes_perm_dict(bn)
end
get_bind_regimes_dict(args...; kwargs...) = get_bind_perm_dict(args...; kwargs...)
get_binding_perm_dict(args...; kwargs...) = get_bind_perm_dict(args...; kwargs...)
get_binding_regimes_dict(args...; kwargs...) = get_binding_perm_dict(args...; kwargs...)

function get_catalysis_perm_dict(args...; kwargs...)
    cn = get_catalysis_network(args...; kwargs...)
    ensure_catalysis_regimes!(cn)
    _catalysis_regimes_perm_dict(cn)
end
get_catalysis_regimes_dict(args...; kwargs...) = get_catalysis_perm_dict(args...; kwargs...)

function get_bnc_perm_dict(args...; kwargs...)
    bn = get_binding_network(args...; kwargs...)
    cn = get_catalysis_network(args...; kwargs...)
    ensure_binding_regimes!(bn)
    ensure_catalysis_regimes!(cn)
    return _bind_regimes_perm_dict(bn), _catalysis_regimes_perm_dict(cn)
end

#========================================================================================#
# Catalysis projection helpers
#========================================================================================#

function get_Lcat(model)
    bn = get_binding_network(model)
    cn = model.catalysis
    if isnothing(cn)
        @warn "Model does not contain a catalysis network. Returning an empty sparse matrix for Lcat."
        return spzeros(Int, 0, bn.n)
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
    names = Symbol[
        fieldnames(typeof(model))...,
        :d_para,
    ]
    return private ? Tuple(unique(names)) : Tuple(sym for sym in unique(names) if !startswith(String(sym), "_"))
end
#========================================================================================#
# Regime object fetchers and data materialization
#========================================================================================#

function get_bind_regime(vtx::BindRegime; inv_info::Bool=true, kwargs...)::BindRegime
    _initialize_regime!(vtx)
    if inv_info
        _fill_all_info!(vtx)
    end
    return vtx
end
get_bind_regime(rgm::BncRegime) = rgm.bind_rgm

"""
    ensure_regime_data!(rgm; affine=true, conditions=true) -> nothing

Materialize cached data for a regime object. For binding regimes, the basic
dominance data are always initialized; affine maps and qK conditions are
materialized when either `affine` or `conditions` is true. Catalysis regimes
materialize their dominance and flux-balance data. Bnc regimes delegate to the
parent model-level cache.
"""
function ensure_regime_data!(
    rgm::BindRegime;
    affine::Bool=true,
    conditions::Bool=true,
)
    _initialize_regime!(rgm)
    if affine || conditions
        _fill_all_info!(rgm)
    end
    return nothing
end

function ensure_regime_data!(
    rgm::CatalysisRegime;
    affine::Bool=true,
    conditions::Bool=true,
)
    _initialize_regime!(rgm)
    return nothing
end

function ensure_regime_data!(
    rgm::BncRegime;
    affine::Bool=true,
    conditions::Bool=true,
)
    ensure_bnc_regimes!(get_binding_network(rgm))
    return nothing
end

function ensure_regime_data!(
    rgms::AbstractVector{<:AbstractRegime};
    kwargs...,
)
    foreach(rgm -> ensure_regime_data!(rgm; kwargs...), rgms)
    return nothing
end


function get_bind_regime(model::AbstractBnc, idx::Integer; kwargs...)
    bn = get_binding_network(model)
    ensure_binding_regimes!(bn)
    return get_bind_regime(_bind_regimes_data(bn)[idx]; kwargs...)
end

function get_bind_regime(model::AbstractBnc, perm::AbstractVector; kwargs...)
    bn = get_binding_network(model)
    ensure_binding_regimes!(bn)
    key = eltype(perm) <: Integer ? perm : locate_sym_x.(Ref(bn), perm)
    idx = _bind_regimes_perm_dict(bn)[key]
    return get_bind_regime(_bind_regimes_data(bn)[idx]; kwargs...)
end
function get_bind_regime(model::AbstractBnc, vtx::BindRegime; kwargs...)
    return get_bind_regime(vtx; kwargs...)
end

"""
    get_binding_regime(args...; kwargs...) -> BindRegime

Return a binding regime. This is the maintained full-name API; `get_bind_regime`
is kept as a compatibility alias.
"""
get_binding_regime(args...; kwargs...) = get_bind_regime(args...; kwargs...)


function get_catalysis_regime(rgm::CatalysisRegime; kwargs...)::CatalysisRegime
    return _initialize_regime!(rgm)
end
get_catalysis_regime(rgm::BncRegime) = rgm.catalysis_rgm
function get_catalysis_regime(model::AbstractBnc, idx::Integer; kwargs...)
    cn = get_catalysis_network(model)
    ensure_catalysis_regimes!(cn)
    return get_catalysis_regime(_catalysis_regimes_data(cn)[idx]; kwargs...)
end
function get_catalysis_regime(model::AbstractBnc, perm::AbstractVector{<:Integer}; kwargs...)
    cn = get_catalysis_network(model)
    ensure_catalysis_regimes!(cn)
    key = perm
    idx = _catalysis_regimes_perm_dict(cn)[key]
    return get_catalysis_regime(_catalysis_regimes_data(cn)[idx]; kwargs...)
end
function get_catalysis_regime(model::AbstractBnc, vtx::CatalysisRegime; kwargs...)
    return get_catalysis_regime(vtx; kwargs...)
end

function get_bnc_regime(rgm::BncRegime; kwargs...)::BncRegime
    return rgm
end

function get_bnc_regime(model::Bnc, bind, cat; check::Bool=false)
    ensure_bnc_regimes!(model)
    idx = get_bnc_idx(model, bind, cat; check=check)
    rgm = model.BncRegimes[idx]
    return rgm
end
function get_bnc_regime(model::Bnc, idx::Integer; kwargs...)
    ensure_bnc_regimes!(model)
    return model.BncRegimes[idx]
end
#========================================================================================#
# Regime identity helpers: permutations and indices
#========================================================================================#

get_bind_perm(args...; kwargs...) = get_bind_regime(args...; kwargs...).perm
get_catalysis_perm(args...; kwargs...) = get_catalysis_regime(args...; kwargs...).perm
get_bnc_perm(args...; kwargs...) = get_bnc_regime(args...; kwargs...).perm
get_binding_perm(args...; kwargs...) = get_bind_perm(args...; kwargs...)
get_steady_state_perm(args...; kwargs...) = get_fixed_point_perm(args...; kwargs...)


function get_bind_perm(Bnc::Bnc, perm::AbstractVector; check::Bool=false)
    key = eltype(perm) <: Integer ? perm : locate_sym_x.(Ref(Bnc), perm)
    check && @assert haskey(get_bind_regimes_dict(Bnc), key) "The given perm is not in Bnc"
    return Vector{Int}(perm)
end
get_bind_perm(Bnc::Bnc, idx::Integer; kwargs...) =
    (ensure_binding_regimes!(Bnc); _bind_regimes_data(Bnc)[idx].perm)

function get_catalysis_perm(model::CatalysisData, perm::AbstractVector{<:Integer}; check::Bool=false)
    check && @assert haskey(get_catalysis_regimes_dict(model), perm) "The given catalysis perm is not in the model."
    return Vector{Int}(perm)
end
get_catalysis_perm(model::CatalysisData, idx::Integer; kwargs...) =
    (ensure_catalysis_regimes!(model); _catalysis_regimes_data(model)[idx].perm)


get_bind_idx(args...; kwargs...) = get_bind_regime(args...; kwargs...).idx
get_catalysis_idx(args...; kwargs...) = get_catalysis_regime(args...; kwargs...).idx
get_binding_index(args...; kwargs...) = get_bind_idx(args...; kwargs...)
get_catalysis_index(args...; kwargs...) = get_catalysis_idx(args...; kwargs...)
get_bnc_index(args...; kwargs...) = get_bnc_idx(args...; kwargs...)
get_idx(rgm::BindRegime) = get_bind_idx(rgm)
get_idx(rgm::CatalysisRegime) = get_catalysis_idx(rgm)
get_idx(rgm::BncRegime) = get_bnc_idx(rgm)
get_idx(model::Bnc, arg; kwargs...) = get_bind_idx(model, arg; kwargs...)
get_idx(model::CatalysisData, arg; kwargs...) = get_catalysis_idx(model, arg; kwargs...)
get_idx(model::Bnc, bind, cat; kwargs...) = get_bnc_idx(model, bind, cat; kwargs...)
get_perm(rgm::BindRegime) = get_bind_perm(rgm)
get_perm(rgm::CatalysisRegime) = get_catalysis_perm(rgm)
get_perm(rgm::BncRegime) = get_bnc_perm(rgm)
get_perm(model::Bnc, arg; kwargs...) = get_bind_perm(model, arg; kwargs...)
get_perm(model::CatalysisData, arg; kwargs...) = get_catalysis_perm(model, arg; kwargs...)
Base.:(==)(perm::AbstractVector{<:Integer}, rgm::CatalysisRegime) = perm == get_catalysis_perm(rgm)
Base.:(==)(rgm::CatalysisRegime, perm::AbstractVector{<:Integer}) = get_catalysis_perm(rgm) == perm

"""
    get_idx(bnc::Bnc, idx::Integer; check=false) -> Integer

Return the binding regime index, optionally validating it.
"""
function get_bind_idx(Bnc::Bnc, idx::T; check::Bool=false) where T<:Integer
    check && (ensure_binding_regimes!(Bnc); @assert idx ≥ 1 && idx ≤ n_regimes(Bnc) "The given index is out of range.")
    return idx
end
get_bind_idx(Bnc::Bnc, perm::AbstractVector; kwargs...) = get_bind_perm_dict(Bnc)[get_perm(Bnc, perm)]

function get_catalysis_idx(model::CatalysisData, idx::T; check::Bool=false) where T<:Integer
    check && (ensure_catalysis_regimes!(model); @assert idx >= 1 && idx <= n_regimes(model) "The given catalysis index is out of range.")
    return idx
end
get_catalysis_idx(model::CatalysisData, perm::AbstractVector; kwargs...) =
    get_catalysis_perm_dict(model)[get_perm(model, perm)]
function get_bnc_idx(model::Bnc, bind, cat; check::Bool=false)
    cat_idx = get_catalysis_idx(model, cat; check=check)
    bind_idx = get_bind_idx(model, bind; check=check)
    return _bnc_linear_index(n_bind_regimes(model), bind_idx, cat_idx)
end
get_bnc_idx(rgm::BncRegime) = _bnc_linear_index(
    n_bind_regimes(get_binding_network(rgm)),
    get_bind_idx(rgm.bind_rgm),
    get_catalysis_idx(rgm.catalysis_rgm),
)


#========================================================================================#
# Regime predicates
#========================================================================================#

get_nullity(rgm::BindRegime) = rgm.nullity
get_nullity(rgm::CatalysisRegime) = get_catalysis_network(rgm).r_v
get_nullity(rgm::BncRegime) = rgm.nlt

get_bind_nullity(args...; kwargs...) = get_nullity(get_bind_regime(args...; kwargs...))
get_bnc_nullity(args...; kwargs...) = get_nullity(get_bnc_regime(args...; kwargs...))

is_singular(rgm::BindRegime) = get_nullity(rgm) > 0
is_singular(::CatalysisRegime) = false
is_singular(rgm::BncRegime) = get_nullity(rgm) > 0

is_feasible(::BindRegime) = true
is_feasible(::CatalysisRegime) = true
is_feasible(rgm::BncRegime) = rgm.is_feasible

is_bind_singular(args...; kwargs...) = is_singular(get_bind_regime(args...; kwargs...))
is_bnc_singular(args...; kwargs...) = is_singular(get_bnc_regime(args...; kwargs...))

#===================================Check if a regime is asymptotic==========================================================================================#

is_asymptotic(rgm::BindRegime) = rgm.is_asymptotic
is_asymptotic(rgm::CatalysisRegime) = rgm.is_asymptotic
is_asymptotic(rgm::BncRegime) = (is_asymptotic(rgm.bind_rgm), is_asymptotic(rgm.catalysis_rgm))

is_bind_asymptotic(args...; kwargs...) = is_asymptotic(get_bind_regime(args...; kwargs...))
is_catalysis_asymptotic(args...; kwargs...) = is_asymptotic(get_catalysis_regime(args...; kwargs...))
is_bnc_asymptotic(args...; kwargs...) = is_asymptotic(get_bnc_regime(args...; kwargs...))

"""
Check if given key is valid for regime fetching.
"""
have_perm(model::Bnc, perm::AbstractVector) = haskey(get_bind_perm_dict(model), get_bind_perm(model, perm))
have_perm(model::Bnc, idx::Integer) = (ensure_binding_regimes!(model); 1 <= idx <= n_bind_regimes(model))
have_perm(model::CatalysisData, perm::AbstractVector) = haskey(get_catalysis_perm_dict(model), get_catalysis_perm(model, perm))
have_perm(model::CatalysisData, idx::Integer) = (ensure_catalysis_regimes!(model); idx >= 1 && idx <= n_catalysis_regimes(model))

have_perm(model::AbstractBnc, rgm::BindRegime) = get_binding_network(rgm) === model
have_perm(model::AbstractBnc, rgm::CatalysisRegime) = get_catalysis_network(rgm) === model

have_perm(model::AbstractBnc, bind, cat) =
    have_perm(get_binding_network(model), bind) && have_perm(get_catalysis_network(model), cat)
#========================================================================================#
# Regime filtering and collection fetchers
#========================================================================================#

function _get_filter(;
    singular::Union{Bool,Nothing}=nothing,
    nullity::Union{Integer, Tuple{<:Integer,<:Integer},Nothing}=nothing,
    max_nullity::Union{Integer,Nothing}=nothing,
    asymptotic::Union{Bool,Nothing}=nothing,
    stable::Union{Bool,Nothing}=nothing,
    feasible::Union{Bool,Nothing}=nothing,
    add_filter::Union{Function,Nothing}=nothing,
)
    judge_asymptotic(x) = isnothing(asymptotic) || is_asymptotic(x) == asymptotic
    judge_nullity(x) = if !isnothing(nullity)
        nullity isa Integer ? get_nullity(x) == nullity : nullity[1] <= get_nullity(x) <= nullity[2]
    elseif !isnothing(max_nullity)
        get_nullity(x) <= max_nullity
    else
        true
    end
    judge_singular(x) = isnothing(singular) || (singular ? get_nullity(x) > 0 : get_nullity(x) == 0)
    judge_stable(x) = isnothing(stable) || is_stable(x) == stable
    judge_feasible(x) = isnothing(feasible) || is_feasible(x) == feasible
    add_filter_func = isnothing(add_filter) ? (x -> true) : add_filter
    return x -> judge_asymptotic(x) &&
                judge_nullity(x) &&
                judge_singular(x) &&
                judge_stable(x) &&
                judge_feasible(x) &&
                add_filter_func(x)
end





function _get_mask(rgms::AbstractVector{<:AbstractRegime}; kwargs...)
    filter_func = _get_filter(; kwargs...)
    masks = falses(length(rgms))
    @inbounds for i in eachindex(rgms)
        masks[i] = filter_func(rgms[i])
    end
    return masks
end

function _get_mask(model::AbstractBnc, rgms::AbstractVector; kwargs...)
    bn = get_binding_network(model)
    ensure_binding_regimes!(bn)
    bind_rgms = get_bind_regime.(Ref(bn), rgms)
    return _get_mask(bind_rgms; kwargs...)
end



function get_binding_regimes(rgms::AbstractVector{<:BindRegime}; kwargs...)
    filter_func = _get_filter(; kwargs...)
    return filter(filter_func, rgms)
end

function get_bind_regimes(rgms::AbstractVector{<:BindRegime}; return_idx::Bool=false, kwargs...)
    selected = get_binding_regimes(rgms; kwargs...)
    return return_idx ? get_binding_index.(selected) : selected
end

function get_catalysis_regimes(rgms::AbstractVector{<:CatalysisRegime}; kwargs...)
    filter_func = _get_filter(; kwargs...)
    return filter(filter_func, rgms)
end

function get_bnc_regimes(rgms::AbstractArray{<:BncRegime}; feasible::Union{Bool,Nothing}=true, kwargs...)
    return filter(_get_filter(; feasible=feasible, kwargs...), vec(rgms))
end

"""
    get_binding_regimes(bnc::Bnc; singular=nothing, asymptotic=nothing) -> Vector

Return cached `BindRegime`s that satisfy singularity/asymptotic filters.
Use `get_binding_perms` or `get_binding_indices` for permutation/index lists.
"""
function get_binding_regimes(Bnc::AbstractBnc, rgms::Union{Nothing,AbstractVector}=nothing; kwargs...)
    bn = get_binding_network(Bnc)
    ensure_binding_regimes!(bn)
    rgms = isnothing(rgms) ? _bind_regimes_data(bn) : get_bind_regime.(Ref(bn), rgms)
    return get_binding_regimes(rgms; kwargs...)
end

function get_bind_regimes(Bnc::AbstractBnc, rgms::Union{Nothing,AbstractVector}=nothing; return_idx::Bool=false, kwargs...)
    return return_idx ? get_binding_indices(Bnc, rgms; kwargs...) : get_binding_regimes(Bnc, rgms; kwargs...)
end

function filter_regimes_mask(model::Bnc, candidates::AbstractVector; kwargs...)
    bn = get_binding_network(model)
    ensure_binding_regimes!(bn)
    idxs = [get_bind_idx(bn, x) for x in candidates]
    rgms = [get_bind_regime(bn, i) for i in idxs]
    return _get_mask(rgms; kwargs...)
end

function filter_regimes(model::Bnc, candidates::AbstractVector; kwargs...)
    mask = filter_regimes_mask(model, candidates; kwargs...)
    return [get_bind_idx(get_binding_network(model), x) for x in candidates][mask]
end

function filter_regimes_with_mask(model::Bnc, candidates::AbstractVector; kwargs...)
    mask = filter_regimes_mask(model, candidates; kwargs...)
    selected = [get_bind_idx(get_binding_network(model), x) for x in candidates][mask]
    return selected, mask
end

function get_catalysis_regimes(Bnc::AbstractBnc, rgms::Union{Nothing,AbstractVector}=nothing; kwargs...)
    cn = get_catalysis_network(Bnc)
    ensure_catalysis_regimes!(cn)
    rgms = isnothing(rgms) ? _catalysis_regimes_data(cn) : get_catalysis_regime.(Ref(cn), rgms)
    return get_catalysis_regimes(rgms; kwargs...)
end

function get_bnc_regimes(model::AbstractBnc, rgms::Union{Nothing,AbstractArray}=nothing; kwargs...)
    bn = get_binding_network(model)
    ensure_bnc_regimes!(bn)
    rgms = isnothing(rgms) ? _bnc_regimes_data(bn) : get_bnc_regime.(Ref(model), rgms)
    return get_bnc_regimes(rgms; kwargs...)
end

n_bnc_regimes(model::Bnc; feasible::Union{Bool,Nothing}=true, kwargs...) =
    length(get_bnc_regimes(model; feasible=feasible, kwargs...))

get_binding_perms(args...; kwargs...) = get_binding_perm.(get_binding_regimes(args...; kwargs...))
get_binding_indices(args...; kwargs...) = get_binding_index.(get_binding_regimes(args...; kwargs...))
get_bind_perms(args...; kwargs...) = get_binding_perms(args...; kwargs...)
get_bind_indices(args...; kwargs...) = get_binding_indices(args...; kwargs...)
get_catalysis_perms(args...; kwargs...) = get_catalysis_perm.(get_catalysis_regimes(args...; kwargs...))
get_catalysis_indices(args...; kwargs...) = get_catalysis_idx.(get_catalysis_regimes(args...; kwargs...))
get_bnc_perms(args...; kwargs...) = get_bnc_perm.(get_bnc_regimes(args...; kwargs...))
get_bnc_indices(args...; kwargs...) = get_bnc_idx.(get_bnc_regimes(args...; kwargs...))
get_bind_nullities(args...; kwargs...) = get_bind_nullity.(get_bind_regimes(args...; kwargs...))
get_bnc_nullities(args...; kwargs...) = get_bnc_nullity.(get_bnc_regimes(args...; kwargs...))
get_nullities(args...; kwargs...) = get_bind_nullities(args...; kwargs...)

#========================================================================================#
# Affine maps and constraints across coordinate systems
#========================================================================================#

function get_affine_x2q(rgm::BindRegime)
    get_bind_regime(rgm; inv_info=false)
    return rgm.P, rgm.P0
end
function get_affine_x2qK(rgm::BindRegime)
    get_bind_regime(rgm; inv_info=false)
    return rgm.M, rgm.M0
end # Binding is equilibrium
get_affine_xk2qKk(rgm::BindRegime) = let 
    n_k = get_catalysis_network(rgm).n_k
    M, M0 = get_affine_x2qK(rgm)
    M_xk = blockdiag(M, spdiagm(0 => ones(Int, n_k)))
    M0_xk = vcat(M0, zeros(eltype(rgm.M0), n_k))
    return M_xk, M0_xk
end
get_affine_x2qcat(rgm::BindRegime) = let
    r_v = get_catalysis_network(rgm).r_v
    P, P0 = get_affine_x2q(rgm)
    P = P[1:r_v, :]
    P0 = P0[1:r_v]
    return P, P0
end
get_affine_x2w(rgm::BindRegime) = let
    r_v = get_catalysis_network(rgm).r_v
    P, P0 = get_affine_x2q(rgm)
    P = P[r_v+1:end, :]
    P0 = P0[r_v+1:end]
    return P, P0
end
get_affine_x2K(rgm::BindRegime) = get_affine_x2K(get_binding_network(rgm))



get_affine_xk2v(rgm::CatalysisRegime) = get_affine_xk2v(get_catalysis_network(rgm))
function get_affine_v2f(rgm::CatalysisRegime)
    get_catalysis_regime(rgm)
    return rgm.P_pos_neg, rgm.P0_pos_neg
end
get_affine_xk2f(rgm::CatalysisRegime) = let 
    P, P0 = get_affine_v2f(rgm)
    z,z0 = get_affine_xk2v(rgm)
    P_xk = P * z
    P0_xk = P * z0 + P0
    return P_xk, P0_xk
end

get_affine_qK2x(rgm::BindRegime) = let
    get_bind_regime(rgm; inv_info=true);
    rgm.nullity > 1 && @error("BindRegime's nullity is bigger than 1, cannot get H0")
    rgm.H, rgm.H0
end
get_affine_qKk2xk(rgm::BindRegime) = let
    H,H0 = get_affine_qK2x(rgm)
    n_k = get_catalysis_network(rgm).n_k
    H_xk = blockdiag(H, spdiagm(0 => ones(Int, n_k)))
    H0_xk = vcat(H0, zeros(eltype(H0), n_k))
    return H_xk, H0_xk
end

get_affine_qKk2v(rgm::BncRegime) = let
    H,H0 = get_affine_qK2x(get_bind_regime(rgm))
    n_k = get_catalysis_network(rgm).n_k
    H_cat = blockdiag(H, spdiagm(0 => ones(Int, n_k)))
    H0_cat = vcat(H0, zeros(eltype(H0), n_k))
    z, z0 = get_affine_xk2v(get_catalysis_regime(rgm))
    H_v = z * H_cat
    H0_v = z * H0_cat + z0
    return H_v, H0_v
end


@inline function _maybe_remove_h_redundancy(
    C::AbstractMatrix{<:Real},
    C0::AbstractVector,
    nullity::Integer=0;
    remove_h_redundancy::Bool=false,
)
    remove_h_redundancy || return C, C0, nullity
    poly = _build_polyhedron_from_C_C0(C, C0, nullity; canonicalize=true)
    return _polyhedron_to_C_C0_nullity(poly)
end

@inline function _maybe_remove_h_redundancy_pair(
    C::AbstractMatrix{<:Real},
    C0::AbstractVector;
    remove_h_redundancy::Bool=false,
)
    C_new, C0_new, _ = _maybe_remove_h_redundancy(
        C,
        C0,
        0;
        remove_h_redundancy=remove_h_redundancy,
    )
    return C_new, C0_new
end

function get_C_C0_x(rgm::BindRegime; remove_h_redundancy::Bool=false)
    get_bind_regime(rgm; inv_info=false)
    return _maybe_remove_h_redundancy_pair(
        rgm.C_x,
        rgm.C0_x;
        remove_h_redundancy=remove_h_redundancy,
    )
end
get_C_C0_xk(rgm::BindRegime; remove_h_redundancy::Bool=false) = let
    n_k = get_catalysis_network(rgm).n_k
    C_x, C0_x = get_C_C0_x(rgm)
    C_xk = hcat(C_x, spzeros(eltype(C_x), size(C_x, 1), n_k))
    return _maybe_remove_h_redundancy_pair(
        C_xk,
        C0_x;
        remove_h_redundancy=remove_h_redundancy,
    )
end
function get_C_C0_v(rgm::CatalysisRegime; remove_h_redundancy::Bool=false)
    get_catalysis_regime(rgm)
    return _maybe_remove_h_redundancy_pair(
        rgm.C,
        rgm.C0;
        remove_h_redundancy=remove_h_redundancy,
    )
end
get_C_v(rgm::CatalysisRegime; kwargs...) = get_C_C0_v(rgm; kwargs...)[1]
get_C0_v(rgm::CatalysisRegime; kwargs...) = get_C_C0_v(rgm; kwargs...)[2]
function get_CΠ(rgm::CatalysisRegime)
    get_catalysis_regime(rgm)
    return rgm.CΠ
end
function get_PΠ(rgm::CatalysisRegime)
    get_catalysis_regime(rgm)
    return rgm.PΠ
end
get_C_k(rgm::CatalysisRegime) = get_C_v(rgm)
get_P_xk(rgm::CatalysisRegime) = get_N_N0_xk(rgm)[1]
get_C_xk(rgm::CatalysisRegime; kwargs...) = get_C_C0_xk(rgm; kwargs...)[1]
get_C0_xk(rgm::CatalysisRegime; kwargs...) = get_C_C0_xk(rgm; kwargs...)[2]
get_C_C0_xk(rgm::CatalysisRegime; remove_h_redundancy::Bool=false)=let
    C, C0 = get_C_C0_v(rgm)
    z, z0 = get_affine_xk2v(rgm)
    C_xk = C * z
    C0_xk = C * z0 + C0
    return _maybe_remove_h_redundancy_pair(
        C_xk,
        C0_xk;
        remove_h_redundancy=remove_h_redundancy,
    )
end

# This is the steady-state.
get_C_C0_nullity_xk(rgm::CatalysisRegime; remove_h_redundancy::Bool=false) = let
    P_xk, P0 = get_N_N0_xk(rgm)
    C_xk, C0 = get_C_C0_xk(rgm)
    return _maybe_remove_h_redundancy(
        vcat(P_xk, C_xk),
        vcat(P0, C0),
        size(P_xk, 1);
        remove_h_redundancy=remove_h_redundancy,
    )
end

get_C_C0_xk(rgm::BncRegime; remove_h_redundancy::Bool=false) = let
    bind_rgm = get_bind_regime(rgm.bind_rgm; inv_info=false)
    catalysis_rgm = get_catalysis_regime(rgm.catalysis_rgm)

    C_bind, C0_bind = get_C_C0_xk(bind_rgm)
    C_cat, C0_cat = get_C_C0_xk(catalysis_rgm)
    C_xk = vcat(C_bind, C_cat)
    C0_xk = vcat(C0_bind, C0_cat)
    return _maybe_remove_h_redundancy_pair(
        C_xk,
        C0_xk;
        remove_h_redundancy=remove_h_redundancy,
    )
end

function get_C_C0_nullity_xk(
    rgm::BncRegime,
    kind::Symbol=:combined;
    remove_h_redundancy::Bool=false,
)
    if kind === :binding
        C, C0 = get_C_C0_xk(get_bind_regime(rgm))
        return _maybe_remove_h_redundancy(
            C,
            C0,
            0;
            remove_h_redundancy=remove_h_redundancy,
        )
    elseif kind === :catalysis
        return get_C_C0_nullity_xk(
            get_catalysis_regime(rgm);
            remove_h_redundancy=remove_h_redundancy,
        )
    elseif kind === :combined || kind === :all
        C_bind, C0_bind = get_C_C0_xk(get_bind_regime(rgm))
        C_cat, C0_cat, nlt_cat = get_C_C0_nullity_xk(get_catalysis_regime(rgm))
        return _maybe_remove_h_redundancy(
            vcat(C_bind, C_cat),
            vcat(C0_bind, C0_cat),
            nlt_cat;
            remove_h_redundancy=remove_h_redundancy,
        )
    else
        error("Unsupported kind=$kind. Use :binding, :catalysis, or :combined.")
    end
end

function get_C_C0_nullity_qK(rgm::BindRegime; remove_h_redundancy::Bool=false)
    get_bind_regime(rgm; inv_info=true)
    return _maybe_remove_h_redundancy(
        rgm.C_qK,
        rgm.C0_qK,
        rgm.nullity;
        remove_h_redundancy=remove_h_redundancy,
    )
end
get_C_C0_nullity_qKk(rgm::BindRegime; remove_h_redundancy::Bool=false) = let
    C_qK, C0_qK, nullity = get_C_C0_nullity_qK(rgm)
    n_k = get_catalysis_network(rgm).n_k
    C = hcat(C_qK, spzeros(eltype(C_qK), size(C_qK, 1), n_k))
    return _maybe_remove_h_redundancy(
        C,
        C0_qK,
        nullity;
        remove_h_redundancy=remove_h_redundancy,
    )
end
function get_C_C0_nullity_qKk(
    rgm::BncRegime,
    kind::Symbol=:combined;
    remove_h_redundancy::Bool=false,
)
    if kind === :binding
        return get_C_C0_nullity_qKk(
            get_bind_regime(rgm);
            remove_h_redundancy=remove_h_redundancy,
        )
    elseif kind === :catalysis
        return _maybe_remove_h_redundancy(
            rgm.C_qKk_cat,
            rgm.C0_qKk_cat,
            rgm.nlt_qKk_cat;
            remove_h_redundancy=remove_h_redundancy,
        )
    elseif kind === :combined || kind === :all
        C_bind, C0_bind, nlt_bind = get_C_C0_nullity_qKk(get_bind_regime(rgm))
        C_cat, C0_cat, nlt_cat = rgm.C_qKk_cat, rgm.C0_qKk_cat, rgm.nlt_qKk_cat
        return _maybe_remove_h_redundancy(
            vcat(C_bind, C_cat),
            vcat(C0_bind, C0_cat),
            max(nlt_bind, nlt_cat);
            remove_h_redundancy=remove_h_redundancy,
        )
    else
        error("Unsupported kind=$kind. Use :binding, :catalysis, or :combined.")
    end
end


# ================================The following will assume the flux is also balanced=================================================#


# Flux comparation
# N\log v + N0 =0 is the flux balance condition
function get_N_N0_v(rgm::CatalysisRegime)
    get_catalysis_regime(rgm)
    return rgm.P, rgm.P0
end
# N\log xk + N0 =0  is the same flux balance condition in xk space
function get_N_N0_xk(rgm::CatalysisRegime)
    get_catalysis_regime(rgm)
    z, z0 = get_affine_xk2v(rgm)
    return rgm.P * z, rgm.P * z0 + rgm.P0
end

#  ̃p := -P\logk = PΠ \log x + P0
get_affine_x2k̃(rgm::CatalysisRegime) = let 
    get_catalysis_regime(rgm)
    return rgm.PΠ, rgm.P0
end

#  ̃p := -P\logk
get_affine_k2k̃(rgm::CatalysisRegime) = let 
    get_catalysis_regime(rgm)
    cn = get_catalysis_network(rgm)
    return -(rgm.P * cn.F), -(rgm.P * cn.F0)
end

get_affine_x2Kk̃(rgm::BncRegime) = let 
    bind_rgm = get_bind_regime(rgm.bind_rgm; inv_info=false)
    cat_rgm = get_catalysis_regime(rgm.catalysis_rgm)

    N_bind,N0_bind = get_affine_x2K(bind_rgm)
    N_ss,N0_ss = get_affine_x2k̃(cat_rgm)

    N = vcat(N_bind, N_ss)
    N0 = vcat(N0_bind, N0_ss)
    return N,N0
end


get_affine_x2wKk̃(rgm::BncRegime) = let
    bind_rgm = get_bind_regime(rgm.bind_rgm; inv_info=false)
    cat_rgm = get_catalysis_regime(rgm.catalysis_rgm)

    P_w, P0_w = get_affine_x2w(bind_rgm)
    N_bind,N0_bind = get_affine_x2K(bind_rgm)
    N_ss,N0_ss = get_affine_x2k̃(cat_rgm)

    M = vcat(P_w, N_bind, N_ss)
    M0 = vcat(P0_w,N0_bind, N0_ss)
    return M,M0
end

get_affine_xk2wKk̃k(rgm::BncRegime) = let
    M,M0 = get_affine_x2wKk̃(rgm)
    n_k = get_catalysis_network(rgm).n_k
    M_xk = blockdiag(M, spdiagm(0 => ones(Int, n_k)))
    M0_xk = vcat(M0,zeros(eltype(M0), n_k))
    return M_xk, M0_xk
end

get_affine_wKk2wKk̃k(rgm::BncRegime)= let 
    d_w = get_catalysis_network(rgm).d_w
    r = get_binding_network(rgm).r
    n_k = get_catalysis_network(rgm).n_k
    blk1 = spdiagm(0 => ones(Int, d_w))
    blk2 = spdiagm(0 => ones(Int, r))
    z,z0 = get_affine_k2k̃(get_catalysis_regime(rgm))
    blk3 = let
        upper = z
        lower = spdiagm(0 => ones(Int, n_k))
        vcat(upper, lower)
    end
    M = blockdiag(blk1, blk2, blk3)
    M0 = vcat(zeros(eltype(z0), d_w + r), z0, zeros(eltype(z0), n_k))
    return M, M0
end
get_affine_wKk2wKk̃(rgm::BncRegime)= let
    d_w = get_catalysis_network(rgm).d_w
    r = get_binding_network(rgm).r
    blk1 = spdiagm(0 => ones(Int, d_w))
    blk2 = spdiagm(0 => ones(Int, r))
    z,z0 = get_affine_k2k̃(get_catalysis_regime(rgm))
    M = blockdiag(blk1, blk2, z)
    M0 = vcat(zeros(eltype(z0), d_w + r), z0)
    return M, M0
end

get_affine_wKk̃2x(rgm::BncRegime)= let 
    H = rgm.H_inner
    H0 = rgm.H0_inner
    return H, H0
end

get_affine_wKk̃k2xk(rgm::BncRegime)= let 
    H = rgm.H_inner
    H0 = rgm.H0_inner
    n_k = get_catalysis_network(rgm).n_k
    H_full = blockdiag(H, spdiagm(0 => ones(Int, n_k)))
    H0_full = vcat(H0, zeros(eltype(H0), n_k))
    return H_full, H0_full
end

get_affine_wKk2x(rgm::BncRegime) = let
    get_bnc_regime(rgm; inv_info=true);
    rgm.nlt > 1 && error("BncRegime's nullity is bigger than 1, cannot get H0")
    rgm.H, rgm.H0
end
get_affine_wKk2xk(rgm::BncRegime) = let
    H,H0 = get_affine_wKk2x(rgm)
    n_k = get_catalysis_network(rgm).n_k
    lower = let 
        left = zeros(eltype(H), n_k, size(H,2)-n_k)
        right = spdiagm(0 => ones(Int, n_k))
        hcat(left, right)
    end
    H_full = vcat(H, lower)
    H0_full = vcat(H0, zeros(eltype(H0), n_k))
    return H_full, H0_full
end

get_affine_wKk2v(rgm::BncRegime) = let
    H,H0 = get_affine_wKk2x(rgm)
    n_k = get_catalysis_network(rgm).n_k
    H_cat = blockdiag(H, spdiagm(0 => ones(Int, n_k)))
    H0_cat = vcat(H0, zeros(eltype(H0), n_k))
    z, z0 = get_affine_xk2v(get_catalysis_regime(rgm))
    H_v = z * H_cat
    H0_v = z * H0_cat + z0
    return H_v, H0_v
end

get_affine_wKk2qcat(rgm::BncRegime) = let
    rgm.nlt == 0 || error("The reduced steady-state system is singular, so q_cat has no affine expression in (w, K, k).")
    H,H0 = get_affine_wKk2x(rgm)
    r_v = get_catalysis_network(rgm).r_v
    P,P0 = get_affine_x2q(get_bind_regime(rgm.bind_rgm; inv_info=false))
    P_cat = P[1:r_v, :]
    P0_cat = P0[1:r_v]
    F = P_cat * H
    F0 = P_cat * H0 + P0_cat
    return F, F0
end

get_F_F0(rgm::BncRegime) = get_affine_wKk2qcat(rgm)

## Consistency condition
get_C_C0_nullity_wKk(rgm::BncRegime; remove_h_redundancy::Bool=false) =
    _maybe_remove_h_redundancy(
        rgm.C_wKk,
        rgm.C0_wKk,
        rgm.nlt_wKk;
        remove_h_redundancy=remove_h_redundancy,
    )





get_P_P0_x(rgm::BindRegime) = get_affine_x2q(rgm)
get_M_M0(rgm::BindRegime) = get_affine_x2qK(rgm)
get_P_P0_v(rgm::CatalysisRegime) = get_affine_v2f(rgm)  # how v is dominant f (flux)
get_P_P0_xk(rgm::CatalysisRegime) = get_affine_xk2f(rgm) # how xk is dominant f (flux)
get_P_P0_pos_neg(rgm::CatalysisRegime) = get_affine_v2f(rgm)
get_P(rgm::CatalysisRegime) = get_N_N0_v(rgm)[1]
get_P0(rgm::CatalysisRegime) = get_N_N0_v(rgm)[2]
get_C0(rgm::CatalysisRegime) = get_C0_v(rgm)

get_H_H0(rgm::BindRegime) = get_affine_qK2x(rgm)
get_H_H0(rgm::BncRegime) = get_affine_wKk2x(rgm)

get_C_C0_qKk(rgm, args...; kwargs...) = get_C_C0_nullity_qKk(rgm, args...; kwargs...)[1:2]
get_C_qKk(rgm, args...; kwargs...) = get_C_C0_nullity_qKk(rgm, args...; kwargs...)[1]
get_C0_qKk(rgm, args...; kwargs...) = get_C_C0_nullity_qKk(rgm, args...; kwargs...)[2]
get_C_C0_wKk(rgm; kwargs...) = get_C_C0_nullity_wKk(rgm; kwargs...)[1:2]
get_C_wKk(rgm; kwargs...) = get_C_C0_nullity_wKk(rgm; kwargs...)[1]
get_C0_wKk(rgm; kwargs...) = get_C_C0_nullity_wKk(rgm; kwargs...)[2]



get_P_P0_x(args...; kwargs...) = get_P_P0_x(get_bind_regime(args...; inv_info=false,kwargs...)) 
get_P_P0(args...; kwargs...) = get_P_P0_x(args...; kwargs...)
get_P(args...; kwargs...) = get_P_P0_x(args...; kwargs...)[1]
get_P0(args...; kwargs...) = get_P_P0_x(args...; kwargs...)[2]

get_P_P0_v(args...; kwargs...) = get_P_P0_v(get_catalysis_regime(args...; kwargs...))
get_P_v(args...; kwargs...) = get_P_P0_v(args...; kwargs...)[1]
get_P0_v(args...; kwargs...) = get_P_P0_v(args...; kwargs...)[2]

get_P_P0_pos_neg(args...; kwargs...) = get_P_P0_pos_neg(get_catalysis_regime(args...; kwargs...))
get_P_pos_neg(args...; kwargs...) = get_P_P0_pos_neg(args...; kwargs...)[1]
get_P0_pos_neg(args...; kwargs...) = get_P_P0_pos_neg(args...; kwargs...)[2]

get_PΠ(args...; kwargs...) = get_PΠ(get_catalysis_regime(args...; kwargs...))


# dominance matrix with N

get_M_M0(args...; kwargs...) = get_M_M0(get_bind_regime(args...; inv_info=false, kwargs...))
get_M(args...) = get_M_M0(args...)[1]
get_M0(args...) = get_M_M0(args...)[2]


# Basic Condition matrix in x space and v space


get_C_C0_x(args...; remove_h_redundancy::Bool=false, kwargs...) =
    get_C_C0_x(
        get_bind_regime(args...; inv_info=false, kwargs...);
        remove_h_redundancy=remove_h_redundancy,
    )
get_C_x(args...; kwargs...) = get_C_C0_x(args...; kwargs...)[1]
get_C0_x(args...; kwargs...) = get_C_C0_x(args...; kwargs...)[2]

get_C_C0_xk(args...; remove_h_redundancy::Bool=false, kwargs...) =
    get_C_C0_xk(
        get_bind_regime(args...; inv_info=false, kwargs...);
        remove_h_redundancy=remove_h_redundancy,
    )

get_C_C0_nullity_qK(args...; remove_h_redundancy::Bool=false, kwargs...) =
    get_C_C0_nullity_qK(
        get_bind_regime(args...; inv_info=true, kwargs...);
        remove_h_redundancy=remove_h_redundancy,
    )
get_C_C0_qK(args...; kwargs...) = get_C_C0_nullity_qK(args...; kwargs...)[1:2]
get_C_qK(args...; kwargs...) = get_C_C0_nullity_qK(args...; kwargs...)[1]
get_C0_qK(args...; kwargs...) = get_C_C0_nullity_qK(args...; kwargs...)[2]

get_C_C0_nullity(args...;kwargs...) = get_C_C0_nullity_qK(args...;kwargs...)
get_C_C0(args...;kwargs...) = get_C_C0_nullity(args...;kwargs...) |> x->(x[1], x[2]) 
get_C(args...;kwargs...) = get_C_C0_nullity(args...;kwargs...)[1]
get_C0(args...;kwargs...) = get_C_C0_nullity(args...;kwargs...)[2]


get_H_H0(args...; kwargs...) = get_H_H0(get_bind_regime(args...; kwargs...))
get_H(args...) = get_H_H0(args...)[1]
get_H0(args...) = get_H_H0(args...)[2]
