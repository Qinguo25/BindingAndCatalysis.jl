export reparameterize_logk_constraints
export get_C_C0_nullity_qKtheta, get_C_C0_qKtheta, get_C_qKtheta, get_C0_qKtheta
export get_C_C0_nullity_wKtheta, get_C_C0_wKtheta, get_C_wKtheta, get_C0_wKtheta
export is_feasible_under_logkmap, feasible_bnc_regimes_under_logkmap

"""
    reparameterize_logk_constraints(C, C0, nullity, n_prefix; R, b=nothing)

Apply the log-space reparameterization `log k = R * log θ + b` to constraints written in
the coordinates `[prefix; k]`, where `n_prefix` is the number of non-`k` coordinates.
Returns the transformed `(C_new, C0_new, nullity)` in `[prefix; θ]` coordinates.
"""
function reparameterize_logk_constraints(
    C::AbstractMatrix{<:Real},
    C0::AbstractVector,
    nullity::Integer,
    n_prefix::Integer;
    R::AbstractMatrix{<:Real},
    b::Union{Nothing,AbstractVector}=nothing,
)
    n_k = size(C, 2) - n_prefix
    @assert n_prefix >= 0 && n_k >= 0 "Invalid prefix dimension."
    @assert size(R, 1) == n_k "R must have one row for each original k coordinate."

    bvec = isnothing(b) ? fill(zero(eltype(C0)), n_k) : [zero(eltype(C0)) + x for x in b]
    @assert length(bvec) == n_k "b must have one entry for each original k coordinate."

    C_prefix = @view C[:, 1:n_prefix]
    C_k = @view C[:, n_prefix + 1:end]
    C_new = hcat(C_prefix, C_k * R)
    C0_new = C0 + C_k * bvec
    return C_new, C0_new, Int(nullity)
end

@inline _qKtheta_prefix_dim(rgm::BncRegime) = length(qK_sym(rgm))
@inline _wKtheta_prefix_dim(rgm::BncRegime) = length(w_sym(rgm)) + length(K_sym(rgm))

function get_C_C0_nullity_qKtheta(
    rgm::BncRegime;
    R::AbstractMatrix{<:Real},
    b::Union{Nothing,AbstractVector}=nothing,
    kind::Symbol=:combined,
)
    return reparameterize_logk_constraints(
        get_C_C0_nullity_qKk(rgm, kind)...,
        _qKtheta_prefix_dim(rgm);
        R=R,
        b=b,
    )
end

function get_C_C0_qKtheta(rgm::BncRegime; kwargs...)
    C, C0, _ = get_C_C0_nullity_qKtheta(rgm; kwargs...)
    return C, C0
end
get_C_qKtheta(rgm::BncRegime; kwargs...) = get_C_C0_nullity_qKtheta(rgm; kwargs...)[1]
get_C0_qKtheta(rgm::BncRegime; kwargs...) = get_C_C0_nullity_qKtheta(rgm; kwargs...)[2]

function get_C_C0_nullity_wKtheta(
    rgm::BncRegime;
    R::AbstractMatrix{<:Real},
    b::Union{Nothing,AbstractVector}=nothing,
)
    return reparameterize_logk_constraints(
        get_C_C0_nullity_wKk(rgm)...,
        _wKtheta_prefix_dim(rgm);
        R=R,
        b=b,
    )
end

function get_C_C0_wKtheta(rgm::BncRegime; kwargs...)
    C, C0, _ = get_C_C0_nullity_wKtheta(rgm; kwargs...)
    return C, C0
end
get_C_wKtheta(rgm::BncRegime; kwargs...) = get_C_C0_nullity_wKtheta(rgm; kwargs...)[1]
get_C0_wKtheta(rgm::BncRegime; kwargs...) = get_C_C0_nullity_wKtheta(rgm; kwargs...)[2]

"""
    is_feasible_under_logkmap(rgm::BncRegime; R, b=nothing, space=:wKtheta, kind=:combined)

Return whether the mixed regime remains feasible after applying the log-space
reparameterization `log k = R * log θ + b`.
"""
function is_feasible_under_logkmap(
    rgm::BncRegime;
    R::AbstractMatrix{<:Real},
    b::Union{Nothing,AbstractVector}=nothing,
    space::Symbol=:wKtheta,
    kind::Symbol=:combined,
)
    C, C0, nlt = if space === :wKtheta
        get_C_C0_nullity_wKtheta(rgm; R=R, b=b)
    elseif space === :qKtheta
        get_C_C0_nullity_qKtheta(rgm; R=R, b=b, kind=kind)
    else
        error("Unsupported space=$space. Use :wKtheta or :qKtheta.")
    end
    return !isempty(get_polyhedron(C, C0, nlt))
end

"""
    feasible_bnc_regimes_under_logkmap(model::Bnc; R, b=nothing, space=:wKtheta, kind=:combined, return_idx=false, kwargs...)

Filter mixed regimes that remain feasible after applying the log-space
reparameterization `log k = R * log θ + b`.
"""
function feasible_bnc_regimes_under_logkmap(
    model::Bnc;
    R::AbstractMatrix{<:Real},
    b::Union{Nothing,AbstractVector}=nothing,
    space::Symbol=:wKtheta,
    kind::Symbol=:combined,
    return_idx::Bool=false,
    kwargs...,
)
    rgms = get_bnc_regimes(model; kwargs...)
    keep = filter(rgm -> is_feasible_under_logkmap(rgm; R=R, b=b, space=space, kind=kind), rgms)
    return return_idx ? get_idx.(keep) : keep
end
