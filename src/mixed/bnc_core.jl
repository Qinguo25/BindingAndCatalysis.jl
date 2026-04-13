@inline _spI(T, n) = spdiagm(0 => ones(T, n))
@inline _zeros_like(A::AbstractMatrix{T}, m::Int, n::Int) where {T<:Real} = spzeros(T, m, n)
@inline _zeros_like(v::AbstractVector{T}, n::Int) where {T<:Real} = zeros(T, n)
@inline function _det_sign_exact(A::AbstractMatrix{<:Integer})
    detA = _bareiss_det_big(Matrix{Int}(A))
    return detA > 0 ? 1 : detA < 0 ? -1 : 0
end

get_binding_regime(rgm::BncRegime) = rgm.bind_rgm
get_catalysis_regime(rgm::BncRegime) = rgm.catalysis_rgm

get_binding_perm(rgm::BncRegime) = get_perm(rgm.bind_rgm)
get_catalysis_perm(rgm::BncRegime) = get_perm(rgm.catalysis_rgm)

function get_perm(rgm::BncRegime)
    r_v = size(rgm.catalysis_rgm.P, 1)
    return rgm.bind_rgm.perm[r_v + 1:end]
end
get_steady_state_perm(rgm::BncRegime) = get_perm(rgm)

get_idx(rgm::BncRegime) = CartesianIndex(get_idx(rgm.catalysis_rgm), get_idx(rgm.bind_rgm))
@inline is_bnc_regimes_built(model::Bnc) = !isnothing(model.BncRegimes)

function _build_BncRegime(cat_rgms::Regimes, bind_rgms::Regimes)
    n_cat_rgms = length(cat_rgms.vertices_data)
    n_bind_rgms = length(bind_rgms.vertices_data)
    bncrgms = Matrix{Union{BncRegime,Nothing}}(undef, n_cat_rgms, n_bind_rgms)

    @info "Matching Catalysis Regimes and Binding Regimes to build BncRegimes..."
    Threads.@threads for i in 1:n_cat_rgms
        cat_rgm = cat_rgms.vertices_data[i]
        for j in 1:n_bind_rgms
            bind_rgm = bind_rgms.vertices_data[j]
            bncrgms[i, j] = bind_rgm.nullity > 1 ? nothing : BncRegime(bind_rgm, cat_rgm)
        end
    end
    @info "Finished matching BncRegimes."
    return bncrgms
end

function get_idx(model::Bnc, bind, cat; check::Bool=false)
    cat_idx = get_idx(_require_catalysis_network(model), cat; check=check)
    bind_idx = get_idx(model, bind; check=check)
    return CartesianIndex(cat_idx, bind_idx)
end

function have_perm(model::Bnc, bind, cat)
    if !have_perm(model, bind)
        return false
    end
    cn = get_catalysis_network(model)
    if isnothing(cn) || !have_perm(cn, cat)
        return false
    end
    match_regimes!(model)
    return !isnothing(model.BncRegimes[get_idx(model, bind, cat)])
end

function get_bnc_regime(model::Bnc, bind, cat; check::Bool=false)
    match_regimes!(model)
    idx = get_idx(model, bind, cat; check=check)
    rgm = model.BncRegimes[idx]
    if isnothing(rgm)
        check && error("No BncRegime is stored for the requested binding/catalysis pair.")
        return nothing
    end
    return rgm
end
get_regime(model::Bnc, bind, cat; kwargs...) = get_bnc_regime(model, bind, cat; kwargs...)
get_regime(rgm::BncRegime; kwargs...) = rgm

function get_bnc_regimes(model::Bnc; return_idx::Bool=false, singular::Union{Bool,Integer,Nothing}=nothing)
    match_regimes!(model)
    idxs = CartesianIndex[]
    rgms = BncRegime[]

    for I in CartesianIndices(model.BncRegimes)
        rgm = model.BncRegimes[I]
        isnothing(rgm) && continue

        nlt = rgm.nlt
        keep = isnothing(singular) || (
            (singular === true && nlt > 0) ||
            (singular === false && nlt == 0) ||
            (singular isa Int && nlt <= singular)
        )
        keep || continue

        push!(idxs, I)
        push!(rgms, rgm)
    end

    return return_idx ? idxs : rgms
end

n_bnc_regimes(model::Bnc; kwargs...) = length(get_bnc_regimes(model; kwargs...))

function get_H_H0(rgm::BncRegime)
    rgm.nlt <= 1 || error("BncRegime nullity is bigger than 1, cannot get H0.")
    return rgm.H, rgm.H0
end
get_H(rgm::BncRegime) = rgm.H
get_H0(rgm::BncRegime) = get_H_H0(rgm)[2]
get_H_bd(rgm::BncRegime) = rgm.H_bd

function judge_stability!(rgm::BncRegime; kwargs...)
    rgm.is_stable = Int8(judge_dstable(rgm.H_bd; kwargs...))
    return rgm.is_stable
end

function is_stable(rgm::BncRegime; recalculate::Bool=false, return_code::Bool=false, kwargs...)
    code = (recalculate || rgm.is_stable < 0) ? judge_stability!(rgm; kwargs...) : rgm.is_stable
    return return_code ? code : (code == 1 ? true : code == 0 ? false : missing)
end
is_stable(model::Bnc, bind, cat; kwargs...) = is_stable(get_bnc_regime(model, bind, cat; check=true); kwargs...)

function _binding_C_qKk(bind_rgm::BindRegime, n_v::Int)
    C_qK, C0_qK, nlt = get_C_C0_nullity_qK(bind_rgm)
    C = hcat(C_qK, _zeros_like(C_qK, size(C_qK, 1), n_v))
    return C, C0_qK, nlt
end

function get_C_C0_nullity_xk(rgm::BncRegime, kind::Symbol=:combined)
    bind_rgm = rgm.bind_rgm
    cat_rgm = rgm.catalysis_rgm
    n_v = size(cat_rgm.P, 2)

    if kind === :binding
        C_x, C0_x = get_C_C0_x(bind_rgm)
        C = hcat(C_x, _zeros_like(C_x, size(C_x, 1), n_v))
        return C, C0_x, 0
    elseif kind === :catalysis
        return get_C_C0_nullity_xk(cat_rgm)
    elseif kind === :combined
        Ceq = get_P_xk(cat_rgm)
        Ccat = get_C_xk(cat_rgm)
        Cbind_x, C0bind_x = get_C_C0_x(bind_rgm)
        Cbind = hcat(Cbind_x, _zeros_like(Cbind_x, size(Cbind_x, 1), n_v))
        C = vcat(Ceq, Cbind, Ccat)
        C0 = vcat(get_P0(cat_rgm), C0bind_x, get_C0(cat_rgm))
        return C, C0, size(Ceq, 1)
    else
        error("Unsupported kind=$kind. Use :binding, :catalysis, or :combined.")
    end
end

function get_C_C0_xk(rgm::BncRegime, kind::Symbol=:combined)
    C, C0, _ = get_C_C0_nullity_xk(rgm, kind)
    return C, C0
end
get_C_xk(rgm::BncRegime, kind::Symbol=:combined) = get_C_C0_nullity_xk(rgm, kind)[1]
get_C0_xk(rgm::BncRegime, kind::Symbol=:combined) = get_C_C0_nullity_xk(rgm, kind)[2]

function get_C_C0_nullity_qKk(rgm::BncRegime, kind::Symbol=:combined)
    n_v = size(rgm.catalysis_rgm.P, 2)

    if kind === :binding
        return _binding_C_qKk(rgm.bind_rgm, n_v)
    elseif kind === :catalysis
        return _calc_C_qKk_catalysis_only(rgm.bind_rgm, rgm.catalysis_rgm)
    elseif kind === :combined
        return rgm.C_qKk_cat, rgm.C0_qKk_cat, rgm.nlt_qKk_cat
    else
        error("Unsupported kind=$kind. Use :binding, :catalysis, or :combined.")
    end
end

function get_C_C0_qKk(rgm::BncRegime, kind::Symbol=:combined)
    C, C0, _ = get_C_C0_nullity_qKk(rgm, kind)
    return C, C0
end
get_C_qKk(rgm::BncRegime, kind::Symbol=:combined) = get_C_C0_nullity_qKk(rgm, kind)[1]
get_C0_qKk(rgm::BncRegime, kind::Symbol=:combined) = get_C_C0_nullity_qKk(rgm, kind)[2]

function get_C_C0_nullity_qssKk(rgm::BncRegime)
    return rgm.C_qKk_ss, rgm.C0_qKk_ss, rgm.nlt
end

function get_C_C0_qssKk(rgm::BncRegime)
    C, C0, _ = get_C_C0_nullity_qssKk(rgm)
    return C, C0
end
get_C_qssKk(rgm::BncRegime) = get_C_C0_nullity_qssKk(rgm)[1]
get_C0_qssKk(rgm::BncRegime) = get_C_C0_nullity_qssKk(rgm)[2]

function get_qcat_F_F0(rgm::BncRegime)
    rgm.nlt == 0 || error("The reduced steady-state system is singular, so q_cat has no affine expression in (q_ss, K, k).")
    r_v = size(rgm.catalysis_rgm.P, 1)
    P_cat = rgm.bind_rgm.P[1:r_v, :]
    P0_cat = rgm.bind_rgm.P0[1:r_v]
    F = P_cat * rgm.H
    F0 = P0_cat + P_cat * rgm.H0
    return F, vec(F0)
end
