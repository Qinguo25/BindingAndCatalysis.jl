@inline _bnc_prefer_fastpath(network::AbstractBnc) = backend_prefers_fastpath(_affine_is_exact(network))

function _project_bnc_singular_condition(
    network::AbstractBnc,
    C::AbstractMatrix,
    C0::AbstractVector,
    n_eq::Integer,
    delset::BitSet,
)
    p = get_polyhedron(C, C0, n_eq)
    p2 = backend_eliminate(p, delset; prefer_fastpath=_bnc_prefer_fastpath(network))
    return get_C_C0_nullity(p2)
end

function _calc_C_qKk_catalysis_only_regular(bind_rgm::BindRegime, cat_rgm::CatalysisRegime)
    H, H0 = get_H_H0(bind_rgm)
    CΠ = get_CΠ(cat_rgm)
    Cθ = get_C_k(cat_rgm)
    C0θ = get_C0(cat_rgm)
    C = hcat(CΠ * H, Cθ)
    C0 = CΠ * H0 + C0θ
    return C, vec(C0), 0
end

function _calc_C_qKk_catalysis_only_singular(bind_rgm::BindRegime, cat_rgm::CatalysisRegime)
    CΠ = get_CΠ(cat_rgm)
    Cθ = get_C_k(cat_rgm)
    C0θ = get_C0(cat_rgm)
    M, M0 = get_M_M0(bind_rgm)

    n_qK = size(M, 1)
    n_x = size(M, 2)
    n_v = size(Cθ, 2)
    d_cat = size(CΠ, 1)

    Eq = hcat(-_spI(Int, n_qK), _zeros_like(M, n_qK, n_v), M)
    In_cat = hcat(_zeros_like(CΠ, d_cat, n_qK), Cθ, CΠ)

    C = vcat(Eq, In_cat)
    C0 = vcat(M0, C0θ)

    return _project_bnc_singular_condition(
        bind_rgm.network,
        C,
        C0,
        n_qK,
        BitSet((n_qK + n_v + 1):(n_qK + n_v + n_x)),
    )
end

function _calc_C_qKk_catalysis_only(bind_rgm::BindRegime, cat_rgm::CatalysisRegime)
    if is_singular(bind_rgm)
        return _calc_C_qKk_catalysis_only_singular(bind_rgm, cat_rgm)
    else
        return _calc_C_qKk_catalysis_only_regular(bind_rgm, cat_rgm)
    end
end

function _first_nonempty_regime(rgms::AbstractMatrix{<:Union{BncRegime,Nothing}})
    pos = findfirst(x -> !isnothing(x), rgms)
    pos === nothing && return nothing
    return rgms[pos]::BncRegime
end

function _row_valid_columns(rgms::AbstractMatrix{<:Union{BncRegime,Nothing}}, i::Int)
    return [j for j in axes(rgms, 2) if !isnothing(rgms[i, j])]
end

function _row_unique_perm_data(perms)
    perm_keys = Vector{Tuple{Vararg{Int}}}(undef, length(perms))
    unique_keys = Tuple{Vararg{Int}}[]
    first_pos = Int[]
    key_to_pos = Dict{Tuple{Vararg{Int}},Int}()

    for (k, perm) in enumerate(perms)
        key = Tuple(Int.(perm))
        perm_keys[k] = key
        if !haskey(key_to_pos, key)
            key_to_pos[key] = length(unique_keys) + 1
            push!(unique_keys, key)
            push!(first_pos, k)
        end
    end

    return perm_keys, unique_keys, first_pos
end

function _steady_state_offsets(vtx::BncRegime, r_v::Int, N_ss)
    P0_ss = vtx.bind_rgm.P0[r_v + 1:end]
    M0_ss = vcat(P0_ss, zeros(eltype(P0_ss), size(N_ss, 1)))
    return P0_ss, M0_ss
end

function _expand_Hss_to_qssKk(H_ss, H0_ss, Pθ, P0θ)
    r_v = size(Pθ, 1)
    split = size(H_ss, 2) - r_v
    H_left = H_ss[:, 1:split]
    H_right = H_ss[:, split + 1:end]
    H_ssk = hcat(H_left, -(H_right * Pθ))
    H0_ssk = H0_ss - H_right * P0θ
    return H_ssk, vec(H0_ssk)
end

function _calc_C_qKk_cat_regular(bind_rgm::BindRegime, cat_rgm::CatalysisRegime)
    H, H0 = get_H_H0(bind_rgm)
    C_qK, C0_qK = get_C_C0_qK(bind_rgm)
    CΠ = get_CΠ(cat_rgm)
    Cθ = get_C_k(cat_rgm)
    C0θ = get_C0(cat_rgm)

    n_v = size(Cθ, 2)
    C1 = hcat(C_qK, _zeros_like(C_qK, size(C_qK, 1), n_v))
    C2 = hcat(CΠ * H, Cθ)

    C = vcat(C1, C2)
    C0 = vcat(C0_qK, CΠ * H0 + C0θ)

    return C, C0, 0
end

function _calc_C_qKk_cat_singular(bind_rgm::BindRegime, cat_rgm::CatalysisRegime)
    C_x, C0_x = get_C_C0_x(bind_rgm)
    CΠ = get_CΠ(cat_rgm)
    Cθ = get_C_k(cat_rgm)
    C0θ = get_C0(cat_rgm)
    M, M0 = get_M_M0(bind_rgm)

    n_qK = size(M, 1)
    n_x = size(M, 2)
    n_v = size(Cθ, 2)
    d_bind = size(C_x, 1)
    d_cat = size(CΠ, 1)

    Eq = hcat(-_spI(Int, n_qK), _zeros_like(M, n_qK, n_v), M)
    In_bind = hcat(_zeros_like(C_x, d_bind, n_qK + n_v), C_x)
    In_cat = hcat(_zeros_like(CΠ, d_cat, n_qK), Cθ, CΠ)

    C = vcat(Eq, In_bind, In_cat)
    C0 = vcat(M0, C0_x, C0θ)

    return _project_bnc_singular_condition(
        bind_rgm.network,
        C,
        C0,
        n_qK,
        BitSet((n_qK + n_v + 1):(n_qK + n_v + n_x)),
    )
end

function _calc_C_qKk_cat(bind_rgm::BindRegime, cat_rgm::CatalysisRegime)
    if is_singular(bind_rgm)
        return _calc_C_qKk_cat_singular(bind_rgm, cat_rgm)
    else
        return _calc_C_qKk_cat_regular(bind_rgm, cat_rgm)
    end
end

function _calc_C_qKk_ss_regular(
    bind_rgm::BindRegime,
    cat_rgm::CatalysisRegime,
    H_ssk,
    H0_ssk,
)
    C_x_bind, C0_x_bind = get_C_C0_x(bind_rgm)
    CΠ = get_CΠ(cat_rgm)
    Cθ = get_C_k(cat_rgm)
    C0θ = get_C0(cat_rgm)

    n_v = size(Cθ, 2)
    C_bind = C_x_bind * H_ssk
    C0_bind = C0_x_bind + C_x_bind * H0_ssk

    C_cat = copy(CΠ * H_ssk)
    @views C_cat[:, end - n_v + 1:end] .+= Cθ
    C0_cat = CΠ * H0_ssk + C0θ

    return vcat(C_bind, C_cat), vcat(C0_bind, C0_cat)
end

function _calc_C_qKk_ss_singular(bind_rgm::BindRegime, cat_rgm::CatalysisRegime)
    bn = bind_rgm.network
    r_v = size(cat_rgm.P, 1)

    P_ss = bind_rgm.P[r_v + 1:end, :]
    P0_ss = bind_rgm.P0[r_v + 1:end]
    N = bn.N
    Pθ = cat_rgm.P
    P0θ = get_P0(cat_rgm)
    PΠ = cat_rgm.PΠ

    C_x_bind, C0_x_bind = get_C_C0_x(bind_rgm)
    CΠ = get_CΠ(cat_rgm)
    Cθ = get_C_k(cat_rgm)
    C0θ = get_C0(cat_rgm)

    d_ss = size(P_ss, 1)
    r = size(N, 1)
    n_v = size(Pθ, 2)
    n_x = size(P_ss, 2)
    r_cat = size(Pθ, 1)

    Eq_qss = hcat(-_spI(Int, d_ss), _zeros_like(P_ss, d_ss, r + n_v), P_ss)
    Eq_K = hcat(_zeros_like(N, r, d_ss), -_spI(Int, r), _zeros_like(N, r, n_v), N)
    Eq_cat = hcat(_zeros_like(PΠ, r_cat, d_ss + r), Pθ, PΠ)
    In_bind = hcat(_zeros_like(C_x_bind, size(C_x_bind, 1), d_ss + r + n_v), C_x_bind)
    In_cat = hcat(_zeros_like(CΠ, size(CΠ, 1), d_ss + r), Cθ, CΠ)

    C = vcat(Eq_qss, Eq_K, Eq_cat, In_bind, In_cat)
    C0 = vcat(P0_ss, zeros(eltype(P0_ss), r), P0θ, C0_x_bind, C0θ)

    return _project_bnc_singular_condition(
        bn,
        C,
        C0,
        d_ss + r + r_cat,
        BitSet((d_ss + r + n_v + 1):(d_ss + r + n_v + n_x)),
    )
end
