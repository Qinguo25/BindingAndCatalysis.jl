function _build_row_affine_cache(rgms, i, valid_js, perms, nlt_valid, N_ss, r_v, direction, cache)
    perm_keys, unique_keys, first_pos = _row_unique_perm_data(perms)
    Hs = Vector{Any}(undef, length(unique_keys))
    H0s = Vector{Any}(undef, length(unique_keys))

    Threads.@threads for t in eachindex(unique_keys)
        k = first_pos[t]
        nlt = nlt_valid[k]
        if nlt > 1
            Hs[t] = nothing
            H0s[t] = nothing
            continue
        end

        rgm = rgms[i, valid_js[k]]::BncRegime
        _initialize_regime!(rgm.bind_rgm)
        perm = perms[k]
        _, M0_ss = _steady_state_offsets(rgm, r_v, N_ss)

        H_ss = if rgm.bind_rgm isa BindRegime{ExactLogExpr} && nlt == 0
            _exact_calc_H_regular(perm, N_ss)
        elseif rgm.bind_rgm isa BindRegime{ExactLogExpr}
            _build_singular_H_from_perm_exact(perm, N_ss, Int(direction))[1]
        elseif nlt == 0
            _calc_H(N_ss, cache, perm)
        else
            M_ss = vcat(rgm.bind_rgm.P[r_v + 1:end, :], N_ss)
            if allunique(perm)
                _calc_H(N_ss, cache, perm; scale=direction)
            else
                H_tmp = _adj_singular_matrix(M_ss)[1]
                droptol!(sparse(H_tmp), 1e-10) .* direction
            end
        end

        Hs[t] = H_ss
        H0s[t] = vec(-(H_ss * M0_ss))
    end

    affine_by_perm = Dict{Tuple{Vararg{Int}},Tuple{Any,Any}}()
    for t in eachindex(unique_keys)
        Hs[t] === nothing && continue
        affine_by_perm[unique_keys[t]] = (Hs[t], H0s[t])
    end

    return perm_keys, affine_by_perm
end

function _build_row_context(rgms::AbstractMatrix{<:Union{BncRegime,Nothing}}, i::Int, r_v::Int)
    valid_js = _row_valid_columns(rgms, i)
    isempty(valid_js) && return nothing

    ref_vtx = rgms[i, first(valid_js)]::BncRegime
    bn = ref_vtx.bind_rgm.network

    N_ss = vcat(bn.N, ref_vtx.catalysis_rgm.PΠ)
    L_ss = bn.L[r_v + 1:end, :]
    direction = _det_sign_exact(vcat(L_ss, N_ss))

    perms = [get_perm(rgms[i, j]::BncRegime) for j in valid_js]
    nlt_valid, cache = _calc_nullity(perms, N_ss)
    perm_keys, affine_by_perm = _build_row_affine_cache(rgms, i, valid_js, perms, nlt_valid, N_ss, r_v, direction, cache)

    return (
        bn=bn,
        r_v=r_v,
        N_ss=N_ss,
        L_ss=L_ss,
        direction=direction,
        valid_js=valid_js,
        perms=perms,
        perm_keys=perm_keys,
        nlt_valid=nlt_valid,
        cache=cache,
        affine_by_perm=affine_by_perm,
    )
end

function _init_regular_bnc_regime!(vtx::BncRegime, perm, rowctx)
    C_qKk_cat, C0_qKk_cat, nlt_qKk_cat = _calc_C_qKk_cat(vtx.bind_rgm, vtx.catalysis_rgm)

    H_w, H0_w = rowctx.affine_by_perm[Tuple(Int.(perm))]
    Pθ = vtx.catalysis_rgm.P
    P0θ = get_P0(vtx.catalysis_rgm)
    H_wKk, H0_wKk = _expand_Hw_to_wKk(H_w, H0_w, Pθ, P0θ)
    C_wKk, C0_wKk = _calc_C_wKk_regular(vtx.bind_rgm, vtx.catalysis_rgm, H_wKk, H0_wKk)

    vtx.H = H_wKk
    vtx.H0 = _materialize_real_vector(H0_wKk)
    vtx.C_qKk_cat = C_qKk_cat
    vtx.C0_qKk_cat = _materialize_real_vector(C0_qKk_cat)
    vtx.nlt_qKk_cat = nlt_qKk_cat
    vtx.C_wKk = C_wKk
    vtx.C0_wKk = _materialize_real_vector(C0_wKk)
    return nothing
end

function _calc_singular_H_ss(bind_rgm::BindRegime, cat_rgm::CatalysisRegime, perm, rowctx)
    r_v = rowctx.r_v
    M_ss = vcat(bind_rgm.P[r_v + 1:end, :], rowctx.N_ss)
    P0_ss = bind_rgm.P0[r_v + 1:end]
    M0_ss = vcat(P0_ss, zeros(eltype(P0_ss), size(rowctx.N_ss, 1)))

    H_ray = if allunique(perm)
        _calc_H(rowctx.N_ss, rowctx.cache, perm; scale=rowctx.direction)
    else
        H_tmp = _adj_singular_matrix(M_ss)[1]
        droptol!(sparse(H_tmp), 1e-10) .* rowctx.direction
    end

    H0_ray = vec(-(H_ray * M0_ss))
    return M_ss, H_ray, H0_ray
end

function _init_singular_bnc_regime!(vtx::BncRegime, perm, rowctx)
    C_qKk_cat, C0_qKk_cat, nlt_qKk_cat = _calc_C_qKk_cat(vtx.bind_rgm, vtx.catalysis_rgm)

    H_w, H0_w = rowctx.affine_by_perm[Tuple(Int.(perm))]
    Pθ = get_P(vtx.catalysis_rgm)
    P0θ = get_P0(vtx.catalysis_rgm)
    H_wKk, H0_wKk = _expand_Hw_to_wKk(H_w, H0_w, Pθ, P0θ)

    C_wKk, C0_wKk, _ = _calc_C_wKk_singular(vtx.bind_rgm, vtx.catalysis_rgm)

    vtx.H = H_wKk
    vtx.H0 = _materialize_real_vector(H0_wKk)
    vtx.C_qKk_cat = C_qKk_cat
    vtx.C0_qKk_cat = _materialize_real_vector(C0_qKk_cat)
    vtx.nlt_qKk_cat = nlt_qKk_cat
    vtx.C_wKk = C_wKk
    vtx.C0_wKk = _materialize_real_vector(C0_wKk)
    return nothing
end

function _initialize_regime!(rgms::AbstractMatrix{<:Union{BncRegime,Nothing}})
    first_vtx = _first_nonempty_regime(rgms)
    first_vtx === nothing && return nothing

    r_v = size(first_vtx.catalysis_rgm.P, 1)
    @info "Initializing BncRegimes..."

    for i in axes(rgms, 1)
        rowctx = _build_row_context(rgms, i, r_v)
        rowctx === nothing && continue

        valid_js = rowctx.valid_js
        perms = rowctx.perms
        nlt_valid = rowctx.nlt_valid

        Threads.@threads for k in eachindex(valid_js)
            j = valid_js[k]
            vtx = rgms[i, j]::BncRegime
            perm = perms[k]
            nlt = nlt_valid[k]

            vtx.nlt = nlt
            nlt > 1 && continue

            if nlt == 0
                _init_regular_bnc_regime!(vtx, perm, rowctx)
            else
                _init_singular_bnc_regime!(vtx, perm, rowctx)
            end
        end
    end

    @info "Finished initializing BncRegimes."
    return nothing
end
