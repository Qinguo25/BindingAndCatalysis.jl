function _build_BncRegime(cat_rgms::Regimes, bind_rgms::Regimes)
    n_cat_rgms = length(cat_rgms.vertices_data)
    n_bind_rgms = length(bind_rgms.vertices_data)
    bncrgms = Matrix{Union{BncRegime,Nothing}}(undef, n_cat_rgms, n_bind_rgms)

    @info "Matching Catalysis Regimes and Binding Regimes to build BncRegimes..."
    Threads.@threads for i in 1:n_cat_rgms
        cat_rgm = cat_rgms.vertices_data[i]
        for j in 1:n_bind_rgms
            bind_rgm = bind_rgms.vertices_data[j]
            bncrgms[i, j] = BncRegime(bind_rgm, cat_rgm)
        end
    end
    @info "Finished matching BncRegimes."
    return bncrgms
end

function _steady_state_affine(
    bind_rgm::BindRegime,
    perm,
    N_ss,
    r_v::Int,
    direction::Int,
    nlt::Int,
)
    nlt > 1 && return nothing

    P0_ss = bind_rgm.P0[r_v + 1:end]
    M0_ss = vcat(P0_ss, zeros(eltype(P0_ss), size(N_ss, 1)))

    H_ss = if nlt == 0
        _exact_calc_H_regular(perm, N_ss)
    else
        _build_singular_H_from_perm_exact(perm, N_ss, direction)[1]
    end
    isnothing(H_ss) && error("Failed to build steady-state affine map for a mixed regime with nullity $nlt.")

    return H_ss, vec(-(H_ss * M0_ss))
end

function _build_row_affine_cache(rgms, i, valid_js, perms, nlt_valid, N_ss, r_v, direction)
    unique_keys, first_pos = _row_unique_perm_data(perms)
    affine_pairs = Union{Nothing,Tuple{Any,Any}}[nothing for _ in eachindex(unique_keys)]

    Threads.@threads for t in eachindex(unique_keys)
        k = first_pos[t]
        nlt = nlt_valid[k]
        nlt > 1 && continue

        rgm = rgms[i, valid_js[k]]::BncRegime
        _initialize_regime!(rgm.bind_rgm)
        affine_pairs[t] = _steady_state_affine(
            rgm.bind_rgm,
            perms[k],
            N_ss,
            r_v,
            direction,
            nlt,
        )
    end

    affine_by_perm = Dict{Tuple{Vararg{Int}},Tuple{Any,Any}}()
    for t in eachindex(unique_keys)
        isnothing(affine_pairs[t]) && continue
        affine_by_perm[unique_keys[t]] = affine_pairs[t]
    end

    return affine_by_perm
end

function _build_row_context(rgms::AbstractMatrix{<:Union{BncRegime,Nothing}}, i::Int, r_v::Int)
    valid_js = _row_valid_columns(rgms, i)
    isempty(valid_js) && return nothing

    ref_vtx = rgms[i, first(valid_js)]::BncRegime
    bn = ref_vtx.bind_rgm.network

    N_ss = vcat(bn.N, ref_vtx.catalysis_rgm.PΠ)
    direction = _det_sign_exact(vcat(bn.L[r_v + 1:end, :], N_ss))

    perms = [get_perm(rgms[i, j]::BncRegime) for j in valid_js]
    nlt_valid, _ = _calc_nullity(perms, N_ss)
    affine_by_perm = _build_row_affine_cache(rgms, i, valid_js, perms, nlt_valid, N_ss, r_v, direction)

    return (
        valid_js=valid_js,
        perms=perms,
        nlt_valid=nlt_valid,
        affine_by_perm=affine_by_perm,
    )
end



function _init_regular_bnc_regime!(vtx::BncRegime, perm, rowctx)
    C_qKk_cat, C0_qKk_cat, nlt_qKk_cat = _calc_C_qKk_cat(vtx.bind_rgm, vtx.catalysis_rgm)
    H_ss, H0_ss = rowctx.affine_by_perm[Tuple(Int.(perm))]
    H_wKk, H0_wKk = _expand_Hw_to_wKk(H_ss, H0_ss, get_P(vtx.catalysis_rgm), get_P0(vtx.catalysis_rgm))
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

function _init_singular_bnc_regime!(vtx::BncRegime, perm, rowctx)
    C_qKk_cat, C0_qKk_cat, nlt_qKk_cat = _calc_C_qKk_cat(vtx.bind_rgm, vtx.catalysis_rgm)
    H_ss, H0_ss = rowctx.affine_by_perm[Tuple(Int.(perm))]
    H_wKk, H0_wKk = _expand_Hw_to_wKk(H_ss, H0_ss, get_P(vtx.catalysis_rgm), get_P0(vtx.catalysis_rgm))
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

function _init_consistency_only_bnc_regime!(vtx::BncRegime)
    C_qKk_cat, C0_qKk_cat, nlt_qKk_cat = _calc_C_qKk_cat(vtx.bind_rgm, vtx.catalysis_rgm)
    C_wKk, C0_wKk, _ = _calc_C_wKk_singular(vtx.bind_rgm, vtx.catalysis_rgm)

    # vtx.H_bd = nothing
    # vtx.is_stable = Int8(0) initial value is In8(0)
    vtx.H = nothing
    vtx.H0 = nothing
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

        @showprogress Threads.@threads for k in eachindex(valid_js)
            j = valid_js[k]
            vtx = rgms[i, j]::BncRegime
            perm = perms[k]
            nlt = nlt_valid[k]

            vtx.nlt = nlt
            if nlt == 0
                _init_regular_bnc_regime!(vtx, perm, rowctx)
            elseif nlt == 1
                _init_singular_bnc_regime!(vtx, perm, rowctx)
            else
                _init_consistency_only_bnc_regime!(vtx)
            end
        end
    end

    @info "Finished initializing BncRegimes."

    return nothing
end
