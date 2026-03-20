#============================================================================================#
#                            Fucntions related to BncRegime
#
#============================================================================================#

@inline _spI(T, n) = spdiagm(0 => ones(T, n))

"""
    get_perm(rgm::BncRegime)

Extract the binding permutation restricted to the steady-state coordinates
q_ss = (w, q_para), i.e. drop the first r_v catalysis-active rows.
"""
function get_perm(rgm::BncRegime)
    r_v = size(rgm.catalysis_rgm.P, 1)
    perm = rgm.bind_rgm.perm[r_v+1:end]
    return perm
end





"""
    _first_nonempty_regime(rgms)

Return the first non-`nothing` BncRegime in a matrix of candidate mixed regimes.
Useful for reading shared dimensions safely.
"""
function _first_nonempty_regime(rgms::AbstractMatrix{<:Union{BncRegime,Nothing}})
    pos = findfirst(x -> !isnothing(x), rgms)
    pos === nothing && return nothing
    return rgms[pos]::BncRegime
end
"""
    _row_valid_columns(rgms, i)

For a fixed catalysis regime row `i`, return the binding-regime columns that are
actually present (i.e. not `nothing`).
"""
function _row_valid_columns(rgms::AbstractMatrix{<:Union{BncRegime,Nothing}}, i::Int)
    return [j for j in axes(rgms, 2) if !isnothing(rgms[i, j])]
end


"""
    _build_row_context(rgms, i, r_v)

Build row-shared data for the i-th catalysis regime. All mixed regimes in the same
row share the same catalysis regime, hence they share:
- N_ss = [N; PΠ]
- direction = sign(det([L_ss; N_ss]))
- nullity/cache data computed from the steady-state permutations.
"""
function _build_row_context(rgms::AbstractMatrix{<:Union{BncRegime,Nothing}}, i::Int, r_v::Int)
    valid_js = _row_valid_columns(rgms, i)
    isempty(valid_js) && return nothing

    ref_vtx = rgms[i, first(valid_js)]::BncRegime
    bn = ref_vtx.bind_rgm.network

    N_ss = vcat(bn.N, ref_vtx.catalysis_rgm.PΠ)
    L_ss = bn.L[r_v+1:end, :]
    direction = sign(det(Matrix{Float64}(vcat(L_ss, N_ss))))

    perms = [get_perm(rgms[i, j]::BncRegime) for j in valid_js]
    nlt_valid, cache = _calc_nullity(perms, N_ss)

    return (
        bn = bn,
        r_v = r_v,
        N_ss = N_ss,
        L_ss = L_ss,
        direction = direction,
        valid_js = valid_js,
        perms = perms,
        nlt_valid = nlt_valid,
        cache = cache,
    )
end



"""
    _steady_state_offsets(vtx, r_v, N_ss)

Extract P0_ss and M0_ss for the steady-state reduced binding network:
    M_ss  = [P_ss; N_ss]
    M0_ss = [P0_ss; 0]
where P_ss is obtained from the binding regime by dropping the first r_v rows.
"""
function _steady_state_offsets(vtx::BncRegime, r_v::Int, N_ss)
    P0_ss = vtx.bind_rgm.P0[r_v+1:end]
    M0_ss = vcat(P0_ss, zeros(eltype(P0_ss), size(N_ss, 1)))
    return P0_ss, M0_ss
end



"""
    _expand_Hss_to_qssKk(H_ss, Pθ)

Convert
    log x = H_ss * log(q_ss, K_ss) + H0_ss
with
    log K_ss = [log K; -Pθ * log k]
into
    log x = H_ssk * log(q_ss, K, k) + H0_ss.
"""
function _expand_Hss_to_qssKk(H_ss, Pθ)
    r_v = size(Pθ, 1)
    split = size(H_ss, 2) - r_v
    @views H_left = H_ss[:, 1:split]
    @views H_right = H_ss[:, split+1:end]
    return hcat(H_left, -(H_right * Pθ))
end



# ------------------------------------------------
# Catalytic consistency conditions in (q, K, k)
# ------------------------------------------------

"""
    _calc_C_qKk_cat_regular(bind_rgm, cat_rgm)

Regular binding regime:
- binding regime condition is already available in (q, K) coordinates:
      C_qK * log(q, K) + C0_qK >= 0
- catalytic dominance condition is
      CΠ * log x + Cθ * log k >= 0
  and we substitute
      log x = H * log(q, K) + H0.

Output variables are ordered as (q, K, k).
"""
function _calc_C_qKk_cat_regular(bind_rgm::BindRegime, cat_rgm::CatalysisRegime)
    H, H0 = get_H_H0(bind_rgm)
    C_qK, C0_qK = get_C_C0_qK(bind_rgm)
    CΠ, Cθ = get_C_xk(cat_rgm)

    n_v = size(Cθ, 2)

    C1 = hcat(C_qK, spzeros(size(C_qK, 1), n_v))
    C2 = hcat(CΠ * H, Cθ)

    C = vcat(C1, C2)
    C0 = vcat(C0_qK, CΠ * H0)

    return C, C0, 0
end



"""
    _calc_C_qKk_cat_singular(bind_rgm, cat_rgm)

Singular binding regime:
we work in the extended variables
    z = (log(q, K), log k, log x)
and encode
    -I * log(q, K) + M * log x + M0 = 0
    C_x * log x + C0_x >= 0
    CΠ * log x + Cθ * log k >= 0
then eliminate log x.

Output variables are ordered as (q, K, k).
"""
function _calc_C_qKk_cat_singular(bind_rgm::BindRegime, cat_rgm::CatalysisRegime)
    C_x, C0_x = get_C_C0_x(bind_rgm)
    CΠ, Cθ = get_C_xk(cat_rgm)
    M, M0 = get_M_M0(bind_rgm)

    n_qK = size(M, 1)
    n_x = size(M, 2)
    n_v = size(Cθ, 2)
    d_bind = size(C_x, 1)
    d_cat = size(CΠ, 1)

    Eq = hcat(-_spI(Int, n_qK), spzeros(n_qK, n_v), M)
    In_bind = hcat(spzeros(d_bind, n_qK + n_v), C_x)
    In_cat = hcat(spzeros(d_cat, n_qK), Cθ, CΠ)

    C = vcat(Eq, In_bind, In_cat)
    C0 = vcat(M0, C0_x, zeros(eltype(M0), d_cat))

    p = get_polyhedron(C, C0, n_qK)
    delset = BitSet((n_qK + n_v + 1):(n_qK + n_v + n_x))
    p2 = eliminate(p, delset)

    return get_C_C0_nullity(p2)
end



"""
    _calc_C_qKk_cat(bind_rgm, cat_rgm)

Dispatch to the regular or singular implementation.
"""
function _calc_C_qKk_cat(bind_rgm::BindRegime, cat_rgm::CatalysisRegime)
    if is_singular(bind_rgm)
        return _calc_C_qKk_cat_singular(bind_rgm, cat_rgm)
    else
        return _calc_C_qKk_cat_regular(bind_rgm, cat_rgm)
    end
end


# ------------------------------------------------
# Steady-state consistency conditions in (q_ss, K, k)
# ------------------------------------------------

"""
    _calc_C_qKk_ss_regular(bind_rgm, cat_rgm, H_ssk, H0_ss)

Regular steady-state reduced regime:
    log x = H_ssk * log(q_ss, K, k) + H0_ss

We combine
1) binding dominance condition in x-space
2) catalytic dominance condition in x,k-space
and push them into the variables (q_ss, K, k).
"""
function _calc_C_qKk_ss_regular(
    bind_rgm::BindRegime,
    cat_rgm::CatalysisRegime,
    H_ssk,
    H0_ss,
)
    C_x_bind, C0_x_bind = get_C_C0_x(bind_rgm)
    CΠ, Cθ = get_C_xk(cat_rgm)

    n_v = size(Cθ, 2)

    # Binding dominance: C_x * x + C0_x >= 0
    C_bind = C_x_bind * H_ssk
    C0_bind = C0_x_bind + C_x_bind * H0_ss

    # Catalytic dominance: CΠ * x + Cθ * k >= 0
    C_cat = copy(CΠ * H_ssk)
    @views C_cat[:, end-n_v+1:end] .+= Cθ
    C0_cat = CΠ * H0_ss

    return vcat(C_bind, C_cat), vcat(C0_bind, C0_cat)
end

"""
    _calc_C_qKk_ss_singular(bind_rgm, cat_rgm)

Singular steady-state reduced regime:
we work in the extended variables
    z = (log q_ss, log K, log k, log x)
and encode
    -I * log q_ss + P_ss * log x + P0_ss = 0
    -I * log K    + N    * log x        = 0
     Pθ * log k   + PΠ   * log x        = 0
    C_x * log x + C0_x >= 0
    CΠ * log x + Cθ * log k >= 0
then eliminate log x.

Output variables are ordered as (q_ss, K, k).
"""
function _calc_C_qKk_ss_singular(bind_rgm::BindRegime, cat_rgm::CatalysisRegime)
    bn = bind_rgm.network
    r_v = size(cat_rgm.P, 1)

    P_ss = bind_rgm.P[r_v+1:end, :]
    P0_ss = bind_rgm.P0[r_v+1:end]
    N = bn.N
    Pθ = cat_rgm.P
    PΠ = cat_rgm.PΠ

    C_x_bind, C0_x_bind = get_C_C0_x(bind_rgm)
    CΠ, Cθ = get_C_xk(cat_rgm)

    d_ss = size(P_ss, 1)
    r = size(N, 1)
    n_v = size(Pθ, 2)
    n_x = size(P_ss, 2)
    r_cat = size(Pθ, 1)

    Eq_qss = hcat(-_spI(Int, d_ss), spzeros(d_ss, r + n_v), P_ss)
    Eq_K = hcat(spzeros(r, d_ss), -_spI(Int, r), spzeros(r, n_v), N)
    Eq_cat = hcat(spzeros(r_cat, d_ss + r), Pθ, PΠ)

    In_bind = hcat(spzeros(size(C_x_bind, 1), d_ss + r + n_v), C_x_bind)
    In_cat = hcat(spzeros(size(CΠ, 1), d_ss + r), Cθ, CΠ)

    C = vcat(Eq_qss, Eq_K, Eq_cat, In_bind, In_cat)
    C0 = vcat(
        P0_ss,
        zeros(eltype(P0_ss), r + r_cat),
        C0_x_bind,
        zeros(eltype(P0_ss), size(CΠ, 1)),
    )

    n_eq = d_ss + r + r_cat
    p = get_polyhedron(C, C0, n_eq)
    delset = BitSet((d_ss + r + n_v + 1):(d_ss + r + n_v + n_x))
    p2 = eliminate(p, delset)

    return get_C_C0_nullity(p2)
end


# ------------------------------------------------
# Per-regime initialization: regular / singular
# ------------------------------------------------

"""
    _init_regular_bnc_regime!(vtx, perm, rowctx)

Initialize one mixed regime whose steady-state reduced matrix M_ss is invertible.
This computes:
- H  : map (q_ss, K, k) -> x
- H0 : affine offset
- C_qKk_cat / C0_qKk_cat : catalysis consistency in (q, K, k)
- C_qKk_ss  / C0_qKk_ss  : steady-state consistency in (q_ss, K, k)
"""
function _init_regular_bnc_regime!(vtx::BncRegime, perm, rowctx)
    r_v = rowctx.r_v
    N_ss = rowctx.N_ss

    Pθ = vtx.catalysis_rgm.P
    _, M0_ss = _steady_state_offsets(vtx, r_v, N_ss)

    # 1) Catalysis consistency on (q, K, k)
    C_qKk_cat, C0_qKk_cat, nlt_qKk_cat = _calc_C_qKk_cat(vtx.bind_rgm, vtx.catalysis_rgm)

    # 2) Steady-state reduced affine inverse on (q_ss, K_ss)
    H_ss = _calc_H(N_ss, rowctx.cache, perm)
    H0_ss = -(H_ss * M0_ss)

    # 3) Replace K_ss by (K, k)
    H_ssk = _expand_Hss_to_qssKk(H_ss, Pθ)

    # 4) Steady-state consistency on (q_ss, K, k)
    C_qKk_ss, C0_qKk_ss = _calc_C_qKk_ss_regular(vtx.bind_rgm, vtx.catalysis_rgm, H_ssk, H0_ss)

    vtx.H = H_ssk
    vtx.H0 = H0_ss
    vtx.C_qKk_cat = C_qKk_cat
    vtx.C0_qKk_cat = C0_qKk_cat
    vtx.nlt_qKk_cat = nlt_qKk_cat
    vtx.C_qKk_ss = C_qKk_ss
    vtx.C0_qKk_ss = C0_qKk_ss

    return nothing
end



"""
    _calc_singular_H_ss(bind_rgm, cat_rgm, perm, rowctx)

Build the nullity-1 ray/adjugate-like matrix for the reduced steady-state system.
This does NOT return an affine offset H0.
"""
function _calc_singular_H_ss(bind_rgm::BindRegime, cat_rgm::CatalysisRegime, perm, rowctx)
    r_v = rowctx.r_v
    M_ss = vcat(bind_rgm.P[r_v+1:end, :], rowctx.N_ss)

    H_ray = if allunique(perm)
        _calc_H(rowctx.N_ss, rowctx.cache, perm; scale = rowctx.direction)
    else
        H_tmp = _adj_singular_matrix(M_ss)[1]
        droptol!(sparse(H_tmp), 1e-10) .* rowctx.direction
    end

    return M_ss, H_ray
end




"""
    _init_singular_bnc_regime!(vtx, perm, rowctx)

Initialize one mixed regime whose steady-state reduced matrix M_ss has nullity 1.
This computes:
- H  : a ray/adjugate-like matrix, not an affine inverse
- H0 : nothing
- C_qKk_cat / C0_qKk_cat : catalysis consistency in (q, K, k)
- C_qKk_ss  / C0_qKk_ss  : steady-state consistency in (q_ss, K, k),
                           obtained by explicit elimination of x
"""
function _init_singular_bnc_regime!(vtx::BncRegime, perm, rowctx)
    # 1) Catalysis consistency on (q, K, k)
    C_qKk_cat, C0_qKk_cat, nlt_qKk_cat = _calc_C_qKk_cat(vtx.bind_rgm, vtx.catalysis_rgm)

    # 2) Ray / adjugate-like H for the reduced steady-state system
    _, H_ray = _calc_singular_H_ss(vtx.bind_rgm, vtx.catalysis_rgm, perm, rowctx)
    H_ssk = _expand_Hss_to_qssKk(H_ray, vtx.catalysis_rgm.P)

    # 3) Steady-state consistency on (q_ss, K, k) by elimination
    C_qKk_ss, C0_qKk_ss, _ = _calc_C_qKk_ss_singular(vtx.bind_rgm, vtx.catalysis_rgm)

    vtx.H = H_ssk
    vtx.H0 = nothing
    vtx.C_qKk_cat = C_qKk_cat
    vtx.C0_qKk_cat = C0_qKk_cat
    vtx.nlt_qKk_cat = nlt_qKk_cat
    vtx.C_qKk_ss = C_qKk_ss
    vtx.C0_qKk_ss = C0_qKk_ss

    return nothing
end




# ------------------------------------------------
# Initialize all mixed regimes
# ------------------------------------------------

"""
    _initialize_regime!(rgms)

Initialize all candidate mixed regimes.
High-level flow:
1) build row-level shared data for each catalysis regime row;
2) for each valid mixed regime in that row:
   - compute its reduced steady-state nullity;
   - dispatch to the regular or singular initializer.
"""
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

        # NOTE:
        # This threading assumes the helpers called below are thread-safe and that
        # rowctx.cache is read-only in _calc_H / _calc_nullity. If not, remove the
        # Threads.@threads here.
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

# ------------------------------------------------
# Top-level entry
# ------------------------------------------------

function match_regimes!(model::Bnc)
    find_all_regimes!(model)
    find_catalysis_regimes!(model)

    model.BncRegimes = _build_BncRegime(
        model.catalysis.CatalysisRegimes,
        model.BindRegimes,
    )
    _initialize_regime!(model.BncRegimes)

    return nothing
end
