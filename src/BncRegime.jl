export match_regimes!, get_bnc_regime, get_bnc_regimes
export get_binding_regime, get_binding_perm, get_catalysis_perm, get_steady_state_perm
export get_C_C0_xk, get_C0_xk, get_C_xk
export get_C_C0_qKk, get_C0_qKk, get_C_qKk, get_C_C0_nullity_qKk
export get_C_C0_wKk, get_C0_wKk, get_C_wKk, get_C_C0_nullity_wKk
export get_H_bd, get_H_bd_numerically, get_qcat_F_F0
export judge_stability!, is_stable



function Base.getproperty(model::BncRegime, sym::Symbol)
    if sym === :perm
        return get_bind_perm(model), get_catalysis_perm(model)
    end
    return getfield(model, sym)
end


@inline _spI(T, n) = spdiagm(0 => ones(T, n))
@inline _zeros_like(A::AbstractMatrix{T}, m::Int, n::Int) where {T<:Real} = spzeros(T, m, n)
@inline _zeros_like(v::AbstractVector{T}, n::Int) where {T<:Real} = zeros(T, n)
@inline function _det_sign_exact(A::AbstractMatrix{<:Integer})
    detA = _bareiss_det_big(Matrix{Int}(A))
    return detA > 0 ? 1 : detA < 0 ? -1 : 0
end




get_fixed_point_perm(args...;kwargs...) = let
    bindperm, catalysisperm = get_bnc_perm(args...;kwargs...)
    r_v = get_catalysis_network(args...;kwargs...).r_v
    return bindperm[r_v+1:end], catalysisperm
end
is_fixed_point_singular(args...;kwargs...)=is_bnc_singular(args...;kwargs...)




get_H_bd(rgm::BncRegime) = rgm.H_bd


# Get an H_bd numerically if the inner binding regime is singular.
function get_H_bd_numerically(rgm::BncRegime)
    bind_rgm = get_binding_regime(rgm)
    cat_rgm = get_catalysis_regime(rgm)

    PΠ = get_PΠ(cat_rgm)
    H_bind = get_H_numerically(bind_rgm)
    r_v = size(PΠ, 1)
    return sparse(Float64.(PΠ * H_bind[:, 1:r_v]))
end



# Determine the stability of a mixed regime
function judge_stability!(rgm::BncRegime; kwargs...)

    if is_singular(get_binding_regime(rgm))
        rgm.H_bd = get_H_bd_numerically(rgm)
        H_bd = rgm.H_bd
    elseif isnothing(rgm.H_bd)
        rgm.H_bd = get_H_bd_numerically(rgm)
        H_bd = rgm.H_bd
    else
        H_bd = rgm.H_bd
    end

    code = judge_dstable(H_bd; kwargs...)

    flag = if code ==1  # d-stable
            Int8(1)
        elseif code == 0 # d-unstable
            Int8(-1)
        else # undetermined
            Int8(2) 
        end

    rgm.is_stable = flag

    return rgm.is_stable
end



function is_stable(rgm::BncRegime; recalculate::Bool=false, kwargs...)
    
    flag = if (recalculate || rgm.is_stable == 0) 
                judge_stability!(rgm; kwargs...)
           else 
            rgm.is_stable
           end

    return flag == 1 ? true : flag == -1 ? false : missing
end

is_stable(model::Bnc, bind, cat; kwargs...) = is_stable(get_bnc_regime(model, bind, cat; check=true); kwargs...)


function get_qcat_F_F0(rgm::BncRegime)
    rgm.nlt == 0 || error("The reduced steady-state system is singular, so q_cat has no affine expression in (w, K, k).")
    r_v = size(rgm.catalysis_rgm.P, 1)
    P_cat = rgm.bind_rgm.P[1:r_v, :]
    P0_cat = rgm.bind_rgm.P0[1:r_v]
    F = P_cat * rgm.H
    F0 = P0_cat + P_cat * rgm.H0
    return F, vec(F0)
end

function get_polyhedron(rgm::BncRegime)
    C, C0, nullity = get_C_C0_nullity_wKk(rgm)
    return _build_polyhedron_from_C_C0(C, C0, nullity)
end





# The following functions is required to digging into to help fix the problem
#========================================================================================#
    #  Functions for  Calcalating conditions.
#========================================================================================#
function _project_bnc_singular_condition(
    network::AbstractBnc,
    C::AbstractMatrix,
    C0::AbstractVector,
    n_eq::Integer,
    delset::BitSet,
)
    p = get_polyhedron(C, C0, n_eq)
    p2 = _poly_eliminate(p, delset)
    return get_C_C0_nullity(p2)
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
    unique_keys = Tuple{Vararg{Int}}[]
    first_pos = Int[]
    key_to_pos = Dict{Tuple{Vararg{Int}},Int}()

    for (k, perm) in enumerate(perms)
        key = Tuple(Int.(perm))
        if !haskey(key_to_pos, key)
            key_to_pos[key] = length(unique_keys) + 1
            push!(unique_keys, key)
            push!(first_pos, k)
        end
    end

    return unique_keys, first_pos
end

function _materialize_real_vector(v)
    vv = vec(v)
    T = isempty(vv) ? Int : foldl(promote_type, map(typeof, vv); init=Int)
    return T[convert(T, x) for x in vv]
end

function _expand_Hw_to_wKk(H_w, H0_w, Pθ, P0θ)
    r_v = size(Pθ, 1)
    split = size(H_w, 2) - r_v
    H_left = H_w[:, 1:split]
    H_right = H_w[:, split + 1:end]
    H_wKk = hcat(H_left, -(H_right * Pθ))
    H0_wKk = H0_w - H_right * P0θ
    return H_wKk, _materialize_real_vector(H0_wKk)
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

    return C, _materialize_real_vector(C0), 0
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

function _calc_C_wKk_regular(
    bind_rgm::BindRegime,
    cat_rgm::CatalysisRegime,
    H_wKk,
    H0_wKk,
)
    C_x_bind, C0_x_bind = get_C_C0_x(bind_rgm)
    CΠ = get_CΠ(cat_rgm)
    Cθ = get_C_k(cat_rgm)
    C0θ = get_C0(cat_rgm)

    n_v = size(Cθ, 2)
    C_bind = C_x_bind * H_wKk
    C0_bind = C0_x_bind + C_x_bind * H0_wKk

    C_cat = copy(CΠ * H_wKk)
    @views C_cat[:, end - n_v + 1:end] .+= Cθ
    C0_cat = CΠ * H0_wKk + C0θ

    return vcat(C_bind, C_cat), _materialize_real_vector(vcat(C0_bind, C0_cat))
end

function _calc_C_wKk_singular(bind_rgm::BindRegime, cat_rgm::CatalysisRegime)
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

    d_w = size(P_ss, 1)
    r = size(N, 1)
    n_v = size(Pθ, 2)
    n_x = size(P_ss, 2)
    r_cat = size(Pθ, 1)

    Eq_w = hcat(-_spI(Int, d_w), _zeros_like(P_ss, d_w, r + n_v), P_ss)
    Eq_K = hcat(_zeros_like(N, r, d_w), -_spI(Int, r), _zeros_like(N, r, n_v), N)
    Eq_cat = hcat(_zeros_like(PΠ, r_cat, d_w + r), Pθ, PΠ)
    In_bind = hcat(_zeros_like(C_x_bind, size(C_x_bind, 1), d_w + r + n_v), C_x_bind)
    In_cat = hcat(_zeros_like(CΠ, size(CΠ, 1), d_w + r), Cθ, CΠ)

    C = vcat(Eq_w, Eq_K, Eq_cat, In_bind, In_cat)
    C0 = vcat(P0_ss, zeros(eltype(P0_ss), r), P0θ, C0_x_bind, C0θ)

    return _project_bnc_singular_condition(
        bn,
        C,
        C0,
        d_w + r + r_cat,
        BitSet((d_w + r + n_v + 1):(d_w + r + n_v + n_x)),
    )
end




#========================================================================================#
  # Functions for initializing BncRegimes
#========================================================================================#
function _build_BncRegime(cat_rgms::Regimes, bind_rgms::Regimes)
    n_cat = n_catalysis_regimes(cat_rgms)
    n_bind = n_binding_regimes(bind_rgms)
    bncrgms = Matrix{BncRegime}(undef, n_cat, n_bind)

    @info "Matching Catalysis Regimes and Binding Regimes to build BncRegimes..."
    Threads.@threads for i in 1:n_cat
        cat_rgm = cat_rgms.vertices_data[i]
        for j in 1:n_bind
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
















#========================================================================================#
    #  Functions for displaying bnc regimes
#========================================================================================#

function summary_regime(rgm::BncRegime)
    rgm = get_regime(rgm)
    println("bind_idx=$(get_idx(rgm.bind_rgm)), cat_idx=$(get_idx(rgm.catalysis_rgm)), nlt=$(rgm.nlt), stable=$(is_stable(rgm))")
    println("Binding / catalysis conditions in (x, k):")
    display.(show_condition_xk(rgm; kind=:binding, log_space=false))
    display.(show_condition_xk(rgm; kind=:catalysis, log_space=false))
    println("Combined consistency in (w, K, k):")
    display.(show_condition_wKk(rgm; log_space=false))
    return nothing
end

summary(rgm::BncRegime) = summary_regime(rgm)
summary_regime(model::Bnc, bind, cat) = summary_regime(get_bnc_regime(model, bind, cat; check=true))
summary(model::Bnc, bind, cat) = summary_regime(model, bind, cat)

@inline function _is_asymptotic(rgm::BncRegime)
    return is_asymptotic(rgm.bind_rgm) && is_asymptotic(rgm.catalysis_rgm)
end

@inline function _regime_display_dominant_mode(rgm::BncRegime)
    return "bind=$(get_binding_perm(rgm)), cat=$(get_catalysis_perm(rgm)), ss=$(get_perm(rgm))"
end

function Base.show(io::IO, rgm::BncRegime)
    print(
        io,
        "BncRegime(",
        _regime_display_dominant_mode(rgm),
        ", nullity=",
        rgm.nlt,
        ", asymptotic=",
        _is_asymptotic(rgm),
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", rgm::BncRegime)
    println(io, "BncRegime")
    println(io, "  dominant mode: ", _regime_display_dominant_mode(rgm))
    println(io, "  nullity: ", rgm.nlt)
    print(io, "  asymptotic: ", _is_asymptotic(rgm))
end

function match_regimes!(model::Bnc)
    if is_bnc_regimes_built(model)
        return model.BncRegimes
    end

    find_all_regimes!(model)
    find_catalysis_regimes!(model)

    model.BncRegimes = _build_BncRegime(
        model.catalysis.CatalysisRegimes,
        model.BindRegimes,
    )
    _initialize_regime!(model.BncRegimes)

    return model.BncRegimes
end
