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
@inline _det_sign_exact(A::AbstractMatrix{<:Integer}) = begin
    detA = _bareiss_det_big(Matrix{Int}(A))
    detA > 0 ? 1 : detA < 0 ? -1 : 0
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
    # if rgm.nlt > 1
    #     return (rgm.is_stable = Int8(0))
    # end

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



function is_stable(rgm::BncRegime; recalculate::Bool=false, return_code::Bool=false, kwargs...)
    
    flag = if (recalculate || rgm.is_stable == 0) 
                judge_stability!(rgm; kwargs...)
           else 
            rgm.is_stable
           end

    return_code && return flag == 1 ? 1 : flag == -1 ? -1 : 0
    return flag == 1 ? true : flag == -1 ? false : missing
end

is_stable(model::Bnc, bind, cat; kwargs...) = is_stable(get_bnc_regime(model, bind, cat; check=true); kwargs...)


get_qcat_F_F0(rgm::BncRegime) = get_affine_wKk2qcat(rgm)

function get_polyhedron(rgm::BncRegime)
    C, C0, nullity = get_C_C0_nullity_wKk(rgm)
    return _build_polyhedron_from_C_C0(C, C0, nullity)
end





# The following functions is required to digging into to help fix the problem
#========================================================================================#
    #  Functions for  Calcalating conditions.
#========================================================================================#




function _row_valid_columns(rgms::AbstractMatrix{<:Union{BncRegime,Nothing}}, i::Int)
    return [j for j in axes(rgms, 2) if !isnothing(rgms[i, j])]
end

function _materialize_real_vector(v)
    vv = vec(v)
    T = isempty(vv) ? Int : foldl(promote_type, map(typeof, vv); init=Int)
    return T[convert(T, x) for x in vv]
end

function _steady_state_affine(bind_rgm::BindRegime, perm, N_ss, r_v::Int, direction::Int, nlt::Int)
    nlt > 1 && return nothing

    _, P0_bind = get_affine_x2q(bind_rgm)
    P0_ss = P0_bind[r_v + 1:end]
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
    bn = get_binding_network(rgms[i, first(valid_js)]::BncRegime)
    bind_grh = get_regimes_graph!(bn; full=false)
    valid_pos = Dict(valid_js[k] => k for k in eachindex(valid_js))
    affine_by_perm = Dict{Tuple{Vararg{Int}},Tuple{Any,Any}}()
    affine_by_bind_idx = Dict{Int,Tuple{Any,Any}}()

    for seed_pos in eachindex(valid_js)
        seed_bind_idx = valid_js[seed_pos]
        haskey(affine_by_bind_idx, seed_bind_idx) && continue
        nlt_valid[seed_pos] == 0 || continue

        seed_rgm = rgms[i, seed_bind_idx]::BncRegime
        get_bind_regime(seed_rgm.bind_rgm; inv_info=false)
        seed_pair = _steady_state_affine(seed_rgm.bind_rgm, perms[seed_pos], N_ss, r_v, direction, 0)
        affine_by_bind_idx[seed_bind_idx] = seed_pair
        affine_by_perm[Tuple(Int.(perms[seed_pos]))] = seed_pair

        queue = [seed_bind_idx]
        while !isempty(queue)
            from_bind_idx = popfirst!(queue)
            from_pos = valid_pos[from_bind_idx]
            from_pair = affine_by_bind_idx[from_bind_idx]
            from_nlt = nlt_valid[from_pos]

            for edge in bind_grh.neighbors[from_bind_idx]
                to_bind_idx = edge.to
                haskey(valid_pos, to_bind_idx) || continue
                haskey(affine_by_bind_idx, to_bind_idx) && continue

                to_pos = valid_pos[to_bind_idx]
                to_nlt = nlt_valid[to_pos]
                to_nlt > 1 && continue

                pair = if edge.i <= r_v
                    from_pair
                elseif from_nlt == 0
                    x_idx, x_sign = _edge_idx_sign(edge, _EDGE_SPACE_X)
                    H_to, H0_to, _, _, _ = _rank1_step_update_from_regular(
                        from_pair[1],
                        from_pair[2],
                        edge.i - r_v,
                        bn._L_helper.hyperplanes[x_idx],
                        x_sign,
                    )
                    (H_to, H0_to)
                else
                    nothing
                end
                pair === nothing && continue

                rgm.H_inner = pair[1]
                rgm.H0_inner = pair[2]

                affine_by_bind_idx[to_bind_idx] = pair
                affine_by_perm[Tuple(Int.(perms[to_pos]))] = pair
                to_nlt == 0 && push!(queue, to_bind_idx)
            end
        end
    end

    for k in eachindex(valid_js)
        nlt = nlt_valid[k]
        nlt > 1 && continue
        key = Tuple(Int.(perms[k]))
        haskey(affine_by_perm, key) && continue

        rgm = rgms[i, valid_js[k]]::BncRegime
        get_bind_regime(rgm.bind_rgm; inv_info=false)
        affine_by_perm[key] = _steady_state_affine(rgm.bind_rgm, perms[k], N_ss, r_v, direction, nlt)
    end

    return affine_by_perm
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


function _init_regular_or_nullity1_bnc_regime!(vtx::BncRegime, perm, rowctx)
    C_qKk, C0_qKk, nlt_qKk = _calc_C_C0_qKk(vtx.bind_rgm, vtx.catalysis_rgm)
    C_wKk, C0_wKk, nlt_wKk = _calc_C_C0_wKk(vtx)
    @assert nlt_wKk == vtx.nlt
    H_inner, H0_inner = get_affine_wKk̃2x(vtx)
    H_wKk, H0_wKk = _expand_Hw_to_wKk(H_inner, H0_inner, get_N_N0_v(vtx.catalysis_rgm)...)
    vtx.H = H_wKk
    vtx.H0 = H0_wKk
    vtx.C_qKk_cat = C_qKk
    vtx.C0_qKk_cat = C0_qKk
    vtx.nlt_qKk_cat = nlt_qKk
    vtx.C_wKk = C_wKk
    vtx.C0_wKk = C0_wKk
    return nothing
end

function _init_consistency_only_bnc_regime!(vtx::BncRegime)
    C_qKk_cat, C0_qKk_cat, nlt_qKk_cat = _calc_C_C0_qKk(vtx.bind_rgm, vtx.catalysis_rgm)
    C_wKk, C0_wKk, _ = _calc_C_wKk_singular(vtx.bind_rgm, vtx.catalysis_rgm)

    vtx.H = nothing
    vtx.H0 = nothing
    vtx.C_qKk_cat = C_qKk_cat
    vtx.C0_qKk_cat = _materialize_real_vector(C0_qKk_cat)
    vtx.nlt_qKk_cat = nlt_qKk_cat
    vtx.C_wKk = C_wKk
    vtx.C0_wKk = _materialize_real_vector(C0_wKk)
    return nothing
end


function _calc_C_C0_qKk(bind_rgm::BindRegime, cat_rgm::CatalysisRegime)

    C_xk_bind, C0_xk_bind = get_C_C0_xk(bind_rgm)
    C_xk_cat, C0_xk_cat = get_C_C0_xk(cat_rgm)
    C= vcat(C_xk_bind, C_xk_cat)
    C0 = vcat(C0_xk_bind,C0_xk_cat)

    if is_singular(bind_rgm)
        M,M0 = get_affine_xk2qKk(bind_rgm)    
        C_qKk, C0_qKk, nlt = _affine_mapping_polyhedra(C,C0,0,M,M0)
        return C_qKk, C0_qKk, nlt
    else
        H,H0 = get_affine_qKk2xk(bind_rgm)
        C_qKk = C*H
        droptol!(C_qKk, 1e-10)
        C0_qKk = C0 + C*H0
        return C_qKk, C0_qKk, 0
    end
end

function _calc_C_C0_wKk(rgm::BncRegime)
    bind_rgm = get_bind_regime(rgm)
    cat_rgm = get_catalysis_regime(rgm)

    C_xk_bind, C0_xk_bind = get_C_C0_xk(bind_rgm)
    C_xk_cat, C0_xk_cat = get_C_C0_xk(cat_rgm)
    C= vcat(C_xk_bind, C_xk_cat)
    C0 = vcat(C0_xk_bind, C0_xk_cat)

    Z,Z0 = get_affine_wKk2wKk̃k(rgm) # not depends on initialization
    if is_singular(rgm)
        M,M0 = get_affine_xk2wKk̃k(rgm) # not depends on initialization
        C_wKkk, C0_wKkk, nlt = _affine_mapping_polyhedra(C, C0, 0, M, M0)
    else
        H,H0 = get_affine_wKk̃k2xk(rgm) # not depends on initialization
        C_wKkk = C*H
        C0_wKkk = C*H0 + C0
        nlt = 0
    end

    C_wKk = C_wKkk*Z
    droptol!(C_wKk, 1e-10)
    C0_wKk = C_wKkk*Z0 + C0_wKkk 
    
    return C_wKk, C0_wKk, nlt # change bases back to wKk
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



#=================================================THE main ENTRY=============================#

function match_regimes!(model::Bnc)
    if is_bnc_regimes_built(model)
        return model.BncRegimes
    end

    find_all_regimes!(model)
    find_catalysis_regimes!(model)

    model.BncRegimes = _build_BncRegime(
        _bind_regimes_data(model),
        _catalysis_regimes_data(model),
    )

    _initialize_regime!(model.BncRegimes) # The real calculation

    return model.BncRegimes
end

function _build_BncRegime(cat_rgms::Regimes, bind_rgms::Regimes)
    n_cat = n_catalysis_regimes(cat_rgms)
    n_bind = n_bind_regimes(bind_rgms)
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

function _initialize_regime!(rgms::AbstractMatrix{BncRegime})
    r_v = size(rgms[1,1].catalysis_rgm.P, 1)

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


function _build_row_context(rgms::AbstractMatrix{<:Union{BncRegime,Nothing}}, i::Int, r_v::Int)
    valid_js = _row_valid_columns(rgms, i)
    isempty(valid_js) && return nothing

    ref_vtx = rgms[i, first(valid_js)]::BncRegime
    bn = get_binding_network(ref_vtx)
    
    L_cat = get_Lcat(bn)
    N_ss = vcat(bn.N, get_PΠ(ref_vtx.catalysis_rgm)) # The shared N for 
    
    direction = _det_sign_exact(vcat(L_cat, N_ss))

    perms = [get_fixed_point_perm(rgms[i, j]::BncRegime)[1] for j in valid_js]
    nlt_valid, _ = _calc_nullity(perms, N_ss)
    affine_by_perm = _build_row_affine_cache(rgms, i, valid_js, perms, nlt_valid, N_ss, r_v, direction)

    return (
        valid_js=valid_js,
        perms=perms,
        nlt_valid=nlt_valid,
        affine_by_perm=affine_by_perm,
    )
end