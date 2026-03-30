# #--------------Matrix inverse helpers-------------------------

const _EMPTY_SPM64 = spzeros(Float64, 0, 0)
const _EMPTY_VEC64 = Float64[]
const NρKey = Tuple{Vararg{Int}}
const NρCache = Dict{NρKey,NρCacheEntry}

@inline _entry_only_def(def::Int) = NρCacheEntry(def, 0x00, _EMPTY_SPM64, 0.0, _EMPTY_VEC64, _EMPTY_VEC64)
@inline _entry_inv(inv::SparseMatrixCSC{Float64,Int}) = NρCacheEntry(0, 0x01, inv, 0.0, _EMPTY_VEC64, _EMPTY_VEC64)
@inline _entry_rank1(def::Int, α::Float64, u::Vector{Float64}, v::Vector{Float64}) =
    NρCacheEntry(def, 0x02, _EMPTY_SPM64, α, u, v)

# -----------------------------------------------------------------------------
# perm -> key / perm nullity
# -----------------------------------------------------------------------------

@inline function _key_from_perm!(
    keybuf::Vector{Int},
    seen::Vector{UInt8},
    touched::Vector{Int},
    perm::AbstractVector{<:Integer},
    n::Int,
)
    ntouched = 0
    nunique  = 0

    @inbounds for p0 in perm
        p = Int(p0)
        if seen[p] == 0x00
            seen[p] = 0x01
            ntouched += 1
            touched[ntouched] = p
            nunique += 1
        end
    end

    k = 0
    @inbounds for j in 1:n
        if seen[j] == 0x00
            k += 1
            keybuf[k] = j
        end
    end

    @inbounds for t in 1:ntouched
        seen[touched[t]] = 0x00
    end

    return k, (length(perm) - nunique)
end

@inline function _tuple_from_prefix(buf::Vector{Int}, k::Int)::NρKey
    return ntuple(i -> @inbounds(buf[i]), k)
end

function _get_Nρ_key(perm::AbstractVector{<:Integer}, n::Int)::Vector{Int}
    seen    = zeros(UInt8, n)
    touched = Vector{Int}(undef, length(perm))
    keybuf  = Vector{Int}(undef, n)
    k, _    = _key_from_perm!(keybuf, seen, touched, perm, n)
    return copy(@view keybuf[1:k])
end

function _get_Nρ_key_and_perm_nullity(perm::AbstractVector{<:Integer}, n::Int)
    seen    = zeros(UInt8, n)
    touched = Vector{Int}(undef, length(perm))
    keybuf  = Vector{Int}(undef, n)
    k, pdef = _key_from_perm!(keybuf, seen, touched, perm, n)
    return copy(@view keybuf[1:k]), pdef
end

# @inline function _calc_perm_nullity(perm::AbstractVector{<:Integer}, n::Int)::Int
#     _, pdef = _get_Nρ_key_and_perm_nullity(perm, n)
#     return pdef
# end

# -----------------------------------------------------------------------------
# permutation sign for exact adj(A) when A is singular and A * Π = M
# -----------------------------------------------------------------------------

function _perm_sign(p::AbstractVector{<:Integer})::Float64
    n = length(p)
    visited = falses(n)
    s = 1.0

    @inbounds for i in 1:n
        if !visited[i]
            j = i
            clen = 0
            while !visited[j]
                visited[j] = true
                j = Int(p[j])
                clen += 1
            end
            if isodd(clen - 1)
                s = -s
            end
        end
    end
    return s
end

# -----------------------------------------------------------------------------
# Nρ analysis / factorization cache
# -----------------------------------------------------------------------------

function _rank1_adjugate_data!(A::Matrix{Float64}; atol::Float64=1e-12, rtol::Float64=1e-10)
    F = svd!(A)
    S = F.S

    σmax = isempty(S) ? 0.0 : maximum(S)
    tol  = max(atol, rtol * σmax)
    rk   = count(σ -> σ > tol, S)
    def  = size(A, 1) - rk

    if size(A, 1) == size(A, 2) && def == 1
        k = findfirst(σ -> σ <= tol, S)
        @assert k !== nothing

        logσprod = 0.0
        @inbounds for i in eachindex(S)
            if i != k
                logσprod += log(S[i])
            end
        end

        # For square SVD, adj(A) = det(U) * det(V) * prod(nonzero singular values) * v * u'
        # Since det(Vt) == det(V), we can use det(Vt) directly.
        α = exp(logσprod) * (det(F.U) * det(F.Vt))
        u = copy(@view F.U[:, k])
        v = copy(@view F.Vt[k, :])  # kth row of Vt == v_k'
        return def, α, u, v
    else
        return def, 0.0, _EMPTY_VEC64, _EMPTY_VEC64
    end
end

function _factor_Nρ(
    Nρ::SparseMatrixCSC{Tv,Int};
    atol::Float64=1e-12,
    rtol::Float64=1e-10,
    drop_tol::Float64=1e-12,
) where {Tv<:Real}
    r, c = size(Nρ)

    # Square case: try sparse LU first. If it succeeds, cache the explicit inverse.
    if r == c
        F = lu(Nρ; check=false)
        if issuccess(F)
            X = F \ Matrix{Float64}(I, r, r)
            Xsp = sparse(X)
            drop_tol > 0 && droptol!(Xsp, drop_tol)
            return _entry_inv(Xsp)
        end

        # Singular square case: dense SVD only on the failure path.
        A = Matrix{Float64}(Nρ)
        def, α, u, v = _rank1_adjugate_data!(A; atol=atol, rtol=rtol)
        if def == 1
            return _entry_rank1(def, α, u, v)
        else
            return _entry_only_def(def)
        end
    end

    # Rectangular case: only the deficiency matters for nullity prefiltering.
    # We deliberately do NOT cache any inverse-like object here.
    A   = Matrix{Float64}(Nρ)
    rk  = rank(A; atol=atol, rtol=rtol)
    def = r - rk
    return _entry_only_def(def)
end

function _get_Nρ_entry!(
    cache::NρCache,
    N::AbstractMatrix{Tv},
    key::AbstractVector{<:Integer};
    atol::Float64=1e-12,
    rtol::Float64=1e-10,
    drop_tol::Float64=1e-12,
) where {Tv<:Real}
    tkey = Tuple(Int.(key))
    return get!(cache, tkey) do
        Nρ = sparse(N[:, collect(key)])
        _factor_Nρ(Nρ; atol=atol, rtol=rtol, drop_tol=drop_tol)
    end
end

@inline function _get_Nρ_entry_from_perm!(
    cache::NρCache,
    N::AbstractMatrix{Tv},
    perm;
    kwargs...,
) where {Tv<:Real}
    key = _get_Nρ_key(perm, size(N, 2))
    return _get_Nρ_entry!(cache, N, key; kwargs...), key
end

function _build_Nρ_cache_parallel!(
    cache::NρCache,
    N::AbstractMatrix{Tv},
    perms::Vector{<:AbstractVector{<:Integer}};
    atol::Float64=1e-12,
    rtol::Float64=1e-10,
    drop_tol::Float64=1e-12,
) where {Tv<:Real}
    nperm = length(perms)
    n = size(N, 2)
    d = n - size(N, 1)

    perm_keys = Vector{NρKey}(undef, nperm)
    perm_defs = Vector{Int}(undef, nperm)

    uniq_index = Dict{NρKey,Int}()
    keys = Vector{Vector{Int}}()

    seen    = zeros(UInt8, n)
    touched = Vector{Int}(undef, max(d, 1))
    keybuf  = Vector{Int}(undef, n)

    sizehint!(uniq_index, nperm)
    sizehint!(keys, nperm)

    # Single-thread pass: create the canonical key for each perm and deduplicate.
    for i in eachindex(perms)
        perm = perms[i]
        k, pdef = _key_from_perm!(keybuf, seen, touched, perm, n)
        tkey = _tuple_from_prefix(keybuf, k)
        perm_keys[i] = tkey
        perm_defs[i] = pdef

        if !haskey(uniq_index, tkey)
            uniq_index[tkey] = length(keys) + 1
            push!(keys, copy(@view keybuf[1:k]))
        end
    end

    # Parallel factorization of all unique Nρ blocks.
    entries = Vector{NρCacheEntry}(undef, length(keys))
    Threads.@threads for i in eachindex(keys)
        Nρ = sparse(N[:, keys[i]])
        entries[i] = _factor_Nρ(Nρ; atol=atol, rtol=rtol, drop_tol=drop_tol)
    end

    empty!(cache)
    sizehint!(cache, length(keys))
    for i in eachindex(keys)
        cache[Tuple(keys[i])] = entries[i]
    end

    return perm_keys, perm_defs
end

# -----------------------------------------------------------------------------
# Batch nullity interface
# -----------------------------------------------------------------------------

function _calc_nullity(
    perms::Vector{<:AbstractVector{<:Integer}},
    N::AbstractMatrix{Tv},
    atol::Float64=1e-12,
    rtol::Float64=1e-10,
    drop_tol::Float64=1e-12,
) where {Tv<:Real}
    cache = NρCache()
    perm_keys, perm_defs = _build_Nρ_cache_parallel!(
        cache,
        N,
        perms;
        atol=atol,
        rtol=rtol,
        drop_tol=drop_tol,
    )

    nullity = Vector{Int}(undef, length(perms))
    Threads.@threads for i in eachindex(perms)
        nullity[i] = perm_defs[i] + cache[perm_keys[i]].deficiency
    end

    return nullity,cache
end


function _calc_nullity(
    perms::Vector{<:AbstractVector{<:Integer}},
    model::Bnc;
    kwargs...,
)
    nullity,cache = _calc_nullity(
        perms,
        model.N;
        kwargs...,
    )
    model._vertices_Nρ_inv_dict = cache
    return nullity
end

# -----------------------------------------------------------------------------
# Sparse H assembly for the nonsingular case
# -----------------------------------------------------------------------------

function _append_block_triplets!(
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{Float64},
    p::Int,
    rowmap::Vector{Int},
    coloffset::Int,
    A::SparseMatrixCSC{Float64,Int},
)
    @inbounds for col in 1:size(A, 2)
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            p += 1
            I[p] = rowmap[A.rowval[ptr]]
            J[p] = coloffset + col - 1
            V[p] = A.nzval[ptr]
        end
    end
    return p
end

function _assemble_H_from_blocks(
    perm::AbstractVector{<:Integer},
    key::Vector{Int},
    BL::SparseMatrixCSC{Float64,Int},
    BR::SparseMatrixCSC{Float64,Int},
    n::Int,
)
    d = length(perm)
    nnzH = d + nnz(BL) + nnz(BR)

    I = Vector{Int}(undef, nnzH)
    J = Vector{Int}(undef, nnzH)
    V = Vector{Float64}(undef, nnzH)

    p = 0
    @inbounds for i in 1:d
        p += 1
        I[p] = Int(perm[i])
        J[p] = i
        V[p] = 1.0
    end

    p = _append_block_triplets!(I, J, V, p, key, 1,     BL)
    p = _append_block_triplets!(I, J, V, p, key, d + 1, BR)

    @assert p == nnzH
    return sparse(I, J, V, n, n)
end

# -----------------------------------------------------------------------------
# Singular case: apply adj([P;N]) to a known direction without materializing H
# -----------------------------------------------------------------------------

function _apply_rank1_adjugate_to_direction(
    perm::AbstractVector{<:Integer},
    key::Vector{Int},
    Nc::AbstractMatrix{Tv},
    α::Float64,
    u::Vector{Float64},
    v::Vector{Float64},
    direction::AbstractVector{<:Real};
    exact_adj_sign::Bool=true,
) where {Tv<:Real}
    d = length(perm)
    r = length(key)
    n = d + r

    @assert length(direction) == n

    rhsP = @view direction[1:d]
    rhsN = @view direction[(d + 1):n]

    # adj(M) = [0; α v] * [ -u' * Nc   u' ]
    # Therefore adj(M) * direction = [0; α v] * (u'*(rhsN - Nc*rhsP)).
    # Use Nc' * u to avoid forming Nc*rhsP when d is the smaller side.
    tmp = dot(u, rhsN) - dot(rhsP, Nc' * u)

    s = 1.0
    if exact_adj_sign
        # A * Π = M, so adj(A) = det(Π) * Π * adj(M).
        s = _perm_sign([Int.(perm); key])
    end

    out = zeros(Float64, length(direction))
    scale = s * α * tmp
    @inbounds for j in 1:r
        out[key[j]] = scale * v[j]
    end
    return out
end

function _materialize_rank1_adjugate(
    perm::AbstractVector{<:Integer},
    key::Vector{Int},
    Nc::AbstractMatrix{Tv},
    α::Float64,
    u::Vector{Float64},
    v::Vector{Float64};
    scale::Real=1.0,
) where {Tv<:Real}
    d = length(perm)
    r = length(key)
    n = d + r

    # right = [ -Nc' * u ; u ]
    right = Vector{Float64}(undef, n)
    tmp = Nc' * u

    @inbounds for i in 1:d
        right[i] = -tmp[i]
    end
    @inbounds for j in 1:r
        right[d + j] = u[j]
    end

    s = _perm_sign([Int.(perm); key])  # 如果你要符号
    σ = Float64(scale) * s

    AdjA = zeros(Float64, n, n)

    @inbounds for j in 1:r
        row = key[j]
        coeff = σ * α * v[j]
        for k in 1:n
            AdjA[row, k] = coeff * right[k]
        end
    end

    return sparse(AdjA)
end


# -----------------------------------------------------------------------------
# Rank-k / rank-1 affine update helpers
# -----------------------------------------------------------------------------

function _sparse_outer(
    c::SparseVector{Float64,Int},
    s::SparseVector{Float64,Int},
    scale::Float64,
)
    nrow = length(c)
    ncol = length(s)
    Ic, Vc = findnz(c)
    Js, Vs = findnz(s)

    if isempty(Ic) || isempty(Js)
        return spzeros(Float64, nrow, ncol)
    end

    nnzA = length(Ic) * length(Js)
    I = Vector{Int}(undef, nnzA)
    J = Vector{Int}(undef, nnzA)
    V = Vector{Float64}(undef, nnzA)

    p = 0
    @inbounds for a in eachindex(Ic)
        ia = Ic[a]
        va = scale * Vc[a]
        for b in eachindex(Js)
            p += 1
            I[p] = ia
            J[p] = Js[b]
            V[p] = va * Vs[b]
        end
    end

    return sparse(I, J, V, nrow, ncol)
end

# """
#     _rank1_update_H_H0(H, H0, i, j_from, j_to, δ0; atol=1e-12, drop_tol=1e-10)

# Apply the rank-1 affine update

#     M'  = M  + e_i (e_{j_to} - e_{j_from})'
#     M0' = M0 + δ0 e_i

# to an existing affine inverse

#     log x = H log y + H0.

# Returns `(H′, H0′, δ)`. If `δ ≈ 0`, `H′` and `H0′` are returned as `nothing`.
# """
# function _rank1_update_H_H0(
#     H::SparseMatrixCSC{Float64,Int},
#     H0::AbstractVector{<:Real},
#     i::Int,
#     j_from::Int,
#     j_to::Int,
#     δ0::Real;
#     atol::Float64=1e-12,
#     drop_tol::Float64=1e-10,
# )
#     c = H[:, i]
#     s = H[j_to, :] - H[j_from, :]
#     δ = 1.0 + H[j_to, i] - H[j_from, i]

#     if !isfinite(δ) || abs(δ) <= atol
#         return nothing, nothing, δ
#     end

#     update = _sparse_outer(c, s, 1 / δ, size(H, 1), size(H, 2))
#     H_new = sparse(H - update)
#     drop_tol > 0 && droptol!(H_new, drop_tol)

#     shift = (Float64(H0[j_to]) - Float64(H0[j_from]) + Float64(δ0)) / δ
#     H0_new = Float64.(copy(H0))
#     Ic, Vc = findnz(c)
#     @inbounds for t in eachindex(Ic)
#         H0_new[Ic[t]] -= Vc[t] * shift
#     end

#     return H_new, H0_new, δ
# end
function droptol!(A::AbstractArray, tol)
    @inbounds for i in eachindex(A)
        if abs(A[i]) < tol
            A[i] = zero(eltype(A))
        end
    end
    return A
end

function droptol!(A::Real, tol)
    return A = abs(A) < tol ? zero(eltype(A)) : A
end

function _rank1_step_update_from_regular(
    H::SparseMatrixCSC{Float64,Int},
    H0::AbstractVector{<:Real},
    
    i::Int, 
    c_c0::Hyperplane_perm,

    sign::Int8,

    atol::Float64=1e-12,
    drop_tol::Float64=1e-10,
)
    c_qK = c_c0 * H .* sign  
    c0_qK = c_c0 * H0 * sign

    drop_tol > 0 && droptol!(c_qK, drop_tol) 

    H_i = H[:, i]
    a = c_qK[i]

    if abs(1 + a) <= atol
        H_to = _sparse_outer(H_i, c_qK, -1.0)
        H0_to = H_i * c0_qK
        nlt_to = 1
    else
        H_to = H - _sparse_outer(H_i, c_qK, 1 / (1 + a))
        H0_to = H0 .- H_i ./(1 + a) * c0_qK
        nlt_to = 0
    end 

    if drop_tol > 0
        droptol!(H_to, drop_tol)
        droptol!(H0_to, drop_tol)
    end

    return H_to, H0_to, nlt_to, c_qK, c0_qK
end








# """
#     _rank1_step_H_H0_from_regular(H, H0, M0_to, i, j_from, j_to, δ0; kwargs...)

# Transition across one x-neighbor edge when the source regime is regular.

# Returns `(H_to, H0_to, nullity_to, δ)`:

# - if `δ ≠ 0`, the target regime is regular and Sherman-Morrison is used
# - if `δ = 0`, the target regime is singular with nullity exactly `1`, and a
#   rank-1 adjugate-like ray is materialized directly from the edge data
# """
# function _rank1_step_H_H0_from_regular(
#     H::SparseMatrixCSC{Float64,Int},
#     H0::AbstractVector{<:Real},
#     M0_to::AbstractVector{<:Real},
#     i::Int,
#     j_from::Int,
#     j_to::Int,
#     δ0::Real;
#     atol::Float64=1e-12,
#     drop_tol::Float64=1e-10,
# )
#     H_to, H0_to, δ = _rank1_update_H_H0(
#         H,
#         H0,
#         i,
#         j_from,
#         j_to,
#         δ0;
#         atol=atol,
#         drop_tol=drop_tol,
#     )
#     if !isnothing(H_to)
#         return H_to, H0_to, 0, δ
#     end

#     c = H[:, i]
#     s = H[j_to, :] - H[j_from, :]
#     H_sing = _sparse_outer(c, s, -1.0, size(H, 1), size(H, 2))
#     drop_tol > 0 && droptol!(H_sing, drop_tol)
#     H0_sing = vec(-(H_sing * Float64.(M0_to)))

#     return H_sing, H0_sing, 1, δ
# end

"""
    _lowrank_update_H_H0(H, H0, U, V, δ0; kwargs...)

Woodbury-style affine update for

    M'  = M  + U V'
    M0' = M0 + U δ0

with

    H'  = H - H U (I + V' H U)^(-1) V' H
    H0' = H0 - H U (I + V' H U)^(-1) (V' H0 + δ0).

This helper is mainly here to centralize the formula used by the rank-1 edge
update and future row-replacement updates.
"""
function _lowrank_update_H_H0(
    H::SparseMatrixCSC{Float64,Int},
    H0::AbstractVector{<:Real},
    U::SparseMatrixCSC{Float64,Int},
    V::SparseMatrixCSC{Float64,Int},
    δ0::AbstractVector{<:Real};
    atol::Float64=1e-12,
)
    HU = Matrix(H * U)
    VtH = Matrix(transpose(V) * H)
    K = Matrix{Float64}(I, size(U, 2), size(U, 2)) + Matrix(transpose(V) * sparse(HU))
    abs(det(K)) <= atol && return nothing, nothing, K

    KVtH = K \ VtH
    H_new = sparse(Matrix(H) - HU * KVtH)

    rhs0 = Vector{Float64}(transpose(V) * Float64.(H0)) + Float64.(δ0)
    H0_new = Float64.(H0) - HU * (K \ rhs0)

    return H_new, vec(H0_new), K
end


# -----------------------------------------------------------------------------
# Main H interface
# -----------------------------------------------------------------------------

function _calc_H(
    N::AbstractMatrix{Tv},
    cache::NρCache,
    perm::AbstractVector{<:Integer};

    scale::Real=1.0,
    atol::Float64=1e-12,
    rtol::Float64=1e-10,
    drop_tol::Float64=1e-12,
    kwargs...
) where {Tv<:Real}
    n = size(N, 2)
    key, perm_def = _get_Nρ_key_and_perm_nullity(perm, n)
    perm_def == 0 || error("_calc_H only supports perms with unique entries; use _calc_nullity first to prefilter invalid perms.")

    entry = _get_Nρ_entry!(cache, N, key; atol=atol, rtol=rtol, drop_tol=drop_tol)

    if entry.deficiency == 0
        entry.kind == 0x01 || error("Internal cache inconsistency: expected explicit inverse for deficiency == 0.")

        BR = entry.inv
        Nc = sparse(N[:, Int.(perm)])
        BL = -(BR * Nc)
        drop_tol > 0 && droptol!(BL, drop_tol)
        return _assemble_H_from_blocks(perm, key, BL, BR, n)
    end

    if entry.deficiency == 1
        entry.kind == 0x02 || error("Internal cache inconsistency: expected rank-1 adjugate factors for deficiency == 1.")

        Nc = sparse(N[:, Int.(perm)])

        return _materialize_rank1_adjugate(
                perm,
                key,
                Nc,
                entry.α,
                entry.u,
                entry.v;
                scale=scale,
                kwargs...,
            )
    end

    error("nullity([P;N]) >= 2 is not supported by _calc_H; call _calc_nullity first and skip those perms.")
end

function _calc_H(
    model::Bnc,
    perm::AbstractVector{<:Integer};
    kwargs...
)
    if isnothing(model._vertices_Nρ_inv_dict)
        error("Nρ cache is not initialized. Call _calc_nullity first to populate the cache before calling _calc_H.")
    end

    return _calc_H(
        model.N,
        model._vertices_Nρ_inv_dict,
        perm;
        scale= model.direction,
        kwargs...
    )
end


# helper funtions to taking inverse when the matrix is singular.
"""
    _adj_singular_matrix(A::AbstractMatrix; atol=1e-12) -> (SparseMatrixCSC, Int)

Compute a sparse adjugate-like matrix for a near-singular square matrix using
its smallest singular vector, and return the inferred nullity.

# Arguments
- A: Square matrix to analyze.

# Keyword Arguments
- atol: Absolute tolerance for identifying zero singular values.

# Returns
- Tuple (adj_A, nullity).
"""
function _adj_singular_matrix(A::AbstractMatrix; atol=1e-12)::Tuple{SparseMatrixCSC,Int}
    n, m = size(A)
    @assert n == m "A must be square"
    F = svd(Array(A))
    S = F.S
    thresh = atol * maximum(S)
    zero_ids = findall(σ -> σ ≤ thresh, S)
    nullity = length(zero_ids)
    if nullity == 1
        k = zero_ids[1]
        logσprod = sum(log, S[setdiff(1:n, [k])])
        σprod = exp(logσprod)
        sign_correction = det(F.U) * det(F.V) # to ensure the sign is right!!!!!!
        u = F.U[:, k]   # 左奇异向量
        v = F.V[:, k]   # 右奇异向量
        adj_A = (sign_correction * σprod) * (sparsevec(v) * sparsevec(u)')
        return adj_A, 1  # rank-1 矩阵
    else
        return spzeros(0, 0), nullity
    end
end


function direct_inverse_or_adjugate(A::AbstractMatrix; atol::Float64=1e-12)::Tuple{SparseMatrixCSC,Int}
    n, m = size(A)
    @assert n == m "A must be square"
    F = lu(sparse(A); check=false)

    if issuccess(F)
        H = luFac(F) \ spdiagm(0=>ones(Float64,n))
        return  H, 0
    else
        adj_A, nullity = _adj_singular_matrix(A; atol=atol)
        return adj_A, nullity
    end
end
