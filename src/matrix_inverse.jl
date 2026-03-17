#--------------Matrix inverse helpers-------------------------

# helper funtions to taking inverse when the matrix is singular.
"""
    _adj_singular_matrix(A::AbstractMatrix; atol=1e-12) -> (SparseMatrixCSC, Int)

Compute a sparse adjugate-like matrix for a near-singular square matrix using
its smallest singular vector, and return the inferred nullity.

# Arguments
- `A`: Square matrix to analyze.

# Keyword Arguments
- `atol`: Absolute tolerance for identifying zero singular values.

# Returns
- Tuple `(adj_A, nullity)`.
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
        return droptol!(adj_A, 1e-10), 1  # rank-1 矩阵
        # return σprod * (v * u'), 1  # rank-1 矩阵
    else
        return spzeros(0, 0), nullity
    end
end

"""
    _calc_Nρ_inverse(Nρ) -> (SparseMatrixCSC, Int)

Compute a sparse inverse-like matrix for `Nρ` and its nullity.

# Arguments
- `Nρ`: Square or rectangular submatrix of `N`.

# Returns
- Tuple `(Nρ_inv, nullity)` where `Nρ_inv` is sparse and the
  inferred nullity.
"""
function _calc_Nρ_inverse(Nρ)::Tuple{SparseMatrixCSC,Int}
    r, r_ncol = size(Nρ)
    if r != r_ncol
        return spzeros(0, 0), r - rank(Nρ)
    end
    Nρ_lu = lu(Nρ; check=false)
    if issuccess(Nρ_lu)
        return sparse(inv(Array(Nρ))), 0
    else
        return _adj_singular_matrix(Nρ)
    end
end

"""
    _get_Nρ_key(bnc::Bnc, perm) -> Vector

Indices of columns not in `perm` (the complement) used to form `Nρ`.
"""
function _get_Nρ_key(Bnc::Bnc{T}, perm)::Vector{T} where T
    return [i for i in 1:Bnc.n if i ∉ perm]
end

"""
    _get_Nρ_inv!(bnc::Bnc, key) -> (SparseMatrixCSC, Int)

Get `(Nρ_inv, nullity)` from cache or compute and store it.
"""
function _get_Nρ_inv!(Bnc::Bnc{T}, key::AbstractVector{<:Integer}) where T
    get!(Bnc._vertices_Nρ_inv_dict, key) do
        Nρ = @view Bnc.N[:, key]
        _calc_Nρ_inverse(Nρ)
    end
end

_get_Nρ_inv_from_perm!(Bnc, perm) = _get_Nρ_inv!(Bnc, _get_Nρ_key(Bnc, perm))

"""
    _build_Nρ_cache_parallel!(bnc::Bnc, perms) -> nothing

Precompute and cache `Nρ` inverse information for all unique permutations.
"""
function _build_Nρ_cache_parallel!(Bnc::Bnc{T}, perms::Vector{Vector{T}}) where T
    perm_set = Set(Set(perm) for perm in perms) # Unique sets of permutations
    keys = [_get_Nρ_key(Bnc, perm) for perm in perm_set]

    nk = length(keys)
    inv_list = Vector{SparseMatrixCSC{Float64,Int}}(undef, nk)
    nullity_list = Vector{T}(undef, nk)

    @showprogress Threads.@threads for i in eachindex(keys)
        key = keys[i]
        Nρ = @view Bnc.N[:, key]
        inv_list[i], nullity_list[i] = _calc_Nρ_inverse(Nρ)
    end

    for i in eachindex(keys)
        Bnc._vertices_Nρ_inv_dict[keys[i]] = (inv_list[i], nullity_list[i])
    end
    return nothing
end

"""
    _calc_H(bnc::Bnc, perm) -> SparseMatrixCSC

Compute the `H` mapping for a vertex permutation using cached `Nρ_inv`.

# Arguments
- `bnc`: Binding network model.
- `perm`: Regime permutation vector.

# Returns
- Sparse matrix `H` mapping log(qK) to log(x).
"""
function _calc_H(Bnc::Bnc, perm::Vector{<:Integer})::SparseMatrixCSC
    key = _get_Nρ_key(Bnc, perm)
    Nρ_inv, Nρ_nullity = _get_Nρ_inv!(Bnc, key) # get Nρ_inv from cache or calculate it. # sparse matrix
    Nc = @view Bnc.N[:, perm] # dense matrix
    Nρ_inv_Nc_neg = -Nρ_inv * Nc

    H_un_perm = if Nρ_nullity == 0
        [[I(Bnc.d) zeros(Bnc.d, Bnc.r)];
         [Nρ_inv_Nc_neg Nρ_inv]]
    elseif Nρ_nullity == 1
        [zeros(Bnc.d, Bnc.n);
         [Nρ_inv_Nc_neg Nρ_inv]]
    else
        error("Nullity greater than 1 not supported")
    end
    perm_inv = invperm([perm; key]) # get the inverse permutation to reorder H
    H = H_un_perm[perm_inv, :]
    H = droptol!(sparse(H), 1e-10)
    return H
end
