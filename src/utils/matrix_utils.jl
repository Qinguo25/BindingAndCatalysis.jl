"""
    L_from_N(N::Matrix{Int}) -> Matrix{Int}

Compute a conservation matrix `L` from a stoichiometry matrix `N` such that
`N * L' = 0`.
"""
function L_from_N(N::Matrix{Int})::Matrix{Int}
    r, n = size(N)
    d = n - r
    N_1 = @view N[:, 1:d]
    N_2 = @view N[:, d+1:n]
    hcat(Matrix(I, d, d), -(N_2 \ N_1)')
end

"""
    N_from_L(L::Matrix{Int}) -> Matrix{Int}

Recover a stoichiometry matrix `N` from a conservation matrix `L`.
"""
function N_from_L(L::Matrix{Int})::Matrix{Int}
    d, n = size(L)
    r = n - d
    L2 = @view L[:, d+1:n]
    hcat(L2', Matrix(-I, r, r))
end

"""
    rowmask_indices(A::SparseMatrixCSC, start_row::Int, end_row::Int)

Return the row indices, column indices, and nzval positions for nonzeros in a
row range of a sparse matrix.
"""
function rowmask_indices(A::SparseMatrixCSC, start_row::Int, end_row::Int)
    rows = Int[]
    cols = Int[]
    idxs = Int[]

    for j in 1:size(A, 2)
        for k in A.colptr[j]:(A.colptr[j + 1] - 1)
            i = A.rowval[k]
            if i >= start_row && i <= end_row
                push!(rows, i)
                push!(cols, j)
                push!(idxs, k)
            end
        end
    end
    return rows, cols, idxs
end

"""
    diag_indices(A::SparseMatrixCSC, end_row::Int) -> Vector{Int}

Return indices into `A.nzval` for diagonal entries up to `end_row`.
"""
function diag_indices(A::SparseMatrixCSC, end_row::Int)
    idxs = Int[]
    for j in 1:size(A, 2)
        for k in A.colptr[j]:(A.colptr[j + 1] - 1)
            i = A.rowval[k]
            if i == j && i <= end_row
                push!(idxs, k)
            end
        end
    end
    return idxs
end

"""
    rebase_mat_lgK(N::AbstractMatrix) -> AbstractMatrix

Return a rebase matrix for `logK`.
"""
function rebase_mat_lgK(N::AbstractMatrix)
    N2 = N_from_L(L_from_N(N))
    Q_inv = Rational.(round.(Int, N2 / N))
    return sparse(inv(Q_inv))
end

"""
    independent_row_idx(N::AbstractMatrix)

Return indices of linearly independent rows in `N`.
"""
function independent_row_idx(N::AbstractMatrix{T}) where T
    Nt_lu = lu(N', check=false)
    issuccess(Nt_lu) && return collect(1:size(N, 1))
    tol = 1e-8
    pivot_indices = findall(abs.(diag(Nt_lu.U)) .> tol)
    return pivot_indices
end

"""
    get_int_type(n) -> Type

Select the smallest signed integer type that can represent `n + 1`.
"""
function get_int_type(n)
    m = n + 1
    if m <= typemax(Int8)
        return Int8
    elseif m <= typemax(Int16)
        return Int16
    elseif m <= typemax(Int32)
        return Int32
    elseif m <= typemax(Int64)
        return Int64
    else
        return Int128
    end
end

"""
    _Mtx2idx_val(Mtx::Matrix) -> (Vector{Int}, Vector)

Convert a single-nonzero-per-row matrix into index and value vectors.
"""
function _Mtx2idx_val(Mtx::Matrix{<:T}) where T
    row_num, col_num = size(Mtx)
    idx = Vector{Int}(undef, row_num)
    val = Vector{T}(undef, row_num)
    for i in 1:row_num
        for j in 1:col_num
            if Mtx[i, j] != 0
                idx[i] = j
                val[i] = Mtx[i, j]
                break
            end
        end
    end
    return idx, val
end

"""
    _idx_val2Mtx(idx::Vector{Int}, val=1; col_num=nothing) -> Matrix

Create a matrix with one nonzero per row from index and value vectors.
"""
function _idx_val2Mtx(idx::Vector{Int}, val::T=1, col_num::Union{Int,Nothing}=nothing) where T
    n = length(idx)
    col_num = isnothing(col_num) ? n : col_num
    Mtx = zeros(T, n, col_num)
    for i in 1:n
        if idx[i] != 0
            Mtx[i, idx[i]] = val
        end
    end
    return Mtx
end

"""
    _idx_val2Mtx(idx::Vector{Int}, val::Vector; col_num=nothing) -> Matrix

Create a matrix with one nonzero per row using per-row values.
"""
function _idx_val2Mtx(idx::Vector{Int}, val::Vector{<:T}, col_num::Union{Int,Nothing}=nothing) where T
    n = length(idx)
    col_num = isnothing(col_num) ? n : col_num
    @assert length(val) == n "val must have the same length as idx"
    Mtx = zeros(T, n, col_num)
    for i in 1:n
        if idx[i] != 0
            Mtx[i, idx[i]] = val[i]
        end
    end
    return Mtx
end

"""
    matrix_iter(f, M; byrow=true, multithread=true) -> Matrix

Apply a function to each row or column of a matrix and collect results.
"""
function matrix_iter(f::Function, M::AbstractArray{<:Any,2}; byrow::Bool=true, multithread::Bool=true)
    if byrow
        num_rows = size(M, 1)
        if num_rows == 0
            return Matrix{Any}(undef, 0, 0)
        end
        first_row = first(eachrow(M))
        first_result = f(first_row)
        result_cols = length(first_result)
        result_rows = num_rows
        result = Matrix{eltype(first_result)}(undef, result_rows, result_cols)
        result[1, :] = first_result
        if multithread
            current_BLAS_threads = BLAS.get_num_threads()
            BLAS.set_num_threads(1)
            Threads.@threads for i in 2:num_rows
                result[i, :] = f(@view M[i, :])
            end
            BLAS.set_num_threads(current_BLAS_threads)
        else
            for i in 2:num_rows
                result[i, :] = f(@view M[i, :])
            end
        end
        return result
    else
        num_cols = size(M, 2)
        if num_cols == 0
            return Matrix{Any}(undef, 0, 0)
        end
        first_col = first(eachcol(M))
        first_result = f(first_col)
        result_rows = length(first_result)
        result_cols = num_cols
        result = Matrix{eltype(first_result)}(undef, result_rows, result_cols)
        result[:, 1] = first_result
        if multithread
            current_BLAS_threads = BLAS.get_num_threads()
            Threads.@threads for j in 2:num_cols
                result[:, j] = f(@view M[:, j])
            end
            BLAS.set_num_threads(current_BLAS_threads)
        else
            for j in 2:num_cols
                result[:, j] = f(@view M[:, j])
            end
        end
        return result
    end
end

"""
    rref_exact(A)

Row-reduced echelon form over an arbitrary exact scalar type.
"""
function rref_exact(A::AbstractMatrix{T}) where {T<:Number}
    M = copy(A)
    m, n = size(M)
    pivotcols = Int[]
    row = 1

    @inbounds for col in 1:n
        row > m && break
        pivot = 0
        for r in row:m
            if M[r, col] != 0
                pivot = r
                break
            end
        end
        pivot == 0 && continue

        if pivot != row
            for j in 1:n
                M[row, j], M[pivot, j] = M[pivot, j], M[row, j]
            end
        end

        piv = M[row, col]
        if piv != one(T)
            for j in col:n
                M[row, j] /= piv
            end
        end

        for r in 1:m
            r == row && continue
            c = M[r, col]
            c == 0 && continue
            for j in col:n
                M[r, j] -= c * M[row, j]
            end
        end

        push!(pivotcols, col)
        row += 1
    end

    return M, pivotcols
end

"""
    primitive_integer(v)

Convert a rational vector to a primitive integer vector.
"""
function primitive_integer(v::AbstractVector{<:Rational})
    dens = denominator.(v)
    L = foldl(lcm, dens; init=1)
    w = Int.(L .* v)
    g = foldl(gcd, abs.(w); init=0)
    g = g == 0 ? 1 : g
    w = div.(w, g)
    for x in w
        if x != 0
            if x < 0
                w = .-w
            end
            break
        end
    end
    return w
end

"""
    left_nullspace_integer(S::AbstractMatrix{Int})

Return primitive integer basis vectors for the left nullspace of `S`.
"""
function left_nullspace_integer(S::AbstractMatrix{Int})
    A = Rational{Int}.(transpose(S))
    M, pivotrows = rref_exact(A)

    m, n = size(M)
    ispivot = falses(n)
    for c in pivotrows
        ispivot[c] = true
    end
    freecols = findall(!, ispivot)

    B = Matrix{Int}(undef, n, length(freecols))
    @inbounds for (j, fc) in enumerate(freecols)
        x = zeros(Rational{Int}, n)
        x[fc] = 1
        for (i, pc) in enumerate(pivotrows)
            x[pc] = -M[i, fc]
        end
        B[:, j] = primitive_integer(x)
    end

    return B, pivotrows
end

"""
    S_to_S_pos_neg(S::SparseMatrixCSC)

Split signed sparse rows into positive/negative duplicated rows.
"""
function S_to_S_pos_neg(S::SparseMatrixCSC{T,Ti}) where {T<:Real,Ti<:Integer}
    m, n = size(S)
    nnzS = nnz(S)

    colptr = Vector{Ti}(undef, n + 1)
    rowval = Vector{Ti}(undef, nnzS)
    nzval  = Vector{T}(undef, nnzS)

    pos = 1
    colptr[1] = 1

    @inbounds for j in 1:n
        for p in S.colptr[j]:(S.colptr[j + 1] - 1)
            i = S.rowval[p]
            v = S.nzval[p]

            if v > zero(T)
                rowval[pos] = i
                nzval[pos] = v
                pos += 1
            elseif v < zero(T)
                rowval[pos] = i + m
                nzval[pos] = -v
                pos += 1
            end
        end
        colptr[j + 1] = pos
    end

    resize!(rowval, pos - 1)
    resize!(nzval, pos - 1)

    return SparseMatrixCSC(2m, n, colptr, rowval, nzval)
end
