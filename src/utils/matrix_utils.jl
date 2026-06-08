"""
    L_from_N(N::Matrix{Int}) -> Matrix{Int}

Compute a conservation matrix `L` from a stoichiometry matrix `N` such that
`N * L' = 0`.
"""
function L_from_N(N::Matrix{Int})::Matrix{Int}
    r, n = size(N)
    d = n - r
    N_1 = @view N[:, 1:d]
    N_2 = @view N[:, (d + 1):n]
    return hcat(Matrix(I, d, d), -(N_2 \ N_1)')
end

"""
    N_from_L(L::Matrix{Int}) -> Matrix{Int}

Recover a stoichiometry matrix `N` from a conservation matrix `L`.
"""
function N_from_L(L::Matrix{Int})::Matrix{Int}
    d, n = size(L)
    r = n - d
    L2 = @view L[:, (d + 1):n]
    return hcat(L2', Matrix(-I, r, r))
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

function _sparse_top_incidence(A::SparseMatrixCSC, end_row::Int)
    rows_for_col = [Int[] for _ in axes(A, 2)]
    nzidx_for_pair = Dict{Tuple{Int, Int}, Int}()

    for col in axes(A, 2)
        for nzidx in A.colptr[col]:(A.colptr[col + 1] - 1)
            row = A.rowval[nzidx]
            if row <= end_row && !iszero(A.nzval[nzidx])
                push!(rows_for_col[col], row)
                nzidx_for_pair[(row, col)] = nzidx
            end
        end
    end

    return rows_for_col, nzidx_for_pair
end

function _transversal_independent(
    rows_for_col::AbstractVector{<:AbstractVector{<:Integer}},
    cols::AbstractVector{<:Integer},
    end_row::Int,
)
    length(cols) <= end_row || return false
    row_to_col = zeros(Int, end_row)

    function try_match(col::Int, seen::AbstractVector{Bool})
        for row in rows_for_col[col]
            seen[row] && continue
            seen[row] = true
            if row_to_col[row] == 0 || try_match(row_to_col[row], seen)
                row_to_col[row] = col
                return true
            end
        end
        return false
    end

    for col in cols
        try_match(col, falses(end_row)) || return false
    end
    return true
end

function _dual_bottom_independent(
    A::SparseMatrixCSC, selected_cols::AbstractVector{<:Integer}, end_row::Int
)
    n_rows, n_cols = size(A)
    bottom_rank = n_rows - end_row
    bottom_rank < 0 && throw(ArgumentError("end_row must be <= size(A, 1)"))
    bottom_rank == 0 && return true

    selected = falses(n_cols)
    selected[selected_cols] .= true
    kept_cols = findall(!, selected)
    length(kept_cols) >= bottom_rank || return false

    bottom = Matrix(A[(end_row + 1):n_rows, kept_cols])
    return rank(bottom) == bottom_rank
end

function _with_added_col(cols::Vector{Int}, col::Int)
    out = copy(cols)
    push!(out, col)
    return out
end

function _with_swapped_col(cols::Vector{Int}, old_col::Int, new_col::Int)
    out = copy(cols)
    idx = findfirst(==(old_col), out)
    idx === nothing && error("old_col must be selected")
    out[idx] = new_col
    return out
end

function _matroid_intersection_diag_cols(A::SparseMatrixCSC, end_row::Int)
    n_rows, n_cols = size(A)
    n_rows == n_cols || throw(ArgumentError("A must be square"))
    0 <= end_row <= n_rows ||
        throw(ArgumentError("end_row must be between 0 and size(A, 1)"))
    end_row == 0 && return Int[]

    rows_for_col, _ = _sparse_top_incidence(A, end_row)
    ground_cols = [col for col in axes(A, 2) if !isempty(rows_for_col[col])]
    selected = Int[]

    m1(cols) = _transversal_independent(rows_for_col, cols, end_row)
    m2(cols) = _dual_bottom_independent(A, cols, end_row)

    while length(selected) < end_row
        selected_set = Set(selected)
        unselected = [col for col in ground_cols if !(col in selected_set)]

        prev = Dict{Int, Int}()
        queue = Int[]
        found = 0

        for col in unselected
            if m1(_with_added_col(selected, col))
                prev[col] = 0
                push!(queue, col)
            end
        end

        head = 1
        while head <= length(queue) && found == 0
            col = queue[head]
            head += 1

            if col in selected_set
                for new_col in unselected
                    haskey(prev, new_col) && continue
                    if m1(_with_swapped_col(selected, col, new_col))
                        prev[new_col] = col
                        push!(queue, new_col)
                    end
                end
            else
                if m2(_with_added_col(selected, col))
                    found = col
                    break
                end
                for old_col in selected
                    haskey(prev, old_col) && continue
                    if m2(_with_swapped_col(selected, old_col, col))
                        prev[old_col] = col
                        push!(queue, old_col)
                    end
                end
            end
        end

        found == 0 && throw(
            ArgumentError("Could not find a nonsingular top-row perturbation pattern.")
        )

        path = Int[]
        col = found
        while col != 0
            push!(path, col)
            col = prev[col]
        end

        selected_set = Set(selected)
        for path_col in path
            if path_col in selected_set
                filter!(!=(path_col), selected)
                delete!(selected_set, path_col)
            else
                push!(selected, path_col)
                push!(selected_set, path_col)
            end
        end
    end

    return selected
end

function _perfect_top_matching(
    rows_for_col::AbstractVector{<:AbstractVector{<:Integer}},
    cols::AbstractVector{<:Integer},
    end_row::Int,
)
    row_to_col = zeros(Int, end_row)

    function try_match(col::Int, seen::AbstractVector{Bool})
        for row in rows_for_col[col]
            seen[row] && continue
            seen[row] = true
            if row_to_col[row] == 0 || try_match(row_to_col[row], seen)
                row_to_col[row] = col
                return true
            end
        end
        return false
    end

    for col in cols
        try_match(col, falses(end_row)) ||
            throw(ArgumentError("Selected columns do not match all top rows."))
    end

    all(!iszero, row_to_col) ||
        throw(ArgumentError("Selected columns do not match all top rows."))
    return row_to_col
end

"""
    diag_indices(A::SparseMatrixCSC, end_row::Int) -> Vector{Int}

Return indices into `A.nzval` for one nonzero entry in each of the first
`end_row` rows such that perturbing those entries keeps the bottom rows'
complementary columns nonsingular. The selected pattern is found by matroid
intersection between the top-row transversal matroid and the dual column
matroid of the bottom block.
"""
function diag_indices(A::SparseMatrixCSC, end_row::Int)
    rows_for_col, nzidx_for_pair = _sparse_top_incidence(A, end_row)
    cols = _matroid_intersection_diag_cols(A, end_row)
    matched_cols = _perfect_top_matching(rows_for_col, cols, end_row)
    return [nzidx_for_pair[(row, matched_cols[row])] for row in 1:end_row]
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
function independent_row_idx(N::AbstractMatrix{T}) where {T}
    Nt_lu = lu(N'; check=false)
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
function _Mtx2idx_val(Mtx::Matrix{<:T}) where {T}
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
function _idx_val2Mtx(
    idx::Vector{Int}, val::T=1, col_num::Union{Int, Nothing}=nothing
) where {T}
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
function _idx_val2Mtx(
    idx::Vector{Int}, val::Vector{<:T}, col_num::Union{Int, Nothing}=nothing
) where {T}
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
function matrix_iter(
    f::Function, M::AbstractArray{<:Any, 2}; byrow::Bool=true, multithread::Bool=true
)
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
function rref_exact(A::AbstractMatrix{T}) where {T <: Number}
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
function S_to_S_pos_neg(S::SparseMatrixCSC{T, Ti}) where {T <: Real, Ti <: Integer}
    m, n = size(S)
    nnzS = nnz(S)

    I = Vector{Ti}(undef, nnzS)
    J = Vector{Ti}(undef, nnzS)
    V = Vector{T}(undef, nnzS)

    pos = 0
    @inbounds for j in 1:n
        for p in S.colptr[j]:(S.colptr[j + 1] - 1)
            i = S.rowval[p]
            v = S.nzval[p]

            if v > zero(T)
                pos += 1
                I[pos] = i
                J[pos] = j
                V[pos] = v
            elseif v < zero(T)
                pos += 1
                I[pos] = i + m
                J[pos] = j
                V[pos] = -v
            end
        end
    end

    resize!(I, pos)
    resize!(J, pos)
    resize!(V, pos)

    return sparse(I, J, V, 2m, n)
end
