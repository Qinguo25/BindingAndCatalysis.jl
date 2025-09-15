using JSON3

#helper function to convert N matrix to L matrix if L is not provided.
function L_from_N(N::Matrix{Int})::Matrix{Int}
    r, n = size(N)
    d = n - r
    N_1 = @view N[:, 1:d]
    N_2 = @view N[:, d+1:n]
    hcat(Matrix(I, d, d), -(N_2 \ N_1)')
end

function N_from_L(L::Matrix{Int})::Matrix{Int}
    d, n = size(L)
    r = n - d
    L2 = @view L[:,d+1:n]
    hcat(L2',Matrix(-I,r,r))
end

function name_converter(name::Vector{<:T})::Vector{Num} where T 
    if T <: Num
        return name
    else
        return [Symbolics.variable(x; T=Real) for x in name]
    end
end


function rowmask_indices(A::SparseMatrixCSC, start_row::Int, end_row::Int)
    # 获取稀疏矩阵 A 中前 d行 的非零元素的行、列索引及其在 nzval 中的位置
    rows = Int[]        # 存储行坐标
    cols = Int[]        # 存储列坐标
    idxs = Int[]        # 存储 nzval 的索引位置

    for j in 1:size(A,2)              # 遍历列
        for k in A.colptr[j]:(A.colptr[j+1]-1)  # 遍历列的非零
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

function diag_indices(A::SparseMatrixCSC,end_row::Int)
    # 获取稀疏矩阵 A 中对角线前end_row行元素在 nzval 中的位置
    # rows = Int[]        # 存储行坐标
    # cols = Int[]        # 存储列坐标
    idxs = Int[]        # 存储 nzval 的索引位置

    for j in 1:size(A,2)              # 遍历列
        for k in A.colptr[j]:(A.colptr[j+1]-1)  # 遍历列的非零
            i = A.rowval[k]
            if i == j && i <= end_row
                push!(idxs, k)
            end
        end
    end
    return idxs
end

function log_sum_exp10(L::AbstractMatrix,logx::AbstractArray)
    m = maximum(logx)
    z = exp10.(x .-m)
    y = L * z
    return log10.(y) .+ m
end

function log_sum_exp10!(logq::AbstractVector, L::SparseMatrixCSC, logx::AbstractVector)
    d, n = size(L)
    m = maximum(logx)
    fill!(logq, 0.0)
    for col = 1:n
        xj = logx[col] - m
        ej = exp10(xj)
        for idx = L.colptr[col]:(L.colptr[col+1]-1)
            row = L.rowval[idx]
            logq[row] += L.nzval[idx] * ej
        end
    end
    @inbounds for i in 1:d
        logq[i] = log10(logq[i]) + m
    end
    return logq
end

# helper funtions to taking inverse when the matrix is singular.
function _adj_singular_matrix(M::Matrix{<:Real})::Tuple{Matrix{<:Real},Int}
    """
    Calculate the adjoint of a singular matrix M, and return the singularity count.
    """
    n, m = size(M)
    @assert n == m "Matrix must be square"
    M_svd = svd(M;full=true)
    singular_pos = findall(M_svd.S .< 1e-7)
    singularity = length(singular_pos)
    adj = zeros(eltype(M), n, n)
    if length(singular_pos) > 1
        @warn("Multiple singular values found")
        return zeros(n, n), singularity
    else # sigularity of 1
        line = M_svd.Vt[singular_pos[1],:] * M_svd.U[:, singular_pos[1]]'
        # @show line
        for i in 1:n
            if abs(line[i,i]) >1e-7
                idx = vcat(1:i-1,i+1:n)
                # @show idx
                amp = det(M[idx,idx])/ line[i,i]
                # @show amp
                @. adj = round(line * amp)
                return adj, singularity
            end
        end
        @error("No non-zero diagonal element found in the singular vector")
    end
end

# function inv_singularity_matrix(M::Matrix{<:Real})
#     M_lu = lu(M,check=false)
#     if issuccess(M_lu) # Lu successfully.
#         return inv(M_lu),0  # singularity is 0, not singular
#     else
#         return _adj_singular_matrix(M)  # calculate the adj matrix, singularity is calculated and returned,
#     end
# end


function randomize(size::Tuple{Int,Int}, log_lower=-6, log_upper=6; log_space::Bool=true)::Matrix{<:Real}
    """
    Generate a random matrix of size (m, n) with values between 10^log_lower and 10^log_upper, in log space
    """
    m, n = size
    if log_space
        return exp10.(rand(m, n) .* (log_upper - log_lower) .+ log_lower)
    else
        return rand(m, n) .* (exp10(log_upper) - exp10(log_lower)) .+ exp10(log_lower)
    end
end
function randomize(n::Int, log_lower=-6, log_upper=6; log_space::Bool=true)::Vector{<:Real}
    """
    Generate a random vector of size n with values between 10^log_lower and 10^log_upper, in log space
    """
    # Generate a random vector of size n with values between 10^log_lower and 10^log_upper

    #turn lowerbound and upperbound into bases of e
    if log_space
        exp10.(rand(n) .* (log_upper - log_lower) .+ log_lower)
    else
        rand(n) .* (exp10(log_upper) - exp10(log_lower)) .+ exp10(log_lower)
    end
end


function arr_to_vector(arr)
    """convert a multidimensional array to a vector or list,
    eg: a matrix to a vector of vector, which contains each colums as a vector,
    """
    d = ndims(arr)
    if d == 0
        return arr[]  # 处理0维数组（标量）
    elseif d == 1
        return [x for x in arr]  # 1维数组转列表
    else
        # 沿第一维切片，递归处理每个切片（降维后）
        return [arr_to_vector(s) for s in eachslice(arr, dims=1)]
    end
end
function pythonprint(arr)
    txt = JSON3.write(arr_to_vector(arr), pretty=true, indent=4, escape_unicode=false)
    println(txt)
    return nothing
end



function N_generator(r::Int, n::Int; min_binder::Int=2, max_binder::Int=2)::Matrix{Int}
    @assert n > r "n must be greater than r"
    @assert min_binder >= 1 && max_binder >= min_binder "min_binder and max_binder must be at least 1"
    @assert min_binder <= n - r "min_binder must be smaller than n-r"
    #initialize the matrix
    d = n-r
    N = [zeros(r,d) -I(r)]
    Threads.@threads for i in 1:r
        idx = sample(1:d+i-1,rand(min_binder:max_binder); replace=true)
        for j in idx
            N[i,j] +=1
        end
    end
    return N
end

function L_generator(d::Int, n::Int; kwargs...)::Matrix{Int}
    N = N_generator(n - d, n; kwargs...)
    L = L_from_N(N)
    return L
end

function independent_row_idx(N::AbstractMatrix{T}) where T
    # find linear independent rows of a matrix N and return the index
    Nt_lu = lu(N',check=false)
    issuccess(Nt_lu) && return collect(1:size(N, 1))
    tol = 1e-8
    pivot_indices = findall(abs.(diag(Nt_lu.U)) .> tol)
    return pivot_indices
end

function _ode_solution_wrapper(
    solution::ODESolution
    )::Tuple{Vector{Float64}, Vector{Vector{Float64}}}
    """
    A wrapper function to convert the ODESolution to a t vector and u matrix
    """
    return solution.t, solution.u
end


function pairwise_distance(data::AbstractVector, dist_func::Function; is_symmetric::Bool=true)
    n = length(data)
    # Determine the output type by calculating one distance value first.
    # This creates a type-stable matrix, which is more efficient.
    T = typeof(dist_func(data[1], data[1]))
    dist_matrix = Matrix{T}(undef, n, n)
    if is_symmetric
        Threads.@threads for i in 1:n
            for j in i:n
                # Calculate the distance
                d = dist_func(data[i], data[j])
                # Assign to both [i, j] and [j, i]
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
            end
        end
    else
        Threads.@threads for i in 1:n
            for j in 1:n
                dist_matrix[i, j] = dist_func(data[i], data[j])
            end
        end
    end
    return dist_matrix
end


log10_sym(x) = x==1 ? Num(0) : Symbolics.wrap(Symbolics.Term(log10, [x,]))
exp10_sym(x) = Symbolics.wrap(Symbolics.Term(exp10, [x,]))


function get_int_type(n)
    # Get the integer type based on the number of bits
    m = n+1
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



#Pure helper functions for converting between matrix and index-value pairs.
function _Mtx2idx_val(Mtx::Matrix{<:T}) where T
    row_num, col_num  = size(Mtx)
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
    return idx,val
end
function _idx_val2Mtx(idx::Vector{Int}, val::T=1, col_num::Union{Int,Nothing}=nothing) where T
    n = length(idx)
    col_num = isnothing(col_num) ? n : col_num # if col_num is not provided, use the maximum idx value
    Mtx = zeros(T, n, col_num)
    for i in 1:n
        if idx[i] != 0
            Mtx[i, idx[i]] = val
        end
    end
    return Mtx
    
end
function _idx_val2Mtx(idx::Vector{Int}, val::Vector{<:T}, col_num::Union{Int,Nothing}=nothing) where T
    # Convert idx and val to a matrix of size (d, n)
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

function _check_valid_idx(idx::Vector{Int},Mtx::Matrix{<:Any})
    # Check if the idx is valid for the given Mtx
    @assert length(idx) == size(Mtx, 1) "idx must have the same length as the number of rows in Mtx"
    for i in 1:length(idx)
        @assert Mtx[i, idx[i]] != 0 "Mtx must have non-zero entries at the idx positions"
    end
    return true
end



# function _update_Jt_ignore_val!(Jt,Bnc::Bnc, x::AbstractArray{<:Real}, q::AbstractArray{<:Real})
#     # helper functions to speed up the calculation of lu decompostion of logder_qK_x.
#     Jt = copy(Bnc._LNt_sparse)
#     Jt_left = @view(Jt.nzval[1:Bnc._val_num_L])
#     x_view = @view(x[Bnc._I])
#     # q_view = @view(q[Bnc._J])
#     @. Jt_left = x_view #/ q_view
#     return nothing
#     # lu!(Jt_lu, Jt)
# end


function find_max_indices_per_column(S::SparseMatrixCSC{Tv, Ti}, first_n_col::Union{Int,Nothing}=nothing) where {Tv, Ti}
    first_n_col = isnothing(first_n_col) ? size(S, 2) : first_n_col
    max_indices = zeros(Ti, first_n_col)

    colptr = S.colptr
    rowval = S.rowval
    nzval  = S.nzval

    @inbounds for j in 1:first_n_col
        col_start = colptr[j]
        col_end   = colptr[j+1] - 1
        if col_start <= col_end
            max_val = typemin(Tv)
            max_row = rowval[col_start]
            @inbounds for idx in col_start:col_end
                v = nzval[idx]
                if v > max_val
                    max_val = v
                    max_row = rowval[idx]
                end
            end
            max_indices[j] = max_row
        end
    end
    return max_indices
end


function matrix_iter(f::Function, M::AbstractArray{<:Any,2}; byrow::Bool=true,multithread::Bool=true)
    # Get the number of rows from the input matrix
    if byrow
        num_rows = size(M, 1)
        # Check if the matrix is empty
        if num_rows == 0
            return Matrix{Any}(undef, 0, 0) # Or return an appropriately typed empty matrix
        end
        first_row = first(eachrow(M))
        # Apply the function to the first row to get a sample result
        first_result = f(first_row)
        # Determine the size of the output matrix
        # The number of rows in the result matrix is the length of the vector returned by f.
        # The number of columns is the number of rows in the input matrix.
        result_cols = length(first_result)
        result_rows = num_rows
        # Pre-allocate the result matrix with the correct type and size
        # This is key for performance!
        result = Matrix{eltype(first_result)}(undef, result_rows, result_cols)
        # Place the first result in the first column
        result[1,:] = first_result
        # Loop through the rest of the rows (from the 2nd row onwards)
        # We use `enumerate` to get the index `i` (starting from 2)
        # and `Iterators.drop` to skip the first row which we've already processed.
        if multithread
            current_BLAS_threads = BLAS.get_num_threads()
            BLAS.set_num_threads(1) # Set to 1 to avoid multithreading
            # --- FIXED PART (byrow) ---
            Threads.@threads for i in 2:num_rows
                result[i, :] = f(@view M[i, :])
            end
            BLAS.set_num_threads(current_BLAS_threads) # Restore the original number of threads
        else
            # This part was already okay, but we use views for consistency
            for i in 2:num_rows
                result[i, :] = f(@view M[i, :])
            end
        end
        return result
    else
        num_cols = size(M, 2)
        # Check if the matrix is empty
        if num_cols == 0
            return Matrix{Any}(undef, 0, 0) # Or return an appropriately typed empty matrix
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
Symbolic helper function to convert a sum of log10 terms into a product form.
from ∑a log b to log ∏b^a

The final expression contains ∏b^a term.
"""
function handle_log_weighted_sum(expr)
    get_single = @rule(+(~~xs) => [~~xs...])
    get_coeff = SymbolicUtils.Chain([@rule(~x => exp10(~x)),@rule(~c * log10(~b) => (~b)^(~c)), @rule(log10(~b) => (~b))])
    terms = get_single(expr)
    subs_expr = get_coeff.(terms) |> prod
    if contains(string(subs_expr), "log10")
        @error "Failed to convert the expression to product form"
    else
        return subs_expr
    end
end


# """
# Helper funtions to taking a classification vector and return a vector of colors by :viridis color map.
# """
# function classification_to_colors(classification::Vector{Int}, color_map::String="viridis")
#     # Convert a classification vector to a vector of colors using the specified color map
#     # The color_map can be "viridis", "plasma", "inferno", "magma", or "cividis"
#     if color_map == "viridis"
#         return cgrad(:viridis, length(unique(classification)))[classification]
#     elseif color_map == "plasma"
#         return cgrad(:plasma, length(unique(classification)))[classification]
#     elseif color_map == "inferno"
#         return cgrad(:inferno, length(unique(classification)))[classification]
#     elseif color_map == "magma"
#         return cgrad(:magma, length(unique(classification)))[classification]
#     elseif color_map == "cividis"
#         return cgrad(:cividis, length(unique(classification)))[classification]
#     else
#         @error "Unknown color map: $color_map"
#     end
# end

"""
Helper functions to find difference between two vectors
"""
function vector_difference(v1::AbstractVector{T}, v2::AbstractVector{T}) where T
    diff_index = findall(v1 .!= v2)
    mp = countmap(zip(v1[diff_index], v2[diff_index]))
    mp_sort = sort(collect(mp), by=x->x.second, rev=true)
    return mp_sort
end

"""
CUDA helper to get access to SM number and maximum threads per SM.
"""
function GPU_SM_threads_num()
    dev = CUDA.device()
    SM = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    max_threads = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    return SM, max_threads
end


#---------------------------------------------------
#
#---------------------------------------------------