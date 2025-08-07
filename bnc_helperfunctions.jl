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
    @assert n >= 3 "at least 3 species"
    @assert r >= 1 "at least 1 reactions"
    @assert min_binder >= 1 && max_binder >= min_binder "min_binder and max_binder must be at least 1"
    @assert min_binder <= n - r "min_binder must be smaller than n-r"
    #initialize the matrix
    N = zeros(Int, r, n)
    i_temp = max(max_binder + 1 + r - n, 1) # the row num before which max_binder <= n-r+i-1 

    for i in 1:i_temp # iter across row
        N[i, n-r+i] = -1
        binder_num = sample(min_binder:(n-r+i-1))
        N[i, sample(1:(n-r+i-1), binder_num; replace=false)] .= 1
    end
    for i in (i_temp+1):r # iter acorss row
        N[i, n-r+i] = -1
        binder_num = sample(min_binder:max_binder)
        N[i, sample(1:(n-r+i-1), binder_num; replace=false)] .= 1
    end
    return N
end

function L_generator(d::Int, n::Int; min_binder::Int=2, max_binder::Int=2)::Matrix{Int}
    N = N_generator(n - d, n; min_binder=min_binder, max_binder=max_binder)
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
    )::Tuple{Vector{Float64}, Matrix{Float64}}
    """
    A wrapper function to convert the ODESolution to a t vector and u matrix
    """
    return solution.t, stack(solution.u)'
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

function _update_Jt!(Jt,Bnc::Bnc, x::AbstractArray{<:Real}, q::AbstractArray{<:Real})
    # helper functions to speed up the calculation of lu decompostion of logder_qK_x.
    # One shall initialize Jt_lu and Jt before calling this function.
    # eg:
    # Jt = copy(Bnc._LNt_sparse)
    # Jt_lu = copy(Bnc._LNt_lu)
    Jt_left = @view(Jt.nzval[1:Bnc._val_num_L])
    x_view = @view(x[Bnc._I])
    q_view = @view(q[Bnc._J])
    @. Jt_left = x_view * Bnc._Lt_sparse.nzval / q_view
    return nothing
    # lu!(Jt_lu, Jt)
end

function find_max_indices_per_column(S::SparseMatrixCSC{Tv, Ti},n::Union{Int,Nothing}=nothing) where {Tv, Ti}
    # Get the number of columns from the sparse matrix
    n = isnothing(n) ? size(S, 2) : n
    # Pre-allocate the result
    max_indices = zeros(Ti, n)
    # Access the raw CSC data structures for efficiency
    colptr = S.colptr
    rowval = S.rowval
    nzval = S.nzval
    # Iterate over each column of the matrix
    for j in 1:n
        # Determine the range of indices in nzval and rowval for the current column j
        # This range points to the non-zero elements of column j.
        col_range = colptr[j] : (colptr[j+1] - 1)
        if !isempty(col_range) #skip empty columns
            col_values = view(nzval, col_range)
            # We only need the index, so we ignore the value with _.
            _, local_max_idx = findmax(col_values)
            # The global index is the start of the column's range plus the local index minus one.
            global_max_idx = col_range[1] + local_max_idx - 1
            max_indices[j] = rowval[global_max_idx]
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

# function adjoint_matrix(A)
#     # Calculate the adjoint of a square matrix A ,for calculate the ray if non-feasible regime
#     n = size(A, 1)
#     if size(A, 1) != size(A, 2)
#         error("square matrix required for adjoint calculation")
#     end
#     adj_A = zeros(eltype(A), n, n)
#     for i in 1:n, j in 1:n
#         minor_matrix = @view A[[1:i-1; i+1:end], [1:j-1; j+1:end]]
#         adj_A[j, i] = (-1)^(i + j) * det(minor_matrix)
#     end
#     return adj_A
# end

# function L_generator(d::Int, n::Int; min_nonatom::Int=0, max_nonatom::Union{Nothing,Int}=nothing, allow_multi::Bool = true)::Matrix{Int}
#     max_nonatom = isnothing(max_nonatom) ? n-d : max_nonatom
#     @assert n >= d "n must be greater than d"
#     @assert min_nonatom >= 0 && max_nonatom >= min_nonatom "min_nonatom and max_nonatom must be at least 0"
#     @assert max_nonatom <= n-d "max_nonatom must be smaller than n-d"
#     #initialize the matrix 
#     L1 = Diagonal(ones(Int, d)) # the first d rows are identity matrix
#     L2 = zeros(Int, d, n-d)
#     for i in 1:d
#         nonatom_num = sample(min_nonatom:max_nonatom)
#         if allow_multi
#             for j in 1:nonatom_num
#                 L2[i, sample(1:(n-d))] += 1
#             end
#         else
#             L2[i, sample(1:(n-d), nonatom_num; replace=false)] .= 1
#         end
#     end
#     return hcat(L1, L2)
# end

# S_generator(n::Int,r_cat::Int; d::Union{Int,nothing} = nothing, transform_only::bool=false, involving_K::Bool=false)::Matrix{Int}
#     S = zeros(Int, n, r_cat)
#     idx = Vector{Int}(undef, 2)
#     row_end = involving_K ? n : n - d
#     for i in 1:r_cat
#         S[sample!(1:row_end,idx),i] = 
#     # assign first 
#     end  
# end 