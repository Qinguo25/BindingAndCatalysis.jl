#------------------------------------------------------------------------------
#             1. Functions find all regimes and return properties
# ------------------------------------------------------------------------------
"""
    find_all_vertices(L; eps=1e-9, dominance_ratio=nothing, mode="auto")

Find all feasible "vertex" regimes for matrix `L` (d × n).
Returns Vector{Vector{Int}} where each inner Vector is a length-d assignment (1-based column indices).
Options:
- eps: small slack for weighted mode
- dominance_ratio: Float64
- asymptotic::Bool
"""

find_all_vertices_nonasym(L;kwargs...) = _vtxs_nonasym(L,Val(get_int_type(size(L)[2]));kwargs...) 
function _vtxs_nonasym(L, ::Val{T} ;eps=1e-9) where T
    d, n = size(L)
    # T = get_int_type(n)  # Determine integer type based on n
    # Nonzero indices of each row
    J = [findall(!iszero, row) for row in eachrow(L)]
    
    order = sortperm(J, by=length, rev=true)
    inv_order = invperm(order)
    J_ord = J[order]

    # Precompute edge weights: Dict[u => edges] for each row
    row_edges = map(1:d) do i
        Ji = J[i]
        logL = log.(L[i, Ji])  # compute once per row
        Dict(Ji[k] => [(Ji[m], -((logL[m] - logL[k]) + eps)) for m in eachindex(Ji) if m != k]
             for k in eachindex(Ji))
    end

    adj = [Vector{Tuple{T,Float64}}() for _ in 1:n]
    results = Vector{Vector{T}}()

    function has_neg_cycle(seeds)
        dist_local = zeros(Float64, n)   # rollback safety
        q = Queue{T}()
        inq = falses(n)
        cnt = zeros(T, n)
        for u in seeds
            enqueue!(q, u)
            inq[u] = true
        end
        while !isempty(q)
            u = dequeue!(q)
            inq[u] = false
            du = dist_local[u]
            for (v, w) in adj[u]
                nd = du + w
                if nd + 1e-15 < dist_local[v]
                    dist_local[v] = nd
                    if !inq[v]
                        enqueue!(q, v)
                        inq[v] = true
                        cnt[v] += 1
                        if cnt[v] > n
                            return true  # negative cycle detected
                        end
                    end
                end
            end
        end
        return false
    end

    function dfs(r, chosen)
        if r > d
            # @show adj
            push!(results, chosen[inv_order])
            return
        end
        for u in J_ord[r]
            oldlen = length(adj[u])
            append!(adj[u], row_edges[order[r]][u])
            if !has_neg_cycle(J_ord[r])
                push!(chosen, u)
                dfs(r+1, chosen)
                pop!(chosen)
            end
            resize!(adj[u], oldlen)  # rollback edges
        end
    end

    dfs(1, Int[])
    return results
end
find_all_vertices_asym(L;kwargs...) = _vtxs_asym(L,Val(get_int_type(size(L)[2]));kwargs...)
function _vtxs_asym(L, ::Val{T}) where T
    d, n = size(L)
    J = [findall(x -> x != 0, row) for row in eachrow(L)]
    order = sortperm(J, by = length, rev=true)
    inv_order = invperm(order)
    J_ord = J[order]

    graph = [T[] for _ in 1:n]
    results = Vector{Vector{T}}()


    stack = Vector{T}(undef, n)                 # DFS stack (reused)
    visited_stamp = zeros(Int, n)                 # visited stamps per node
    stamp_ref = Ref(0)
    function reachable(start, target)::Bool
        stamp_ref[] += 1
        curstamp = stamp_ref[]
        top = 1
        stack[1] = start
        visited_stamp[start] = curstamp
        while top > 0
            u = stack[top]; top -= 1
            if u == target
                return true
            end
            for w in graph[u]
                if visited_stamp[w] != curstamp
                    visited_stamp[w] = curstamp
                    top += 1
                    stack[top] = w
                end
            end
        end
        return false
    end

    # fast reachable using stamp to avoid clearing visited array
    function dfs(r,chosen)
        if r > d
            push!(results, chosen[inv_order])
            # @show graph
            return
        end

        row_choices = J_ord[r]
        any_success = false
        for v in row_choices
            
            # if v can reach some k in this row, adding k->v would create cycle
            bad = false
            for k in row_choices
                k == v && continue
                if reachable(v, k)
                    bad = true
                    break
                end
            end
            bad && continue

            # add edges k -> v for k != v
            for k in row_choices
                k == v && continue
                push!(graph[k], v)
            end

            push!(chosen, v)
            dfs(r + 1,chosen)
            pop!(chosen)

            # remove added edges
            for k in reverse(row_choices)
                k == v && continue
                pop!(graph[k])
            end
            any_success = true
        end
    end
    dfs(1,T[])
    return results
end
function find_all_vertices(L::Matrix{Int} ; eps=1e-9, dominance_ratio=Inf, asymptotic::Union{Bool,Nothing}=nothing)

    asymptotic =  (isnothing(asymptotic) && dominance_ratio == Inf) || (asymptotic == true)
    eps = asymptotic ? nothing : (dominance_ratio == Inf ? eps : log(dominance_ratio))  # extra slack for weighted mode 
    if asymptotic
        return find_all_vertices_asym(L)
    else
        return find_all_vertices_nonasym(L,eps=eps)
    end
end
"""
find all vertices in Bnc, and store them in Bnc.vertices_perm.
With its idx Dict in Bnc.vertices_idx.
Also calculate if the vertex is real or fake, and its nullity.
"""
function find_all_vertices!(Bnc::Bnc{T};) where T # cheap enough for now
    if isempty(Bnc.vertices_perm) || isempty(Bnc.vertices_real_flag)
        print("Start finding all vertices, it may takes a while.\n")
    
        # finding non-asymptotic vettices, which gives all vertices both real and fake, singular and non-singular
        Bnc.vertices_perm = find_all_vertices(Bnc.L; asymptotic=false)
        # Create the idx for each vertex
        Bnc.vertices_idx = Dict(a=>idx for (idx, a) in enumerate(Bnc.vertices_perm)) # Map from vertex to its index
        # finding asymptotic vertices, which is the real vertices.
        real_vtx = Set(find_all_vertices(Bnc.L; asymptotic=true))
        Bnc.vertices_real_flag = Bnc.vertices_perm .∈ Ref(real_vtx)
        
        # Caltulate the nullity for each vertices
        nullity = Vector{T}(undef, length(Bnc.vertices_perm))
        function calc_nullity(perm)
            # helper function for helping calculate nullity when vertex is real.
            length(perm) -length(Set(perm))
        end

        Threads.@threads for i in  1:length(Bnc.vertices_perm)
            perm = Bnc.vertices_perm[i]
            # is_real = Bnc.vertices_real_flag[i]
            # if is_real
            #     nullity[i] = calc_nullity(perm)
            # else
                nullity_P = calc_nullity(perm)
                _ , nullity_N =  _get_Nρ_inv_from_perm!(Bnc,perm) 
                nullity[i] = nullity_P + nullity_N
            # end
        end
        # @show nullity
        Bnc.vertices_nullity = nullity
        print("Done, with $(length(Bnc.vertices_perm)) vertices found and $(length(real_vtx)) real vertices.\n")
    end
    return Bnc.vertices_perm
end



"""
Return a dict with key: vertex perm, value: its index in Bnc.vertices_perm
"""
function get_vertices_mapping_dict(Bnc::Bnc)
    """
    get vertices mapping dict
    """
    find_all_vertices!(Bnc) # Ensure vertices are calculated
    return Bnc.vertices_idx
end


"""
Get the nullity of all vertices in Bnc.
"""
function get_all_vertices_nullity!(Bnc::Bnc)
    """
    Calculate the nullity of all vertices in Bnc.
    """
    find_all_vertices!(Bnc)
    return Bnc.vertices_nullity
end


#--------------Helper functions speeding up inverse calculation and singular detection---------------
function _get_Nρ_inv_from_perm!(Bnc::Bnc{T},perm::AbstractVector{<:Integer}) where T
    perm_set = Set(perm)
    key = [i for i in 1:Bnc.n if i ∉ perm_set]
    return _get_Nρ_inv!(Bnc,key)
end
function _get_Nρ_inv!(Bnc::Bnc{T}, key::AbstractVector{<:Integer}) where T
    function _calc_Nρ_inv(Nρ)
        r, r_ncol = size(Nρ)
        if r != r_ncol
            nullity = r - rank(Nρ)
            return spzeros(0, 0), nullity
        else
            Nρ_lu = lu(Nρ; check=false)
            if issuccess(Nρ_lu)
                nullity = 0
                Nρ_inv = sparse(inv(Array(Nρ)))
                return Nρ_inv, nullity
            else
                nullity = r - rank(Nρ)
                return spzeros(0, 0), nullity
            end
        end
    end

    get!(Bnc._vertices_Nρ_inv_dict, key) do
        Nρ = @view Bnc.N[:, key]
        _calc_Nρ_inv(Nρ)
    end
end


#---------------------------------------------------------------------------------------------
#   Functions involving vertices relationships, (neighbors finding and changedir finding)
#---------------------------------------------------------------------------------------------

function _calc_neighbor_mat(data::Vector{<:AbstractVector{T}}) where {T}
    n = length(data)
    # Each thread collects pairs for the upper triangle of the matrix
    thread_rows = [Int[] for _ in 1:Threads.nthreads()]
    thread_cols = [Int[] for _ in 1:Threads.nthreads()]
    thread_diff_rows = [T[] for _ in 1:Threads.nthreads()]

    Threads.@threads for i in 1:n
        tid = Threads.threadid()
        local_r = thread_rows[tid]
        local_c = thread_cols[tid]
        local_diff_r = thread_diff_rows[tid]
        # Iterate only over the upper triangle (j > i) to avoid redundant checks
        vi = data[i]
        for j in i+1:n
            vj = data[j]
            
            dist = Int8(0)

            diff_r = 0
            @inbounds for k in eachindex(vi, vj)
                if vi[k] != vj[k]
                    dist += 1
                    if dist == 1
                        diff_r = k   # record first difference
                    else
                        break
                    end
                end
            end

            if dist == 1
                push!(local_r, i)
                push!(local_c, j)
                push!(local_diff_r,diff_r)
            end
        end
    end
    # Merge results from all threads
    I = reduce(vcat, thread_rows)
    J = reduce(vcat, thread_cols)
    vals = reduce(vcat, thread_diff_rows)

    # Build the full symmetric matrix from the upper triangle parts at the end
    # rows = [I; J]
    # cols = [J; I]
    # vals = ones(Bool, length(I))
    # perms = sortperm(1:length(rows),by=k->(cols[k],rows[k]))
    return SparseArrays.sparse!(I, J, vals, n, n)#, invperm(perms)
end
function get_vertices_neighbor_mat!(Bnc::Bnc;)
    """
    # find the x space neighbor of all vertices in Bnc, the value denotes for two perms, which row they differ at.
    """
    find_all_vertices!(Bnc) # Ensure vertices are calculated
    if isempty(Bnc.vertices_neighbor_mat)
        print("Start calculating vertex neighbor matrix, It may takes a while.\n")
        Bnc.vertices_neighbor_mat = _calc_neighbor_mat(Bnc.vertices_perm)
        print("Done.\n")
    end
    return Bnc.vertices_neighbor_mat
end
function get_vertices_change_dir_x!(Bnc::Bnc{T}) where T #Could be optimized to incoperated into neighbor finding process.
"""
Calculate the x space change needed to change from one vertex to another, based on the calculated neighbor matrix, further could be intrgrated into neighbor finding process.
The value is stroed as SparseVector, with only two non-zero elements, with positive and negative 1. the index denotes the 
Source: row; Target: column.
"""
    if !isempty(Bnc.vertices_change_dir_x)
        return Bnc.vertices_change_dir_x
    end
    # print("1\n")
    d_mtx = get_vertices_neighbor_mat!(Bnc) # upper diagnal mtx
    # Initialize the change direction matrix
    # I,J,_ = findnz(triu(d_mtx))
    I,J,_ = findnz(d_mtx)
    n = Bnc.n
    val_len = length(I)

    vals = Vector{SparseVector{Int8, T}}(undef, val_len)

    Threads.@threads for k in 1:val_len
        from_idx = I[k]
        to_idx = J[k]

        from_vtx = Bnc.vertices_perm[from_idx]
        to_vtx = Bnc.vertices_perm[to_idx]

        # Find the first differing element
        for (x, y) in zip(from_vtx, to_vtx)
            if x == y
                continue
            elseif x < y
                vals[k] = SparseVector(n,T[x,y],Int8[-1,1])
                break
            else
                vals[k] = SparseVector(n,T[y,x],Int8[1,-1])
                break
            end
        end
    end
    Bnc.vertices_change_dir_x = SparseArrays.sparse!(I,J, vals, size(d_mtx)...)
end
function get_change_dir_x(Bnc::Bnc, from, to)
    from = get_idx(Bnc, from)
    to = get_idx(Bnc, to)
    d_mat = get_vertices_change_dir_x!(Bnc)
    if from < to
        return d_mat[from, to]
    else
        return -d_mat[to, from]
    end
end
function get_vertices_change_dir_qK!(Bnc::Bnc{T}) where T
    if !isempty(Bnc.vertices_change_dir_qK)
        return Bnc.vertices_change_dir_qK
    end
    d_mat = get_vertices_change_dir_x!(Bnc)
    I,J,V = findnz(d_mat)
    n_vals = length(I)
    vals = Vector{SparseVector{Float64, T}}(undef, n_vals)

    # Threads.@threads 
    for k in 1:n_vals
        v_source_idx = I[k] #perms_idx
        v_target_idx = J[k] #perms_idx
        dir = V[k]

        print("start calculating from source to target: $v_source_idx -> $v_target_idx \n")

        if get_nullity!(Bnc,v_source_idx) != 0 
            if get_nullity!(Bnc,v_target_idx) != 0
                @warn("Both $v_source_idx and $v_target_idx are singular, cannot get change direction between them for qK")
                total_flow = spzeros(Bnc.n)
            else #
                print("Source vertex $v_source_idx is singular, only use target vertex $v_target_idx to calculate flow \n") 
                # flow_qK_target = get_H!(Bnc, v_target_idx)[dir,:] - get_H!(Bnc, v_target_idx)[dir_back_x,:]
                total_flow = get_H!(Bnc,v_target_idx)' * dir
            end
        else
            if get_nullity!(Bnc,v_target_idx) != 0
                print("Target vertex $v_target_idx is singular, only use source vertex $v_source_idx to calculate flow \n")
                # flow_qK_source = get_H!(Bnc, v_source_idx)[dir,:] - get_H!(Bnc, v_source_idx)[dir_back_x,:]
                total_flow = get_H!(Bnc,v_source_idx)' * dir
            else
                flow_qK_source = get_H!(Bnc,v_source_idx)' * dir
                # @show get_H!(Bnc, v_source_idx)[dir,:]|>Array
                # @show -get_H!(Bnc, v_source_idx)[dir_back_x,:]|>Array

                flow_qK_target = get_H!(Bnc, v_target_idx)' * dir
                # @show get_H!(Bnc, v_target_idx)[dir,:]|>Array
                # @show -get_H!(Bnc, v_target_idx)[dir_back_x,:]|>Array

                # @show Array(flow_qK_source)
                # @show Array(flow_qK_target)
                total_flow = flow_qK_source + flow_qK_target
            end
        end

        print("Total flow is: ")
        print(Array(total_flow))

        tgt_qK1 = droptol!(total_flow,1e-9)
        vals[k] = tgt_qK1
    end
    Bnc.vertices_change_dir_qK = SparseArrays.sparse!(I, J, vals, size(d_mat)...)
end
function get_change_dir_qK(Bnc::Bnc, from, to)
    from = get_idx(Bnc, from)
    to = get_idx(Bnc, to)
    d_mat = get_vertices_change_dir_qK!(Bnc)
    if from < to
        return d_mat[from, to]
    else
        return -d_mat[to, from]
    end
end



#-------------------------------------------------------------------------------------
#         functions involving single vertex and lazy calculate  its properties
# ------------------------------------------------------------------------------------

# """
# Creates the P and P0 matrices from a permutation.
# """
# function _calculate_P_P0(Bnc::Bnc{T}, perm::Vector{<:Integer}) where T
    
#     P = zeros(Int, Bnc.d, Bnc.n)
#     # P0 = zeros(Rational, Bnc.d)
#     P0 = Vector{Float64}(undef, Bnc.d)
#     for i in 1:Bnc.d
#         P[i, perm[i]] = 1
#         P0[i] = log10(Bnc.L[i, perm[i]])
#         # P0[i] = log10_sym(Bnc.L[i, perm[i]])
#     end
#     return P, P0
# end
"""
Creates the P and P0 matrices from a permutation.
"""
function _calculate_P_P0(Bnc::Bnc{T}, perm::Vector{<:Integer}) where T
    d, n = Bnc.d, Bnc.n
    
    # The non-zero elements are at rows i and columns perm[i].
    I = 1:d                # Row indices are 1, 2, 3, ...
    J = perm               # Column indices are given by the permutation vector
    V = ones(Int, d)       # The value at each location is 1
    P = sparse(I, J, V, d, n)

    # The calculation for P0 remains the same
    P0 = Vector{Float64}(undef, d)
    for i in 1:d
        P0[i] = log10(Bnc.L[i, perm[i]])
    end
    
    return P, P0
end

"""
Creates the C and C0 matrices from a permutation.
"""
function _calculate_C_C0_x(Bnc::Bnc{T}, perm::Vector{<:Integer}) where T
    # This fucntion is created by chatgpt with numeric verification from dense version.
    # Is the lowest level, maximum-speed version.
    num_ineq = Bnc._val_num_L - Bnc.d
    nnz = 2 * num_ineq  # exactly two entries per inequality

    # Preallocate row indices + values
    rowval = Vector{Int}(undef, nnz)
    nzval  = Vector{Int}(undef, nnz)
    c0     = Vector{Float64}(undef, num_ineq)

    # Count nonzeros per column first
    colcounts = zeros(Int, Bnc.n)
    for i in 1:Bnc.d
        valid_idx = Bnc._valid_L_idx[i]
        rgm = perm[i]
        for col in valid_idx
            if col != rgm
                colcounts[col] += 1
                colcounts[rgm] += 1
            end
        end
    end
    # Build colptr from counts
    colptr = Vector{Int}(undef, Bnc.n+1)
    colptr[1] = 1
    for j in 1:Bnc.n
        colptr[j+1] = colptr[j] + colcounts[j]
    end

    # Position trackers for each column
    nextpos = copy(colptr)

    # Fill in rowval, nzval
    row = 1
    for i in 1:Bnc.d
        valid_idx = Bnc._valid_L_idx[i]
        rgm = perm[i]
        for col in valid_idx
            if col != rgm
                # insert (-1) at (row,col)
                pos = nextpos[col]
                rowval[pos] = row
                nzval[pos]  = -1
                nextpos[col] += 1

                # insert (+1) at (row,rgm)
                pos = nextpos[rgm]
                rowval[pos] = row
                nzval[pos]  = 1
                nextpos[rgm] += 1

                # compute c0 entry
                c0[row] = log10(Bnc.L[i, rgm] / Bnc.L[i, col])
                row += 1
            end
        end
    end

    c_mtx = SparseMatrixCSC(num_ineq, Bnc.n, colptr, rowval, nzval)
    return c_mtx, c0
end

"""
Creates a new, partially-filled Vertex object.
This function performs the initial, less expensive calculations.
"""
function _create_vertex(Bnc::Bnc{T}, perm::Vector{<:Integer})::Vertex where T
    find_all_vertices!(Bnc)
    idx = Bnc.vertices_idx[perm] # Index of the vertex in the Bnc.vertices_perm list
    real = Bnc.vertices_real_flag[idx] # Check if the vertex is real or fake
    nullity = Bnc.vertices_nullity[idx] # Get the nullity of the vertex
    
    P, P0 = _calculate_P_P0(Bnc, perm); 
    C_x, C0_x = _calculate_C_C0_x(Bnc, perm)

    F = eltype(P0)

    M = vcat(P, Bnc._N_sparse)
    M0 = vcat(P0, zeros(F, Bnc.r))
    # Initialize a partial vertex. "Full" properties are empty placeholders.
    return Vertex{F,T}(
        nullity = nullity,
        idx = idx,
        perm = perm, 
        real = real,
        M = M, M0 = M0, P = P, P0 = P0, C_x = C_x, C0_x = C0_x
    )
end

function _calc_H(Bnc::Bnc,perm::Vector{<:Integer})
    perm_set = Set(perm)
    key = [i for i in 1:Bnc.n if i ∉ perm_set]
    Nρ_inv = _get_Nρ_inv!(Bnc,key)[1] # get Nρ_inv from cache or calculate it.
    Nc = @view Bnc.N[:,perm]
    NcNρ_inv_neg = -Nc * Nρ_inv
    H_un_perm = [[I(Bnc.d) zeros(Bnc.d,Bnc.r)];[NcNρ_inv_neg Nρ_inv]]
    perm_inv = invperm([perm;key]) # get the inverse permutation to reorder H
    H = H_un_perm[perm_inv, :]
    return H
end

function _ensure_full_properties!(Bnc::Bnc, vtx::Vertex)
    # Check if already calculated
    if !isempty(vtx.H)
        return
    end
    if vtx.nullity == 0
        H = inv(Array(vtx.M)) # dense matrix 
        # H = _calc_H(Bnc, vtx.perm)
        vtx.H = droptol!(sparse(H),1e-10) # Calculate the inverse matrix from pre-computed LU decomposition of M
        vtx.H0 = H * vtx.M0
        vtx.C_qK = droptol!(sparse(vtx.C_x * H),1e-10)
        vtx.C0_qK = vtx.C0_x - vtx.C_x * vtx.H0 # Correctly use vtx.C0_x
    end
end

"""
Retrieves a vertex from cache or creates it if it doesn't exist.
"""
function get_vertex!(Bnc::Bnc, perm::Vector{<:Integer};full::Bool=true)::Vertex
    vtx = get!(Bnc.vertices_data, perm) do 
        _create_vertex(Bnc, perm)
    end
    if full
        _ensure_full_properties!(Bnc,vtx)
        get_all_neighbors!(Bnc, perm)
    end
    return vtx
end
function get_vertex!(Bnc::Bnc, idx::Int; kwargs...)
    """
    Get a vertex by its index in Bnc.vertices_perm.
    """
    find_all_vertices!(Bnc)
    return get_vertex!(Bnc, Bnc.vertices_perm[idx]; kwargs...)
end
function get_vertex!(Bnc::Bnc, vtx::Vertex; kwargs...)
    return get_vertex!(Bnc, vtx.perm; kwargs...)
end



"""
Get a vertex's index
"""
function get_idx(Bnc,perm::Vector{<:Integer})
    find_all_vertices!(Bnc)
    return Bnc.vertices_idx[perm]
end
function get_idx(Bnc::Bnc, idx::T) where T<:Integer
   return idx
end
function get_idx(Bnc::Bnc, vtx::Vertex)
    return vtx.idx
end


"""
Get perm of a vertex
"""
function get_perm(Bnc,perm::Vector{<:Integer})
    return perm
end
function get_perm(Bnc::Bnc, idx::Int)
    return find_all_vertices!(Bnc)[idx]
end
function get_perm(Bnc::Bnc, vtx::Vertex)
    return vtx.perm
end



function get_all_neighbors!(Bnc::Bnc, perm; return_idx::Bool=false)
    # Get the neighbors of the vertex represented by perm
    vtx = get_vertex!(Bnc, perm ; full=false)
    if isempty(vtx.neighbors_idx)
        d_mat = Symmetric(get_vertices_neighbor_mat!(Bnc), :U) # generate symmetric view
        vtx.neighbors_idx = findall(!iszero, d_mat[vtx.idx, :])
        finite_neighbors = Int[]
        infinite_neighbors = Int[]
        for idx in vtx.neighbors_idx
            if get_nullity!(Bnc, Bnc.vertices_perm[idx]) == 0
                push!(finite_neighbors, idx)
            else
                push!(infinite_neighbors, idx)
            end
        end

        vtx.finite_neighbors_idx = finite_neighbors
        vtx.infinite_neighbors_idx = infinite_neighbors
    end

    idx = vtx.neighbors_idx
    return return_idx ? idx : Bnc.vertices_perm[idx]
end
function get_finite_neighbors!(Bnc::Bnc, perm; return_idx::Bool=false)
    vtx = get_vertex!(Bnc, perm;full=false)
    if isempty(vtx.neighbors_idx)
        get_all_neighbors!(Bnc, perm)
    end
    idx = vtx.finite_neighbors_idx
    return return_idx ? idx : Bnc.vertices_perm[idx]
end
function get_infinite_neighbors!(Bnc::Bnc, perm; return_idx::Bool=false,nullity_max::Union{Int,Nothing}=nothing)
    vtx = get_vertex!(Bnc, perm;full=false)
    if isempty(vtx.neighbors_idx)
        get_all_neighbors!(Bnc, perm)
    end
    idx = vtx.infinite_neighbors_idx
    idx = isnothing(nullity_max) ? idx : filter(i->Bnc.vertices_nullity[i] <= nullity_max, idx)
    return return_idx ? idx : Bnc.vertices_perm[idx]
end

"""
Gets P and P0, creating the vertex if necessary.
"""
function get_P_P0!(Bnc::Bnc, perm)
    vtx = get_vertex!(Bnc, perm; full=false)
    return vtx.P, vtx.P0
end
get_P!(Bnc::Bnc, perm) = get_vertex!(Bnc, perm; full=false).P
get_P0!(Bnc::Bnc, perm) = get_vertex!(Bnc, perm; full=false).P0

"""
Gets M and M0, creating the vertex if necessary.
"""
function get_M_M0!(Bnc::Bnc, perm)
    vtx = get_vertex!(Bnc, perm; full=false)
    return vtx.M, vtx.M0
end
get_M!(Bnc::Bnc, perm) = get_vertex!(Bnc, perm; full=false).M
get_M0!(Bnc::Bnc, perm) = get_vertex!(Bnc, perm; full=false).M0


"""
Gets C_x and C0_x, creating the vertex if necessary.
"""
function get_C_C0_x!(Bnc::Bnc, perm)
    vtx = get_vertex!(Bnc, perm; full=false)
    return vtx.C_x, vtx.C0_x
end
get_C_x!(Bnc::Bnc, perm) = get_vertex!(Bnc, perm; full=false).C_x
get_C0_x!(Bnc::Bnc, perm) = get_vertex!(Bnc, perm; full=false).C0_x


"""
Gets C_qK and C0_qK, ensuring the full vertex is calculated.
"""
function get_C_C0_qK!(Bnc::Bnc, perm)
    vtx = get_vertex!(Bnc, perm; full=false)
    _ensure_full_properties!(Bnc,vtx)
    if vtx.nullity >= 1
        @error("Vertex got nullity $(vtx.nullity), currently doesn't support get C_qK and C0_qK")
    end
    return vtx.C_qK, vtx.C0_qK
end
get_C_qK!(Bnc::Bnc, perm) = get_C_C0_qK!(Bnc, perm)[1]
get_C0_qK!(Bnc::Bnc, perm) = get_C_C0_qK!(Bnc, perm)[2]

"""
Gets the nullity of a vertex
"""
function get_nullity!(Bnc::Bnc,perm)
    find_all_vertices!(Bnc)
    idx = get_idx(Bnc, perm)
    return Bnc.vertices_nullity[idx]
end


"""
Gets H and H0, ensuring the full vertex is calculated.
"""
function get_H_H0!(Bnc::Bnc, perm)
    vtx = get_vertex!(Bnc, perm; full=false)
    _ensure_full_properties!(Bnc,vtx)
    if vtx.nullity > 0
        @error("Vertex is singular, cannot get H0")
    end # This will compute if needed
    return vtx.H, vtx.H0
end

function get_H!(Bnc::Bnc, perm)
    vtx = get_vertex!(Bnc, perm; full=false)
    _ensure_full_properties!(Bnc,vtx)
    if vtx.nullity > 1
        @error("Vertex's nullity is bigger than 1, cannot get H")
    end # This will compute if needed
    return vtx.H
end
get_H0!(Bnc::Bnc, perm) = get_H_H0!(Bnc, perm)[2]


#-------------------------------------------------------------------------------------
#         functions of getting vertices with certain properties
# -------------------------------------------------------------------------------------

function get_singular_vertices(Bnc::Bnc; return_idx::Bool=false)
    """
    Get the indices of all singular vertices.
    Default to return perms
    """
    idx = findall(!iszero, get_all_vertices_nullity!(Bnc))
    return return_idx ? idx : Bnc.vertices_perm[idx]
end
get_singular_vertices_idx(Bnc::Bnc) = get_singular_vertices(Bnc; return_idx=true)


function get_nonsingular_vertices(Bnc::Bnc; return_idx::Bool=false)
    """
    Get the indices of all nonsingular vertices.
    """
    nullity = get_all_vertices_nullity!(Bnc)
    idx =  findall(iszero, nullity)
    return return_idx ? idx : Bnc.vertices_perm[idx]
end
get_nonsingular_vertices_idx(Bnc::Bnc) = get_nonsingular_vertices(Bnc; return_idx=true)


function get_real_vertices(Bnc::Bnc; return_idx::Bool=false)
    """
    Get the idx of all real vertices.
    """
    find_all_vertices!(Bnc)
    idx = findall(Bnc.vertices_real_flag)
    return return_idx ? idx : Bnc.vertices_perm[idx]
end
get_real_vertices_idx(Bnc::Bnc) = get_real_vertices(Bnc; return_idx=true)


function get_fake_vertices(Bnc::Bnc; return_idx::Bool=false)
    """
    Get the idx of all fake vertices.
    """
    find_all_vertices!(Bnc)
    idx = findall(!, Bnc.vertices_real_flag)
    return return_idx ? idx : Bnc.vertices_perm[idx]
end
get_fake_vertices_idx(Bnc::Bnc) = get_fake_vertices(Bnc; return_idx=true)


function get_vertices(Bnc::Bnc; singular::Union{Bool,Nothing}=nothing, real::Union{Bool,Nothing}=nothing, return_idx::Bool=false)
    """
    get vertices with certain properties.
    """ 
    singular_flag = isnothing(singular) ? trues(length(Bnc.vertices_perm)) : (singular .== (get_all_vertices_nullity!(Bnc) .> 0))
    real_flag = isnothing(real) ? trues(length(Bnc.vertices_perm)) : (real .== Bnc.vertices_real_flag)
    flag = singular_flag .& real_flag
    idx = findall(flag)
    return return_idx ? idx : Bnc.vertices_perm[idx]
end



function merge_conditions(Bnc::Bnc, perms...)
    Result = Vector{Tuple{SparseMatrixCSC{Float64, Int64}, Vector{Float64}}}()

end

# function ∂logqK_∂logx_regime(Bnc::Bnc; regime::Union{Vector{Int}, Nothing}=nothing,
#     Mtd::Union{Matrix{Int}, Nothing}=nothing,
#     M::Union{Matrix{Int}, Nothing}=nothing,
#     check::Bool=true)::Matrix{<:Real}
#     """
#     Calculate the derivative of log(qK) with respect to log(x) given regime
#     check: if true, check if the regime is valid for the L matrix
#     regime: the regime vector, if not provided , Mtd  will be derived from Mtd or M
    
#     Return:
#     - logder_qK_x: the derivative of log(qK) with respect to log(x)
#     """
#     if isnothing(Mtd)
#         if isnothing(regime)
#             if isnothing(M)
#                 @error("Either regime or M/Mtd must be provided")
#             else
#                 Mtd = sign.(M)
#             end
#         else
#             (Mtd , _) = P_P0_from_vertex(Bnc, regime; check=check)
#         end
#     end

#     return vcat(Mtd, Bnc.N)
# end









# fucntion c_mtx_C0_x(Bnc::Bnc; regime::Vertex,kwargs) 
#     if Bnc.regimes
#     else 

# c_mtx_C0_x(Bnc; regime=regime.regime, check=check)


# function x_ineq_mtx(Bnc::Bnc; regime::Vector{Int}, check::Bool=true)::Tuple{Matrix{Int}, Vector{Rational{Int64}}}
#     """
#     return a matrix of ineq in x space for regime expressed as Ax < 0
#     """
#     !check || _check_valid_idx(regime, Bnc.L) # check if the regime is valid for the given L matrix
#     idx = Set{Tuple{Int,Int}}()
#     for (valid_idx,rgm) in zip(Bnc._valid_L_idx, regime)
#         for i in valid_idx
#             if i != rgm
#             push!(idx, (rgm,i))
#             end
#         end
#     end
#     mtx = zeros(Int,length(idx), Bnc.n)
#     c0 = Vector{Rational{Int}}(undef, length(idx))
#     for (i, (rgm, j)) in enumerate(idx)
#         mtx[i, rgm] = -1
#         mtx[i, j] = 1
#         c0[i] = Bnc.L[i,rgm] // Bnc.L[i,j]
#     end
#     return mtx,c0
# end



# using Base.Threads
# function find_valid_regime_parallel(idx::Vector{Vector{Int}}, d::Int, n::Int)::Vector{Vector{Int}}
#     # A Channel is a thread-safe FIFO queue to collect results from all threads.
#     results_channel = Channel{Vector{Int}}(Inf)

#     # Use @threads to parallelize the loop over the first dimension's choices.
#     # Each thread will handle one or more initial choices from `idx[1]`.
#     @threads for v_initial in idx[1]
#         # --- THREAD-LOCAL STATE ---
#         # Each thread gets its own graph and choices array to prevent race conditions.
#         graph = [Vector{Int}() for _ in 1:n]
#         choices = Vector{Int}(undef, d)

#         # --- INITIAL SETUP FOR THIS THREAD ---
#         # Set the first choice and update the graph accordingly.
#         choices[1] = v_initial
#         target_nodes_initial = [w for w in idx[1] if w != v_initial]
#         for node in target_nodes_initial
#             push!(graph[node], v_initial)
#         end

#         # If the initial choice already creates a cycle, this path is invalid.
#         if have_cyclic_at_node(graph, v_initial, n)
#             continue # This thread moves to its next assigned v_initial.
#         end

#         # --- RECURSIVE BACKTRACKING (within a single thread) ---
#         # This recursive function is defined locally and closes over its
#         # thread-local `graph` and `choices`.
#         function backtrack_from_level2!(i)
#             # Base case: A valid configuration for all `d` dimensions is found.
#             if i == d + 1
#                 put!(results_channel, copy(choices))
#                 return
#             end

#             # Explore choices for the current level `i`.
#             for v in idx[i]
#                 target_nodes = [w for w in idx[i] if w != v]
                
#                 # 1. MAKE MOVE: Add edges for the current choice.
#                 for node in target_nodes
#                     push!(graph[node], v)
#                 end

#                 # 2. CHECK & RECURSE: If no cycle, recurse to the next level.
#                 if !have_cyclic_at_node(graph, v, n)
#                     choices[i] = v
#                     backtrack_from_level2!(i + 1)
#                 end

#                 # 3. BACKTRACK: Revert the graph to its previous state.
#                 for node in target_nodes
#                     pop!(graph[node])
#                 end
#             end
#         end

#         # Start the recursive search from the second level.
#         backtrack_from_level2!(2)
#     end

#     # All threads are finished, so no more results will be added.
#     close(results_channel)

#     # Collect all results from the channel into a final vector.
#     return collect(results_channel)
# end

# using Graphs
# function find_valid_regime(L::Matrix{Int})
#     # Slower may because of repeatedly inherently repeat check. 
#     (d,n) = size(L)
#     idx = [[idx for (idx, value) in enumerate(row) if value != 0] for row in eachrow(L) ] #!!! extremely key, avoid repeated add, or bug when removing it.
#     graph = SimpleDiGraph(n)
#     ict = IncrementalCycleTracker(graph, dir = :out)
#     choices = Vector{Int}(undef, d)
#     results = Vector{Vector{Int}}()
#     function backtrack!(i)
#         if i == d+1 
#             push!(results, copy(choices))  # 使用副本避免后续修改影响结果
#             return nothing
#         end

#         for v in idx[i]
#             target_nodes = [w for w in idx[i] if w != v && ~(w in outneighbors(graph,v))] # target_nodes
#             if add_edge_checked!(ict, v, target_nodes) # add successfully
#                 choices[i] = v
#                 backtrack!(i + 1)
#             end

#             for node in target_nodes
#                 rem_edge!(graph, v, node)
#             end
#         end
#     end
#     backtrack!(1)
#     return results
# end


#------------------------------------------------------------------------------
#   Function to finding dominence regime，one of the hardest part to optimize
# -------------------------------------------------------------------------------


# function have_cyclic_at_node(g::Vector{Vector{Int}}, node::Int, len::Int)::Bool
#     # DFS algorithm, 
#     # modified from https://github.com/JuliaGraphs/Graphs.jl/blob/2d6f4d56b06cb597ebd5c40c5a8db783f1b83991/src/traversals/dfs.jl#L4-L11
#     # 0 if not visited, 1 if in the current dfs path, 2 if fully explored
#     vcolor = zeros(UInt8, len)
#     vertex_stack = [node]
#     while !isempty(vertex_stack)
#         u = vertex_stack[end]
#         if vcolor[u] == 0
#             vcolor[u] = 1
#             for n in g[u]
#                 # we hit a loop when reaching back a vertex of the main path
#                 if vcolor[n] == 1
#                     return true
#                 elseif vcolor[n] == 0
#                     # we store neighbors, but these are not yet on the path
#                     push!(vertex_stack, n)
#                 end
#             end
#         else
#             pop!(vertex_stack)
#             if vcolor[u] == 1
#                 vcolor[u] = 2
#             end
#         end
#     end
#     return false
# end

# function find_all_vertices(idx::Vector{Vector{Int}}, d::Int, n::Int)::Vector{Vector{Int}}
#     graph = [Vector{Int}() for _ in 1:n]
#     choices = Vector{Int}(undef, d)
#     results = Vector{Vector{Int}}()
#     """
#     A backtracking algorithm to find all valid regimes given the index of valid choices for each row.
#     idx: valid choices index for each row,
#     d: number of rows of initial matrix,
#     n: number of columns of initial matrix, or we can say number of nodes for this application.
#     """
#     function backtrack!(i)
#         # All rows are fine
#         if i == d + 1
#             # @show choices
#             push!(results, copy(choices))
#             return nothing
#         end

#         for v in idx[i]
#             # add edges for current row. and record.

#             target_nodes = [w for w in idx[i] if w != v] # target_nodes

#             for node in target_nodes
#                 push!(graph[node], v) # add edge node -> v
#             end

#             if ~have_cyclic_at_node(graph, v, n)
#                 choices[i] = v
#                 backtrack!(i + 1)
#             end

#             for node in target_nodes
#                 pop!(graph[node])
#             end
#         end
#     end
#     backtrack!(1)
#     return results
# end

# function find_all_vertices(L::Matrix{Int}; kwargs...)
#     d, n = size(L)
#     idx = [[idx for (idx, value) in enumerate(row) if value != 0] for row in eachrow(L)] #[findall(!iszero, row) for row in eachrow(L)]
#     find_all_vertices(idx, d, n;kwargs...)
# end

# function pairwise_distance_cpu(data::Vector{<:AbstractVector{T}}) where {T}
#     n = length(data)
#     dist_matrix = zeros(T, n, n)
#     Threads.@threads for i in 1:n
#         for j in i+1:n
#             d = sum(data[i] .!= data[j])
#             dist_matrix[i,j] = d
#             dist_matrix[j,i] = d
#         end
#     end
#     return dist_matrix
# end
# function pairwise_distance_gpu(data::Vector{<:AbstractVector{T}}) where {T}
#     mat = CuArray(reduce(hcat,data))  # d × n, each column is a vertex
#     d, n = size(mat)
#     dist_matrix = CUDA.zeros(T, n, n)
#     SM, threads = GPU_SM_threads_num()
#     blocks = min(cld(n * n, threads), SM*2)
#     @cuda threads=threads blocks=blocks kernel_hamming!(dist_matrix, mat, n, d)
#     return Array(dist_matrix)
# end
# function kernel_hamming!(dist_matrix, mat, n, d)
#     # 1. Calculate a 1D global index and stride
#     # This is the standard grid-stride loop pattern
#     linear_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride = gridDim().x * blockDim().x
#     # 2. Loop over the entire workload
#     for k in linear_idx:stride:(n*n)
#         i = mod1(k, n)        # Column index (from 1 to n)
#         j = cld(k, n)         # Row index (from 1 to n)
#         # The core calculation remains the same
#         s = 0
#         @inbounds for l in 1:d
#             s += (mat[l,i] != mat[l,j])
#         end
#         dist_matrix[i,j] = s
#     end
#     return nothing
# end

# """
# Calculates the distance between two vertexes.
# """
# function get_vertices_distance!(Bnc::Bnc; use_gpu::Bool=true)
#     # Calculate the distance matrix for all vertices in Bnc
#     find_all_vertices!(Bnc) # Ensure vertices are calculated
#     if isempty(Bnc.vertices_distance)
#         if use_gpu
#             Bnc.vertices_distance = pairwise_distance_gpu(Bnc.vertices_perm)
#         else
#             Bnc.vertices_distance = pairwise_distance_cpu(Bnc.vertices_perm)
#         end
#     end
#     return Bnc.vertices_distance
# end

# function _get_Nρ_inv!(Bnc::Bnc{T},key::AbstractVector{<:Integer}) where T

#     function _calc_Nρ_inv(Nρ)
#         #calc [Nρ^{-1}] and if singular/non-square return nullity
#         #check if Nρ is square
#         r,r_ncol = size(Nρ)
#         if r != r_ncol # non-square_matrix
#             nullity = r - rank(Nρ)
#             return spzeros(0,0), nullity
#         else
#             Nρ_lu = lu(Nρ,check=false)
#             if issuccess(Nρ_lu)
#                 nullity = 0
#                 Nρ_inv = sparse(inv(Array(Nρ)))
#                 return Nρ_inv, nullity
#             else
#                 nullity = r - rank(Nρ)
#                 return spzeros(0,0), nullity
#             end
#         end
#     end

#     get!(Bnc._vertices_Nρ_inv_dict, key) do _
#         Nρ = @view Bnc.N[:,key]
#         _calc_Nρ_inv(Nρ)
#     end
# end