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

#------------------------------------------------------------------------------
#             Functions find all regimes
# ------------------------------------------------------------------------------

"""
    find_all_vertices(L; eps=1e-9, dominance_ratio=nothing, mode="auto")

Find all feasible "vertex" regimes for matrix `L` (d × n).
Returns Vector{Vector{Int}} where each inner Vector is a length-d assignment (1-based column indices).
Options:
- eps: small slack for weighted mode
- dominance_ratio: Float64
- asymtotic::Bool
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


function find_all_vertices(L::Matrix{Int} ; eps=1e-9, dominance_ratio=Inf, asymtotic::Union{Bool,Nothing}=nothing)

    asymtotic =  (isnothing(asymtotic) && dominance_ratio == Inf) || (asymtotic == true)
    eps = asymtotic ? nothing : (dominance_ratio == Inf ? eps : log(dominance_ratio))  # extra slack for weighted mode 
    if asymtotic
        return find_all_vertices_asym(L)
    else
        return find_all_vertices_nonasym(L,eps=eps)
    end
end

function calc_singularity(perm)
    length(perm) -length(Set(perm))
end
"""
find all vertices in Bnc, and store them in Bnc.vertices_perm.
With its idx Dict in Bnc.vertices_idx.
Also calculate if the vertex is real or fake, and its singularity.
"""
function find_all_vertices!(Bnc::Bnc;) # cheap enough for now
    if isempty(Bnc.vertices_perm) || isempty(Bnc.vertices_real_flag)
        Bnc.vertices_perm = find_all_vertices(Bnc.L; asymtotic=false)
        Bnc.vertices_idx = Dict(a=>idx for (idx, a) in enumerate(Bnc.vertices_perm)) # Map from vertex to its index
        real_vtx = Set(find_all_vertices(Bnc.L; asymtotic=true))
        Bnc.vertices_real_flag = Bnc.vertices_perm .∈ Ref(real_vtx)
        Bnc.vertices_singularity = Bnc.vertices_perm .|> calc_singularity
    end
    return Bnc.vertices_perm
end


#-----------------------------------------------------------------------------------
#   Functions to calculate the relationship between vertices
#-----------------------------------------------------------------------------------
"""
get vertices mapping dict
"""
function get_vertices_mapping_dict(Bnc::Bnc)
    find_all_vertices!(Bnc) # Ensure vertices are calculated
    return Bnc.vertices_idx
end

"""
Calculates the distance between two vertexes.
"""

function get_vertices_distance!(Bnc::Bnc)
    # Calculate the distance matrix for all vertices in Bnc
    find_all_vertices!(Bnc) # Ensure vertices are calculated
    
    function _distance_between(vtx1::Vector{T},vtx2::Vector{T})::T where T 
        return sum(vtx1 .!= vtx2)
    end
    
    if isempty(Bnc.vertices_distance)
        Bnc.vertices_distance = pairwise_distance(Bnc.vertices_perm, _distance_between; is_symmetric=true)
    end
    return Bnc.vertices_distance
end

"""
Calculate the x space change needed to change from one vertex to another.
The number idx of (i,j) deontes if regime i want to reach regime j, it shall rise its idx species.And also decrease its (j,i) species.
Source: row; Target: column.
"""
function get_vertices_change_dir_x!(Bnc::Bnc{T}) where T
    if !isempty(Bnc.vertices_change_dir_x)
        return Bnc.vertices_change_dir_x
    end
    d_mtx = get_vertices_distance!(Bnc)
    num_vertices = size(d_mtx, 1)
    # Initialize the change direction matrix
    vertices_change_dir_x = zeros(T, num_vertices, num_vertices)
    Threads.@threads for i in 1:num_vertices
        v_source = Bnc.vertices_perm[i]
        for j in (i+1):num_vertices
            if d_mtx[i, j] == 1 #neighbors
                v_target = Bnc.vertices_perm[j]
                for (x, y) in zip(v_source, v_target)
                    if x != y
                        vertices_change_dir_x[i, j] = y
                        vertices_change_dir_x[j, i] = x 
                        break
                    end
                end
            end
        end
    end
    Bnc.vertices_change_dir_x = vertices_change_dir_x
    return vertices_change_dir_x
end


#-------------------------------------------------------------------------------------
#         fucntions with single vertex and lazy calculate  its properties
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

# """
# Creates the C and C0 matrices from a permutation.
# """
# function _calculate_C_C0_x(Bnc::Bnc{T}, perm::Vector{<:Integer}) where T
#     """
#     return a matrix of ineq in x space for regime expressed as Clogx+ c0> 0
#     (logic seems to be complicate.)
#     """
#     # , check::Bool=true
#     # !check || _check_valid_idx(regime, Bnc.L) # check if the regime is valid for the given L matrix
    
#     num_ineq = Bnc._val_num_L - Bnc.d # inequalities counts from L's number of values minus d.
#     c_mtx = zeros(Int, num_ineq, Bnc.n) # initialize the c_mtx
#     c0 = Vector{Float64}(undef, num_ineq) # initialize the c0 vector
#     row_ptr = Bnc._Lt_sparse.colptr .- (0:Bnc.d) # From _Lt_sparse.colptr, we can get the row start index for each original row.

#     for (i,valid_idx,rgm,row_block_start) in zip(1:Bnc.d, Bnc._valid_L_idx, perm, row_ptr)
#         # Within block
#         row = row_block_start # current row index in c_mtx, start from the block start index.
#         for col in valid_idx
#             if col != rgm
#                 # Calculate the correct row index for the output matrix
#                 c_mtx[row, col] = -1
#                 c_mtx[row, rgm] = 1
#                 # c0[row] = log10(Bnc.L[i, rgm] / Bnc.L[i, col])
#                 c0[row] = log10(Bnc.L[i, rgm] // Bnc.L[i, col])
#                 row += 1
#             end
#         end
#     end
#     return c_mtx,c0
# end
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
# function _calculate_P_P0_sym(Bnc::Bnc, perm::Vector{Int})::Tuple{Matrix{Int}, Vector{Num}}
#     P = zeros(Int, Bnc.d, Bnc.n)
#     P0 = Vector{Num}(undef, Bnc.d)
#     for i in 1:Bnc.d
#         P[i, perm[i]] = 1
#         P0[i] = log10_sym(Bnc.L[i, perm[i]])
#     end
#     return P, P0
# end
# function _calculate_C_C0_x_sym(Bnc::Bnc,perm::Vector{Int})::Tuple{Matrix{Int}, Vector{Num}}
#     """
#     return a matrix of ineq in x space for regime expressed as Clogx+ c0> 0
#     (logic seems to be complicate.)
#     """
#     # , check::Bool=true
#     # !check || _check_valid_idx(regime, Bnc.L) # check if the regime is valid for the given L matrix
    
#     num_ineq = Bnc._val_num_L - Bnc.d # inequalities counts from L's number of values minus d.
#     c_mtx = zeros(Int, num_ineq, Bnc.n) # initialize the c_mtx
#     c0 = Vector{Num}(undef, num_ineq) # initialize the c0 vector
#     row_ptr = Bnc._Lt_sparse.colptr .- (0:Bnc.d) # From _Lt_sparse.colptr, we can get the row start index for each original row.

#     for (i,valid_idx,rgm,row_block_start) in zip(1:Bnc.d, Bnc._valid_L_idx, perm, row_ptr)
#         # i: block_index, each original L's row is a block.
#         # valid_idx: all the valid indices for the current block, Vector{Vector{Int}}.
#         # rgm: the current regime index.
#         # row_block_start: the start row idx for the current block in c_mtx.
#         # k = 0 # finished rows count for the current block, used to update row.
        
#         # Within block
#         row = row_block_start # current row index in c_mtx, start from the block start index.
#         for col in valid_idx
#             if col != rgm
#                 # Calculate the correct row index for the output matrix
#                 c_mtx[row, col] = -1
#                 c_mtx[row, rgm] = 1
#                 # c0[row] = log10(Bnc.L[i, rgm] / Bnc.L[i, col])
#                 c0[row] = log10_sym(Bnc.L[i, rgm] // Bnc.L[i, col])
#                 row += 1
#             end
#         end
#     end
#     return c_mtx,c0
# end




"""
Creates a new, partially-filled Vertex object.
This function performs the initial, less expensive calculations.
"""
function _create_vertex(Bnc::Bnc{T}, perm::Vector{<:Integer})::Vertex where T
    find_all_vertices!(Bnc)

    idx = Bnc.vertices_idx[perm]
    real = Bnc.vertices_real_flag[idx] # Check if the vertex is real or fake
    singularity = Bnc.vertices_singularity[idx] # Get the singularity of the vertex
     # Index of the vertex in the Bnc.vertices_perm list
    P, P0 = _calculate_P_P0(Bnc, perm); 
    C_x, C0_x = _calculate_C_C0_x(Bnc, perm)

    F = eltype(P0)

    M = vcat(P, Bnc._N_sparse)
    M0 = vcat(P0, zeros(F, Bnc.r))
    # Initialize a partial vertex. "Full" properties are empty placeholders.
    return Vertex{F,T}(
        singularity = singularity,
        idx = idx,
        perm = perm, 
        real = real,
        M = M, M0 = M0, P = P, P0 = P0, C_x = C_x, C0_x = C0_x
    )
end

function _ensure_full_properties!(vtx::Vertex)
    # Check if already calculated
    if !isempty(vtx.H)
        return
    end
    if vtx.singularity == 0
        vtx.H = inv(vtx._M_lu) # Calculate the inverse matrix from pre-computed LU decomposition of M
        vtx.H0 = vtx.H * vtx.M0
        vtx.C_qK = vtx.C_x * vtx.H
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
        _ensure_full_properties!(vtx)
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





function get_vertex_idx(Bnc,perm::Vector{Int})
    find_all_vertices!(Bnc)
    return Bnc.vertices_idx[perm]
end

function get_vertex_idx(Bnc::Bnc, idx::Int)
   return idx
end



function get_all_neighbors!(Bnc::Bnc, perm)
    # Get the neighbors of the vertex represented by perm
    vtx = get_vertex!(Bnc, perm ; full=false)
    if isempty(vtx.neighbors_idx)
        d_mat = get_vertices_distance!(Bnc)

        vtx.neighbors_idx = findall(d_mat[vtx.idx, :] .== 1)

        finite_neighbors = Int[]
        infinite_neighbors = Int[]
        for idx in vtx.neighbors_idx
            if get_singularity!(Bnc, Bnc.vertices_perm[idx]) == 0
                push!(finite_neighbors, idx)
            else
                push!(infinite_neighbors, idx)
            end
        end

        vtx.finite_neighbors_idx = finite_neighbors
        vtx.infinite_neighbors_idx = infinite_neighbors
    end
    return Bnc.vertices_perm[vtx.neighbors_idx]
end

function get_finite_neighbors!(Bnc::Bnc, perm)
    get_all_neighbors!(Bnc, perm)
    return Bnc.vertices_perm[vtx.finite_neighbors_idx]
end

function get_infinite_neighbors!(Bnc::Bnc, perm)
    vtx = get_vertex!(Bnc, perm)
    if isempty(vtx.neighbors_idx)
        get_all_neighbors!(Bnc, perm)
    end
    return Bnc.vertices_perm[vtx.infinite_neighbors_idx]
end

"""
Gets P and P0, creating the vertex if necessary.
"""
function get_P_P0!(Bnc::Bnc, perm)
    vtx = get_vertex!(Bnc, perm; full=false)
    return vtx.P, vtx.P0
end

"""
Gets M and M0, creating the vertex if necessary.
"""
function get_M_M0!(Bnc::Bnc, perm)
    vtx = get_vertex!(Bnc, perm; full=false)
    return vtx.M, vtx.M0
end

"""
Gets C_x and C0_x, creating the vertex if necessary.
"""
function get_C_C0_x!(Bnc::Bnc, perm)
    vtx = get_vertex!(Bnc, perm; full=false)
    return vtx.C_x, vtx.C0_x
end

"""
Gets C_qK and C0_qK, ensuring the full vertex is calculated.
"""
function get_C_C0_qK!(Bnc::Bnc, perm)
    vtx = get_vertex!(Bnc, perm; full=false)
    _ensure_full_properties!(vtx)
    if vtx.singularity >= 2
        @error("Vertex got singluarity $(vtx.singularity), cannot get C_qK and C0_qK")
    end
    return vtx.C_qK, vtx.C0_qK
end

function get_singularity!(Bnc::Bnc,perm)
    vtx = get_vertex!(Bnc, perm)
    if vtx.singularity == -1 # non-singular, uninitialized
        _ensure_full_properties!(vtx)
    end
    return vtx.singularity
end


"""
Gets H and H0, ensuring the full vertex is calculated.
"""
function get_H_H0!(Bnc::Bnc, perm::Vector{Int})
    vtx = get_vertex!(Bnc, perm)
    _ensure_full_properties!(vtx)
    if vtx.singularity > 0
        @error("Vertex is singular, cannot get H0")
    end # This will compute if needed
    return vtx.H, vtx.H0
end

function get_H!(Bnc::Bnc, perm::Vector{Int})
    vtx = get_vertex!(Bnc, perm)
    _ensure_full_properties!(vtx)
    if vtx.singularity > 1
        @error("Vertex's singularity is bigger than 1, cannot get H")
    end # This will compute if needed
    return vtx.H
end

#-------------------------------------------------------------------------------------
#         fucntions of getting vertex with certein properties
# ------------------------------------------------------------------------------------

function get_all_vertices_singularity!(Bnc::Bnc)
    """
    Calculate the singularity of all vertices in Bnc.
    """
    if isempty(Bnc.vertices_singularity)
        vtx = find_all_vertices!(Bnc)
        Bnc.vertices_singularity = [get_singularity!(Bnc, v) for v in vtx]
    end
    return Bnc.vertices_singularity
end

function get_singular_vertices_idx(Bnc::Bnc)
    """
    Get the indices of all singular vertices in Bnc.
    """
    singularity = get_all_vertices_singularity!(Bnc)
    return findall(!iszero, singularity)
end

function get_nonsingular_vertices_idx(Bnc::Bnc)
    """
    Get the indices of all singular vertices in Bnc.
    """
    singularity = get_all_vertices_singularity!(Bnc)
    return findall(iszero, singularity)
end

function get_singular_vertex(Bnc::Bnc)
    idx = get_singular_vertices_idx!(Bnc)
    return Bnc.vertices_perm[idx]
end

function get_nonsingular_vertex(Bnc::Bnc)
    idx = get_nonsingular_vertices_idx!(Bnc)
    return Bnc.vertices_perm[idx]
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