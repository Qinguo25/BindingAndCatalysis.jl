#--------------Core computation functions-------------------------
function _vtxs_nonasym(L, ::Val{T} ;eps=1e-9) where T
    d, n = size(L)
    # T = get_int_type(n)  # Determine integer type based on n
    # Nonzero indices of each row
    J = [findall(!iszero, row) for row in eachrow(L)]
    
    order = sortperm(J, by=length, rev=true)
    inv_order = invperm(order)
    J_ord = J[order]

    # Precompute VertexEdge weights: Dict[u => edges] for each row
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
function _calc_Nρ_inv(Nρ) # Core function to calculate Nρ_inv or adj matrix
    r, r_ncol = size(Nρ)
    if r != r_ncol
        return spzeros(0,0), r - rank(Nρ)
    end
    Nρ_lu = lu(Nρ; check=false)
    if issuccess(Nρ_lu)
        return sparse(inv(Array(Nρ))), 0
    else
        return _adj_singular_matrix(Nρ)
    end
end
function _calc_vertices_graph(data::Vector{<:AbstractVector{T}})::VertexGraph where {T}
    n_vtxs = length(data)
    n=length(data[1])

    # Tuple(source, VertexEdge)
    thread_edges = [Vector{Tuple{Int, VertexEdge{T}}}() for _ in 1:Threads.nthreads()] # threads_num of eges.

    Threads.@threads for i in 1:n_vtxs
        tid = Threads.threadid() #threadid
        local_edges = thread_edges[tid] # Vector{Tuple{Int, VertexEdge{T}}}()
        vi = data[i] # source

        for j in i+1:n_vtxs
            vj = data[j] # target
            dist = 0 
            diff_r = 0
            @inbounds for k in eachindex(vi, vj)
                if vi[k] != vj[k]
                    dist += 1
                    diff_r = k
                    if dist > 1
                        break
                    end
                end
            end

            if dist == 1
                # calculate the dx
                x, y = vi[diff_r], vj[diff_r]
                dx = x < y ? SparseVector(n, [x,y], Int8[-1,1]) :
                             SparseVector(n, [y,x], Int8[1,-1])

                # 双向记录（反对称）
                push!(local_edges, (i, VertexEdge(j, diff_r, dx)))
                push!(local_edges, (j, VertexEdge(i, diff_r, -dx)))
            end
        end
    end

    all_edges = reduce(vcat, thread_edges)
    neighbors = [Vector{VertexEdge{T}}() for _ in 1:n_vtxs]

    for (from, e) in all_edges
        push!(neighbors[from], e)
    end

    return VertexGraph(neighbors)
end

function _calc_H(Bnc::Bnc,perm::Vector{<:Integer})
    key = _get_Nρ_key(Bnc, perm)
    Nρ_inv,nullity = _get_Nρ_inv!(Bnc,key) # get Nρ_inv from cache or calculate it. # sparse matrix
    Nc = @view Bnc.N[:,perm] # dense matrix
    Nρ_inv_Nc_neg = - Nρ_inv * Nc 
    
    H_un_perm = if nullity == 0 
         [[I(Bnc.d) zeros(Bnc.d,Bnc.r)];
         [Nρ_inv_Nc_neg Nρ_inv]]
    elseif nullity == 1
        [zeros(Bnc.d,Bnc.n);
        [Nρ_inv_Nc_neg Nρ_inv]] # Seems there are sign problem within this part.(fixed)
    else
        error("Nullity greater than 1 not supported")
    end
    perm_inv = invperm([perm;key]) # get the inverse permutation to reorder H
    H = H_un_perm[perm_inv, :]
    return H
end

function _calculate_P_P0(Bnc::Bnc{T}, perm::Vector{<:Integer}) where T
    """
    Creates the P and P0 matrices from a permutation.
    """
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

function _calculate_C_C0_x(Bnc::Bnc{T}, perm::Vector{<:Integer}) where T # highly optimized version
    """
    Creates the C and C0 matrices from a permutation.
    """
    # This fucntion is created by chatgpt with numeric verification from dense version.
    # Is the lowest level, maximum-speed version.
    num_ineq = length(Bnc._L_sparse.nzval) - Bnc.d
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
function _calc_C_C0_qK_singular(Bnc::Bnc, vtx)
    M,M0 = get_M_M0!(Bnc,vtx)
    C,C0 = get_C_C0_x!(Bnc,vtx)
    # n = Bnc.n
    poly_x = hrep(-C,C0) |> x->polyhedron(x,CDDLib.Library())
    poly_elim = M * poly_x
    rlt = MixedMatHRep(hrep(poly_elim))
    A, b, linset = (rlt.A, rlt.b, rlt.linset)
    # @show linset
    @assert linset == BitSet(1:maximum(linset)) "linear rows are not the first top n rows, code fix is needed"
    # perm = [collect(linset) ; [i for i in 1:size(A,1) if i ∉ linset]]
    CqK = sparse(-A) |> x->droptol!(x,1e-10)
    C0qK = (b+A*M0)
    return CqK, C0qK
end

#------------------Storage layer functions -----------------------------

function _build_Nρ_cache_parallel!(Bnc::Bnc{T}) where T
    perm_set = Set(Set(perm) for perm in Bnc.vertices_perm) # Unique sets of permutations
    keys = [_get_Nρ_key(Bnc, perm) for perm in perm_set]

    nk = length(keys)
    inv_list = Vector{SparseMatrixCSC{Float64,Int}}(undef, nk)
    nullity_list = Vector{T}(undef, nk)

    Threads.@threads for i in eachindex(keys)
        key = keys[i]
        Nρ = @view Bnc.N[:, key]
        inv_list[i], nullity_list[i] = _calc_Nρ_inv(Nρ)
    end

    for i in eachindex(keys)
        Bnc._vertices_Nρ_inv_dict[keys[i]] = (inv_list[i], nullity_list[i])
    end
    return nothing
end
function _get_Nρ_inv!(Bnc::Bnc{T}, key::AbstractVector{<:Integer}) where T
    get!(Bnc._vertices_Nρ_inv_dict, key) do
        Nρ = @view Bnc.N[:, key]
        _calc_Nρ_inv(Nρ)
    end
end
function _build_vertices_graph!(Bnc::Bnc)
    find_all_vertices!(Bnc) # Ensure vertices are calculated
    if isnothing(Bnc.vertices_graph)
        print("Start calculating verteices graph, It may takes a while.\n")
        Bnc.vertices_graph = _calc_vertices_graph(Bnc.vertices_perm)
        print("Done.\n")
    end
end
function _fulfill_vertices_graph!(Bnc::Bnc)::VertexGraph
    """
    fill the qK space change dir matrix for all vertices in Bnc.
    """
    vtx_graph = get_vertices_graph!(Bnc;fulfill=false)
    if vtx_graph.change_dir_qK_computed
        return vtx_graph
    end
    function _calc_change_dir_qK(Bnc, vi, vj, i, j1, j2)
        n1 = get_nullity!(Bnc, vi)
        n2 = get_nullity!(Bnc, vj)
        if n1 > 1 || n2 > 1
            return nothing
        end

        # unit vector (Float64) at position i, reused where needed
        ei = SparseVector(Bnc.n, [i], [1])

        if n1 == 0
            H1 = get_H!(Bnc, vi)
            dir = H1[j2, :] - ei
        elseif n2 == 0
            H2 = get_H!(Bnc, vj)
            dir = ei - H2[j1, :]
        else
            # n1 == 1 && n2 == 1
            H1 = get_H!(Bnc, vi)
            dir = H1[j2, :]
        end

        return droptol!(dir, 1e-10)
    end
    Threads.@threads for (vi, edges) in enumerate(vtx_graph.neighbors)
        if get_nullity!(Bnc,vi) > 1 # jump off those regimes with nullity >1
            continue
        end
        for e in edges

            if !isnothing(e.change_dir_qK) # pass if have been computed
                continue
            end

            # from vi to vj, and change happens on ith row that "1" goes from j1 position to j2 position.
            vj = e.to # target 
            i = e.diff_r # different row
            I,V = findnz(e.change_dir_x) # should be two elements
            (j1,j2) = V[1] > V[2] ? (I[2], I[1]) : (I[1], I[2])
            
            # calculate their direction based on formula
            dir = _calc_change_dir_qK(Bnc, vi, vj,i,j1,j2)
            e.change_dir_qK = dir
        end
    end
    vtx_graph.change_dir_qK_computed = true
end

#------------------Helper functions -------------------------------------------
function _get_Nρ_key(Bnc::Bnc{T}, perm)::Vector{T} where T 
   return [i for i in 1:Bnc.n if i ∉ perm]
end
_get_Nρ_inv_from_perm!(Bnc, perm) = _get_Nρ_inv!(Bnc, _get_Nρ_key(Bnc, perm))

function _vertexgraph_to_sparsematrix(G::VertexGraph{T}; weight_fn = e -> 1) where T
    n = length(G.neighbors)
    # 预分配估计：平均度 × n
    nnz = sum(length(v) for v in G.neighbors)
    I = Vector{Int}(undef, nnz)
    J = Vector{Int}(undef, nnz)
    V = Vector{Float64}(undef, nnz)
    idx = 0
    for i in 1:n
        for e in G.neighbors[i]
            idx += 1
            I[idx] = i
            J[idx] = e.to
            V[idx] = weight_fn(e)
        end
    end
    return sparse(I,J,V, n, n)
end

function _create_vertex(Bnc::Bnc{T}, perm::Vector{<:Integer})::Vertex where T
    """
    Creates a new, partially-filled Vertex object.
    This function performs the initial, less expensive calculations.
    """
    find_all_vertices!(Bnc)
    idx = Bnc.vertices_perm_dict[perm] # Index of the vertex in the Bnc.vertices_perm list
    real = Bnc.vertices_asymptotic_flag[idx] # Check if the vertex is real or fake
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

function _ensure_full_properties!(Bnc::Bnc, vtx::Vertex)
    # Check if already calculated
    if !isempty(vtx.H)
        return nothing
    end
    if vtx.nullity == 0
        # H = inv(Array(vtx.M)) # dense matrix 
        H = _calc_H(Bnc, vtx.perm)
        vtx.H = droptol!(sparse(H),1e-10) # Calculate the inverse matrix from pre-computed LU decomposition of M
        vtx.H0 = H * vtx.M0
        vtx.C_qK = droptol!(sparse(vtx.C_x * H),1e-10)
        vtx.C0_qK = vtx.C0_x - vtx.C_x * vtx.H0 # Correctly use vtx.C0_x
    else
        if vtx.nullity ==1
            # we need to check where this nullity comes from.
            if length(Set(vtx.perm)) == Bnc.d # the nullity comes from N
                H = _calc_H(Bnc, vtx.perm) 
                vtx.H = droptol!(sparse(H),1e-10)#.* Bnc.direction 
            else # the nullity comes from P
                H = _adj_singular_matrix(vtx.M)[1]
                vtx.H = droptol!(sparse(H),1e-10)#.* Bnc.direction
            end
        else # nullity>1 , H, HO is nolonger avaliable
            vtx.H = spzeros(Bnc.n, Bnc.n) # fill value as a sign that this regime is fully computed
        end
        vtx.C_qK, vtx.C0_qK = _calc_C_C0_qK_singular(Bnc, vtx.perm)
    end
end









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

find_all_vertices_asym(L;kwargs...) = _vtxs_asym(L,Val(get_int_type(size(L)[2]));kwargs...)

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
With its idx Dict in Bnc.vertices_perm_dict.
Also calculate if the vertex is real or fake, and its nullity.
"""
function find_all_vertices!(Bnc::Bnc{T};) where T # cheap enough for now
    if isempty(Bnc.vertices_perm) || isempty(Bnc.vertices_asymptotic_flag)
        print("Start finding all vertices, it may takes a while.\n")
        
        # all vertices
        # finding non-asymptotic vettices, which gives all vertices both real and fake, singular and non-singular
        Bnc.vertices_perm = find_all_vertices(Bnc.L; asymptotic=false)
        # Create the idx for each vertex
        Bnc.vertices_perm_dict = Dict(a=>idx for (idx, a) in enumerate(Bnc.vertices_perm)) # Map from vertex to its index
        # finding asymptotic vertices, which is the real vertices.

        # asymptotic vertices
        real_vtx = Set(find_all_vertices(Bnc.L; asymptotic=true))
        Bnc.vertices_asymptotic_flag = Bnc.vertices_perm .∈ Ref(real_vtx)
        print("Done, with $(length(Bnc.vertices_perm)) vertices found and $(length(real_vtx)) asymptotic vertices.\n")
        println("Start calculating nullity for each vertex, it may takes a while.")
        # build Nρ_inv cache in parallel
        _build_Nρ_cache_parallel!(Bnc)
        # Caltulate the nullity for each vertices
        nullity = Vector{T}(undef, length(Bnc.vertices_perm))
        Threads.@threads for i in  1:length(Bnc.vertices_perm)
            perm_set = Set(Bnc.vertices_perm[i])
            nullity_P =  Bnc.d -length(perm_set)
            _ , nullity_N =  _get_Nρ_inv_from_perm!(Bnc,perm_set) 
            nullity[i] = nullity_P + nullity_N # this is true as we can permute the matrix into diagnal block matrix.
        end
        # @show nullity
        Bnc.vertices_nullity = nullity
        println("Done.")
    end
    return Bnc.vertices_perm
end




"""
Return a dict with key: vertex perm, value: its index in Bnc.vertices_perm
"""
function get_vertices_perm_dict(Bnc::Bnc)
    """
    get vertices mapping dict
    """
    find_all_vertices!(Bnc) # Ensure vertices are calculated
    return Bnc.vertices_perm_dict
end

"""
Get the nullity of all vertices in Bnc.
"""
function get_vertices_nullity(Bnc::Bnc)
    """
    Calculate the nullity of all vertices in Bnc.
    """
    find_all_vertices!(Bnc)
    return Bnc.vertices_nullity
end

#---------------------------------------------------------------------------------------------
#   Functions involving vertices relationships, (neighbors finding and changedir finding)
#---------------------------------------------------------------------------------------------

"""
get the neighbor of vertices formed graph.
"""
function get_vertices_graph!(Bnc::Bnc; fulfill::Bool=false)::VertexGraph
    """
    get the neighbor of vertices formed graph.
    """
    if fulfill
        _fulfill_vertices_graph!(Bnc)
    else
        _build_vertices_graph!(Bnc)
    end
    return Bnc.vertices_graph
end

function get_vertices_neighbor_mat!(Bnc::Bnc)
    """
    # find the x space neighbor of all vertices in Bnc, the value denotes for two perms, which row they differ at.
    """
    grh = get_vertices_graph!(Bnc;fulfill=true)
    spmat = _vertexgraph_to_sparsematrix(grh; weight_fn = e -> 1)
    return spmat
end


function get_change_dir_x(Bnc::Bnc, from, to)
    from = get_idx(Bnc, from)
    to = get_idx(Bnc, to)
    vtx_grh = get_vertices_graph!(Bnc)
    for VertexEdge in vtx_grh.neighbors[from]
        if VertexEdge.to == to
            return VertexEdge.change_dir_x
        end
    end
    @error("Vertices $get_perm(Bnc, from) and $get_perm(Bnc, to) are not neighbors in x space.")
end


# function get_change_dir_qK(Bnc::Bnc, from, to)
#     from = get_idx(Bnc, from)
#     to = get_idx(Bnc, to)
#     d_mat = get_vertices_change_dir_qK!(Bnc)
#     if from < to
#         return d_mat[from, to]
#     else
#         return -d_mat[to, from]
#     end
# end


function vertexgraph_to_graph(vg::VertexGraph)
    n = length(vg.neighbors)
    g = SimpleDiGraph(n)
    for (i, edges) in enumerate(vg.neighbors)
        for e in edges
            add_edge!(g, i, e.to)
        end
    end
    return g
end

function vertexgraph_to_graph(Bnc::Bnc)
    vtx_graph = get_vertices_graph!(Bnc)
    return vertexgraph_to_graph(vtx_graph)
end




#-------------------------------------------------------------------------------------
#         functions involving single vertex and lazy calculate  its properties
# ------------------------------------------------------------------------------------
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
    return Bnc.vertices_perm_dict[perm]
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
        vtx_grh = get_vertices_graph!(Bnc)
        vtx.neighbors_idx = vtx_grh.neighbors[vtx.idx] .|> e -> e.to
    end

    idx = vtx.neighbors_idx
    return return_idx ? idx : Bnc.vertices_perm[idx]
end
function get_finite_neighbors(Bnc::Bnc, perm; return_idx::Bool=false)
    nb_idx = get_all_neighbors!(Bnc, perm; return_idx=true)
    idx = filter(i->Bnc.vertices_nullity[i] == 0, nb_idx)
    return return_idx ? idx : Bnc.vertices_perm[idx]
end
function get_infinite_neighbors(Bnc::Bnc, perm; return_idx::Bool=false) # nullity_max::Union{Int,Nothing}=nothing
    nb_idx = get_all_neighbors!(Bnc, perm; return_idx=true)
    idx = filter(i->Bnc.vertices_nullity[i] > 0, nb_idx)
    return return_idx ? idx : Bnc.vertices_perm[idx]
end
function get_neighbors(Bnc::Bnc, perm; singular::Union{Bool,Nothing}=nothing, asymptotic::Union{Bool,Nothing}=nothing, return_idx::Bool=false)
    """
    Get the neighbors of a vertex with certain properties.
    - singular: true for singular, false for non-singular, nothing for all
    - asymptotic: true for real, false for fake, nothing for all
    """
    idx = get_all_neighbors!(Bnc, perm; return_idx=true)
    idx = filter(i -> (
        (isnothing(singular) || (singular == (Bnc.vertices_nullity[i] > 0))) &&
        (isnothing(asymptotic) || (asymptotic == Bnc.vertices_asymptotic_flag[i]))
    ), idx)
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
    if vtx.nullity >= 2
        
        # @error("Vertex got nullity $(vtx.nullity), currently doesn't support get C_qK and C0_qK")
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
    if vtx.nullity > 0
        @error("Vertex is singular, cannot get H0")
    end # This will compute if needed
    _ensure_full_properties!(Bnc,vtx)
    return vtx.H, vtx.H0
end

function get_H!(Bnc::Bnc, perm)
    vtx = get_vertex!(Bnc, perm; full=false)
    if vtx.nullity > 1
        @error("Vertex's nullity is bigger than 1, cannot get H")
    end # This will compute if needed
    _ensure_full_properties!(Bnc,vtx)
    return vtx.H
end
get_H0!(Bnc::Bnc, perm) = get_H_H0!(Bnc, perm)[2]


function get_polyhedra(Bnc::Bnc,perm)::Polyhedron 
    A, b = get_C_C0_qK!(Bnc,perm)
    nullity = get_nullity!(Bnc,perm)
    if nullity ==0
        return hrep(-A,b) |> x-> polyhedron(x,CDDLib.Library())
    else
        linset = BitSet(1:nullity)
        return hrep(-A,b,linset) |> x-> polyhedron(x,CDDLib.Library())
    end
end


function is_neighbor_direct(Bnc::Bnc,vtx1,vtx2)::Bool
    if get_nullity!(Bnc, vtx1) > 1 || get_nullity!(Bnc, vtx2) > 1
        @warn "Currently we doesn't care neighbor relationships less than your model's dim - 1  ,return false by default"
        return false
    end
    p1 = get_polyhedra(Bnc, vtx1)
    p2 = get_polyhedra(Bnc, vtx2)
    p = intersect(p1,p2)
    # detecthlinearity!(p)
    # if nhyperplanes(p) > null || isempty(p)
    if dim(p)==Bnc.n-1 
        return true
    else
        return false
    end
end

function is_neighbor(Bnc::Bnc,vtx1,vtx2)::Bool
    @assert get_nullity!(Bnc, vtx1) <= 1 "Currently we only support neighbor detection for vertices with nullity less than or equal to 1"
    @assert get_nullity!(Bnc, vtx2) <= 1 "Currently we only support neighbor detection for vertices with nullity less than or equal to 1"
    nbs = get_all_neighbors!(Bnc, vtx1; return_idx=true)
    idx2 = get_idx(Bnc, vtx2)
    if idx2 in nbs
        return true
    else
        return is_neighbor_direct(Bnc,vtx1,vtx2)
    end
end



#-------------------------------------------------------------------------------------
#         functions of getting vertices with certain properties
# -------------------------------------------------------------------------------------

function get_singular_vertices(Bnc::Bnc; return_idx::Bool=false)
    """
    Get the indices of all singular vertices.
    Default to return perms
    """
    idx = findall(!iszero, get_vertices_nullity(Bnc))
    return return_idx ? idx : Bnc.vertices_perm[idx]
end
get_singular_vertices_idx(Bnc::Bnc) = get_singular_vertices(Bnc; return_idx=true)
function get_nonsingular_vertices(Bnc::Bnc; return_idx::Bool=false)
    """
    Get the indices of all nonsingular vertices.
    """
    nullity = get_vertices_nullity(Bnc)
    idx =  findall(iszero, nullity)
    return return_idx ? idx : Bnc.vertices_perm[idx]
end
get_nonsingular_vertices_idx(Bnc::Bnc) = get_nonsingular_vertices(Bnc; return_idx=true)
function get_real_vertices(Bnc::Bnc; return_idx::Bool=false)
    """
    Get the idx of all real vertices.
    """
    find_all_vertices!(Bnc)
    idx = findall(Bnc.vertices_asymptotic_flag)
    return return_idx ? idx : Bnc.vertices_perm[idx]
end
get_real_vertices_idx(Bnc::Bnc) = get_real_vertices(Bnc; return_idx=true)
function get_fake_vertices(Bnc::Bnc; return_idx::Bool=false)
    """
    Get the idx of all fake vertices.
    """
    find_all_vertices!(Bnc)
    idx = findall(!, Bnc.vertices_asymptotic_flag)
    return return_idx ? idx : Bnc.vertices_perm[idx]
end
get_fake_vertices_idx(Bnc::Bnc) = get_fake_vertices(Bnc; return_idx=true)
function get_vertices(Bnc::Bnc; singular::Union{Bool,Nothing}=nothing, asymptotic::Union{Bool,Nothing}=nothing, return_idx::Bool=false)
    """
    get vertices with certain properties.
    """ 
    singular_flag = isnothing(singular) ? trues(length(Bnc.vertices_perm)) : (singular .== (get_vertices_nullity(Bnc) .> 0))
    real_flag = isnothing(asymptotic) ? trues(length(Bnc.vertices_perm)) : (asymptotic .== Bnc.vertices_asymptotic_flag)
    flag = singular_flag .& real_flag
    idx = findall(flag)
    return return_idx ? idx : Bnc.vertices_perm[idx]
end







#-------------------------------------------------------------
# Functions using Polyhedra.jl  to calculate and fufill the 
#polyhedron helper functions
#--------------------------------------------------
function hyperplane_project_func(polyhedra::T)::Function where T<:Polyhedron
    if !hashyperplanes(polyhedra)
        error("polyhedra doesn't have hyperplanes")
    end
    # A^⊤y =b to project to this subspace   
    A = stack([i.a for i in hyperplanes(polyhedra)])
    b = stack([i.β for i in hyperplanes(polyhedra)])
    @show A,b
    # Now we need to generate a function to project a point into this hyperplanes
    AAtA_inv = A*pinv(A'*A)
    b0 = AAtA_inv*b
    P0 = I(size(A,1))-AAtA_inv*A'
    return x -> P0*x+b0
end
function get_one_inner_point(Bnc::Bnc,perm;kwargs...)
    poly = get_polyhedra(Bnc,perm)
    return get_one_inner_point(poly;kwargs...)
end

function get_one_inner_point(poly::T;rand_line=true,rand_ray=true) where T<:Polyhedron
    vrep_poly = MixedMatVRep(vrep(poly))
    point = [mean(p) for p in eachcol(vrep_poly.V)]
    ray_avg = zeros(size(point,1))
    for (i, ray) in enumerate(eachrow(vrep_poly.R))
        if i ∉ vrep_poly.Rlinset
            norm_ray = norm(ray)
            sigma = rand_ray ? rand()-0.5 : 0
            ray_avg .+= (ray ./ norm_ray .* (1+sigma) )
        else
            if rand_line
                norm_ray = norm(ray)
                sigma = rand()-0.5
                ray_avg .+= (ray ./ norm_ray * sigma)
            end
        end
    end
    norm_ray_avg = norm(ray_avg)
    @. ray_avg = ray_avg / norm_ray_avg .* 3
    return (point.+ ray_avg)
end

#-------------------------------------------------------------
#Other higher lever functions
#----------------------------------------------------------------
function summary_vertex(Bnc::Bnc, perm)
    idx= get_idx(Bnc, perm)
    perm = get_perm(Bnc, idx)
    is_real = get_vertex!(Bnc, idx).real
    nullity = get_nullity!(Bnc, idx)
    println("idx=$idx,perm=$perm, is_real=$is_real, nullity=$nullity")
    return nothing
end
function summary_vertices(Bnc::Bnc;kwargs...)
    vtx = get_vertices(Bnc;kwargs...)
    vtx .|> x->summary_vertex(Bnc,x)
    return nothing
end


# function _calc_C_C0_qK_singular(model, vtx)
#     M,M0 = get_M_M0!(model,vtx)
#     C,C0 = get_C_C0_x!(model,vtx)
#     n = model.n
#     n_C = size(C0,1)
#     rg_y = BitSet(1:n)
#     poly = hrep([[-M I(n)];[-C zeros(n_C, n)]], [M0;C0],rg_y) |> x->polyhedron(x,CDDLib.Library())
#     poly_elim = eliminate(poly,rg_y)
#     rlt = MixedMatHRep(hrep(poly_elim))
#     # perm = [rlt.linset, ]
#     A, b, linset = (rlt.A, rlt.b, rlt.linset)
#     perm = [collect(linset) ; [i for i in 1:5 if i ∉ linset]] # make sure the eqrelation is at the beginning
#     CqK = sparse(-A[perm,:]) |> x->droptol!(x,1e-5)
#     C0qK = b[perm]
#     return CqK, C0qK
# end

# function _calc_C_C0_qk_nullity1(Bnc::Bnc,perm, atol=1e-10) # this algorithm can not drop duplication.
#     @assert get_nullity!(Bnc, perm) == 1 "The nullity of the system is not 1"
    
#     M, M_0 = get_M_M0!(Bnc, perm)
#     C, C_0 = get_C_C0_x!(Bnc, perm)
#     M_array = Array(M)
#     F = svd(M_array)
#     # 找到零奇异值的索引
#     zero_idx = findfirst(s -> s < atol * F.S[1], F.S) 
#     # 计算 nullspace 向量和伪逆
#     U_n = F.V[:, zero_idx:zero_idx]  # 右零空间 (nullspace of M)
#     V_n = F.U[:, zero_idx:zero_idx]  # 左零空间 (nullspace of M')
    
#     # 计算伪逆：M_inv = V * S^+ * U'
#     S_pinv = zeros(size(F.S))
#     for i in 1:zero_idx-1
#         S_pinv[i] = 1.0 / F.S[i]
#     end
#     M_inv = F.V * Diagonal(S_pinv) * F.U'
    
#     # 计算核心矩阵
#     A = C * U_n  
#     B = C * M_inv
#     C_vec = C_0 - B * M_0  
    
#     # 找到正负索引
#     A = vec(A) # as A have to get only one column
#     posi = findall(x -> x > atol, A)
#     nega = findall(x -> x < -atol, A)
#     zero = findall(x -> abs(x) <= atol, A)

#     # original constraints
#     C_qK_org = B[zero, :]
#     C0_qK_org = C_vec[zero]
#     for i in 1:size(C_qK_org,1)
#         val_to_norm =Inf
#         for j in 1:size(C_qK_org,2)
#             val = abs(C_qK_org[i,j])
#             if val < atol
#                 C_qK_org[i,j] = 0.0
#                 continue
#             elseif val < val_to_norm
#                 val_to_norm = val       
#             end
#         end
#         C_qK_org[i,:] ./= val_to_norm
#         C0_qK_org[i] = abs(C0_qK_org[i]) > atol ? C0_qK_org[i]/val_to_norm : 0.0
#     end

#     # Equality constraints
#     C_qK_eq = V_n'
#     C0_qK_eq = (V_n' * M_0)[1] # Only one value
#     val_to_norm = Inf
#     for i in eachindex(C_qK_eq)
#         val = abs(C_qK_eq[i])
#         if val < atol
#             C_qK_eq[i] = 0.0
#             continue
#         elseif val < val_to_norm
#             val_to_norm = val
#         end
#     end
#     C_qK_eq ./= val_to_norm
#     C0_qK_eq = abs(C0_qK_eq) > atol ? C0_qK_eq/val_to_norm : 0.0

#     # inequality constraints
#     # 预计算不等式约束的数量
#     n_posi = length(posi)
#     n_nega = length(nega)
#     n_ineq = n_posi * n_nega
    
#     # 预分配不等式约束数组
#     d = size(B, 2)  # B 的列数
    
#     # 预计算正负约束的系数
#     posi_coeffs = Matrix{Float64}(undef, n_posi, d)
#     posi_consts = Vector{Float64}(undef, n_posi)
#     for (idx, i) in enumerate(posi)
#         a_val = A[i]
#         posi_coeffs[idx, :] = -B[i, :] / a_val
#         posi_consts[idx] = -C_vec[i] / a_val
#     end
    
#     nega_coeffs = Matrix{Float64}(undef, n_nega, d)
#     nega_consts = Vector{Float64}(undef, n_nega)
#     for (idx, i) in enumerate(nega)
#         a_val = A[i]
#         nega_coeffs[idx, :] = -B[i, :] / a_val
#         nega_consts[idx] = -C_vec[i] / a_val
#     end
    

#     C_qK_ineq = Matrix{Float64}(undef, n_ineq, d)
#     C0_qK_ineq = Vector{Float64}(undef, n_ineq)
#     # 构建不等式约束：negamat[j] - posimat[i] 和 negac[j] - posic[i]
#     constraint_idx = 1
#     for i in 1:n_posi
#         for j in 1:n_nega
#             row = nega_coeffs[j, :] - posi_coeffs[i, :]
#             cons = nega_consts[j] - posi_consts[i]

#             # Normalized to the smallest non-zero value
#             val_to_norm = Inf
#             for i in eachindex(row)
#                 val = abs(row[i])
#                 if val < atol
#                     row[i] = 0.0
#                     continue
#                 elseif val < val_to_norm
#                     val_to_norm = val
#                 end
#             end

#             row ./= val_to_norm
#             cons = abs(cons) > atol ? cons/val_to_norm : 0.0

#             C_qK_ineq[constraint_idx, :] = row
#             C0_qK_ineq[constraint_idx] = cons

#             constraint_idx += 1
#         end
#     end
#     # 合并等式和不等式约束
#     C_final = [C_qK_eq; C_qK_ineq; C_qK_org]
#     C0_final = [C0_qK_eq; C0_qK_ineq; C0_qK_org]

#     return C_final, C0_final
# end