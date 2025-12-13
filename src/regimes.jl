#--------------Core computation functions-------------------------

"""
    _calc_Nρ_inverse(Nρ) -> (Nρ_inv::SparseMatrixCSC, nullity::Int)

Compute inverse or adjacency for possibly singular Nρ.
- If Nρ is square and factorizable: returns sparse(inv(Nρ)), nullity = 0
- If singular: delegates to _adj_singular_matrix(Nρ) to get adjacency and nullity.
"""
function _calc_Nρ_inverse(Nρ)::Tuple{SparseMatrixCSC,Int}

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
"""
    _calc_vertices_graph_from_perms(data::Vector{<:AbstractVector{T}}, n::Int) where T

Build VertexGraph from a list of vertex permutations.
- Groups vertices differing in exactly one row, creates bidirectional edges with change_dir_x.
"""
function  _calc_vertices_graph_from_perms(perms::Vector{<:AbstractVector{T}},n::Int) where {T} # optimized by GPT-5, not fullly understood yet.
    n_vtxs = length(perms)
    d=length(perms[1])# n = maximum(v -> maximum(v), data)
    # n = maximum(v -> maximum(v), data)
    # 线程本地边集，最后归并
    thread_edges = [Vector{Tuple{Int, VertexEdge{T}}}() for _ in 1:Threads.nthreads()]

    # 按行分桶：key 为去掉该行后的签名（Tuple），值为该签名下的 (顶点索引, 该行取值)
    for r in 1:d
        buckets = Dict{Tuple{Vararg{T}}, Vector{Tuple{Int,T}}}()

        # 构建桶
        @inbounds for i in 1:n_vtxs
            v = perms[i]
            sig = if r == 1
                Tuple(v[2:end])
            elseif r == d
                Tuple(v[1:end-1])
            else
                Tuple((v[1:r-1]..., v[r+1:end]...))
            end
            push!(get!(buckets, sig) do
                Vector{Tuple{Int,T}}()
            end, (i, v[r]))
        end

        groups = collect(values(buckets))

        # 并行生成边：同桶内所有不同取值的顶点两两相连
        Threads.@threads for gi in 1:length(groups)
            tid = Threads.threadid()
            local_edges = thread_edges[tid]
            group = groups[gi]  # ::Vector{Tuple{Int,T}}
            m = length(group)
            m <= 1 && continue

            @inbounds for a in 1:m-1
                i, xi = group[a]
                for b in a+1:m
                    j, xj = group[b]
                    xi == xj && continue

                    if xi < xj
                        dx = SparseVector(n, [xi, xj], Int8[-1, 1])
                        push!(local_edges, (i, VertexEdge(j, r, dx)))
                        push!(local_edges, (j, VertexEdge(i, r, -dx)))
                    else
                        dx = SparseVector(n, [xj, xi], Int8[-1, 1])
                        push!(local_edges, (i, VertexEdge(j, r, -dx)))
                        push!(local_edges, (j, VertexEdge(i, r, dx)))
                    end
                end
            end
        end
    end

    # 归并线程本地边
    all_edges = reduce(vcat, thread_edges; init=Tuple{Int, VertexEdge{T}}[])
    neighbors = [Vector{VertexEdge{T}}() for _ in 1:n_vtxs]
    for (from, e) in all_edges
        push!(neighbors[from], e)
    end
    return VertexGraph(neighbors)
end
"""
    _calc_H(Bnc, perm)

Compute H for a vertex permutation using cached Nρ_inv where possible.
Returns a dense H matrix; caller may sparsify.
"""
function _calc_H(Bnc::Bnc,perm::Vector{<:Integer})::SparseMatrixCSC
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
    H = droptol!(sparse(H),1e-10)
    return H
end
"""
    _calc_P_and_P0(Bnc, perm)

Build sparse selection matrix P (d×n) and P0 (d-vector) for a permutation.
"""
function _calc_P_and_P0(Bnc::Bnc{T}, perm::Vector{<:Integer})::Tuple{SparseMatrixCSC{Int,Int}, Vector{Float64}} where T
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
"""
    _calc_C_C0_x(Bnc, perm)

Construct x-space inequality matrix C_x (num_ineq×n) and c0_x (num_ineq).
Exactly two nonzeros per inequality: (-1 at col, +1 at chosen rgm column).
"""
function _calc_C_C0_x(Bnc::Bnc{T}, perm::Vector{<:Integer})::Tuple{SparseMatrixCSC{Int,Int}, Vector{Float64}} where T # highly optimized version
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
"""
    _calc_C_C0_qK_singular(Bnc, vtx_perm)

Build qK-space constraints (C_qK, C0_qK) for singular vertices via affine mapping.
Returns: (C_qK::SparseMatrixCSC, C0_qK::Vector)
"""
function _calc_C_C0_qK_singular(Bnc::Bnc, vtx)
    M,M0 = get_M_M0!(Bnc,vtx)
    C,C0 = get_C_C0_x!(Bnc,vtx)
    # n = Bnc.n
    poly_x = hrep(-C,C0) |> x->polyhedron(x,CDDLib.Library())
    poly_elim = M * poly_x  # If for convenience, one can write `translate(M * poly_x, M0)`, and then C0qK = b
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

"""
    _build_Nρ_cache_parallel!(Bnc)

Precompute and cache Nρ inverse info for all distinct complements of vertices.
Returns nothing.
"""

function _build_Nρ_cache_parallel!(Bnc::Bnc{T},perms::Vector{Vector{T}}) where T
    perm_set = Set(Set(perm) for perm in perms) # Unique sets of permutations
    keys = [_get_Nρ_key(Bnc, perm) for perm in perm_set]

    nk = length(keys)
    inv_list = Vector{SparseMatrixCSC{Float64,Int}}(undef, nk)
    nullity_list = Vector{T}(undef, nk)

    Threads.@threads for i in eachindex(keys)
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
    _get_Nρ_inv!(Bnc, key)

Get (Nρ_inv, nullity) from cache or compute.
"""
function _get_Nρ_inv!(Bnc::Bnc{T}, key::AbstractVector{<:Integer}) where T
    get!(Bnc._vertices_Nρ_inv_dict, key) do
        Nρ = @view Bnc.N[:, key]
        _calc_Nρ_inverse(Nρ)
    end
end
"""
    _build_vertices_graph!(Bnc)

Ensure vertices are discovered and vertex graph is built and cached in Bnc.
Returns nothing.
"""

function _fulfill_vertices_graph!(Bnc::Bnc, vtx_graph::VertexGraph)
    """
    fill the qK space change dir matrix for all vertices in Bnc.
    """
    function _calc_change_dir_qK(Bnc, vi, vj, i, j1, j2)
        n1 = get_nullity!(Bnc, vi)
        # println("trigger one")
        n2 = get_nullity!(Bnc, vj)
        # println("trigger two")
        if n1 > 1 || n2 > 1
            return nothing
        end

        # unit vector (Float64) at position i, reused where needed
        ei = SparseVector(Bnc.n, [i], [1.0])

        if n1 == 0
            H1 = get_H!(Bnc, vi)
            # println("trigger three")
            dir = H1[j2, :] - ei
        elseif n2 == 0
            H2 = get_H!(Bnc, vj)
            # println("trigger four")
            dir = ei - H2[j1, :]
        else
            # n1 == 1 && n2 == 1
            H1 = get_H!(Bnc, vi)
            dir = H1[j2, :]
        end
        droptol!(dir, 1e-10)
        return nnz(dir)==0 ? nothing : dir
    end
    # pre compute H for all vertices with nullity 0 or 1
    Threads.@threads for idx in eachindex(vtx_graph.neighbors)
        if Bnc.vertices_nullity[idx] <= 1
            get_H!(Bnc, idx)
        end
    end

    Threads.@threads for vi in eachindex(vtx_graph.neighbors)
        edges = vtx_graph.neighbors[vi]
        if Bnc.vertices_nullity[vi] > 1 # jump off those regimes with nullity >1
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
end

"""
    _locate_C_row(Bnc, i ,j1, j2)
Locate the row index in C matrix for 1 move from L[i,j1] to L[i,j2].
(Warning, for CqK, works only for invertible regime as singular will change the row order)
"""
function _locate_C_row(Bnc::Bnc, i ,j1, j2)
    cls_start = Bnc._C_partition_idx[i]
    i_j1= findfirst(x-> x == j1, Bnc._valid_L_idx[i])
    i_j2= findfirst(x-> x == j2, Bnc._valid_L_idx[i])
    if isnothing(i_j1) || isnothing(i_j2)
        error("Either j1 or j2 is not a valid change direction for regime $i")
    end
    i = i_j2 < i_j1 ? i_j2 : i_j2 - 1
    return cls_start + i-1
end

function _calc_change_col(from::Vector{T},to::Vector{T}) where T<:Integer
    j1 = 0
    j2 = 0
    inconsis = Int[]
    for (i , (val_a,val_b)) in enumerate(zip(from,to))
        if val_a == val_b
            continue
        else
            push!(inconsis, i)
        end
    end
    target_inconsis = Set(to[inconsis])
    if target_inconsis |> length == 1
        j1,j2 = from[inconsis[1]], to[inconsis[1]]
        return j1,j2    
    end
    for (val1,i1) in zip(from[inconsis], inconsis)
        if val1 ∈ target_inconsis
            j2 = to[i1]
            i2 = inconsis[findfirst(x -> x == val1, to[inconsis])]
            j1 = from[i2]
            return j1,j2
        end
    end
end

function _get_i_j_perms(from::Vector{T},to::Vector{T}) where T<:Integer
    """
    We shall find the source col+row and target col+row directly from two permutations.
    """
    inconsis_idx = findall(from .!= to)
    if length(inconsis_idx) == 1
        i1 = inconsis_idx[1]
        i2 = i1
    else
        intersect_val = Set(from[inconsis_idx]) ∩ Set(to[inconsis_idx])
        @assert length(intersect_val) == 1 "More than one intersected value found in inconsistent positions."
        for i in inconsis_idx
            if from[i] ∈ intersect_val
                i2 = i
            end
            if to[i] ∈ intersect_val
                i1 = i
            end
        end
    end
    j1 = from[i1]
    j2 = to[i2]
    return i1,i2,j1,j2
end
#------------------Helper functions -------------------------------------------
"""
    _nrho_key_for_perm(Bnc, perm) -> Vector{Int}

Indices of columns not in `perm` (complement) used to form Nρ.
"""
function _get_Nρ_key(Bnc::Bnc{T}, perm)::Vector{T} where T 
   return [i for i in 1:Bnc.n if i ∉ perm]
end
_get_Nρ_inv_from_perm!(Bnc, perm) = _get_Nρ_inv!(Bnc, _get_Nρ_key(Bnc, perm))
"""
    _vertex_graph_to_sparse(G; weight_fn = e->1.0)

Convert VertexGraph to sparse adjacency (weights from weight_fn).
"""
function _vertex_graph_to_sparse(G::VertexGraph{T}; weight_fn = e -> 1) where T
    n = length(G.neighbors)
    # 预分配估计：平均度 × n
    nnz = sum(length(v) for v in G.neighbors)
    I = Vector{Int}(undef, nnz)
    J = Vector{Int}(undef, nnz)
    V = Vector{Float64}(undef, nnz)
    idx = 0
    for i in 1:n
        for e in G.neighbors[i] #Edge
            idx += 1
            I[idx] = i
            J[idx] = e.to
            V[idx] = weight_fn(e)
        end
    end
    return sparse(I,J,V, n, n) |> dropzeros!
end
"""
    _create_vertex(Bnc, perm) -> Vertex

Create a partially-filled Vertex (P/P0, M/M0, C_x/C0_x are ready).
"""
function _create_vertex(Bnc::Bnc{T}, perm::Vector{<:Integer})::Vertex where T
    """
    Creates a new, partially-filled Vertex object.
    This function performs the initial, less expensive calculations.
    """
    find_all_vertices!(Bnc)
    idx = Bnc.vertices_perm_dict[perm] # Index of the vertex in the Bnc.vertices_perm list
    real = Bnc.vertices_asymptotic_flag[idx] # Check if the vertex is real or fake
    nullity = Bnc.vertices_nullity[idx] # Get the nullity of the vertex
    
    P, P0 = _calc_P_and_P0(Bnc, perm); 
    C_x, C0_x = _calc_C_C0_x(Bnc, perm)

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
"""
    _fill_inv_info!(Bnc, vtx)

Ensure Vertex has H/H0 and qK constraints computed and cached.
Mutates vtx. Returns nothing.
"""
function _fill_inv_info!(Bnc::Bnc, vtx::Vertex)
    # Check if already calculated
    if !isempty(vtx.H)
        return nothing
    end
    if vtx.nullity == 0
        H = _calc_H(Bnc, vtx.perm) 
        vtx.H = H # Calculate the inverse matrix from pre-computed LU decomposition of M H=M^-1
        vtx.H0 = - H * vtx.M0  # H0 = -M^-1 * M0
        vtx.C_qK = droptol!(sparse(vtx.C_x * H),1e-10) # C_qK = C_x * H
        vtx.C0_qK = vtx.C0_x + vtx.C_x * vtx.H0 # C0_qK = C0_x + C_x * H0 
    else
        if vtx.nullity ==1
            # we need to check where this nullity comes from.
            if length(Set(vtx.perm)) == Bnc.d # the nullity comes from N
                vtx.H = _calc_H(Bnc, vtx.perm)#.* Bnc.direction 
            else # the nullity comes from P
                H = _adj_singular_matrix(vtx.M)[1]
                vtx.H = droptol!(sparse(H),1e-10)#.* Bnc.direction
            end
        else # nullity>1 , H, HO is nolonger avaliable
            vtx.H = spzeros(Bnc.n, Bnc.n) # fill value as a sign that this regime is fully computed
        end
        vtx.C_qK, vtx.C0_qK = _calc_C_C0_qK_singular(Bnc, vtx.perm)
    end
    return nothing
end

function _fill_neighbor_info!(Bnc::Bnc, vtx::Vertex)
    """
    Fill the neighbor info for a given vertex.
    """
    if isempty(vtx.neighbors_idx)
        vtx_grh = get_vertices_graph!(Bnc;full=false)
        vtx.neighbors_idx = vtx_grh.neighbors[vtx.idx] .|> e -> e.to
    end
    return nothing
end


#------------------------------------------------------------------------------
#             1. Functions find all regimes and return properties
# ------------------------------------------------------------------------------

"""
    find_all_vertices!(Bnc)

Compute and cache:
- all vertex permutations and index dict
- asymptotic flags
- Nρ inverse cache (parallel)
- vertex nullity

Returns Vector{Vector{Int}} of vertex permutations.
"""
function find_all_vertices!(Bnc::Bnc{T};) where T # cheap enough for now
    if isempty(Bnc.vertices_perm) || isempty(Bnc.vertices_asymptotic_flag)
        println("---------------------Start finding all vertices, it may takes a while.--------------------")
        # all vertices
        # finding non-asymptotic vettices, which gives all vertices both real and fake, singular and non-singular
        all_vertices = find_all_vertices(Bnc.L; asymptotic=false)
        n_vertices = length(all_vertices)
        # finding asymptotic vertices, which is the real vertices.
        real_vtx = Set(find_all_vertices(Bnc.L; asymptotic=true))
        n_real_vtx = length(real_vtx)

        println("Finished, with $(n_vertices) vertices found and $(n_real_vtx) asymptotic vertices.\n")
        println("-------------Start calculating nullity for each vertex, it also takes a while.------------")
        println("1.Building Nρ_inv cache in parallel...")
        _build_Nρ_cache_parallel!(Bnc, all_vertices) # build Nρ_inv cache in parallel
        # Caltulate the nullity for each vertices
        nullity = Vector{T}(undef, length(all_vertices))
        println("2.Calculating nullity for each vertex in parallel...")
        Threads.@threads for i in  eachindex(all_vertices)
            perm_set = Set(all_vertices[i])
            nullity_P =  Bnc.d -length(perm_set)
            _ , nullity_N =  _get_Nρ_inv_from_perm!(Bnc,perm_set) 
            nullity[i] = nullity_P + nullity_N # this is true as we can permute the matrix into diagnal block matrix.
        end
        println("3.Storing all vertices information...")
        Bnc.vertices_perm = all_vertices
        Bnc.vertices_asymptotic_flag = Bnc.vertices_perm .∈ Ref(real_vtx)
        Bnc.vertices_perm_dict = Dict(a=>idx for (idx, a) in enumerate(Bnc.vertices_perm)) # Map from vertex to its index
        Bnc.vertices_nullity = nullity
        Bnc.vertices_data = Vector{Vertex}(undef, n_vertices)
        Bnc._vertices_is_initialized = falses(n_vertices)
        Bnc._vertices_volume_is_calced = falses(n_vertices)
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

"""
Get the volume of all vertices in Bnc.
"""
function get_vertices_volume!(Bnc::Bnc,vtxs=nothing; recalculate::Bool=false, kwargs...)
    """
    Calculate the volume of all vertices in Bnc. (currently calc all the voluems, but unnecessary)
    """
    all_vtxs = isnothing(vtxs) ? get_vertices(Bnc;return_idx=true) : [get_idx(Bnc, vtx) for vtx in vtxs]

    vtxs_to_calc = 
        if recalculate
            all_vtxs
        else
            filter(i -> !Bnc._vertices_volume_is_calced[i], all_vtxs)
        end
    
    if !isempty(vtxs_to_calc)
        rlts = calc_volume(Bnc,vtxs_to_calc;kwargs...)
        for (i,idx) in enumerate(vtxs_to_calc)
            vtx = get_vertex!(Bnc,idx; inv_info=false,neighbor_info=false)
            vtx.volume = rlts[i][1]
            vtx.eps_volume = rlts[i][2]
            Bnc._vertices_volume_is_calced[idx]=true
        end
    end

    return [(vtx.volume, vtx.eps_volume) for vtx in Bnc.vertices_data[all_vtxs]]
end

#---------------------------------------------------------------------------------------------
#   Functions involving vertices relationships, (neighbors finding and changedir finding)
#---------------------------------------------------------------------------------------------
"""
    get_vertices_graph!(Bnc; full=false) -> VertexGraph

Ensure vertex graph is built; if full=true, also compute qK change directions on edges.
Returns the cached VertexGraph.
"""
function get_vertices_graph!(Bnc::Bnc; full::Bool=false)::VertexGraph
    """
    get the neighbor of vertices formed graph.
    """
    if full
        vtx_graph = get_vertices_graph!(Bnc; full=false)
        if !vtx_graph.change_dir_qK_computed
            println("-------Start calculating vertices neighbor graph with qK change dir, It may takes a while.------------")
            _fulfill_vertices_graph!(Bnc, vtx_graph)
            vtx_graph.change_dir_qK_computed = true
            println("Done.\n")
        end
    else
        if isnothing(Bnc.vertices_graph)
            find_all_vertices!(Bnc)# Ensure vertices are calculated
            println("----------------Start calculating vertices neighbor graph, It may takes a while.----------------")
            Bnc.vertices_graph =  _calc_vertices_graph_from_perms(Bnc.vertices_perm,Bnc.n)
            println("Done.\n")
        end
    end
    return Bnc.vertices_graph
end
"""
    get_vertices_neighbor_mat!(Bnc) -> SparseMatrixCSC

Return x-space adjacency matrix of the vertex graph (unweighted → 1.0).
"""
function get_vertices_neighbor_mat_x!(Bnc::Bnc)
    """
    # find the x space neighbor of all vertices in Bnc, the value denotes for two perms, which row they differ at.
    """
    grh = get_vertices_graph!(Bnc;full=false)
    spmat = _vertex_graph_to_sparse(grh; weight_fn = e -> 1)
    return spmat
end

function get_vertices_neighbor_mat_qK!(Bnc::Bnc)
    """
    # find the x space neighbor of all vertices in Bnc, the value denotes for two perms, which row they differ at.
    """
    grh = get_vertices_graph!(Bnc;full=true)
    f(x::VertexEdge) = isnothing(x.change_dir_qK) ? 0.0 : 1.0
    spmat = _vertex_graph_to_sparse(grh; weight_fn = f)
    return spmat
end


#-------------------------------------------------------------------------------------
#         functions involving single vertex and lazy calculate  its properties
# ------------------------------------------------------------------------------------
"""
Get a vertex's index
"""
function get_idx(Bnc,perm::Vector{<:Integer};check::Bool=false)
    find_all_vertices!(Bnc)
    return Bnc.vertices_perm_dict[perm]
end
function get_idx(Bnc::Bnc, idx::T;check::Bool=false) where T<:Integer
    if check
        find_all_vertices!(Bnc)
        @assert idx ≥ 1 && idx ≤ length(Bnc.vertices_perm) "The given index is out of range."
    end
   return idx
end
function get_idx(Bnc::Bnc, vtx::Vertex;check::Bool=false)
    return vtx.idx
end

"""
Get perm of a vertex
"""
function get_perm(Bnc,perm::Vector{<:Integer};check::Bool=false)
    if check
        find_all_vertices!(Bnc)
        @assert haskey(Bnc.vertices_perm_dict, perm) "The given perm is not in Bnc"
    end
    return perm
end
function get_perm(Bnc::Bnc, idx::T;check::Bool=false) where T<:Integer
    find_all_vertices!(Bnc)
    if check
        @assert idx ≥ 1 && idx ≤ length(Bnc.vertices_perm) "The given index is out of range."
    end
    return Bnc.vertices_perm[idx]
end
function get_perm(Bnc::Bnc, vtx::Vertex;check::Bool=false)
    return vtx.perm
end

"""
Retrieves a vertex from cache or creates it if it doesn't exist.
"""
function get_vertex!(Bnc::Bnc, perm; inv_info::Bool=true, neighbor_info::Bool=true, check::Bool=false)::Vertex
    find_all_vertices!(Bnc) #initialize perm_data
    idx = get_idx(Bnc, perm; check=check)
    if Bnc._vertices_is_initialized[idx]
        vtx = Bnc.vertices_data[idx]
    else
        perm = Bnc.vertices_perm[idx]
        vtx = _create_vertex(Bnc, perm)
        Bnc.vertices_data[idx] = vtx
        Bnc._vertices_is_initialized[idx] = true
    end

    if inv_info
        _fill_inv_info!(Bnc,vtx)
    end
    if neighbor_info
        _fill_neighbor_info!(Bnc,vtx)
    end
    return vtx
end

function have_perm(Bnc::Bnc, perm::Vector{<:Integer})
    """
    Check if the vertex represented by perm is within the Bnc.
    """
    find_all_vertices!(Bnc)
    return haskey(Bnc.vertices_idx, perm)
end


# """
#     get_all_neighbors!(Bnc::Bnc, perm; return_idx::Bool=false)

# Return all neighbors of the vertex represented by `perm`.

# If the neighbors are not cached in `Bnc`, they will be computed
# and stored in `vtx.neighbors_idx`.

# # Keyword arguments
# - `return_idx`: if `true`, return neighbor indices; otherwise, return `vertices_perm[idx]`.

# # Returns
# A vector of neighbor indices or vertex permutations.
# """
# function get_all_neighbors!(Bnc::Bnc, perm; return_idx::Bool=false)
#     # Get the neighbors of the vertex represented by perm
#     vtx = get_vertex!(Bnc, perm ; inv_info=false)
#     if isempty(vtx.neighbors_idx)
#         vtx_grh = get_vertices_graph!(Bnc;full=false)
#         vtx.neighbors_idx = vtx_grh.neighbors[vtx.idx] .|> e -> e.to
#     end

#     idx = vtx.neighbors_idx
#     return return_idx ? idx : Bnc.vertices_perm[idx]
# end
# """
#     get_finite_neighbors(Bnc::Bnc, perm; return_idx::Bool=false)

# Return neighbors of the vertex `perm` that are **finite** (nullity == 0).
# """
# function get_finite_neighbors(Bnc::Bnc, perm; return_idx::Bool=false)
#     nb_idx = get_all_neighbors!(Bnc, perm; return_idx=true)
#     idx = filter(i->Bnc.vertices_nullity[i] == 0, nb_idx)
#     return return_idx ? idx : Bnc.vertices_perm[idx]
# end

# """
#     get_infinite_neighbors(Bnc::Bnc, perm; return_idx::Bool=false)

# Return neighbors of the vertex `perm` that are **infinite** (nullity > 0).
# """
# function get_infinite_neighbors(Bnc::Bnc, perm; return_idx::Bool=false) # nullity_max::Union{Int,Nothing}=nothing
#     nb_idx = get_all_neighbors!(Bnc, perm; return_idx=true)
#     idx = filter(i->Bnc.vertices_nullity[i] > 0, nb_idx)
#     return return_idx ? idx : Bnc.vertices_perm[idx]
# end

"""
    get_neighbors(Bnc::Bnc, perm; singular=nothing, asymptotic::Union{Bool,Nothing}=nothing, return_idx::Bool=false)

Return neighbors of the vertex `perm` that satisfy certain conditions.

# Keyword arguments
- `singular`:
    - `true` → only singular vertices (`nullity > 0`)
    - `false` → only non-singular vertices (`nullity == 0`)
    - `Int` → vertices with `nullity ≤ singular`
    - `nothing` → no filter on nullity
- `asymptotic`:
    - `true` → only asymptotic (real) vertices
    - `false` → only fake vertices
    - `nothing` → no filter
- `return_idx`: if `true`, return neighbor indices; otherwise, return permutations.

# Example
julia
get_neighbors(Bnc, perm)                       # all neighbors
get_neighbors(Bnc, perm; singular=true)        # only singular ones
get_neighbors(Bnc, perm; singular=2)           # nullity ≤ 2
get_neighbors(Bnc, perm; asymptotic=true)      # only asymptotic ones
get_neighbors(Bnc, perm; singular=1, asymptotic=false)

"""
function get_neighbors(Bnc::Bnc, perm; singular::Union{Bool,Int,Nothing}=nothing, asymptotic::Union{Bool,Nothing}=nothing, return_idx::Bool=false)
    idx = get_vertex!(Bnc,perm; inv_info=false, neighbor_info=true).neighbors_idx
    
    idx = filter(idx) do i
        nlt = Bnc.vertices_nullity[i]
        flag_asym = Bnc.vertices_asymptotic_flag[i]

        ok_singular = isnothing(singular) || (
            (singular === true  && nlt > 0) ||
            (singular === false && nlt == 0) ||
            (singular isa Int   && nlt ≤ singular)
        )

        ok_asym = isnothing(asymptotic) || (asymptotic == flag_asym)
        return ok_singular && ok_asym 
    end

    return return_idx ? idx : Bnc.vertices_perm[idx]
end


"""
Gets the nullity of a vertex
"""
function get_nullity!(Bnc::Bnc,perm)
    find_all_vertices!(Bnc)
    idx = get_idx(Bnc, perm)
    return Bnc.vertices_nullity[idx]
end
function is_singular(Bnc::Bnc, perm)::Bool
    nullity = get_nullity!(Bnc, perm)
    return nullity > 0
end

function is_asymptotic(Bnc::Bnc, perm)::Bool
    find_all_vertices!(Bnc)
    idx = get_idx(Bnc, perm)
    return Bnc.vertices_asymptotic_flag[idx]
end

"""
Gets P and P0, creating the vertex if necessary.
"""
get_P_P0!(Bnc::Bnc, perm) = get_vertex!(Bnc, perm; inv_info=false, neighbor_info=false) |> vtx -> (vtx.P, vtx.P0)
get_P!(Bnc::Bnc, perm) = get_vertex!(Bnc, perm; inv_info=false, neighbor_info=false).P
get_P0!(Bnc::Bnc, perm) = get_vertex!(Bnc, perm; inv_info=false, neighbor_info=false).P0

"""
Gets M and M0, creating the vertex if necessary.
"""
get_M_M0!(Bnc::Bnc, perm) = get_vertex!(Bnc, perm; inv_info=false, neighbor_info=false) |> vtx -> (vtx.M, vtx.M0)
get_M!(Bnc::Bnc, perm) = get_vertex!(Bnc, perm; inv_info=false, neighbor_info=false).M
get_M0!(Bnc::Bnc, perm) = get_vertex!(Bnc, perm; inv_info=false, neighbor_info=false).M0


"""
Gets C_x and C0_x, creating the vertex if necessary.
"""
get_C_C0_x!(Bnc::Bnc, perm) = get_vertex!(Bnc, perm; inv_info=false, neighbor_info=false) |> vtx -> (vtx.C_x, vtx.C0_x)
get_C_x!(Bnc::Bnc, perm) = get_vertex!(Bnc, perm; inv_info=false, neighbor_info=false).C_x
get_C0_x!(Bnc::Bnc, perm) = get_vertex!(Bnc, perm; inv_info=false, neighbor_info=false).C0_x


"""
Gets C_qK and C0_qK, ensuring the inv_info  is calculated.
"""
get_C_C0_qK!(Bnc::Bnc, perm) = get_vertex!(Bnc, perm; inv_info=true, neighbor_info=false) |> vtx -> (vtx.C_qK, vtx.C0_qK)
get_C_qK!(Bnc::Bnc, perm) = get_vertex!(Bnc, perm; inv_info=true, neighbor_info=false).C_qK
get_C0_qK!(Bnc::Bnc, perm) = get_vertex!(Bnc, perm; inv_info=true, neighbor_info=false).C0_qK


"""
Gets H and H0, ensuring the full vertex is calculated.
"""

get_H_H0!(Bnc::Bnc, perm) = is_singular(Bnc, perm) ? @error("Vertex is singular, cannot get H0") : get_vertex!(Bnc, perm; inv_info=true, neighbor_info=false) |> vtx -> (vtx.H, vtx.H0)
get_H!(Bnc::Bnc, perm) = get_nullity!(Bnc, perm) > 1 ? @error("Vertex's nullity is bigger than 1, cannot get H") : get_vertex!(Bnc, perm; inv_info=true, neighbor_info=false).H
get_H0!(Bnc::Bnc, perm) = is_singular(Bnc, perm) ? @error("Vertex is singular, cannot get H0") : get_vertex!(Bnc, perm; inv_info=true, neighbor_info=false).H0
# function get_H_H0!(Bnc::Bnc, perm)
#     vtx = get_vertex!(Bnc, perm; full=false, neighbor_info=false)
#     if vtx.nullity > 0
#         @error("Vertex is singular, cannot get H0")
#     end # This will compute if needed
#     _fill_inv_info!(Bnc,vtx)
#     return vtx.H, vtx.H0
# end
# function get_H!(Bnc::Bnc, perm)
#     vtx = get_vertex!(Bnc, perm; full=false, neighbor_info=false)
#     if vtx.nullity > 1
#         @error("Vertex's nullity is bigger than 1, cannot get H")
#     end # This will compute if needed
#     _fill_inv_info!(Bnc,vtx)
#     return vtx.H
# end
# get_H0!(Bnc::Bnc, perm) = get_H_H0!(Bnc, perm)[2]


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

function get_polyhedra(C::AbstractMatrix{<:Real}, C0::AbstractVector{<:Real}, nullity::Int=0)::Polyhedron 
    if nullity ==0
        return hrep(-C,C0) |> x-> polyhedron(x,CDDLib.Library())
    else
        linset = BitSet(1:nullity)
        return hrep(-C,C0,linset) |> x-> polyhedron(x,CDDLib.Library())
    end
end

function get_volume!(Bnc::Bnc, perm;recalculate::Bool=false, kwargs...)
    idx = get_idx(Bnc, perm)
    vtx = get_vertex!(Bnc, perm; inv_info=true, neighbor_info=false)
    if recalculate || !Bnc._vertices_volume_is_calced[idx]
        vol = calc_volume(Bnc, [idx];kwargs...)[1]
        vtx.volume = vol[1]
        vtx.eps_volume = vol[2]
        Bnc._vertices_volume_is_calced[idx] = true
    end
    return (vtx.volume, vtx.eps_volume)
end



#--------------------------------------------------------------------------------------------------------------------------------------
#          Naive code for figuring out  relationships between two vertices 
#----------------------------------------------------------------------------------------------------------------------------------------


# Direct method:
function get_polyhedron_intersect(Bnc::Bnc,vtx1,vtx2;cache::Bool=true)::Polyhedron
    idx1 = get_idx(Bnc, vtx1)
    idx2 = get_idx(Bnc, vtx2)
    
    f(vtx1,vtx2) = begin
        p1 = get_polyhedra(Bnc, vtx1)
        p2 = get_polyhedra(Bnc, vtx2)
        p = intersect(p1,p2)
        return p
    end

    if !cache
        return f(vtx1,vtx2)
    end

    key = Set([idx1, idx2])
    vg = get_vertices_graph!(Bnc; full=false) # May not necessary
    if haskey(vg.edge_map, key)
        idx = vg.edge_map[key]
        if vg.boundary_polys_is_computed[idx]
            return vg.boundary_polys[idx]
        else
            vg.boundary_polys[idx] = f(vtx1,vtx2)
            vg.boundary_polys_is_computed[idx] = true
            return vg.boundary_polys[idx]
        end
    end
    return f(vtx1,vtx2)
end


"""
Directly judge if two vertices are neighbors by polyhedron intersection.
"""
function is_neighbor_direct(Bnc::Bnc,vtx1,vtx2)::Bool
    if get_nullity!(Bnc, vtx1) > 1 || get_nullity!(Bnc, vtx2) > 1
        @warn "Currently we doesn't care neighbor relationships less than your model's dim - 1  ,return false by default"
        return false
    end
    p = get_polyhedron_intersect(Bnc,vtx1,vtx2)
    # detecthlinearity!(p)
    # if nhyperplanes(p) > null || isempty(p)
    if dim(p)==Bnc.n-1 
        return true
    else
        return false
    end
end
function get_change_dir_qK_direct(Bnc::Bnc, from, to;check=false)
    p = get_polyhedron_intersect(Bnc,from,to)
    detecthlinearity!(p)
    if dim(p)< Bnc.n-1
        @error("Vertices $get_perm(Bnc, from) and $get_perm(Bnc, to) do not intersect.")
    end
    hplanes = hyperplanes(p)
    hp = first(hplanes)

    a = droptol!(sparse(hp.a), 1e-10)
    
    if check
        point = get_one_inner_point(p2)
        val = dot(a, point)
        b = hp.β
        if val < b 
            return -a
        end
    end
    return a
    # judge the direction
end

"""
a'x+b =0 is the interface between two neighboring vertices in qK space.
"""
function get_interface_direct(Bnc::Bnc, from, to)
    p = get_polyhedron_intersect(Bnc,from,to)
    detecthlinearity!(p)
    if dim(p)< Bnc.n-1
        @error("Vertices $get_perm(Bnc, from) and $get_perm(Bnc, to) do not intersect.")
    end
    hplanes = hyperplanes(p)
    hp = first(hplanes)
    a = droptol!(sparse(hp.a), 1e-10)
    b = hp.β
    return a, -b
end



# indirect method:
"""
Judge if two vertices are neighbors.
"""
function is_neighbor_x(Bnc::Bnc,vtx1,vtx2)::Bool
    nbs = get_all_neighbors!(Bnc, vtx1; return_idx=true)
    idx2 = get_idx(Bnc, vtx2)
    if idx2 in nbs
        return true
    else
        return is_neighbor_direct(Bnc,vtx1,vtx2)
    end
end

function is_neighbor_qK(Bnc::Bnc,vtx1,vtx2)::Bool
    @assert get_nullity!(Bnc, vtx1) <= 1 "Currently we only support neighbor detection for vertices with nullity less than or equal to 1"
    @assert get_nullity!(Bnc, vtx2) <= 1 "Currently we only support neighbor detection for vertices with nullity less than or equal to 1"
    from = get_idx(Bnc, vtx1)
    to = get_idx(Bnc, vtx2)
    vtx_grh = get_vertices_graph!(Bnc,full=true)
    for ve in vtx_grh.neighbors[from]
        if ve.to == to
            return isnothing(ve.change_dir_qK) ? false : true
        end
    end
    # no directly edge found, judge numerically,
    return is_neighbor_direct(Bnc,vtx1,vtx2)
end
is_neighbor(Bnc::Bnc,vtx1,vtx2) = is_neighbor_qK(Bnc,vtx1,vtx2)


function get_change_dir_x(Bnc::Bnc, from, to)
    from = get_idx(Bnc, from)
    to = get_idx(Bnc, to)
    vtx_grh = get_vertices_graph!(Bnc)
    for ve in vtx_grh.neighbors[from]
        if ve.to == to
            return ve.change_dir_x
        end
    end
    @error("Vertices $get_perm(Bnc, from) and $get_perm(Bnc, to) are not neighbors in x space.")
end

function get_change_dir_qK(Bnc::Bnc, from, to;check=false)
    from = get_idx(Bnc, from)
    to = get_idx(Bnc, to)
    vtx_grh = get_vertices_graph!(Bnc)
    for ve in vtx_grh.neighbors[from]
        if ve.to == to
            if isnothing(ve.change_dir_qK)
                @error("Vertices $get_perm(Bnc, from) and $get_perm(Bnc, to) are not neighbors in qK space.")
            end
            return ve.change_dir_qK
        end
    end
    # no directly edge found, judge numerically, or maybe further optimized to judge based on graph.
    if check && !is_neighbor_direct(Bnc,from,to)
        @error("Vertices $get_perm(Bnc, from) and $get_perm(Bnc, to) are not neighbors in qK space.")
    else
        @assert get_nullity!(Bnc, from) ==0 && get_nullity!(Bnc, to) ==0 "They are neighbor but change direaction is currently not supported"
        (i1, i2, j1, j2) = _get_i_j_perms(get_perm(Bnc, from), get_perm(Bnc, to))
        # n1 = get_nullity!(Bnc, from)
        # n2 = get_nullity!(Bnc, to)
        # a = get_H!(Bnc, from)[j2] .- SparseVector(Bnc.n, [i1], [1.0])
        # b = SparseVector(Bnc.n, [i2], [1.0]) .- get_H!(Bnc, to)[j1]
        # @show a,b
        dir = (get_H!(Bnc, from)[j2, :] - get_H!(Bnc, to)[j1, :]) ./ 2
        return droptol!(dir,1e-10)
    end
end



"""
    get_interface(Bnc::Bnc, from, to)
return the interface (a,b) between two neighboring vertices in qK space, i.e., a'x + b = 0.
(For now the logic is to find the right row in C and C0, and calculate from then is required.)
"""
function get_interface(Bnc::Bnc, from, to)
    if !is_neighbor(Bnc,from,to)
        @error("Vertices $get_perm(Bnc, from) and $get_perm(Bnc, to) are not neighbors.")
    end
    from = get_perm(Bnc, from)
    to = get_perm(Bnc, to)
    n1 = get_nullity!(Bnc, from)
    n2 = get_nullity!(Bnc, to)

    # return interface based on nullity
    if n1 ==1 
        C,C0 = get_C_C0_qK!(Bnc, from)
        return  C[1,:], C0[1]
    elseif n2 ==1
        C,C0 = get_C_C0_qK!(Bnc, to)
        return  C[1,:], C0[1]
    else
        (i1,i2,j1,j2) = _get_i_j_perms(from,to)
        row_idx = _locate_C_row(Bnc, i1, j1, j2) # or i2,j2,j1 while under such case, the next row will change from "from" to "to"
        C,C0 = get_C_C0_qK!(Bnc, from)
        return  C[row_idx,:], C0[row_idx]
    end
end










#-------------------------------------------------------------------------------------
#         functions of getting vertices with certain properties
# -------------------------------------------------------------------------------------

"""
    get_singular_vertices(Bnc::Bnc; return_idx=false)

Return all singular vertices (nullity > 0).
"""
get_singular_vertices(Bnc::Bnc; return_idx::Bool=false) = get_vertices(Bnc; singular=true, return_idx)
get_singular_vertices_idx(Bnc::Bnc) = get_vertices(Bnc; singular=true, return_idx=true)

"""
    get_nonsingular_vertices(Bnc::Bnc; return_idx=false)

Return all nonsingular vertices (nullity == 0).
"""
get_nonsingular_vertices(Bnc::Bnc; return_idx::Bool=false) = get_vertices(Bnc; singular=false, return_idx)
get_nonsingular_vertices_idx(Bnc::Bnc) = get_vertices(Bnc; singular=false, return_idx=true)

"""
    get_real_vertices(Bnc::Bnc; return_idx=false)

Return all real/asymptotic vertices.
"""
get_real_vertices(Bnc::Bnc; return_idx::Bool=false) = get_vertices(Bnc; asymptotic=true, return_idx)
get_real_vertices_idx(Bnc::Bnc) = get_vertices(Bnc; asymptotic=true, return_idx=true)


"""
    get_fake_vertices(Bnc::Bnc; return_idx=false)

Return all fake (non-asymptotic) vertices.
"""
get_fake_vertices(Bnc::Bnc; return_idx::Bool=false) = get_vertices(Bnc; asymptotic=false, return_idx)
get_fake_vertices_idx(Bnc::Bnc) = get_vertices(Bnc; asymptotic=false, return_idx=true)

"""
    get_vertices(Bnc::Bnc; singular=nothing, asymptotic=nothing, return_idx=false)

Return all vertices of `Bnc` that satisfy given filters.

# Keyword arguments
- `singular`:
    - `true` → only singular vertices (`nullity > 0`)
    - `false` → only nonsingular vertices (`nullity == 0`)
    - `Int` → vertices with `nullity ≤ singular`
    - `nothing` → no filter
- `asymptotic`:
    - `true` → only real/asymptotic vertices
    - `false` → only fake vertices
    - `nothing` → no filter
- `return_idx`: if `true`, return vertex indices; otherwise return vertex permutations.

# Example
julia
get_vertices(Bnc)                          # all vertices
get_vertices(Bnc; singular=true)           # singular only
get_vertices(Bnc; singular=2)              # nullity ≤ 2
get_vertices(Bnc; asymptotic=true)         # real/asymptotic vertices
get_vertices(Bnc; singular=false, asymptotic=false)
"""
function get_vertices(Bnc::Bnc; return_idx::Bool=false, kwargs...)
    find_all_vertices!(Bnc)
    idx_all = eachindex(Bnc.vertices_data)
    masks = _get_vertices_mask(Bnc, idx_all; kwargs...)
    return return_idx ? findall(masks) : Bnc.vertices_perm[masks]
end



#
# filter the vtxs accoring to the criteria
#
function _get_vertices_mask(model::Bnc,vtxs::AbstractVector{<:Integer};
     singular::Union{Bool,Integer,Nothing}=nothing, 
     asymptotic::Union{Bool,Nothing}=nothing)::Vector{Bool}
    # ensure nullity and asymptotic flags are calculated
    find_all_vertices!(model)

    nlt = model.vertices_nullity
    flag_asym = model.vertices_asymptotic_flag

    f(nlt) = isnothing(singular) || (
        (singular === true  && nlt > 0) ||
        (singular === false && nlt == 0) ||
        (singular isa Int   && nlt ≤ singular)
    )

    g(flag_asym) = isnothing(asymptotic) || (asymptotic == flag_asym)
    
    return map(vtxs) do i
        f(nlt[i]) && g(flag_asym[i])
    end
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
function get_one_inner_point(poly::T;rand_line=true,rand_ray=true,extend=3) where T<:Polyhedron
    vrep_poly = MixedMatVRep(vrep(poly))
    point = [mean(p) for p in eachcol(vrep_poly.V)]
    ray_avg = zeros(size(point,1))
    for (i, ray) in enumerate(eachrow(vrep_poly.R))
        if i ∉ vrep_poly.Rlinset
            norm_ray = norm(ray)
            sigma = rand_ray ? (rand()+0.5)*extend : extend
            ray_avg .+= (ray ./ norm_ray .* sigma )
        else
            if rand_line
                norm_ray = norm(ray)
                sigma = (rand()-0.5)*extend
                ray_avg .+= (ray ./ norm_ray * sigma)
            end
        end
    end
    return (point.+ ray_avg)
end

"""
    get_C_C0(poly::Polyhedron)
Get the C and C0 matrices from a Polyhedron's H-representation.
Returns C, C0, linset, the original polyhedra can be represented as {x | Cx ≤ C0, Cx = C0 for linset}
"""
function get_C_C0(poly::Polyhedron)
    p = MixedMatHRep(hrep(poly))
    C = -p.A
    C0 = p.b
    linset = p.linset
    if !isempty(linset)
        nullity = maximum(linset)
        @assert linset == BitSet(1:nullity)
    else
        nullity = linset
    end
    return C, C0, nullity
end

function check_feasibility(Bnc::Bnc, perm,C::AbstractMatrix{<:Real},C0::AbstractVector{<:Real},nullity::Int=0)::Bool
    poly_additional = get_polyhedra(C,C0,nullity)
    poly = get_polyhedra(Bnc, perm)
    ins = intersect(poly,poly_additional)
    @show dim(ins)
    return !isempty(ins)
end

function feasible_vertieces_with_constraint(Bnc::Bnc, C::AbstractMatrix{<:Real},C0::AbstractVector{<:Real},nullity::Int=0;kwargs...)
    all_vtx = get_vertices(Bnc;kwargs...)
    feasible_vtx = Vector{eltype(all_vtx)}()
    for perm in all_vtx
        if check_feasibility(Bnc, perm,C,C0,nullity)
            push!(feasible_vtx, perm)
        end
    end
    return feasible_vtx
end

#-------------------------------------------------------------
#Other higher lever functions
#----------------------------------------------------------------
function summary_vertex(Bnc::Bnc, perm)
    idx= get_idx(Bnc, perm)
    perm = get_perm(Bnc, idx)
    is_real = get_vertex!(Bnc, idx).real
    nullity = get_nullity!(Bnc, idx)
    volume = get_volume!(model,perm)
    println("idx=$idx,perm=$perm, asymptotic=$is_real, nullity=$nullity")
    println("volume=$(volume[1]) +- $(volume[2])")
    println("Dominante condition")
    display.(show_dominant_condition(model,perm;log_space=false))
    println("x expression")
    try
        display.(show_expression_x(model,perm;log_space=false))
    catch
    end
    println("condition:")
    display.(show_condition_qK(model,perm;log_space=false))
    
    return nothing
end
function summary_vertices(Bnc::Bnc;kwargs...)
    vtx = get_vertices(Bnc;kwargs...)
    vtx .|> x->summary_vertex(Bnc,x)
    return nothing
end

