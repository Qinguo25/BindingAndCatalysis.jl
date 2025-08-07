#------------------------------------------------------------------------------
#   Function to finding dominence regime，one of the hardest part to optimize
# -------------------------------------------------------------------------------


function have_cyclic_at_node(g::Vector{Vector{Int}}, node::Int, len::Int)::Bool
    # DFS algorithm, 
    # modified from https://github.com/JuliaGraphs/Graphs.jl/blob/2d6f4d56b06cb597ebd5c40c5a8db783f1b83991/src/traversals/dfs.jl#L4-L11
    # 0 if not visited, 1 if in the current dfs path, 2 if fully explored
    vcolor = zeros(UInt8, len)
    vertex_stack = [node]
    while !isempty(vertex_stack)
        u = vertex_stack[end]
        if vcolor[u] == 0
            vcolor[u] = 1
            for n in g[u]
                # we hit a loop when reaching back a vertex of the main path
                if vcolor[n] == 1
                    return true
                elseif vcolor[n] == 0
                    # we store neighbors, but these are not yet on the path
                    push!(vertex_stack, n)
                end
            end
        else
            pop!(vertex_stack)
            if vcolor[u] == 1
                vcolor[u] = 2
            end
        end
    end
    return false
end

function find_all_vertices(idx::Vector{Vector{Int}}, d::Int, n::Int)::Vector{Vector{Int}}
    graph = [Vector{Int}() for _ in 1:n]
    choices = Vector{Int}(undef, d)
    results = Vector{Vector{Int}}()
    """
    A backtracking algorithm to find all valid regimes given the index of valid choices for each row.
    idx: valid choices index for each row,
    d: number of rows of initial matrix,
    n: number of columns of initial matrix, or we can say number of nodes for this application.
    """
    function backtrack!(i)
        # All rows are fine
        if i == d + 1
            # @show choices
            push!(results, copy(choices))
            return nothing
        end

        for v in idx[i]
            # add edges for current row. and record.

            target_nodes = [w for w in idx[i] if w != v] # target_nodes

            for node in target_nodes
                push!(graph[node], v) # add edge node -> v
            end

            if ~have_cyclic_at_node(graph, v, n)
                choices[i] = v
                backtrack!(i + 1)
            end

            for node in target_nodes
                pop!(graph[node])
            end
        end
    end
    backtrack!(1)
    return results
end

function find_all_vertices(L::Matrix{Int})
    d, n = size(L)
    idx = [[idx for (idx, value) in enumerate(row) if value != 0] for row in eachrow(L)]
    find_all_vertices(idx, d, n)
end

function find_all_vertices(Bnc::Bnc; recalculate::Bool=false)
    if isempty(Bnc.vertices) || recalculate
        Bnc.vertices = find_all_vertices(Bnc._valid_L_idx, Bnc.d, Bnc.n)
    end
        return Bnc.vertices
end

#-----------------------------------------------fucntions with dominance regime-------------------------------------------------------

#----end of helper functions.



"""
Creates the P and P0 matrices from a permutation.
"""
function _calculate_P_P0(Bnc::Bnc, perm::Vector{Int})::Tuple{Matrix{Int}, Vector{<:Real}}
    P = zeros(Int, Bnc.d, Bnc.n)
    P0 = zeros(Int, Bnc.d)
    for i in 1:Bnc.d
        P[i, perm[i]] = 1
        P0[i] = log10(Bnc.L[i, perm[i]])
    end
    return P, P0
end

# function _calculate_Ptd(Bnc::Bnc, perm::Vector{Int})::Matrix{Int}
#     # Generate the ̃P matrix from the given regime
#     # !check || _check_valid_idx(perm, Bnc.L)
#     Ptd = zeros(Int, Bnc.d, Bnc.n)
#     for i in 1:Bnc.d
#         Ptd[i, perm[i]] = Bnc.L[i, perm[i]]
#     end
#     Ptd[Bnc.d+1:end, :] .= Bnc.N
#     return Ptd
# end

function _calculate_C_C0_x(Bnc::Bnc,perm::Vector{Int})::Tuple{Matrix{Int}, Vector{<:Real}}
    """
    return a matrix of ineq in x space for regime expressed as Clogx+ c0> 0
    (logic seems to be complicate.)
    """
    # , check::Bool=true
    # !check || _check_valid_idx(regime, Bnc.L) # check if the regime is valid for the given L matrix
    
    num_ineq = Bnc._val_num_L - Bnc.d # inequalities counts from L's number of values minus d.
    c_mtx = zeros(Int, num_ineq, Bnc.n) # initialize the c_mtx
    c0 = Vector{Rational{Int}}(undef, num_ineq) # initialize the c0 vector
    row_ptr = Bnc._Lt_sparse.colptr .- (0:Bnc.d) # From _Lt_sparse.colptr, we can get the row start index for each original row.

    for (i,valid_idx,rgm,row_block_start) in zip(1:Bnc.d, Bnc._valid_L_idx, perm, row_ptr)
        # i: block_index, each original L's row is a block.
        # valid_idx: all the valid indices for the current block, Vector{Vector{Int}}.
        # rgm: the current regime index.
        # row_block_start: the start row idx for the current block in c_mtx.
        # k = 0 # finished rows count for the current block, used to update row.
        
        # Within block
        row = row_block_start # current row index in c_mtx, start from the block start index.
        for col in valid_idx
            if col != rgm
                # Calculate the correct row index for the output matrix
                c_mtx[row, col] = -1
                c_mtx[row, rgm] = 1
                c0[row] = log10(Bnc.L[i, rgm] / Bnc.L[i, col])
                row += 1
            end
        end
    end
    return c_mtx,c0
end


"""
Creates a new, partially-filled Vertex object.
This function performs the initial, less expensive calculations.
"""

function _create_vertex(Bnc::Bnc, perm::Vector{Int})::Vertex
    P, P0 = _calculate_P_P0(Bnc, perm)
    C_x, C0_x = _calculate_C_C0_x(Bnc, perm)

    M = vcat(P, Bnc.N)
    M0 = vcat(P0, zeros(Bnc.r))

    # Initialize a partial vertex. "Full" properties are empty placeholders.
    return Vertex(
        perm = perm,
        M = M, M0 = M0, P = P, P0 = P0, C_x = C_x, C0_x = C0_x
    )
end

function _ensure_full_properties!(vtx::Vertex)
    # Check if already calculated
    if !isempty(vtx.H)
        return
    end

    if vtx.singularity == -1 # unsuccessful LU decomposition, means singular
        vtx.H, vtx.singularity = _adj_singular_matrix(vtx.M)
    elseif vtx.singularity == 0
        vtx.H = inv(vtx._M_lu) # Calculate the inverse matrix from pre-computed LU decomposition of M
        vtx.H0 = vtx.H * vtx.M0
        vtx.C_qK = vtx.C_x * vtx.H
        vtx.C0_qK = vtx.C0_x - vtx.C_x * vtx.H0 # Correctly use vtx.C0_x
    end
end

"""
Retrieves a vertex from cache or creates it if it doesn't exist.
"""
function get_vertex!(Bnc::Bnc, perm::Vector{Int})::Vertex
    return get!(Bnc.vertices_data, perm) do
        _create_vertex(Bnc, perm)
    end
end


"""
Gets P and P0, creating the vertex if necessary.
"""
function get_P_P0!(Bnc::Bnc; perm::Vector{Int})
    vtx = get_vertex!(Bnc, perm)
    return vtx.P, vtx.P0
end

"""
Gets M and M0, creating the vertex if necessary.
"""
function get_M_M0!(Bnc::Bnc; perm::Vector{Int})
    vtx = get_vertex!(Bnc, perm)
    return vtx.M, vtx.M0
end

"""
Gets C_x and C0_x, creating the vertex if necessary.
"""
function get_C_C0_x!(Bnc::Bnc; perm::Vector{Int})
    vtx = get_vertex!(Bnc, perm)
    return vtx.C_x, vtx.C0_x
end

"""
Gets C_qK and C0_qK, ensuring the full vertex is calculated.
"""
function get_C_C0_qK!(Bnc::Bnc; perm::Vector{Int})
    vtx = get_vertex!(Bnc, perm)
    _ensure_full_properties!(vtx)
    if vtx.singularity == 0
        @error("Vertex is singular, cannot get C_qK and C0_qK")
    end
    return vtx.C_qK, vtx.C0_qK
end


"""
Gets H and H0, ensuring the full vertex is calculated.
"""
function get_H_H0!(Bnc::Bnc; perm::Vector{Int})
    vtx = get_vertex!(Bnc, perm)
    _ensure_full_properties!(vtx)
    if vtx.singularity == 0
        @error("Vertex is singular, cannot get H0")
    end # This will compute if needed
    return vtx.H, vtx.H0
end

function get_H!(Bnc::Bnc; perm::Vector{Int})
    vtx = get_vertex!(Bnc, perm)
    _ensure_full_properties!(vtx)
    if vtx.singularity > 1
        @error("Vertex's singularity is bigger than 1, cannot get H")
    end # This will compute if needed
    return vtx.H
end

function get_all_neighbors!(Bnc::Bnc; perm::Vector{Int})
    # Get the neighbors of the vertex represented by perm
    vtx = get_vertex!(Bnc, perm)
    if isempty(vtx.neighbors)
        
    end
    return vtx.P
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