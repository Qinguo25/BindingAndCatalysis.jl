#----------------Function to finding dominence regime，one of the hardest part to optimize----------------------

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
function find_valid_regime(idx::Vector{Vector{Int}}, d::Int, n::Int)::Vector{Vector{Int}}
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

function find_valid_regime(L::Matrix{Int})
    d, n = size(L)
    idx = [[idx for (idx, value) in enumerate(row) if value != 0] for row in eachrow(L)]
    find_valid_regime(idx, d, n)
end

find_valid_regime(Bnc::Bnc) = find_valid_regime(Bnc._valid_L_idx, Bnc.d, Bnc.n)

#-----------------------------------------------fucntions with dominance regime-------------------------------------------------------

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
#----end of helper functions.



function Mtd_a_from_regime(Bnc::Bnc, regime::Vector{Int}; check::Bool=true)::Tuple{Matrix{Int}, Vector{Int}}
    # Generate the ̃M matrix from the given regime
    !check || _check_valid_idx(regime, Bnc.L) # check if the regime is valid for the given L matrix
    M = zeros(Int, Bnc.d, Bnc.n)
    a = zeros(Int, Bnc.d) # buffer for the a values
    for i in 1:Bnc.d
        M[i, regime[i]] = 1
        a[i] = Bnc.L[i, regime[i]]
    end
    return M,a
end

function M_from_regime(Bnc::Bnc, regime::Vector{Int}; check::Bool=true)::Matrix{Int}
    # Generate the ̃M matrix from the given regime
    M = zeros(Int, Bnc.d, Bnc.n)
    !check || _check_valid_idx(regime, Bnc.L)
    for i in 1:Bnc.d
        M[i, regime[i]] = Bnc.L[i, regime[i]]
    end
    M[Bnc.d+1:end, :] .= Bnc.N
    return M
end



function ∂logqK_∂logx_regime(Bnc::Bnc; regime::Union{Vector{Int}, Nothing}=nothing,
    Mtd::Union{Matrix{Int}, Nothing}=nothing,
    M::Union{Matrix{Int}, Nothing}=nothing,
    check::Bool=true)::Matrix{<:Real}
    """
    Calculate the derivative of log(qK) with respect to log(x) given regime
    check: if true, check if the regime is valid for the L matrix
    regime: the regime vector, if not provided , Mtd  will be derived from Mtd or M
    
    Return:
    - logder_qK_x: the derivative of log(qK) with respect to log(x)
    """
    if isnothing(Mtd)
        if isnothing(regime)
            if isnothing(M)
                @error("Either regime or M/Mtd must be provided")
            else
                Mtd = sign.(M)
            end
        else
            (Mtd , _) = Mtd_a_from_regime(Bnc, regime; check=check)
        end
    end

    return vcat(Mtd, Bnc.N)
end

function ∂logx_∂logqK_regime(Bnc::Bnc;
    logder_qK_x::Union{Matrix{<:Real},Nothing}=nothing,
    regime::Union{Vector{Int}, Nothing}=nothing,
    Mtd::Union{Matrix{Int}, Nothing}=nothing,
    M::Union{Matrix{Int}, Nothing}=nothing,
    check::Bool=true)::Tuple{Matrix{<:Real},Int}
    """
    Calculate the derivative of log(qK) with respect to log(x) given regime
    check: if true, check if the regime is valid for the L matrix
    regime: the regime vector, if not provided, will be derived from Mtd or M

    Return:
    - logder_x_qK: the derivative of log(x) with respect to log(qK)
    - singularity: singularity of the logder_qK_x.
    """

    logder_qK_x = isnothing(logder_qK_x) ? ∂logqK_∂logx_regime(Bnc; regime=regime,Mtd=Mtd,M=M, check=check) : logder_qK_x
    logder_qK_x_fac = lu(logder_qK_x,check=false)
    if issuccess(logder_qK_x_fac) # Lu successfully.
        return inv(logder_qK_x_fac),0  # singularity is 0, not singular
    else
        return _adj_singular_matrix(logder_qK_x) .* Bnc.direction # calculate the adj matrix, singularity is calculated and returned,
    end
end



@kwdef struct Regime
    regime::Vector{Int} # The regime vector
    logder_qK_x::Matrix{<:Real} # The derivative of log(qK) with respect to log(x)
    logder_x_qK::Matrix{<:Real} # The derivative of log(x) with respect to log(qK)
    singularity::Int # Whether logder_x_qK is singular
    # direction::Int # If singular, direction defines its direction.
    a::Vector{Int} # Values of L on regime defined position
end



function create_regime(Bnc::Bnc; regime::Vector{Int})::Regime
    Mtd, a = Mtd_a_from_regime(Bnc, regime; check=true)
    logder_qK_x = ∂logqK_∂logx_regime(Bnc; Mtd=Mtd)
    logder_x_qK, singularity = ∂logx_∂logqK_regime(Bnc; logder_qK_x=logder_qK_x)
    return Regime(regime = regime, 
        logder_qK_x = logder_qK_x, 
        logder_x_qK = logder_x_qK, 
        singularity = singularity, 
        # direction = Bnc.direction,
        a = a)
end

x_ineq_mtx(Bnc::Bnc; regime::Regime, check::Bool=true) =  x_ineq_mtx(Bnc; regime=regime.regime, check=check)
function x_ineq_mtx(Bnc::Bnc; regime::Vector{Int}, check::Bool=true)::Matrix{<:Real}
    """
    return a matrix of ineq in x space for regime expressed as Ax < 0
    """
    !check || _check_valid_idx(regime, Bnc.L) # check if the regime is valid for the given L matrix
    idx = Set{Tuple{Int,Int}}()
    for (valid_idx,rgm) in zip(Bnc._valid_L_idx, regime)
        for i in valid_idx
            if i != rgm
            push!(idx, (rgm,i))
            end
        end
    end
    mtx = zeros(eltype(Bnc.L), length(idx), Bnc.n)
    for (i, (rgm, j)) in enumerate(idx)
        mtx[i, rgm] = -1
        mtx[i, j] = 1
    end
    return mtx
end


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