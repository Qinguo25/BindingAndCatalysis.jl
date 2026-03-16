# ============================================================
# Core data structures
# ============================================================

"""
Canonical hyperplane

Stored in canonical form with `u < v`:

    z_u - z_v + log(num/den) = 0

where `(num, den)` is the reduced integer ratio.
"""
struct Hyperplane_perm
    u::Int # fast access 
    v::Int # fast access
    num::Int # reduced positive integer
    den::Int # reduced positive integer
    c0::Float64 # pre-logarithm 
    crow::SparseVector{Int8,Int}      # +1 at u, -1 at v
    crow_neg::SparseVector{Int8,Int}  # +1 at v, -1 at u
end


"""
Helper struct for managing matrix operations.
- `J[i]`: positive columns in row i
- `choice_slot[i][p]`: local slot of column p inside J[i], or 0 if p ∉ J[i]
- `choice_map[i][t]`: all oriented inequalities for choosing p = J[i][t]
- `hyperplanes`: global deduplicated hyperplane pool
- `asymptotic`: all asymptotic regimes
- `feasible`: all regimes feasible under the weighted constraints
"""
struct Matrix_helper
    n::Int
    J::Vector{Vector{Int}}
    choice_slot::Vector{Vector{Int}}
    choice_logcoeff::Vector{Vector{Float64}}
    rowptr::Vector{Int}
    total_constraints::Int
    choice_map::Dict{Tuple{Int,Int},Int}
    hyperplanes::Vector{Hyperplane_perm}
end


"""
One oriented inequality induced by choosing p in row i.
If `sign == +1`, use the canonical side:
    crow * z + c0 > 0
If `sign == -1`, use the opposite side:
    crow_neg * z - c0 > 0
`competitor` is the losing column k compared against the chosen dominant p.
`oriented_c0 = log(L[i,p] / L[i,k])`
so the actual inequality is:
    z_p - z_k + oriented_c0 > 0
"""
struct ChoiceIneq
    hid::Int  # index into global hyperplane pool
    sign::Int8 # +1 for canonical side, -1 for opposite side

    #Fast access
    competitor::Int # fast access
    oriented_c0::Float64 # fast access.
end

"""
A feasible regime.

- `perm[i]` is the chosen dominant column in row i
- `P0[i] = log(L[i, perm[i]])`
- `is_asymptotic` means recession cone is full-dimensional
- `hyperplane_ids` and `signs` encode the regime inequalities row-by-row
"""
mutable struct Vertex{F}
    #--- Parent Bnc model reference ---
    bn::Union{AbstractBnc,Nothing} # Reference to the parent Bnc model

    # --- Initial / Identifying Properties ---
    perm::Vector{Int} # The regime vector
    perm_nullity::Int
    idx::Int # Index of the vertex in the Bnc.vertices list
    is_asymptotic::Bool # Whether the vertex is asymptotic or fake vertex.
    
    # --- Basic Calculated Properties ---
    # P::SparseMatrixCSC{Int, Int}
    # P0::Vector{F} 
    
    # M::SparseMatrixCSC{Int, Int}
    # M0::Vector{F} #

    # x_space_condition
    x_hyperplane_idx::Vector{Int}  # row partition could be checked from rowptr
    x_hyperplane_signs::Vector{Int8}


    # --- Expensive Calculated Properties ---
    
    nullity::Int
    H::SparseMatrixCSC{Float64, Int} # Taking inverse, can have Float.
    H0::Vector{F} 
    C_qK::SparseMatrixCSC{Float64, Int}
    C0_qK::Vector{F} 
    
    #---Realizibility Index
    volume::Volume

    # The inner constructor also needs to be updated for the parametric type
    function Vertex(;bn::Union{AbstractBnc,Nothing}=nothing, perm, P0, is_asymptotic, hyperplane_ids, signs, perm_nullity::T) where {T<:Integer,F<:Real}
        # _M_lu = lu(M, check=false) # It's good practice to ensure M is Float64 for LU
        # Use new{T} to construct an instance of Vertex{T}
        return new{F,T}(bn, perm,perm_nullity, 0 ,is_asymptotic, hyperplane_ids, signs,
            -1,
            SparseMatrixCSC{Float64, Int}(undef, 0, 0), # H
            Vector{F}(undef, 0),          # H0
            SparseMatrixCSC{Float64, Int}(undef, 0, 0), # C_qK
            Vector{F}(undef, 0),          # C0_qK
            Volume(0.0, 0.0) # volume
        )
    end
end


# ============================================================
# Small helpers
# ============================================================

@inline function _reduced_ratio(a::T, b::T) where {T<:Integer}
    g = gcd(a, b)
    return div(a, g), div(b, g)
end



# ============================================================
# Precomputation from L
# ============================================================

function _build_pool_and_choice_map(L::AbstractMatrix{Tv}) where {Tv<:Integer}
    d, n = size(L)
    J = Vector{Vector{Int}}(undef, d)
    choice_slot = [zeros(Int, n) for _ in 1:d]
    choice_logcoeff = Vector{Vector{Float64}}(undef, d)
    choice_map = Vector{Vector{Vector{ChoiceIneq}}}(undef, d)

    # Global deduplicated hyperplane pool
    key_to_id = Dict{Tuple{Int,Int,Tv,Tv}, Int}()
    hyperplanes = Hyperplane_perm{Tv}[]

    # First build J / choice_map
    @inbounds for i in 1:d
        Ji = Int[]
        sizehint!(Ji, n)
        for j in 1:n
            if L[i, j] > 0
                push!(Ji, j)
            end
        end
        isempty(Ji) && throw(ArgumentError("row $i of L has no positive entry"))

        J[i] = Ji
        choice_logcoeff[i] = [log(Float64(L[i, j])) for j in Ji]

        row_choices = Vector{Vector{ChoiceIneq}}(undef, length(Ji))

        for (t, p) in pairs(Ji)
            choice_slot[i][p] = t

            refs = Vector{ChoiceIneq}(undef, max(length(Ji) - 1, 0))
            ptr = 1
            Lp = L[i, p]

            for k in Ji
                k == p && continue
                Lk = L[i, k]

                # Canonicalize the hyperplane by ordering the variable pair.
                if p < k
                    u, v = p, k
                    num, den = _reduced_ratio(Lp, Lk)
                    sign = Int8(+1)
                else
                    u, v = k, p
                    num, den = _reduced_ratio(Lk, Lp)
                    sign = Int8(-1)
                end

                key = (u, v, num, den)
                hid = get(key_to_id, key, 0)

                if hid == 0
                    crow = sparsevec([u, v], Int8[1, -1], n)
                    crow_neg = sparsevec([v, u], Int8[1, -1], n)
                    c0 = log(Float64(num)) - log(Float64(den))
                    push!(hyperplanes, Hyperplane_perm{Tv}(u, v, num, den, c0, crow, crow_neg))
                    hid = length(hyperplanes)
                    key_to_id[key] = hid
                end

                refs[ptr] = ChoiceIneq(
                    hid,
                    sign,
                    k,
                    log(Float64(Lp)) - log(Float64(Lk))
                )
                ptr += 1
            end

            row_choices[t] = refs
        end

        choice_map[i] = row_choices
    end

    # Row block pointers for regime constraints:
    # row i contributes |J_i|-1 inequalities, independent of the chosen dominant.
    rowptr = Vector{Int}(undef, d + 1)
    rowptr[1] = 1
    @inbounds for i in 1:d
        rowptr[i + 1] = rowptr[i] + (length(J[i]) - 1)
    end
    total_constraints = rowptr[end] - 1
    
    return Matrix_helper(
        n, J, choice_slot, choice_logcoeff, rowptr, total_constraints,choice_map,hyperplanes
    )
end


# ============================================================
# Regime materialization
# ============================================================

function _regime_from_ordered_choice(
    order::Vector{Int},
    chosen_ord::Vector{Int},
    choice_slot::Vector{Vector{Int}},
    choice_logcoeff::Vector{Vector{Float64}},
    choice_map::Vector{Vector{Vector{ChoiceIneq}}},
    rowptr::Vector{Int},
    total_constraints::Int,
    is_asymptotic::Bool,
)
    d = length(order)
    n = length(choice_slot[1])   # perm 的取值范围应当是 1:n

    perm = Vector{Int}(undef, d)
    P0 = Vector{Float64}(undef, d)
    hyperplane_ids = Vector{Int}(undef, total_constraints)
    signs = Vector{Int8}(undef, total_constraints)

    @inbounds for r in 1:d
        i = order[r]
        p = chosen_ord[r]
        perm[i] = p

        t = choice_slot[i][p]
        P0[i] = choice_logcoeff[i][t]

        refs = choice_map[i][t]
        block_start = rowptr[i]
        for s in eachindex(refs)
            hyperplane_ids[block_start + s - 1] = refs[s].hid
            signs[block_start + s - 1] = refs[s].sign
        end
    end

    # count redundant elements in perm:
    # equivalent to length(perm) - length(unique(perm))
    perm_nullity = 0
    seen = falses(n)
    @inbounds for p in perm
        if seen[p]
            perm_nullity += 1
        else
            seen[p] = true
        end
    end

    return Vertex(;perm, P0, is_asymptotic, hyperplane_ids, signs, perm_nullity)
end


# ============================================================
# Asymptotic enumeration
# ============================================================

function _enumerate_asymptotic_regimes(L_helper::Matrix_helper)

    @unpack n, J, choice_slot, choice_logcoeff, choice_map, rowptr, total_constraints = L_helper
    d = length(J)

    order = sortperm(J, by = length, rev = true)

    J_ord = J[order]

    # Asymptotic graph: for choosing v in a row, add k -> v for all competitors k != v.
    graph = [Int[] for _ in 1:n]
    chosen_ord = Vector{Int}(undef, d)
    regimes = Regime[]

    # Reusable DFS stack for reachability
    stack = Vector{Int}(undef, n)
    visited_stamp = zeros(Int, n)
    stamp_ref = Ref(0)

    function reachable(start::Int, target::Int)::Bool
        stamp_ref[] += 1
        curstamp = stamp_ref[]
        top = 1
        stack[1] = start
        visited_stamp[start] = curstamp

        while top > 0
            u = stack[top]
            top -= 1
            u == target && return true

            @inbounds for w in graph[u]
                if visited_stamp[w] != curstamp
                    visited_stamp[w] = curstamp
                    top += 1
                    stack[top] = w
                end
            end
        end
        return false
    end

    function dfs(r::Int)
        if r > d
            push!(regimes, _regime_from_ordered_choice(
                order, chosen_ord, choice_slot, choice_logcoeff,
                choice_map, rowptr, total_constraints, true
            ))
            return
        end

        row_choices = J_ord[r]

        @inbounds for v in row_choices
            # Adding edges k -> v is valid iff v cannot already reach any competitor k.
            bad = false
            for k in row_choices
                k == v && continue
                if reachable(v, k)
                    bad = true
                    break
                end
            end
            bad && continue

            for k in row_choices
                k == v && continue
                push!(graph[k], v)
            end

            chosen_ord[r] = v
            dfs(r + 1)

            for k in reverse(row_choices)
                k == v && continue
                pop!(graph[k])
            end
        end
    end

    dfs(1)
    return regimes
end

# ============================================================
# Feasible-regime enumeration
# ============================================================

function _enumerate_all_regimes(
    L_helper::Matrix_helper,
    eps::Float64=Inf,
)
    @unpack n, J, choice_slot, choice_logcoeff, choice_map, rowptr, total_constraints = L_helper
    d = length(J)
    order = sortperm(J, by = length, rev = true)
    J_ord = J[order]

    # Weighted difference-constraint graph for feasibility:
    # from z_p - z_k + c0 > eps  =>  z_k < z_p + c0 - eps
    # so we add edge p -> k with weight (c0 - eps)
    weighted_edges = Vector{Vector{Vector{Tuple{Int,Float64}}}}(undef, d)
    @inbounds for i in 1:d
        by_choice = Vector{Vector{Tuple{Int,Float64}}}(undef, length(J[i]))
        for t in eachindex(J[i])
            refs = choice_map[i][t]
            edges = Vector{Tuple{Int,Float64}}(undef, length(refs))
            for s in eachindex(refs)
                ref = refs[s]
                edges[s] = (ref.competitor, ref.oriented_c0 - eps)
            end
            by_choice[t] = edges
        end
        weighted_edges[i] = by_choice
    end

    # Adjacency for weighted feasibility graph
    adj = [Vector{Tuple{Int,Float64}}() for _ in 1:n]

    # Also maintain the asymptotic graph incrementally,
    # so each feasible regime also gets a correct is_asymptotic flag.
    dag_adj = [Int[] for _ in 1:n]

    chosen_ord = Vector{Int}(undef, d)
    regimes = Regime[]

    # Reusable DFS stack for reachability on dag_adj
    stack = Vector{Int}(undef, n)
    visited_stamp = zeros(Int, n)
    stamp_ref = Ref(0)

    function dag_reachable(start::Int, target::Int)::Bool
        stamp_ref[] += 1
        curstamp = stamp_ref[]
        top = 1
        stack[1] = start
        visited_stamp[start] = curstamp

        while top > 0
            u = stack[top]
            top -= 1
            u == target && return true

            @inbounds for w in dag_adj[u]
                if visited_stamp[w] != curstamp
                    visited_stamp[w] = curstamp
                    top += 1
                    stack[top] = w
                end
            end
        end
        return false
    end

    function has_neg_cycle(seeds::Vector{Int})::Bool
        dist = fill(Inf, n)
        inq = falses(n)
        cnt = zeros(Int, n)
        q = Int[]
        sizehint!(q, max(length(seeds), 8))

        @inbounds for u in seeds
            if !inq[u]
                dist[u] = 0.0
                push!(q, u)
                inq[u] = true
            else
                dist[u] = 0.0
            end
        end

        head = 1
        while head <= length(q)
            u = q[head]
            head += 1
            inq[u] = false
            du = dist[u]

            @inbounds for (v, w) in adj[u]
                nd = du + w
                if nd + 1e-15 < dist[v]
                    dist[v] = nd
                    if !inq[v]
                        push!(q, v)
                        inq[v] = true
                        cnt[v] += 1
                        if cnt[v] > n
                            return true
                        end
                    end
                end
            end
        end

        return false
    end

    function dfs(r::Int, still_acyclic::Bool)
        if r > d
            push!(regimes, _regime_from_ordered_choice(
                order, chosen_ord, choice_slot, choice_logcoeff,
                choice_map, rowptr, total_constraints, still_acyclic
            ))
            return
        end

        i = order[r]
        row_choices = J_ord[r]

        @inbounds for v in row_choices
            t = choice_slot[i][v]

            # This only updates the asymptotic flag; it does NOT prune feasible branches.
            local_acyclic = still_acyclic
            if still_acyclic
                for k in row_choices
                    k == v && continue
                    if dag_reachable(v, k)
                        local_acyclic = false
                        break
                    end
                end
            end

            # Add weighted feasibility edges: v -> competitor
            oldlen_w = length(adj[v])
            append!(adj[v], weighted_edges[i][t])

            # Add asymptotic graph edges: competitor -> v
            for k in row_choices
                k == v && continue
                push!(dag_adj[k], v)
            end

            if !has_neg_cycle(row_choices)
                chosen_ord[r] = v
                dfs(r + 1, local_acyclic)
            end

            # Roll back weighted edges
            resize!(adj[v], oldlen_w)

            # Roll back asymptotic edges
            for k in reverse(row_choices)
                k == v && continue
                pop!(dag_adj[k])
            end
        end
    end

    dfs(1, true)
    return regimes
end

# ============================================================
# Top-level compiler
# ============================================================

"""
    compile_regime_catalog(L; eps=1e-9, dominance_ratio=Inf,
                           enumerate_asymptotic=true,
                           enumerate_feasible=true)

Compile all regime-related structures from a nonnegative integer matrix `L`.

Returns a `RegimeCatalog` containing:

1. `hyperplanes`:
   global deduplicated hyperplane pool

2. `choice_map` + `choice_slot`:
   from `(i,p)` to the corresponding hyperplanes and orientations

3. `asymptotic`:
   all regimes whose recession cone is full-dimensional

4. `feasible`:
   all regimes feasible under the weighted difference constraints

Notes:
- `dominance_ratio == Inf` means use `eps` directly
- otherwise, feasible mode uses `eps_eff = log(dominance_ratio)`
"""
function compile_regime_catalog(
    L::AbstractMatrix{Tv};
    eps::Real = 1e-9,
    dominance_ratio::Real = Inf,
    enumerate_asymptotic_only::Bool = false,
) where {Tv<:Integer}

    n = size(L,2)
    
    L_helper = _build_pool_and_choice_map(L)
    J = L_helper.J

    # Good pruning order: rows with more choices first
    if enumerate_asymptotic_only
        return _enumerate_asymptotic_regimes(L_helper)
    else
        if dominance_ratio != Inf && dominance_ratio < 1
            throw(ArgumentError("dominance_ratio must be >= 1 or Inf"))
        end
        eps_eff = dominance_ratio == Inf ? Float64(eps) : log(Float64(dominance_ratio))
        return _enumerate_all_regimes(L_helper, eps_eff)
    end
end

function find_all_vertices!(
    model::Bnc;
    )
    model.vertices_data = _enumerate_all_regimes(model._L_helper, Inf)
    for (i, v) in enumerate(model.vertices_data)
        v.idx = i
        v.bn = model
    end
    
end








"""
Build a sparse P matrix from a regime's perm.

P[i, perm[i]] = 1.
"""
function regime_P(reg::Regime, n::Int)
    return regime_P(reg.perm, n)
end

function regime_P(perm::AbstractVector{<:Integer}, n::Integer)
    d = length(perm)
    I = collect(1:d)
    J = copy(perm)
    V = ones(Int8, d)
    return sparse(I, J, V, d, n)
end
"""
Convenience: return only the perms.
"""
perms(regimes::Vector{Regime}) = [copy(r.perm) for r in regimes]

"""
Convenience: return only the P0 vectors.
"""
P0s(regimes::Vector{Vertex}) = [copy(r.P0) for r in regimes]
