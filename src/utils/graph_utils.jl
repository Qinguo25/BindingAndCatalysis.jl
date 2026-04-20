"""
    get_sources(g::AbstractGraph) -> Set{Int}

Return source vertices with zero indegree.
"""
get_sources(g::AbstractGraph) = Set(v for v in vertices(g) if indegree(g, v) == 0)
"""
    get_sinks(g::AbstractGraph) -> Set{Int}

Return sink vertices with zero outdegree.
"""
get_sinks(g::AbstractGraph)   = Set(v for v in vertices(g) if outdegree(g, v) == 0)
"""
    get_sources_sinks(g::AbstractGraph) -> (Set{Int}, Set{Int})

Return sources and sinks for a graph.
"""
get_sources_sinks(g::AbstractGraph) = (get_sources(g), get_sinks(g))

"""
    get_sources_sinks(model::Bnc, g::AbstractGraph) -> (Vector{Int}, Vector{Int})

Return sources and sinks while excluding singular regimes.
"""
function get_sources_sinks(model::Bnc, g::AbstractGraph)
    sources_all = get_sources(g) 
    sinks_all   = get_sinks(g) 
    common_vs = intersect(sources_all, sinks_all)
    filter!(common_vs) do v
        get_nullity(model, v) > 0
    end
    sources = setdiff(sources_all, common_vs)
    sinks = setdiff(sinks_all, common_vs)
    return (collect(sources), collect(sinks))
end

# 只遍历子图：sources 可达 & 能到 sinks
"""
    _reachable_from_sources(g::AbstractGraph, sources) -> Vector{Bool}

Return a boolean mask of vertices reachable from sources.
"""
function _reachable_from_sources(g::AbstractGraph, sources::AbstractVector{Int})
    n = nv(g)
    seen = falses(n)
    stack = Int[]
    for s in sources
        if !seen[s]
            seen[s] = true
            push!(stack, s)
            while !isempty(stack)
                v = pop!(stack)
                for nb in outneighbors(g, v)
                    if !seen[nb]
                        seen[nb] = true
                        push!(stack, nb)
                    end
                end
            end
        end
    end
    return seen
end

"""
    _can_reach_sinks(g::AbstractGraph, sinks) -> Vector{Bool}

Return a boolean mask of vertices that can reach sinks.
"""
function _can_reach_sinks(g::AbstractGraph, sinks::AbstractVector{Int})
    n = nv(g)
    seen = falses(n)
    stack = Int[]
    for t in sinks
        if !seen[t]
            seen[t] = true
            push!(stack, t)
            while !isempty(stack)
                v = pop!(stack)
                for nb in inneighbors(g, v)   # 反向走
                    if !seen[nb]
                        seen[nb] = true
                        push!(stack, nb)
                    end
                end
            end
        end
    end
    return seen
end

"""
    _enumerate_paths(g; sources, sinks) -> Vector{Vector{Int}}

Enumerate all paths in a DAG from `sources` to `sinks`.
"""
function _enumerate_paths(
    g::AbstractGraph;
    sources::AbstractVector{Int},
    sinks::AbstractVector{Int},
)::Vector{Vector{Int}}

    @info "sources: $sources"
    @info "sinks: $sinks"
    n = nv(g)

    # 剪枝：只处理相关子图
    fromS = _reachable_from_sources(g, sources)
    toT   = _can_reach_sinks(g, sinks)
    active = fromS .& toT

    is_sink = falses(n)
    @inbounds for t in sinks
        is_sink[t] = true
    end

    # 拓扑排序（DAG）
    topo = topological_sort_by_dfs(g)   # Graphs.jl
    # memo[v] = Vector{Vector{Int}} 或 nothing
    memo = Vector{Union{Nothing, Vector{Vector{Int}}}}(undef, n)
    fill!(memo, nothing)

    @info "Start enumerating paths from sources to sinks. This may take a while if there are many paths."
    # 逆拓扑：先算子节点，再算父节点

    @info "Total vertices to process in topological order: $(length(topo))"
    @showprogress for v in Iterators.reverse(topo)
        active[v] || continue

        if is_sink[v]
            memo[v] = Vector{Vector{Int}}(undef, 1)
            memo[v][1] = [v]
            continue
        end

        # 收集所有 nb 的路径，并在前面加 v
        acc = Vector{Vector{Int}}()
        # 你也可以在这里做 sizehint!（需要先统计 path 数量，会多一次循环；看你取舍）
        for nb in outneighbors(g, v)
            active[nb] || continue
            paths_nb = memo[nb]
            paths_nb === nothing && continue
            for p in paths_nb
                L = length(p)
                np = Vector{Int}(undef, L + 1)
                np[1] = v
                @inbounds copyto!(np, 2, p, 1, L)
                push!(acc, np)
            end
        end

        memo[v] = isempty(acc) ? nothing : acc
    end

    # 汇总 sources 的结果
    @info "Finished enumerating paths. Now collecting paths from sources. Total sources: $(length(sources))"
    out = Vector{Vector{Int}}()
    @showprogress for s in sources
        active[s] || continue
        ps = memo[s]
        ps === nothing && continue
        append!(out, ps)
    end

    sort!(out)
    return out
end


"""
    graph_from_paths(paths; nv=nothing) -> SimpleDiGraph

Construct a directed graph from a collection of vertex paths.
"""
function graph_from_paths(paths::AbstractVector{<:AbstractVector{<:Integer}}, nv=nothing)::SimpleDiGraph
    nv = nv === nothing ? maximum(Iterators.flatten(paths)) : nv
    grh = SimpleDiGraph(nv)
    for p in paths
        n = length(p)
        for i in 1:(n - 1)
            add_edge!(grh, p[i], p[i + 1])
        end
    end
    return grh
end


"""
    vector_difference(v1, v2) -> Vector

Summarize differences between two vectors as counts of value transitions.
"""
function vector_difference(v1::AbstractVector{T}, v2::AbstractVector{T}) where T
    diff_index = findall(v1 .!= v2)
    mp = countmap(zip(v1[diff_index], v2[diff_index]))
    mp_sort = sort(collect(mp), by=x -> x.second, rev=true)
    return mp_sort
end

"""
    norm_vec_space(x::AbstractVector{<:Real}) -> Vector{Float64}

Normalize a vector by the median nonzero magnitude.
"""
function norm_vec_space(x::AbstractVector{<:Real})::Vector{Float64}
    num_to_norm = median!(filter!(>(1e-9), abs.(x)))
    return x ./ num_to_norm
end

function compress_adjacency(
    A::SparseMatrixCSC,
    keep::AbstractVector{<:Integer};
    drop_stored_zeros::Bool=true,
)
    n = size(A, 1)
    size(A, 2) == n || throw(ArgumentError("A must be square"))

    A2 = drop_stored_zeros ? dropzeros(A) : A
    keep_set = Set(keep)
    length(keep_set) == length(keep) || throw(ArgumentError("keep contains duplicates"))
    all(1 <= v <= n for v in keep) || throw(ArgumentError("keep contains out-of-range indices"))

    m = length(keep)
    keep_pos = zeros(Int, n)
    for (i, v) in enumerate(keep)
        keep_pos[v] = i
    end

    iskeep = falses(n)
    isdrop = trues(n)
    for v in keep
        iskeep[v] = true
        isdrop[v] = false
    end

    rows = rowvals(A2)
    I = Int[]
    J = Int[]

    @inbounds for j in keep
        jj = keep_pos[j]
        for p in nzrange(A2, j)
            i = rows[p]
            if i != j && iskeep[i]
                ii = keep_pos[i]
                push!(I, ii)
                push!(J, jj)
            end
        end
    end

    visited = falses(n)
    stack = Int[]
    touched = Int[]
    touched_mark = zeros(Int, m)
    stamp = 0

    @inbounds for s in 1:n
        if !isdrop[s] || visited[s]
            continue
        end
        empty!(stack)
        push!(stack, s)
        visited[s] = true

        empty!(touched)
        stamp += 1

        while !isempty(stack)
            u = pop!(stack)
            for p in nzrange(A2, u)
                v = rows[p]
                v == u && continue
                if isdrop[v]
                    if !visited[v]
                        visited[v] = true
                        push!(stack, v)
                    end
                else
                    kv = keep_pos[v]
                    if kv != 0 && touched_mark[kv] != stamp
                        touched_mark[kv] = stamp
                        push!(touched, kv)
                    end
                end
            end
        end

        t = length(touched)
        for a in 1:(t - 1)
            ia = touched[a]
            for b in (a + 1):t
                ib = touched[b]
                push!(I, ia); push!(J, ib)
                push!(I, ib); push!(J, ia)
            end
        end
    end

    B = sparse(I, J, fill(true, length(I)), m, m, |)
    if nnz(B) > 0
        B = B - spdiagm(0 => diag(B))
        dropzeros!(B)
    end
    return B
end

function connected_components_sparse(A::SparseMatrixCSC)
    n = size(A, 1)
    size(A, 2) == n || throw(ArgumentError("A must be square"))

    rows = rowvals(A)
    visited = falses(n)
    labels = zeros(Int, n)
    groups = Vector{Vector{Int}}()
    cid = 0

    for s in 1:n
        visited[s] && continue
        cid += 1
        labels[s] = cid
        stack = [s]
        visited[s] = true
        comp = Int[]

        while !isempty(stack)
            u = pop!(stack)
            push!(comp, u)
            for p in nzrange(A, u)
                v = rows[p]
                if v != u && !visited[v]
                    visited[v] = true
                    labels[v] = cid
                    push!(stack, v)
                end
            end
        end
        push!(groups, comp)
    end

    return Set.(groups), labels
end

"""
    group_sum(keys, vals; sort_values=true) -> Vector{Tuple}

Group values by keys, returning indices, key, and summed values.
"""
function group_sum(keys::AbstractVector{I}, vals::AbstractVector{J}; sort_values::Bool=true)::Vector{Tuple{Vector{Int}, I, J}} where {I,J}
    @assert length(keys) == length(vals)
    dict = Dict{I,J}()
    index_dict = Dict{I, Vector{Int}}()

    @inbounds for (i, (k, v)) in enumerate(zip(keys, vals))
        dict[k] = get(dict, k, zero(v)) + v
        push!(get!(index_dict, k, Int[]), i)
    end

    dict_vec = collect(dict)
    if sort_values
        sort!(dict_vec, by=x -> x[2], rev=true)
    end

    result = Vector{Tuple{Vector{Int}, I, J}}(undef, length(dict))
    for i in eachindex(dict_vec)
        key, sum_val = dict_vec[i]
        group = index_dict[key]
        result[i] = (group, key, sum_val)
    end
    return result
end

function group_sum(
    keys::AbstractVector{I},
    vals::AbstractVector{Nothing};
    sort_values::Bool=true,
)::Vector{Tuple{Vector{Int}, I, Nothing}} where {I}
    @assert length(keys) == length(vals)

    index_dict = Dict{I, Vector{Int}}()
    order = I[]
    @inbounds for (i, k) in enumerate(keys)
        if !haskey(index_dict, k)
            push!(order, k)
        end
        push!(get!(index_dict, k, Int[]), i)
    end

    if sort_values
        sort!(order, by=k -> length(index_dict[k]), rev=true)
    end
    return [(index_dict[k], k, nothing) for k in order]
end
