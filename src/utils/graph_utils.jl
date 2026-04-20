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
    sources_sinks_from_paths(paths) -> (Vector{Int}, Vector{Int})

Return unique source and sink vertices for a collection of paths.
"""
function sources_sinks_from_paths(paths::AbstractVector{<:AbstractVector{<:Integer}})::Tuple{Vector{Int}, Vector{Int}}
    sources = unique(Int(first(p)) for p in paths)
    sinks = unique(Int(last(p)) for p in paths)
    return sources, sinks
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
