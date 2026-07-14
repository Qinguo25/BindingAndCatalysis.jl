function _calc_polyhedra_for_path(
    model::Bnc, paths::AbstractVector{<:AbstractVector{<:Integer}}, change_qK_idx::Integer
)::Vector{Polyhedron}
    el_dim = BitSet((change_qK_idx,))

    node_polyhedra = let
        unique_rgms = unique(vcat(paths...))
        dic = Dict{Int, Polyhedron}()
        for r in unique_rgms
            dic[Int(r)] = get_polyhedron(model, r)
        end
        dic
    end

    edges, edge_paths = _build_path_edge_index(paths)
    edge_poly = Vector{Polyhedron}(undef, length(edges))
    @info "Start building polyhedra for edges (total: $(length(edges)))"
    @showprogress Threads.@threads for i in eachindex(edges)
        (u, v) = edges[i]
        edge_poly[i] = _poly_intersect_eliminate(
            node_polyhedra[u], node_polyhedra[v], el_dim; canonicalize=false
        )
    end

    out = Vector{Polyhedron}(undef, length(edge_paths))
    @info "Start building polyhedra for paths (total: $(length(edge_paths)))"
    @showprogress Threads.@threads for i in eachindex(edge_paths)
        if isempty(edge_paths[i])
            out[i] = _clean_polyhedron!(
                _poly_eliminate(
                    node_polyhedra[Int(first(paths[i]))], el_dim; canonicalize=false
                ),
            )
        else
            out[i] = _clean_polyhedron!(
                _poly_intersect_many(edge_poly[edge_paths[i]]; canonicalize=false)
            )
        end
    end
    return out
end

function _calc_polyhedra_for_path(
    model::Bnc, path::AbstractVector{<:Integer}, change_qK
)::Polyhedron
    change_qK_idx = change_qK isa Integer ? Int(change_qK) : locate_sym_qK(model, change_qK)
    return _calc_polyhedra_for_path(model, [Int.(path)], change_qK_idx)[1]
end

function _empty_simo_condition(grh::SIMOPaths)::Polyhedron
    n_base = base_dimension(grh.fiber_problem)
    return _build_polyhedron_from_C_C0(
        zeros(Float64, 1, n_base), [-1.0], 0; canonicalize=true
    )
end

function _calc_polyhedra_for_paths_pair_memo_dag!(
    grh::SIMOPaths, path_idxs::AbstractVector{<:Integer}
)::Vector{Polyhedron}
    indices = Int.(path_idxs)
    isempty(indices) && return Polyhedron[]
    backend = grh.condition_backend

    root_pairs = unique(
        (Int(first(grh.rgm_paths[idx])), Int(last(grh.rgm_paths[idx]))) for idx in indices
    )
    sort!(root_pairs)
    uncached_roots = filter(pair -> !_pair_is_cached(backend, pair[1], pair[2]), root_pairs)
    isempty(uncached_roots) || _find_all_path_conditions_dag!(backend, uncached_roots)

    out = Vector{Polyhedron}(undef, length(indices))
    for (position, idx) in enumerate(indices)
        path = grh.rgm_paths[idx]
        conditions = _pair_conditions(backend, Int(first(path)), Int(last(path)))
        condition =
            isnothing(conditions) ? nothing : get(conditions, _path_key(path), nothing)
        if isnothing(condition)
            out[position] = _empty_simo_condition(grh)
            grh.path_feasible[idx] = false
        else
            out[position] = condition
            grh.path_feasible[idx] = !isempty(condition)
        end
    end
    return out
end

function get_polyhedra(
    grh::SIMOPaths, pth_idx::Union{AbstractVector, Nothing}=nothing
)::Vector{Polyhedron}
    path_idxs = get_indices(grh, pth_idx)
    path_idxs_to_calc = _path_indices_to_calculate(grh.path_polys_is_calc, path_idxs)
    isempty(path_idxs_to_calc) && return grh.path_polys[path_idxs]

    polys = _calc_polyhedra_for_paths_pair_memo_dag!(grh, path_idxs_to_calc)
    grh.path_polys[path_idxs_to_calc] .= polys
    grh.path_polys_is_calc[path_idxs_to_calc] .= true

    return grh.path_polys[path_idxs]
end

get_polyhedron(grh::SIMOPaths, pth) = get_polyhedra(grh, [get_idx(grh, pth)])[1]

function is_feasible(grh::SIMOPaths, pth)
    idx = get_idx(grh, pth)
    get_polyhedron(grh, idx)
    return grh.path_feasible[idx]::Bool
end

function get_conditional_slice_types(
    grh::SIMOPaths, pth_idx::Union{AbstractVector, Nothing}=nothing
)
    path_idxs = get_indices(grh, pth_idx)
    conditions = get_polyhedra(grh, path_idxs)
    ambient_dim = base_dimension(grh.fiber_problem)
    return map(path_idxs, conditions) do idx, condition
        status = _poly_dim_status(condition; ambient_dim=ambient_dim)
        ConditionalSliceType(
            OrderedRegimePath(grh.rgm_paths[idx]),
            condition,
            status.feasible,
            status.dim,
            status.full_dim,
        )
    end
end

function _resolve_simo_rebase_mat(grh::SIMOPaths; rebase_K::Bool=false, rebase_mat=nothing)
    if !isnothing(rebase_mat)
        @assert !rebase_K "Cannot specify both rebase_K and providing rebase_mat"
        return rebase_mat
    elseif rebase_K
        bn = get_binding_network(grh)
        Q = rebase_mat_lgK(bn.N)
        return blockdiag(spdiagm(fill(Rational(1), bn.d - 1)), Q)
    else
        return nothing
    end
end

function get_volumes(
    grh::SIMOPaths,
    pth_idx::Union{AbstractVector, Nothing}=nothing;
    rebase_K=false,
    rebase_mat=nothing,
    recompute=false,
    kwargs...,
)
    _reject_renamed_keywords(kwargs)
    path_idxs = get_indices(grh, pth_idx)
    path_idxs_to_calculate = _path_indices_to_calculate(
        grh.path_volume_is_calc, path_idxs; recompute=recompute
    )

    if !isempty(path_idxs_to_calculate)
        resolved_rebase_mat = _resolve_simo_rebase_mat(
            grh; rebase_K=rebase_K, rebase_mat=rebase_mat
        )
        polys = get_polyhedra(grh, path_idxs_to_calculate)
        rlts = calc_volume(polys; rebase_mat=resolved_rebase_mat, kwargs...)
        for (i, idx) in enumerate(path_idxs_to_calculate)
            grh.path_volume[idx] = rlts[i]
            grh.path_volume_is_calc[idx] = true
        end
    end
    return grh.path_volume[path_idxs]
end

function get_volume(grh::SIMOPaths, pth; kwargs...)
    return get_volumes(grh, [get_idx(grh, pth)]; kwargs...)[1]
end
