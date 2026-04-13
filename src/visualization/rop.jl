function _lock_current_limits!(ax::Axis)
    r = ax.finallimits[]
    x0, y0 = r.origin
    wx, wy = r.widths
    limits!(ax, x0, x0 + wx, y0, y0 + wy)
end

function _lock_current_limits!(ax::Axis3)
    r = ax.targetlimits[]
    x0, y0, z0 = r.origin
    wx, wy, wz = r.widths
    limits!(ax, x0, x0 + wx, y0, y0 + wy, z0, z0 + wz)
end

_rop_axis_label(model::Bnc, i, j) = "∂log $(string(x_sym(model)[i]))/∂log $(string(qK_sym(model)[j]))"

function draw_ROP(
    model::Bnc,
    pairs::AbstractVector{<:Tuple{Any,Any}};
    emphasize_regimes::AbstractVector=Int[],
    add_inner_points::Bool=true,
    npoints=50000,
    singular_extends::Float64=2.0,
    singular_color="#CCCCFF",
    asymptotic_color="#FFCCCC",
    regular_color="#CCFFCC",
    emphasize_color="#FF0000",
)
    V = get_regimes(model, singular=1, return_idx=true)
    V_non_singular = filter(v -> !is_singular(model, v), V)
    V_singular = filter(v -> is_singular(model, v), V)

    neighbor_mat = get_regimes_neighbor_mat(model)
    singular_neighbor_mat = neighbor_mat[V_singular, V_singular]
    nonsingular_neighbor_mat = neighbor_mat[V_non_singular, V_non_singular]

    vtx_bag = [(Set{Int}(), Set{Int}()) for _ in eachindex(V_non_singular)]

    rgm_dct = let
        groups, labels = connected_components_sparse(singular_neighbor_mat)
        dct = Dict{Int,Set{Int}}()
        for i in eachindex(V_singular)
            dct[i] = Set(groups[labels[i]])
        end
        dct
    end

    get_direct_neighbor_with_singular_regime(i) = [idx for (idx, j) in enumerate(V_non_singular) if neighbor_mat[i, j] == 1]

    function fill_indirect_adj!(j)
        rgms = getindex.(Ref(rgm_dct), collect(vtx_bag[j][1]))
        all_rgms = isempty(rgms) ? Set{Int}() : union(rgms...)
        union!(vtx_bag[j][2], setdiff(all_rgms, vtx_bag[j][1]))
    end

    for (idx, i) in enumerate(V_singular)
        for nb in get_direct_neighbor_with_singular_regime(i)
            push!(vtx_bag[nb][1], idx)
        end
    end

    for j in eachindex(vtx_bag)
        fill_indirect_adj!(j)
    end

    direct_neighbor_pairs = let
        I, J, _ = findnz(tril(nonsingular_neighbor_mat))
        collect(zip(I, J))
    end

    new_form_neighbor_mat = let
        neighbor_mat_compressed = compress_adjacency(neighbor_mat, V_non_singular)
        dropzeros!(neighbor_mat_compressed .- nonsingular_neighbor_mat)
    end

    indirect_neighbor_pairs = let
        I, J, _ = findnz(tril(new_form_neighbor_mat))
        collect(zip(I, J))
    end

    if length(pairs) > 3
        @warn "More than 3 pairs provided, only the first 3 will be used for 3D visualization."
        pairs = pairs[1:3]
    end
    if length(pairs) < 2
        @error "At least 2 pairs are needed for visualization."
        return nothing
    end

    pairs = pairs .|> x -> (locate_sym_x(model, x[1]), locate_sym_qK(model, x[2]))
    get_val(H) = [H[pair...] for pair in pairs]

    Ptype = length(pairs) == 3 ? Point3f : Point2f
    get_col(i) = is_asymptotic(model, i) ? asymptotic_color : regular_color

    pnts = get_H.(Ref(model), V_non_singular) .|> get_val
    dirs = get_H.(Ref(model), V_singular) .|> get_val
    Points = Ptype.(pnts)
    Points_color = get_col.(V_non_singular)

    direct_lines = Tuple{Ptype,Ptype}[(Points[i], Points[j]) for (i, j) in direct_neighbor_pairs]
    indirect_lines = Tuple{Ptype,Ptype}[(Points[i], Points[j]) for (i, j) in indirect_neighbor_pairs]

    direct_rays, indirect_rays = let
        rays1 = Tuple{Ptype,Ptype}[]
        rays2 = Tuple{Ptype,Ptype}[]
        for i in eachindex(vtx_bag)
            for j in vtx_bag[i][1]
                push!(rays1, (Points[i], Points[i] + dirs[j] * singular_extends))
            end
            for j in vtx_bag[i][2]
                push!(rays2, (Points[i], Points[i] + dirs[j] * singular_extends))
            end
        end
        rays1, rays2
    end

    inner_pnts = if add_inner_points
        x_smp = randomize(model, npoints)
        pnts = x_smp .|> x -> ∂logx_∂logqK(model; x=x, input_logspace=true) |> get_val
        Ptype.(pnts)
    else
        nothing
    end

    emphasize_Points = Ptype[]
    emph_rays_direct = Tuple{Ptype,Ptype}[]
    emph_rays_indirect = Tuple{Ptype,Ptype}[]
    if !isempty(emphasize_regimes)
        idx = get_idx.(Ref(model), emphasize_regimes)
        inv_rgm = Set{Int}()
        singular_rgm = Set{Int}()
        for i in idx
            if is_singular(model, i)
                push!(singular_rgm, findfirst(isequal(i), V_singular))
            else
                push!(inv_rgm, findfirst(isequal(i), V_non_singular))
            end
        end

        emphasize_Points = Points[collect(inv_rgm)]
        for i in eachindex(vtx_bag)
            for j in vtx_bag[i][1]
                j in singular_rgm && push!(emph_rays_direct, (Points[i], Points[i] + dirs[j] * singular_extends))
            end
            for j in vtx_bag[i][2]
                j in singular_rgm && push!(emph_rays_indirect, (Points[i], Points[i] + dirs[j] * singular_extends))
            end
        end
    end

    f = Figure()
    ax = if length(pairs) == 3
        Axis3(
            f[1, 1],
            title="Reaction Order Polyhedra",
            xlabel=_rop_axis_label(model, pairs[1]...),
            ylabel=_rop_axis_label(model, pairs[2]...),
            zlabel=_rop_axis_label(model, pairs[3]...),
        )
    else
        Axis(
            f[1, 1],
            title="Reaction Order Polyhedra",
            xlabel=_rop_axis_label(model, pairs[1]...),
            ylabel=_rop_axis_label(model, pairs[2]...),
        )
    end

    for (p1, p2) in direct_lines
        lines!(ax, [p1, p2]; color=:black, linewidth=2)
    end
    for (p1, p2) in indirect_lines
        lines!(ax, [p1, p2]; color=:black, linewidth=2, linestyle=:dash)
    end
    for (p1, p2) in direct_rays
        lines!(ax, [p1, p2]; color=singular_color, linewidth=5)
    end
    for (p1, p2) in indirect_rays
        lines!(ax, [p1, p2]; color=singular_color, linewidth=5, linestyle=:dash)
    end

    scatter!(ax, Points; color=Points_color, markersize=15)
    autolimits!(ax)
    _lock_current_limits!(ax)

    if !isempty(emphasize_regimes)
        scatter!(ax, emphasize_Points; color=emphasize_color, markersize=20)
        for (p1, p2) in emph_rays_direct
            lines!(ax, [p1, p2]; color=emphasize_color, linewidth=5)
        end
        for (p1, p2) in emph_rays_indirect
            lines!(ax, [p1, p2]; color=emphasize_color, linewidth=5, linestyle=:dash)
        end
    end

    if add_inner_points
        scatter!(ax, inner_pnts; color=(:gray, 0.1), markersize=5)
    end

    return f, ax
end
