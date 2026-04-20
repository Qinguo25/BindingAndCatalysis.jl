function slice_polyhedron(poly::Polyhedron; fixed_idx::AbstractVector{<:Integer}, fixed_value::Real=1.0)::Polyhedron
    n = fulldim(poly)
    all(1 .<= fixed_idx .<= n) || throw(ArgumentError("`fixed_idx` must be in 1:$n"))

    get_hyperplane(i) = let
        aT = zeros(n)
        aT[i] = 1.0
        HyperPlane(aT, fixed_value)
    end

    ps = get_hyperplane.(fixed_idx)
    sliced = intersect(poly, ps...)
    return _poly_eliminate(sliced, BitSet(fixed_idx))
end

function _grid_sample_polyhedron(poly::Polyhedron, bounds; npoints::Int=10000)
    @assert fulldim(poly) == 3 "Only 3D polyhedra are supported for grid sampling."
    pts_each_dim = round(Int, npoints^(1 / 3))
    gridsize = (pts_each_dim, pts_each_dim, pts_each_dim)
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    xs = range(xmin, xmax; length=gridsize[1])
    ys = range(ymin, ymax; length=gridsize[2])
    zs = range(zmin, zmax; length=gridsize[3])

    return Point3f[
        Point3f(x, y, z)
        for x in xs, y in ys, z in zs
        if [x, y, z] ∈ poly
    ]
end

function plot_polyhedron_slices(
    polys::AbstractVector{<:Polyhedron};
    fixed_idx::AbstractVector{<:Integer},
    fixed_value::Real=1.0,
    labels=nothing,
    colors=nothing,
    axis_labels=nothing,
    bounds=[(-6, 6), (-6, 6), (-6, 6)],
    npoints=10000,
    markersize::Real=5,
    alpha::Real=0.35,
    title::AbstractString="Polyhedron slices (grid sampling)",
)
    isempty(polys) && throw(ArgumentError("`polys` must not be empty"))

    sliced = [slice_polyhedron(p; fixed_idx=fixed_idx, fixed_value=fixed_value) for p in polys]
    labels = isnothing(labels) ? ["poly $i" for i in eachindex(polys)] : collect(string.(labels))
    axis_labels = isnothing(axis_labels) ? ["dim 1", "dim 2", "dim 3"] : collect(string.(axis_labels))
    colors = isnothing(colors) ? Makie.wong_colors() : collect(colors)

    samples = [isempty(p) ? Point3f[] : _grid_sample_polyhedron(p, bounds; npoints=npoints) for p in sliced]

    fig = Figure()
    ax = Axis3(fig[1, 1]; title=title, xlabel=axis_labels[1], ylabel=axis_labels[2], zlabel=axis_labels[3])

    for (i, pts) in enumerate(samples)
        isempty(pts) && continue
        scatter!(ax, pts; color=(colors[mod1(i, length(colors))], alpha), markersize=markersize, label=labels[i])
    end

    axislegend(ax; position=:rb)
    return fig, ax
end

function find_bounds(lattice)
    col_asym_x_bounds = imfilter(lattice, Kernel.Laplacian(), "replicate")
    return col_asym_x_bounds .!= 0
end
