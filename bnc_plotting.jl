

using SparseArrays
using LinearAlgebra
using Colors
using Graphs
using CairoMakie
using GraphMakie



# pick dominant component (index, value) by |value|
@inline function _dominant_component(v::SparseVector)
    I, V, _ = findnz(v)
    isempty(I) && return nothing
    j = argmax(abs.(V))
    return (I[j], V[j])
end

"""
visualize_vertex_graph(bnc; layout=:spring, size_range=(1.5, 6.0), palette=nothing)

- Colors: component index of max-abs entry in change_dir_qK
- Direction: negative value flips arrow direction
- Width: proportional to summed magnitude over parallel edges
"""
function visualize_vertex_graph(bnc::Bnc; layout=:spring, size_range=(1.5, 6.0), palette=nothing)
    vg = bnc.vertices_graph
    isnothing(vg) && error("bnc.vertices_graph is nothing")

    nv = length(vg.neighbors)
    # accumulate per displayed directed edge (u,v)
    # value is (sum_magnitude, dominant_idx, dominant_signed_val)
    acc = Dict{Tuple{Int,Int}, Tuple{Float64, Int, Float64}}()

    for u in 1:nv
        for e in vg.neighbors[u]
            v = e.to
            vdir = e.change_dir_qK
            vdir === nothing && continue
            dom = _dominant_component(vdir)
            dom === nothing && continue
            idx, val = dom
            # flip direction for negative dominant component
            if val < 0
                u_, v_ = v, u
                sval = -val
            else
                u_, v_ = u, v
                sval = val
            end
            key = (u_, v_)
            prev = get(acc, key, (0.0, idx, sval))
            mag = prev[1] + abs(sval)
            # keep component of larger |val| for color
            if abs(sval) >= abs(prev[3])
                acc[key] = (mag, idx, sval)
            else
                acc[key] = (mag, prev[2], prev[3])
            end
        end
    end

    # build Graphs.DiGraph
    g = SimpleDiGraph(nv)
    for (u,v) in keys(acc)
        add_edge!(g, u, v)
    end

    # map attributes to Graphs.edge order
    eorder = collect(edges(g))
    mags = Float64[]
    idxs = Int[]
    for e in eorder
        m, idx, _ = acc[(src(e), dst(e))]
        push!(mags, m)
        push!(idxs, idx)
    end

    # widths scaled into size_range
    mmax = maximum(mags; init=0.0)
    lo, hi = size_range
    widths = if mmax == 0
        fill(lo, length(mags))
    else
        @. lo + (hi - lo) * (mags / mmax)
    end

    # palette by component
    ncomp = length(bnc.q_sym) + length(bnc.K_sym)
    if palette === nothing
        palette = distinguishable_colors(ncomp, [RGB(1,1,1)])
    elseif length(palette) < ncomp
        error("palette length $(length(palette)) < number of components $ncomp")
    end
    ecolors = [palette[i] for i in idxs]

    # choose layout
    layoutfun = layout === :spring ? Spring() :
                layout === :spectral ? Spectral() :
                layout isa AbstractLayout ? layout : Spring()

    fig = Figure()
    ax = Axis(fig[1,1])
    graphplot!(ax, g;
        layout = layoutfun,
        edge_color = ecolors,
        edge_width = widths,
        arrow_show = true,
        arrowsize = 12,
        node_size = 10,
        node_color = :gray35,
        node_strokewidth = 0.0,
    )
    hidespines!(ax); hidedecorations!(ax)

    # legend for q/K components
    labels = vcat(string.(bnc.q_sym), string.(bnc.K_sym))
    elems = [LineElement(color=palette[i], linewidth=4) for i in 1:ncomp]
    Legend(fig[1,2], elems, labels, "q/K component")

    display(fig)
    return fig
end
