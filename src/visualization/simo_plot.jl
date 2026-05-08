_txt(x) = replace(sprint(show, MIME"text/plain"(), x), '\n' => ' ')

function _full_qK(params::AbstractVector{<:Real}, change_idx::Integer, q::Real)
    out = Vector{Float64}(undef, length(params) + 1)
    src = 1
    for i in eachindex(out)
        if i == change_idx
            out[i] = Float64(q)
        else
            out[i] = Float64(params[src])
            src += 1
        end
    end
    return out
end

function _simo_logx_grid(model::Bnc, params, change_idx, qvals; regime_line::Bool=false, method::Symbol=:free_energy)
    out = Matrix{Float64}(undef, model.n, length(qvals))
    Threads.@threads for i in eachindex(qvals)
        out[:, i] = qK2x(
            model,
            _full_qK(params, change_idx, qvals[i]);
            input_logspace=true,
            output_logspace=true,
            method=regime_line ? :regime : method,
        )
    end
    return out
end

function _simo_assign_rgms(model::Bnc, logx)
    out = Vector{Int}(undef, size(logx, 2))
    Threads.@threads for i in axes(logx, 2)
        out[i] = assign_regime_x(
            model,
            collect(@view logx[:, i]);
            input_logspace=true,
            asymptotic_only=false,
            return_idx=true,
        )
    end
    return out
end

function _simo_rgm_runs(qvals, rgms)
    runs = NamedTuple[]
    start_i = 1
    for i in 2:(length(rgms) + 1)
        if i > length(rgms) || rgms[i] != rgms[start_i]
            left = start_i == 1 ? qvals[1] : (qvals[start_i - 1] + qvals[start_i]) / 2
            right = i > length(rgms) ? qvals[end] : (qvals[i - 1] + qvals[i]) / 2
            push!(runs, (; rgm=Int(rgms[start_i]), left=Float64(left), right=Float64(right)))
            start_i = i
        end
    end
    return runs
end

function _simo_y_limits(mats...)
    vals = Float64[]
    for mat in mats
        isnothing(mat) && continue
        append!(vals, vec(Float64.(mat)))
    end
    filter!(isfinite, vals)
    isempty(vals) && return (-1.0, 1.0)
    lo, hi = extrema(vals)
    span = hi - lo
    span = iszero(span) ? 1.0 : span
    return lo - 0.08span, hi + 0.12span
end

function SIMO_plot(
    model::Bnc,
    params,
    change_idx;
    start::Real=-6,
    stop::Real=6,
    observe_x=nothing,
    npoints::Integer=300,
    method::Symbol=:free_energy,
    add_regime_line::Bool=true,
    shade_background::Bool=true,
    show_regime_label::Bool=true,
    show_regime_colorbar::Bool=false,
    region_alpha::Real=0.12,
    region_colormap=:rainbow,
    size=(760, 500),
    title=nothing,
)
    change_idx = locate_sym_qK(model, change_idx)
    params = Float64.(collect(params))
    if length(params) == model.d + model.r
        deleteat!(params, change_idx)
    end
    length(params) == model.d + model.r - 1 || throw(ArgumentError("Expected reduced qK parameter length $(model.d + model.r - 1)."))

    qvals = collect(range(Float64(start), Float64(stop); length=max(npoints, 2)))
    x_idx, xsyms, _ = _normalize_simo_observe_x(model, observe_x)

    logx = _simo_logx_grid(model, params, change_idx, qvals; method=method)
    rgm_logx = add_regime_line ? _simo_logx_grid(model, params, change_idx, qvals; regime_line=true) : nothing
    rgms = _simo_assign_rgms(model, logx)
    runs = _simo_rgm_runs(qvals, rgms)
    rgm_cmap = get_color_map(unique(rgms); colormap=region_colormap)
    line_colors = Makie.wong_colors()

    ymin, ymax = _simo_y_limits(logx, rgm_logx)
    fig = Figure(size=size)
    ax = Axis(
        fig[1, 1];
        xlabel="log" * _txt(qK_sym(model)[change_idx]),
        ylabel=length(x_idx) == 1 ? "log" * _txt(only(xsyms)) : "log x",
        title=isnothing(title) ? "Changing " * _txt(qK_sym(model)[change_idx]) : title,
    )
    ylims!(ax, ymin, ymax)

    if shade_background
        label_y = ymax - 0.035 * (ymax - ymin)
        for run in runs
            vspan!(ax, run.left, run.right; color=(rgm_cmap[run.rgm], region_alpha))
            if show_regime_label && isfinite(run.left) && isfinite(run.right)
                text!(
                    ax,
                    (run.left + run.right) / 2,
                    label_y;
                    text="#" * string(run.rgm),
                    align=(:center, :top),
                    fontsize=11,
                    color=(:black, 0.75),
                )
            end
        end
    end

    if add_regime_line
        for (j, xi) in enumerate(x_idx)
            c = line_colors[mod1(j, length(line_colors))]
            lines!(ax, qvals, @view rgm_logx[xi, :]; color=c, linestyle=:dash, linewidth=2, label="$(_txt(xsyms[j])) regime")
        end
    end

    for (j, xi) in enumerate(x_idx)
        c = line_colors[mod1(j, length(line_colors))]
        lines!(ax, qvals, @view logx[xi, :]; color=c, linewidth=2.5, label="$(_txt(xsyms[j])) numeric")
    end

    axislegend(ax; position=:lb)
    show_regime_colorbar && add_rgm_colorbar!(fig, rgm_cmap)
    return fig, ax
end

function SIMO_plot(
    grh::SIMOPaths,
    path_idx::Integer;
    title="Path $path_idx",
    extend::Real=4.0,
    kwargs...,
)
    model = get_binding_network(grh)
    params = get_one_inner_point(get_polyhedron(grh, path_idx); rand_line=false, rand_ray=false, extend=extend)
    return SIMO_plot(model, params, grh.change_qK_idx; title=title, kwargs...)
end

function SIMO_plot(
    grh::SIMOPaths,
    path_idxs::AbstractVector{<:Integer};
    title="Paths $(collect(path_idxs))",
    kwargs...,
)
    isempty(path_idxs) && throw(ArgumentError("path_idxs must not be empty."))
    fig, _ = SIMO_plot(grh, first(path_idxs); title=title, kwargs...)
    return fig
end
