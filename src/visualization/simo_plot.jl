@inline _render_simo_text(x) = replace(sprint(show, MIME"text/plain"(), x), '\n' => ' ')
@inline _simo_expr_rhs(expr) = expr isa Symbolics.Equation ? expr.rhs : expr
@inline _simo_expr_rhs_text(expr) = _render_simo_text(_simo_expr_rhs(expr))

function _format_ro_value(x::Real; digits::Int=3)
    if isnan(x)
        return "NaN"
    elseif isinf(x)
        return signbit(x) ? "-Inf" : "Inf"
    end
    xr = round(Float64(x); digits=digits)
    return isapprox(xr, round(xr); atol=10.0^(-digits), rtol=0) ? string(Int(round(xr))) : string(xr)
end

function _normalize_simo_parameters(model::Bnc, parameters, change_idx::Integer)
    full_dim = model.d + model.r
    params = Float64.(collect(parameters))
    if length(params) == full_dim - 1
        return params
    elseif length(params) == full_dim
        deleteat!(params, change_idx)
        return params
    else
        error("Expected $(full_dim - 1) reduced qK parameters or $(full_dim) full qK values, got $(length(params)).")
    end
end

function _insert_simo_change(params::AbstractVector{<:Real}, change_idx::Integer, value::Real)
    full = Vector{Float64}(undef, length(params) + 1)
    src = 1
    @inbounds for j in eachindex(full)
        if j == change_idx
            full[j] = Float64(value)
        else
            full[j] = Float64(params[src])
            src += 1
        end
    end
    return full
end

function _simo_boundary_value(
    model::Bnc,
    from::Integer,
    to::Integer,
    change_idx::Integer,
    params::AbstractVector{<:Real},
)
    c, c0 = get_interface(model, from, to)
    denom = c[change_idx]
    abs(denom) > 1e-10 || return NaN
    base = _insert_simo_change(params, change_idx, 0.0)
    return -((dot(c, base) + c0) / denom)
end

function _simo_path_boundaries(
    model::Bnc,
    rgm_path::AbstractVector{<:Integer},
    change_idx::Integer,
    params::AbstractVector{<:Real},
)
    n = max(length(rgm_path) - 1, 0)
    vals = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        vals[i] = _simo_boundary_value(model, rgm_path[i], rgm_path[i + 1], change_idx, params)
    end
    return vals
end

function _simo_range_from_boundaries(boundaries::AbstractVector{<:Real}; start=nothing, stop=nothing, pad::Real=4.0)
    finite_bounds = filter(isfinite, Float64.(collect(boundaries)))
    default_start = isempty(finite_bounds) ? -6.0 : minimum(finite_bounds) - pad
    default_stop = isempty(finite_bounds) ? 6.0 : maximum(finite_bounds) + pad
    xstart = isnothing(start) ? default_start : Float64(start)
    xstop = isnothing(stop) ? default_stop : Float64(stop)
    xstart < xstop || error("Expected start < stop, got start=$xstart and stop=$xstop.")
    return xstart, xstop
end

function _simo_sample_qk_grid(params::AbstractVector{<:Real}, change_idx::Integer, qvals::AbstractVector{<:Real})
    full_dim = length(params) + 1
    out = Matrix{Float64}(undef, full_dim, length(qvals))
    @inbounds for (col, q) in enumerate(qvals)
        out[:, col] = _insert_simo_change(params, change_idx, q)
    end
    return out
end

function _simo_numeric_logx(model::Bnc, params::AbstractVector{<:Real}, change_idx::Integer, qvals::AbstractVector{<:Real})
    qk_grid = _simo_sample_qk_grid(params, change_idx, qvals)
    out = Matrix{Float64}(undef, model.n, size(qk_grid, 2))
    for i in axes(qk_grid, 2)
        out[:, i] = qK2x(model, collect(@view qk_grid[:, i]); input_logspace=true, output_logspace=true)
    end
    return out
end

function _simo_archetype_logx(
    model::Bnc,
    params::AbstractVector{<:Real},
    change_idx::Integer,
    qvals::AbstractVector{<:Real},
)
    qk_grid = _simo_sample_qk_grid(params, change_idx, qvals)
    out = Matrix{Float64}(undef, model.n, length(qvals))
    for i in eachindex(qvals)
        out[:, i] = qK2x(model, collect(@view qk_grid[:, i]); input_logspace=true, output_logspace=true, use_vtx=true)
    end
    return out
end

function _simo_dedup_consecutive(vals::AbstractVector{<:Integer})
    isempty(vals) && return Int[]
    out = Int[Int(first(vals))]
    @inbounds for v in @view vals[2:end]
        Int(v) == out[end] || push!(out, Int(v))
    end
    return out
end

function _simo_numeric_rgms(model::Bnc, numeric_logx::AbstractMatrix{<:Real})
    out = Vector{Int}(undef, size(numeric_logx, 2))
    for i in axes(numeric_logx, 2)
        out[i] = assign_regime_x(model, collect(@view numeric_logx[:, i]); input_logspace=true, asymptotic_only=false, return_idx=true)
    end
    return out
end

function _simo_numeric_runs(qvals::AbstractVector{<:Real}, rgms::AbstractVector{<:Integer})
    isempty(rgms) && return NamedTuple[]
    runs = NamedTuple[]
    start_idx = 1
    for i in 2:(length(rgms) + 1)
        if i > length(rgms) || rgms[i] != rgms[start_idx]
            left = start_idx == 1 ? Float64(qvals[1]) : 0.5 * (Float64(qvals[start_idx - 1]) + Float64(qvals[start_idx]))
            right = i > length(rgms) ? Float64(qvals[end]) : 0.5 * (Float64(qvals[i - 1]) + Float64(qvals[i]))
            push!(runs, (rgm=Int(rgms[start_idx]), start_idx=start_idx, stop_idx=i - 1, left=left, right=right))
            start_idx = i
        end
    end
    return runs
end

function _simo_numeric_context(
    model::Bnc,
    rgm_path::AbstractVector{<:Integer},
    params::AbstractVector{<:Real},
    change_idx::Integer;
    npoints::Integer=300,
    start=nothing,
    stop=nothing,
    pad::Real=4.0,
)
    boundaries = _simo_path_boundaries(model, rgm_path, change_idx, params)
    xstart, xstop = _simo_range_from_boundaries(boundaries; start=start, stop=stop, pad=pad)
    qvals = collect(range(xstart, xstop; length=npoints))
    numeric_logx = _simo_numeric_logx(model, params, change_idx, qvals)
    numeric_rgms = _simo_numeric_rgms(model, numeric_logx)
    runs = _simo_numeric_runs(qvals, numeric_rgms)
    return (qvals=qvals, numeric_logx=numeric_logx, runs=runs, rgms=_simo_dedup_consecutive(numeric_rgms))
end

function _simo_path_from_numeric(
    model::Bnc,
    params::AbstractVector{<:Real},
    change_idx::Integer,
    qvals::AbstractVector{<:Real},
)
    numeric_logx = _simo_numeric_logx(model, params, change_idx, qvals)
    return _simo_dedup_consecutive(_simo_numeric_rgms(model, numeric_logx))
end

function _simo_y_limits(numeric_logx, regime_lines)
    vals = Float64[]
    if !(numeric_logx === nothing)
        append!(vals, vec(Float64.(numeric_logx)))
    end
    for line in regime_lines
        line === nothing && continue
        append!(vals, vec(Float64.(line)))
    end
    filter!(isfinite, vals)
    if isempty(vals)
        return (-1.0, 1.0)
    end
    ymin, ymax = extrema(vals)
    span = ymax - ymin
    span = iszero(span) ? 1.0 : span
    return (ymin - 0.08 * span, ymax + 0.08 * span)
end

function _simo_region_label(
    xsyms,
    ro_row::AbstractVector{<:Real},
    expr_row::AbstractVector,
    mode::Symbol,
)
    mode === :none && return nothing
    if mode === :reaction_order
        return join(["$(_render_simo_text(sym)): $(_format_ro_value(ro))" for (sym, ro) in zip(xsyms, ro_row)], "\n")
    elseif mode === :expression
        return join(["$(_render_simo_text(sym)) = $(_simo_expr_rhs_text(expr))" for (sym, expr) in zip(xsyms, expr_row)], "\n")
    else
        error("Unsupported region_label=$mode. Use :none, :reaction_order, or :expression.")
    end
end

function _simo_plot_one_path!(
    ax::Axis,
    model::Bnc,
    rgm_path::AbstractVector{<:Integer},
    params::AbstractVector{<:Real},
    change_idx::Integer,
    observe_x_idx::AbstractVector{<:Integer},
    xsyms,
    rgm_cmap;
    npoints::Integer=300,
    start=nothing,
    stop=nothing,
    pad::Real=4.0,
    show_numeric::Bool=true,
    show_regime::Bool=true,
    region_fill::Bool=true,
    region_label::Symbol=:reaction_order,
    region_alpha::Real=0.12,
    region_fontsize::Real=12,
    numeric_linewidth::Real=2.0,
    regime_linewidth::Real=2.0,
    numeric_linestyle=:solid,
    regime_linestyle=:dash,
    line_colors,
    show_legend_labels::Bool=false,
    numeric_ctx=nothing,
)
    ctx = isnothing(numeric_ctx) ? _simo_numeric_context(model, rgm_path, params, change_idx; npoints=npoints, start=start, stop=stop, pad=pad) : numeric_ctx
    qvals = ctx.qvals
    numeric_logx = ctx.numeric_logx
    runs = ctx.runs

    archetype_logx = show_regime ? _simo_archetype_logx(model, params, change_idx, qvals) : nothing
    ro_cache = Dict{Int, Vector{Float64}}()
    expr_cache = Dict{Int, Any}()
    ymin, ymax = _simo_y_limits(numeric_logx, [archetype_logx])
    ylims!(ax, ymin, ymax)
    xlims!(ax, qvals[1], qvals[end])

    for run in runs
        rgm_idx = run.rgm
        left, right = run.left, run.right
        if region_fill && isfinite(left) && isfinite(right)
            vspan!(ax, left, right; color=(rgm_cmap[rgm_idx], region_alpha))
        end

        ro_row = get!(ro_cache, rgm_idx) do
            vec(_calc_RO_for_single_path(model, [rgm_idx], change_idx, observe_x_idx))
        end
        expr_row = get!(expr_cache, rgm_idx) do
            first(_path_expression_rows(model, [rgm_idx], change_idx, observe_x_idx; log_space=false)[1])
        end
        label_text = _simo_region_label(xsyms, ro_row, expr_row, region_label)
        if !isnothing(label_text) && isfinite(left) && isfinite(right)
            text!(
                ax,
                (left + right) / 2,
                ymax - 0.1 * (ymax - ymin);
                text=label_text,
                align=(:center, :top),
                fontsize=region_fontsize,
                color=:black,
            )
        end
    end
    if !(archetype_logx === nothing)
        for (j, x_idx) in enumerate(observe_x_idx)
            lines!(
                ax,
                qvals,
                @view(archetype_logx[x_idx, :]);
                color=line_colors[j],
                linestyle=regime_linestyle,
                linewidth=regime_linewidth,
                label=show_legend_labels ? "$(_render_simo_text(xsyms[j])) (regime)" : nothing,
            )
        end
    end
    if !(numeric_logx === nothing)
        for (j, x_idx) in enumerate(observe_x_idx)
            lines!(
                ax,
                qvals,
                @view(numeric_logx[x_idx, :]);
                color=line_colors[j],
                linestyle=numeric_linestyle,
                linewidth=numeric_linewidth,
                label=show_legend_labels ? "$(_render_simo_text(xsyms[j])) (numeric)" : nothing,
            )
        end
    end

    return nothing
end

function _resolve_simo_line_colors(n::Integer; colormap=:tab10)
    n <= 0 && return Any[]
    if colormap === :tab10
        base = Makie.wong_colors()
        if n <= length(base)
            return base[1:n]
        end
    end
    return cgrad(colormap, n, categorical=true)[1:n]
end

function _simo_figure_size(n_paths::Integer; size=nothing)
    !isnothing(size) && return size
    width = 980
    height = max(320 * n_paths, 320)
    return (width, height)
end

function _simo_style(fig_size)
    width, _ = Float64.(Tuple(fig_size))
    scale = clamp(width / 980.0, 0.9, 1.35)
    return (
        axis_fontsize = 14 * scale,
        tick_fontsize = 11 * scale,
        title_fontsize = 16 * scale,
        region_fontsize = 11 * scale,
        numeric_lw = 2.2 * scale,
        regime_lw = 2.0 * scale,
    )
end

# function SIMO_plot(
#     grh::SIMOPaths,
#     pth_idx::Union{Nothing,Integer,AbstractVector}=nothing;
#     observe_x=nothing,
#     npoints::Integer=300,
#     start=nothing,
#     stop=nothing,
#     pad::Real=4.0,
#     rand_line::Bool=false,
#     rand_ray::Bool=false,
#     extend::Real=4.0,
#     show_numeric::Bool=true,
#     show_regime::Bool=true,
#     region_fill::Bool=true,
#     region_label::Symbol=:reaction_order,
#     show_regime_colorbar::Bool=false,
#     region_colormap=:rainbow,
#     line_colormap=:tab10,
#     size=nothing,
# )
#     model = get_binding_network(grh)
#     path_idxs = get_indices(grh, pth_idx)
#     observe_x_idx, xsyms, _ = _normalize_simo_observe_x(model, observe_x)
#     line_colors = _resolve_simo_line_colors(length(observe_x_idx); colormap=line_colormap)

#     n_paths = length(path_idxs)
#     fig_size = _simo_figure_size(n_paths; size=size)
#     style = _simo_style(fig_size)
#     F = Figure(size=fig_size, figure_padding=(10, 12, 10, 10))
#     rowgap!(F.layout, 8)
#     colgap!(F.layout, 12)
#     params_by_path = Dict{Int, Vector{Float64}}()
#     ctx_by_path = Dict{Int, Any}()
#     all_rgms = Int[]
#     for path_idx in path_idxs
#         params = get_one_inner_point(get_polyhedron(grh, path_idx); rand_line=rand_line, rand_ray=rand_ray, extend=extend)
#         params_by_path[path_idx] = params
#         ctx = _simo_numeric_context(model, grh.rgm_paths[path_idx], params, grh.change_qK_idx; npoints=npoints, start=start, stop=stop, pad=pad)
#         ctx_by_path[path_idx] = ctx
#         append!(all_rgms, ctx.rgms)
#     end
#     rgm_cmap = get_color_map(all_rgms; colormap=region_colormap)
#     for (row, path_idx) in enumerate(path_idxs)
#         ax = Axis(
#             F[row, 1];
#             xlabel="log" * repr(qK_sym(model)[grh.change_qK_idx]),
#             ylabel=length(observe_x_idx) == 1 ? "log" * repr(only(xsyms)) : "log x",
#             title="Path $(path_idx): $(format_arrow(get_path(grh, path_idx; return_idx=true); prefix="#"))",
#             xlabelsize=style.axis_fontsize,
#             ylabelsize=style.axis_fontsize,
#             titlesize=style.title_fontsize,
#             xticklabelsize=style.tick_fontsize,
#             yticklabelsize=style.tick_fontsize,
#         )

#         params = params_by_path[path_idx]

#         _simo_plot_one_path!(
#             ax,
#             model,
#             grh.rgm_paths[path_idx],
#             params,
#             grh.change_qK_idx,
#             observe_x_idx,
#             xsyms,
#             rgm_cmap;
#             npoints=npoints,
#             start=start,
#             stop=stop,
#             pad=pad,
#             show_numeric=show_numeric,
#             show_regime=show_regime,
#             region_fill=region_fill,
#             region_label=region_label,
#             region_fontsize=style.region_fontsize,
#             numeric_linewidth=style.numeric_lw,
#             regime_linewidth=style.regime_lw,
#             line_colors=line_colors,
#             show_legend_labels=row == 1,
#             numeric_ctx=ctx_by_path[path_idx],
#         )

#         if row == 1 && (show_numeric || show_regime)
#             axislegend(ax; position=:rb, labelsize=style.tick_fontsize)
#         end
#     end

#     if show_regime_colorbar
#         add_rgm_colorbar!(F, rgm_cmap)
#     end
#     return F
# end

# function SIMO_plot(
#     model::Bnc,
#     parameters,
#     change_idx;
#     rgm_path=nothing,
#     observe_x=nothing,
#     npoints::Integer=300,
#     start::Real=-6,
#     stop::Real=6,
#     pad::Real=4.0,
#     show_numeric::Bool=true,
#     show_regime::Bool=true,
#     region_fill::Bool=true,
#     region_label::Symbol=:reaction_order,
#     show_regime_colorbar::Bool=false,
#     region_colormap=:rainbow,
#     line_colormap=:tab10,
#     size=nothing,
# )
#     change_idx = locate_sym_qK(model, change_idx)
#     params = _normalize_simo_parameters(model, parameters, change_idx)
#     qvals = collect(range(Float64(start), Float64(stop); length=max(npoints, 50)))
#     rgm_path = isnothing(rgm_path) ? _simo_path_from_numeric(model, params, change_idx, qvals) : Int.(get_idx.(Ref(model), rgm_path))
#     observe_x_idx, xsyms, _ = _normalize_simo_observe_x(model, observe_x)
#     line_colors = _resolve_simo_line_colors(length(observe_x_idx); colormap=line_colormap)
#     ctx = _simo_numeric_context(model, rgm_path, params, change_idx; npoints=npoints, start=start, stop=stop, pad=pad)
#     rgm_cmap = get_color_map(ctx.rgms; colormap=region_colormap)

#     fig_size = _simo_figure_size(1; size=size)
#     style = _simo_style(fig_size)
#     F = Figure(size=fig_size, figure_padding=(10, 12, 10, 10))
#     colgap!(F.layout, 12)
#     ax = Axis(
#         F[1, 1];
#         xlabel="log" * repr(qK_sym(model)[change_idx]),
#         ylabel=length(observe_x_idx) == 1 ? "log" * repr(only(xsyms)) : "log x",
#         title="Path: $(format_arrow(rgm_path; prefix="#"))",
#         xlabelsize=style.axis_fontsize,
#         ylabelsize=style.axis_fontsize,
#         titlesize=style.title_fontsize,
#         xticklabelsize=style.tick_fontsize,
#         yticklabelsize=style.tick_fontsize,
#     )

#     _simo_plot_one_path!(
#         ax,
#         model,
#         rgm_path,
#         params,
#         change_idx,
#         observe_x_idx,
#         xsyms,
#         rgm_cmap;
#         npoints=npoints,
#         start=start,
#         stop=stop,
#         pad=pad,
#         show_numeric=show_numeric,
#         show_regime=show_regime,
#         region_fill=region_fill,
#         region_label=region_label,
#         region_fontsize=style.region_fontsize,
#         numeric_linewidth=style.numeric_lw,
#         regime_linewidth=style.regime_lw,
#         line_colors=line_colors,
#         show_legend_labels=true,
#         numeric_ctx=ctx,
#     )

#     if show_numeric || show_regime
#         axislegend(ax; position=:rb, labelsize=style.tick_fontsize)
#     end

#     if show_regime_colorbar
#         add_rgm_colorbar!(F, rgm_cmap)
#     end
#     return F
# end




# Light SIMO plot:
# path condition -> one inner point -> sweep changed qK -> logx curves
# background color = regime assigned from numeric x
# optional dashed regime-derived line

_txt(x) = replace(sprint(show, MIME"text/plain"(), x), '\n' => ' ')

_full_qK(params, change_idx, q) =
    Float64.(vcat(params[1:change_idx-1], q, params[change_idx:end]))

function _logx_grid(model, params, change_idx, qvals; regime_line::Bool=false)
    f(q) = regime_line ?
        qK2x(model, _full_qK(params, change_idx, q);
             input_logspace=true, output_logspace=true, use_vtx=true) :
        qK2x(model, _full_qK(params, change_idx, q);
             input_logspace=true, output_logspace=true)

    return reduce(hcat, (f(q) for q in qvals))
end

function _assign_rgms(model, logx)
    return [
        assign_regime_x(model, collect(@view logx[:, i]);
            input_logspace=true,
            asymptotic_only=false,
            return_idx=true)
        for i in axes(logx, 2)
    ]
end

function _rgm_runs(qvals, rgms)
    runs = NamedTuple[]
    s = 1

    for i in 2:(length(rgms) + 1)
        if i > length(rgms) || rgms[i] != rgms[s]
            left  = s == 1 ? qvals[1]  : (qvals[s-1] + qvals[s]) / 2
            right = i > length(rgms) ? qvals[end] : (qvals[i-1] + qvals[i]) / 2
            push!(runs, (; rgm=Int(rgms[s]), left=Float64(left), right=Float64(right)))
            s = i
        end
    end

    return runs
end

function SIMO_plot(
    model::Bnc,
    params,
    change_idx;
    start::Real=-6,
    stop::Real=6,
    observe_x=nothing,
    npoints::Integer=300,
    add_regime_line::Bool=true,
    shade_background::Bool=true,
    region_alpha::Real=0.12,
    region_colormap=:rainbow,
    size=(760, 500),
    title=nothing,
)
    change_idx = locate_sym_qK(model, change_idx)

    params = Float64.(collect(params))
    if length(params) == model.d + model.r
        deleteat!(params, change_idx)   # allow full qK input too
    end

    qvals = collect(range(Float64(start), Float64(stop); length=max(npoints, 2)))

    x_idx, xsyms, _ = _normalize_simo_observe_x(model, observe_x)

    logx = _logx_grid(model, params, change_idx, qvals)
    rgms = _assign_rgms(model, logx)

    rgm_colors = get_color_map(unique(rgms); colormap=region_colormap)
    line_colors = Makie.wong_colors()

    F = Figure(size=size)
    ax = Axis(
        F[1, 1];
        xlabel = "log" * _txt(qK_sym(model)[change_idx]),
        ylabel = length(x_idx) == 1 ? "log" * _txt(only(xsyms)) : "log x",
        title = isnothing(title) ? "Changing " * _txt(qK_sym(model)[change_idx]) : title,
    )

    if shade_background
        for run in _rgm_runs(qvals, rgms)
            vspan!(ax, run.left, run.right; color=(rgm_colors[run.rgm], region_alpha))
        end
    end

    if add_regime_line
        rgm_logx = _logx_grid(model, params, change_idx, qvals; regime_line=true)

        for (j, xi) in enumerate(x_idx)
            c = line_colors[mod1(j, length(line_colors))]
            lines!(ax, qvals, @view rgm_logx[xi, :];
                color=c, linestyle=:dash, linewidth=2,
                label="$(_txt(xsyms[j])) regime")
        end
    end

    for (j, xi) in enumerate(x_idx)
        c = line_colors[mod1(j, length(line_colors))]
        lines!(ax, qvals, @view logx[xi, :];
            color=c, linewidth=2.5,
            label="$(_txt(xsyms[j])) numeric")
    end

    axislegend(ax; position=:lb)
    return F,ax
end

function SIMO_plot(
    grh::SIMOPaths,
    path_idx::Integer;
    title="Path $path_idx",
    extend::Real = 4.0,
    kwargs...,
)
    model = get_binding_network(grh)
    
     
    params = get_one_inner_point(get_polyhedron(grh, path_idx), rand_line=false,rand_ray=false, extend=extend)
    
    let 
        model = get_binding_network(grh)
        qK_s = qK_sym(model)
        deleteat!(qK_s, grh.change_qK_idx)
        dict = Dict{Any,Float64}(sym => val for (sym, val) in zip(qK_s, params))
        @info "$dict"
    end


    return SIMO_plot(
        model,
        params,
        grh.change_qK_idx;
        title=title,
        kwargs...,
    )
end
