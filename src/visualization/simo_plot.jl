@inline _render_simo_text(x) = replace(sprint(show, MIME"text/plain"(), x), '\n' => ' ')
@inline _simo_expr_rhs(expr) = expr isa Symbolics.Equation ? expr.rhs : expr
@inline _simo_expr_lhs(expr) = expr isa Symbolics.Equation ? expr.lhs : nothing
@inline _simo_expr_text(expr) = _render_simo_text(expr)
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

function _simo_range_from_boundaries(boundaries::AbstractVector{<:Real}; start=nothing, stop=nothing, pad::Real=1.0)
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

function _simo_regular_logx(
    model::Bnc,
    rgm_idx::Integer,
    observe_x_idx::AbstractVector{<:Integer},
    params::AbstractVector{<:Real},
    change_idx::Integer,
    qvals::AbstractVector{<:Real},
)
    get_nullity(model, rgm_idx) == 0 || return nothing
    H, H0 = get_H_H0(model, rgm_idx)
    base = _insert_simo_change(params, change_idx, 0.0)
    out = Matrix{Float64}(undef, length(observe_x_idx), length(qvals))
    for (row_idx, x_idx) in enumerate(observe_x_idx)
        slope = Float64(H[x_idx, change_idx])
        intercept = Float64(dot(H[x_idx, :], base) + H0[x_idx])
        @inbounds for (col, q) in enumerate(qvals)
            out[row_idx, col] = slope * q + intercept
        end
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

function _simo_path_from_numeric(
    model::Bnc,
    params::AbstractVector{<:Real},
    change_idx::Integer,
    qvals::AbstractVector{<:Real},
)
    qk_grid = _simo_sample_qk_grid(params, change_idx, qvals)
    rgms = [assign_regime_qK(model, qk_grid[:, i]; input_logspace=true, asymptotic_only=false, return_idx=true) for i in axes(qk_grid, 2)]
    return _simo_dedup_consecutive(rgms)
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

function _simo_panel_text(
    model::Bnc,
    rgm_path::AbstractVector{<:Integer},
    xsyms,
    expr_rows::AbstractVector,
    boundary_exprs::AbstractVector;
    include_region_expr::Bool=true,
    include_boundary_expr::Bool=true,
)
    lines = String["Path: $(format_arrow(rgm_path; prefix="#"))"]
    for i in eachindex(rgm_path)
        if include_region_expr
            push!(lines, "Regime #$(rgm_path[i])")
            expr_row = expr_rows[i]
            if expr_row isa AbstractVector
                append!(lines, ["  $(_render_simo_text(sym)) = $(_simo_expr_rhs_text(expr))" for (sym, expr) in zip(xsyms, expr_row)])
            else
                push!(lines, "  $(_render_simo_text(only(xsyms))) = $(_simo_expr_rhs_text(expr_row))")
            end
        end
        if include_boundary_expr && i <= length(boundary_exprs)
            push!(lines, "Boundary #$(rgm_path[i]) ↔ #$(rgm_path[i + 1])")
            push!(lines, "  $(_simo_expr_text(boundary_exprs[i]))")
        end
    end
    return join(lines, "\n")
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
    pad::Real=1.0,
    show_numeric::Bool=true,
    show_regime::Bool=true,
    region_fill::Bool=true,
    region_label::Symbol=:reaction_order,
    boundary_label::Symbol=:panel,
    region_alpha::Real=0.12,
    boundary_color=:black,
    boundary_lw::Real=1.0,
    numeric_linestyle=:solid,
    regime_linestyle=:dash,
    line_colors,
    show_legend_labels::Bool=false,
)
    boundaries = _simo_path_boundaries(model, rgm_path, change_idx, params)
    xstart, xstop = _simo_range_from_boundaries(boundaries; start=start, stop=stop, pad=pad)
    qvals = collect(range(xstart, xstop; length=npoints))

    numeric_logx = show_numeric ? _simo_numeric_logx(model, params, change_idx, qvals) : nothing
    ro_rows = _calc_RO_for_single_path(model, rgm_path, change_idx, observe_x_idx)
    expr_rows, boundary_exprs = _path_expression_rows(model, rgm_path, change_idx, observe_x_idx; log_space=false)

    regime_lines = Vector{Union{Nothing,Matrix{Float64}}}(undef, length(rgm_path))
    if show_regime
        for (i, rgm_idx) in enumerate(rgm_path)
            regime_lines[i] = _simo_regular_logx(model, rgm_idx, observe_x_idx, params, change_idx, qvals)
        end
    else
        fill!(regime_lines, nothing)
    end

    ymin, ymax = _simo_y_limits(numeric_logx, regime_lines)
    ylims!(ax, ymin, ymax)
    xlims!(ax, xstart, xstop)

    interval_edges = vcat([xstart], Float64.(boundaries), [xstop])
    for i in eachindex(rgm_path)
        left = interval_edges[i]
        right = interval_edges[i + 1]
        if region_fill && isfinite(left) && isfinite(right)
            vspan!(ax, left, right; color=(rgm_cmap[rgm_path[i]], region_alpha))
        end

        if show_regime && !(regime_lines[i] === nothing)
            mask = (qvals .>= left) .& (qvals .<= right)
            if any(mask)
                qseg = qvals[mask]
                line = regime_lines[i]
                for (j, x_idx) in enumerate(observe_x_idx)
                    lines!(
                        ax,
                        qseg,
                        @view(line[j, mask]);
                        color=line_colors[j],
                        linestyle=regime_linestyle,
                        label=show_legend_labels && i == 1 ? "$(_render_simo_text(xsyms[j])) (regime)" : nothing,
                    )
                end
            end
        end

        label_text = _simo_region_label(xsyms, @view(ro_rows[i, :]), expr_rows[i], region_label)
        if !isnothing(label_text) && isfinite(left) && isfinite(right)
            text!(
                ax,
                (left + right) / 2,
                ymax - 0.1 * (ymax - ymin);
                text=label_text,
                align=(:center, :top),
                fontsize=12,
                color=:black,
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
                linewidth=2,
                label=show_legend_labels ? "$(_render_simo_text(xsyms[j])) (numeric)" : nothing,
            )
        end
    end

    for (i, bnd) in enumerate(boundaries)
        isfinite(bnd) || continue
        vlines!(ax, [bnd]; color=boundary_color, linewidth=boundary_lw)
        if boundary_label === :plot
            text!(
                ax,
                bnd,
                ymax - 0.02 * (ymax - ymin);
                text=_simo_expr_text(boundary_exprs[i]),
                rotation=pi / 2,
                align=(:left, :top),
                fontsize=11,
                color=boundary_color,
            )
        end
    end

    return expr_rows, boundary_exprs
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

function SIMO_plot(
    grh::SIMOPaths,
    pth_idx::Union{Nothing,Integer,AbstractVector}=nothing;
    observe_x=nothing,
    npoints::Integer=300,
    start=nothing,
    stop=nothing,
    pad::Real=1.0,
    rand_line::Bool=false,
    rand_ray::Bool=false,
    extend::Real=4.0,
    show_numeric::Bool=true,
    show_regime::Bool=true,
    region_fill::Bool=true,
    region_label::Symbol=:reaction_order,
    boundary_label::Symbol=:panel,
    show_expression_panel::Bool=true,
    show_regime_colorbar::Bool=false,
    region_colormap=:rainbow,
    line_colormap=:tab10,
    size=nothing,
)
    model = get_binding_network(grh)
    path_idxs = get_indices(grh, pth_idx)
    observe_x_idx, xsyms, _ = _normalize_simo_observe_x(model, observe_x)
    line_colors = _resolve_simo_line_colors(length(observe_x_idx); colormap=line_colormap)
    rgm_cmap = get_color_map(vcat(grh.rgm_paths[path_idxs]...); colormap=region_colormap)

    n_paths = length(path_idxs)
    fig_size = isnothing(size) ? (show_expression_panel || boundary_label === :panel ? 1600 : 1200, max(360 * n_paths, 360)) : size
    F = Figure(size=fig_size)

    show_panel = show_expression_panel || boundary_label === :panel
    for (row, path_idx) in enumerate(path_idxs)
        ax = Axis(
            F[row, 1];
            xlabel="log" * repr(qK_sym(model)[grh.change_qK_idx]),
            ylabel=length(observe_x_idx) == 1 ? "log" * repr(only(xsyms)) : "log x",
            title="Path $(path_idx): $(format_arrow(get_path(grh, path_idx; return_idx=true); prefix="#"))",
        )

        params = get_one_inner_point(
            get_polyhedron(grh, path_idx);
            rand_line=rand_line,
            rand_ray=rand_ray,
            extend=extend,
        )

        expr_rows, boundary_exprs = _simo_plot_one_path!(
            ax,
            model,
            grh.rgm_paths[path_idx],
            params,
            grh.change_qK_idx,
            observe_x_idx,
            xsyms,
            rgm_cmap;
            npoints=npoints,
            start=start,
            stop=stop,
            pad=pad,
            show_numeric=show_numeric,
            show_regime=show_regime,
            region_fill=region_fill,
            region_label=region_label,
            boundary_label=boundary_label,
            line_colors=line_colors,
            show_legend_labels=row == 1,
        )

        if row == 1 && (show_numeric || show_regime)
            axislegend(ax; position=:rb)
        end

        if show_panel
            Label(
                F[row, 2],
                _simo_panel_text(
                    model,
                    grh.rgm_paths[path_idx],
                    xsyms,
                    expr_rows,
                    boundary_exprs;
                    include_region_expr=show_expression_panel,
                    include_boundary_expr=boundary_label === :panel,
                );
                justification=:left,
                halign=:left,
                valign=:top,
                tellwidth=false,
            )
        end
    end

    if show_regime_colorbar
        add_rgm_colorbar!(F, rgm_cmap)
    end
    return F
end

function SIMO_plot(
    model::Bnc,
    parameters,
    change_idx;
    rgm_path=nothing,
    observe_x=nothing,
    npoints::Integer=300,
    start::Real=-6,
    stop::Real=6,
    pad::Real=1.0,
    show_numeric::Bool=true,
    show_regime::Bool=true,
    region_fill::Bool=true,
    region_label::Symbol=:reaction_order,
    boundary_label::Symbol=:panel,
    show_expression_panel::Bool=true,
    show_regime_colorbar::Bool=false,
    region_colormap=:rainbow,
    line_colormap=:tab10,
    size=(1600, 400),
)
    change_idx = locate_sym_qK(model, change_idx)
    params = _normalize_simo_parameters(model, parameters, change_idx)
    qvals = collect(range(Float64(start), Float64(stop); length=max(npoints, 50)))
    rgm_path = isnothing(rgm_path) ? _simo_path_from_numeric(model, params, change_idx, qvals) : Int.(get_idx.(Ref(model), rgm_path))
    observe_x_idx, xsyms, _ = _normalize_simo_observe_x(model, observe_x)
    line_colors = _resolve_simo_line_colors(length(observe_x_idx); colormap=line_colormap)
    rgm_cmap = get_color_map(rgm_path; colormap=region_colormap)

    F = Figure(size=size)
    ax = Axis(
        F[1, 1];
        xlabel="log" * repr(qK_sym(model)[change_idx]),
        ylabel=length(observe_x_idx) == 1 ? "log" * repr(only(xsyms)) : "log x",
        title="Path: $(format_arrow(rgm_path; prefix="#"))",
    )

    expr_rows, boundary_exprs = _simo_plot_one_path!(
        ax,
        model,
        rgm_path,
        params,
        change_idx,
        observe_x_idx,
        xsyms,
        rgm_cmap;
        npoints=npoints,
        start=start,
        stop=stop,
        pad=pad,
        show_numeric=show_numeric,
        show_regime=show_regime,
        region_fill=region_fill,
        region_label=region_label,
        boundary_label=boundary_label,
        line_colors=line_colors,
        show_legend_labels=true,
    )

    if show_numeric || show_regime
        axislegend(ax; position=:rb)
    end

    if show_expression_panel || boundary_label === :panel
        Label(
            F[1, 2],
            _simo_panel_text(
                model,
                rgm_path,
                xsyms,
                expr_rows,
                boundary_exprs;
                include_region_expr=show_expression_panel,
                include_boundary_expr=boundary_label === :panel,
            );
            justification=:left,
            halign=:left,
            valign=:top,
            tellwidth=false,
        )
    end

    if show_regime_colorbar
        add_rgm_colorbar!(F, rgm_cmap)
    end
    return F
end
