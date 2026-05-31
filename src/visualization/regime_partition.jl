function _sym_text(x)
    return strip_before_bracket(repr(x))
end

function _axis_indices(syms, axes)
    axes isa Tuple && (axes = collect(axes))
    axes isa AbstractVector || (axes = [axes])
    return Int[locate_sym(syms, ax) for ax in axes]
end

function _fixed_log_values(syms, fixed; default=0.0, input_logspace::Bool=true, axis_idxs=nothing)
    vals = fill(Float64(default), length(syms))
    if fixed isa AbstractVector
        raw = input_logspace ? Float64.(fixed) : log10.(Float64.(fixed))
        if length(raw) == length(syms)
            vals .= raw
        elseif !isnothing(axis_idxs)
            fixed_idxs = setdiff(collect(eachindex(syms)), Int.(axis_idxs))
            length(raw) == length(fixed_idxs) || throw(ArgumentError("fixed vector length must be either $(length(syms)) for all coordinates or $(length(fixed_idxs)) for the non-axis coordinates."))
            vals[fixed_idxs] .= raw
        else
            throw(ArgumentError("fixed vector length must be $(length(syms))."))
        end
    elseif !isnothing(fixed)
        for (k, v) in pairs(fixed)
            vals[locate_sym(syms, k)] = input_logspace ? Float64(v) : log10(Float64(v))
        end
    end
    return vals
end



function _axis_ranges(naxes::Int; ranges=(-3.0, 3.0), n::Integer=80, input_logspace::Bool=true)
    if ranges isa Tuple && length(ranges) == 2 && all(x -> x isa Real, ranges)
        lo, hi = input_logspace ? Float64.(ranges) : log10.(Float64.(ranges))
        return [range(lo, hi; length=n) for _ in 1:naxes]
    end
    length(ranges) == naxes || throw(ArgumentError("ranges must have one range per selected axis."))
    return map(ranges) do rg
        if rg isa Tuple && length(rg) == 2
            lo, hi = input_logspace ? Float64.(rg) : log10.(Float64.(rg))
            range(lo, hi; length=n)
        else
            input_logspace ? Float64.(collect(rg)) : log10.(Float64.(collect(rg)))
        end
    end
end

function _partition_color_matrix(vals; colormap=:Pastel1_9)
    valid = [v for v in vec(vals) if !(v == 0)]
    isempty(valid) && return fill(RGBAf(0, 0, 0, 0), size(vals)), nothing
    cmap = get_color_map(valid; colormap=colormap, appendix="#")
    colors = Array{RGBAf}(undef, size(vals))
    for I in CartesianIndices(vals)
        colors[I] = vals[I] == 0 ? RGBAf(0, 0, 0, 0) : RGBAf(cmap[vals[I]])
    end
    return colors, cmap
end

function _draw_partition_2d(xs, ys, vals; xlabel="", ylabel="", title="", colormap=:Pastel1_9, categorical::Bool=true, colorrange=nothing)
    fig = Figure(size=(760, 620))
    ax = Axis(fig[1, 1], xlabel=xlabel, ylabel=ylabel, title=title)
    if categorical
        colors, cmap = _partition_color_matrix(vals; colormap=colormap)
        heatmap!(ax, xs, ys, colors)
        isnothing(cmap) || add_rgm_colorbar!(fig, cmap)
    else
        kwargs = isnothing(colorrange) ? (; colormap=colormap) : (; colormap=colormap, colorrange=colorrange)
        hm = heatmap!(ax, xs, ys, vals; kwargs...)
        Colorbar(fig[1, 2], hm)
    end
    return fig, ax
end

function _draw_partition_3d(xs, ys, zs, vals; xlabel="", ylabel="", zlabel="", title="", colormap=:Pastel1_9, categorical::Bool=true, colorrange=nothing)
    fig = Figure(size=(820, 720))
    ax = Axis3(fig[1, 1], xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title)
    coords = Point3f[]
    raw = eltype(vals)[]
    for I in CartesianIndices(vals)
        push!(coords, Point3f(xs[I[1]], ys[I[2]], zs[I[3]]))
        push!(raw, vals[I])
    end
    if categorical
        keep = findall(!=(0), raw)
        cmap = isempty(keep) ? nothing : get_color_map(raw[keep]; colormap=colormap, appendix="#")
        !isempty(keep) && meshscatter!(ax, coords[keep]; color=getindex.(Ref(cmap), raw[keep]), markersize=0.04)
        isnothing(cmap) || add_rgm_colorbar!(fig, cmap)
    else
        kwargs = isnothing(colorrange) ? (; colormap=colormap) : (; colormap=colormap, colorrange=colorrange)
        meshscatter!(ax, coords; color=Float64.(raw), markersize=0.04, kwargs...)
    end
    return fig, ax
end

function _binding_partition_value(
    model::Bnc,
    logqK::AbstractVector{<:Real};
    chart::Symbol,
    value_func,
    method::Union{Symbol,Nothing},
    asymptotic_only::Bool,
    tol::Float64,
)
    if isnothing(value_func) && chart === :qK
        return assign_regime_qK(model, logqK; input_logspace=true, asymptotic_only=asymptotic_only, return_idx=true)
    end
    logx = solve_logx_checked(model, logqK; method=method, tol=tol)
    isnothing(logx) && return nothing
    if isnothing(value_func)
        chart === :x || throw(ArgumentError("chart must be :qK or :x."))
        return assign_regime_x(model, logx; input_logspace=true, asymptotic_only=asymptotic_only, return_idx=true)
    end
    return Float64(value_func(logx, logqK))
end

function _fill_binding_partition_direct!(
    vals,
    model::Bnc,
    fixed_logqK,
    idxs,
    rgs;
    chart::Symbol,
    value_func,
    method::Union{Symbol,Nothing},
    asymptotic_only::Bool,
    tol::Float64,
    categorical::Bool,
)
    Threads.@threads for linear_idx in eachindex(vals)
        I = CartesianIndices(vals)[linear_idx]
        logqK = copy(fixed_logqK)
        for (ax_i, qk_i) in enumerate(idxs)
            logqK[qk_i] = rgs[ax_i][I[ax_i]]
        end
        vals[I] = try
            v = _binding_partition_value(
                model,
                logqK;
                chart=chart,
                value_func=value_func,
                method=method,
                asymptotic_only=asymptotic_only,
                tol=tol,
            )
            isnothing(v) ? (categorical ? 0 : NaN) : v
        catch
            categorical ? 0 : NaN
        end
    end
    return vals
end

function _fill_binding_partition_homotopy!(
    vals,
    model::Bnc,
    fixed_logqK,
    idxs,
    rgs;
    chart::Symbol,
    value_func,
    asymptotic_only::Bool,
    tol::Float64,
    categorical::Bool,
)
    length(idxs) in (2, 3) || return _fill_binding_partition_direct!(
        vals, model, fixed_logqK, idxs, rgs;
        chart=chart, value_func=value_func, method=:homotopy,
        asymptotic_only=asymptotic_only, tol=tol, categorical=categorical,
    )
    rest_shape = length(idxs) == 2 ? (length(rgs[2]),) : (length(rgs[2]), length(rgs[3]))
    Threads.@threads for rest_linear in 1:prod(rest_shape)
        rest_I = CartesianIndices(rest_shape)[rest_linear]
        start_logqK = copy(fixed_logqK)
        stop_logqK = copy(fixed_logqK)
        start_logqK[idxs[1]] = first(rgs[1])
        stop_logqK[idxs[1]] = last(rgs[1])
        start_logqK[idxs[2]] = stop_logqK[idxs[2]] = rgs[2][rest_I[1]]
        if length(idxs) == 3
            start_logqK[idxs[3]] = stop_logqK[idxs[3]] = rgs[3][rest_I[2]]
        end
        logxs = try
            x_traj_with_qK_change(
                model,
                start_logqK,
                stop_logqK;
                input_logspace=true,
                output_logspace=true,
                npoints=length(rgs[1]),
            )[2]
        catch
            nothing
        end
        for i in eachindex(rgs[1])
            I = length(idxs) == 2 ? CartesianIndex(i, rest_I[1]) : CartesianIndex(i, rest_I[1], rest_I[2])
            logqK = copy(start_logqK)
            logqK[idxs[1]] = rgs[1][i]
            vals[I] = try
                if isnothing(logxs)
                    categorical ? 0 : NaN
                else
                    logx = logxs[i]
                    resid_ok = maximum(abs.(qK2x_residual(model, logx, logqK; input_logspace=true))) <= tol
                    if !resid_ok
                        categorical ? 0 : NaN
                    elseif isnothing(value_func)
                        chart === :qK ?
                            assign_regime_qK(model, logqK; input_logspace=true, asymptotic_only=asymptotic_only, return_idx=true) :
                            assign_regime_x(model, logx; input_logspace=true, asymptotic_only=asymptotic_only, return_idx=true)
                    else
                        Float64(value_func(logx, logqK))
                    end
                end
            catch
                categorical ? 0 : NaN
            end
        end
    end
    return vals
end

"""
    plot_binding_regime_partition(model; axes, fixed=nothing, ranges=(-6,6), n=100,
        chart=:qK, value_func=nothing, method=nothing)

Plot a 2D or 3D partition over selected `qK` coordinates. `chart=:qK`
classifies by `assign_regime_qK`; `chart=:x` first solves `qK -> x` and
classifies by `assign_regime_x`. If `value_func` is supplied, it is called as
`value_func(logx, logqK)` and plotted as a scalar field.
"""
function plot_binding_regime_partition(
    model::Bnc;
    axes,
    fixed=nothing,
    ranges=(-6.0, 6.0),
    n::Integer=100,
    chart::Symbol=:qK,
    value_func=nothing,
    method::Union{Symbol,Nothing}=nothing,
    input_logspace::Bool=true,
    colormap=:Pastel1_9,
    colorrange=nothing,
    asymptotic_only::Bool=false,
    tol::Float64=1e-6,
)
    method = _resolve_qK2x_method(model, method)
    syms = qK_sym(model)
    idxs = _axis_indices(syms, axes)
    rgs = _axis_ranges(length(idxs); ranges=ranges, n=n, input_logspace=input_logspace)
    fixed_logqK = _fixed_log_values(syms, fixed; input_logspace=input_logspace, axis_idxs=idxs)
    categorical = isnothing(value_func)
    T = categorical ? Int : Float64
    vals = Array{T}(undef, length.(rgs)...)
    find_all_regimes!(model) 

    if method === :homotopy && (chart === :x || !isnothing(value_func))
        _fill_binding_partition_homotopy!(
            vals, model, fixed_logqK, idxs, rgs;
            chart=chart, value_func=value_func, asymptotic_only=asymptotic_only,
            tol=tol, categorical=categorical,
        )
    else
        _fill_binding_partition_direct!(
            vals, model, fixed_logqK, idxs, rgs;
            chart=chart, value_func=value_func, method=method,
            asymptotic_only=asymptotic_only, tol=tol, categorical=categorical,
        )
    end

    labels = _sym_text.(syms[idxs])
    title = isnothing(value_func) ? "Binding regime partition ($chart)" : "Binding scalar field"
    if length(idxs) == 2
        fig, ax = _draw_partition_2d(rgs[1], rgs[2], vals; xlabel=labels[1], ylabel=labels[2], title=title, colormap=colormap, categorical=categorical, colorrange=colorrange)
    elseif length(idxs) == 3
        fig, ax = _draw_partition_3d(rgs[1], rgs[2], rgs[3], vals; xlabel=labels[1], ylabel=labels[2], zlabel=labels[3], title=title, colormap=colormap, categorical=categorical, colorrange=colorrange)
    else
        throw(ArgumentError("Select 2 or 3 qK axes."))
    end
    return fig, ax, (; axes=idxs, ranges=rgs, values=vals)
end

"""
    plot_bnc_regime_partition(model; axes, fixed=nothing, ranges=(-3,3), n=60)

Plot a Bnc regime partition over selected `(w,K,k)` log coordinates.
"""
function plot_bnc_regime_partition(
    model::Bnc;
    axes,
    fixed=nothing,
    ranges=(-6.0, 6.0),
    n::Integer=100,
    chart::Symbol=:wKk,
    input_logspace::Bool=true,
    colormap=:Pastel1_9,
    tol::Float64=1e-6,
    max_nullity::Integer=0,
)
    chart === :wKk || throw(ArgumentError("plot_bnc_regime_partition currently supports chart=:wKk."))
    match_regimes!(model)
    syms = wKk_sym(model)
    idxs = _axis_indices(syms, axes)
    rgs = _axis_ranges(length(idxs); ranges=ranges, n=n, input_logspace=input_logspace)
    fixed_logwKk = _fixed_log_values(syms, fixed; input_logspace=input_logspace, axis_idxs=idxs)
    vals = Array{Int}(undef, length.(rgs)...)

    Threads.@threads for linear_idx in eachindex(vals)
        I = CartesianIndices(vals)[linear_idx]
        logwKk = copy(fixed_logwKk)
        for (ax_i, wk_i) in enumerate(idxs)
            logwKk[wk_i] = rgs[ax_i][I[ax_i]]
        end
        vals[I] = assign_bnc_regime_wKk(model, logwKk; tol=tol, max_nullity=max_nullity)
    end

    labels = _sym_text.(syms[idxs])
    if length(idxs) == 2
        fig, ax = _draw_partition_2d(rgs[1], rgs[2], vals; xlabel=labels[1], ylabel=labels[2], title="Bnc regime partition", colormap=colormap)
    elseif length(idxs) == 3
        fig, ax = _draw_partition_3d(rgs[1], rgs[2], rgs[3], vals; xlabel=labels[1], ylabel=labels[2], zlabel=labels[3], title="Bnc regime partition", colormap=colormap)
    else
        throw(ArgumentError("Select 2 or 3 wKk axes."))
    end
    return fig, ax, (; axes=idxs, ranges=rgs, values=vals)
end

function _qcat_flux_at(
    model::Bnc,
    logqcat::AbstractVector{<:Real},
    logwKk::AbstractVector{<:Real};
    method::Union{Symbol,Nothing}=nothing,
    tol::Float64=1e-6,
)
    method = _resolve_qK2x_method(model, method)
    cn = model.catalysis
    logqK = _logqK_from_logqcat_logwKk(model, logqcat, logwKk)
    logx = solve_logx_checked(model, logqK; method=method, tol=tol)
    isnothing(logx) && return nothing

    logv = cn.Π * logx .+ _old_logk_from_logwKk(model, logwKk)
    vshift = maximum(logv)
    vscaled = exp10.(logv .- vshift)
    qdot_scaled = Vector{Float64}(cn.S * vscaled)
    qdot_log = qdot_scaled .* exp10.(vshift .- Float64.(logqcat)) ./ log(10.0)
    return (; qdot_scaled, qdot_log)
end

function _finite_difference_jacobian(f, x::Vector{Float64}; h::Float64=1e-4)
    fx = f(x)
    isnothing(fx) && return nothing
    J = Matrix{Float64}(undef, length(fx), length(x))
    for j in eachindex(x)
        xp = copy(x)
        xm = copy(x)
        xp[j] += h
        xm[j] -= h
        fp = f(xp)
        fm = f(xm)
        (isnothing(fp) || isnothing(fm)) && return nothing
        J[:, j] .= (fp .- fm) ./ (2h)
    end
    return J
end


function _newton_refine_qcat_root(
    model::Bnc,
    seed::AbstractVector{<:Real},
    logwKk::AbstractVector{<:Real},
    bounds;
    method::Union{Symbol,Nothing}=nothing,
    tol::Float64=1e-6,
    root_tol::Float64=1e-7,
    maxiters::Integer=30,
)
    x = Vector{Float64}(seed)
    f_scaled(u) = begin
        out = _qcat_flux_at(model, u, logwKk; method=method, tol=tol)
        isnothing(out) ? nothing : out.qdot_scaled
    end
    for _ in 1:maxiters
        fx = f_scaled(x)
        isnothing(fx) && return nothing
        norm(fx, Inf) <= root_tol && break
        J = _finite_difference_jacobian(f_scaled, x)
        isnothing(J) && return nothing
        step = try
            -(J \ fx)
        catch
            return nothing
        end
        all(isfinite, step) || return nothing

        accepted = false
        fx_norm = norm(fx, Inf)
        damping = 1.0
        while damping >= 1 / 64
            trial = x .+ damping .* step
            in_bounds = all(i -> bounds[i][1] <= trial[i] <= bounds[i][2], eachindex(trial))
            if in_bounds
                ftrial = f_scaled(trial)
                if !isnothing(ftrial) && norm(ftrial, Inf) <= fx_norm
                    x = trial
                    accepted = true
                    break
                end
            end
            damping /= 2
        end
        accepted || return nothing
    end
    out = _qcat_flux_at(model, x, logwKk; method=method, tol=tol)
    (isnothing(out) || norm(out.qdot_scaled, Inf) > root_tol) && return nothing
    return x
end

function _is_numerically_stable_qcat_root(
    model::Bnc,
    logqcat::Vector{Float64},
    logwKk::AbstractVector{<:Real};
    method::Union{Symbol,Nothing}=nothing,
    tol::Float64=1e-6,
)
    f_log(u) = begin
        out = _qcat_flux_at(model, u, logwKk; method=method, tol=tol)
        isnothing(out) ? nothing : out.qdot_log
    end
    J = _finite_difference_jacobian(f_log, logqcat)
    isnothing(J) && return missing
    vals = eigvals(J)
    all(real.(vals) .< 0) ? true : false
end

function _push_unique_qcat_root!(
    roots::Vector{Vector{Float64}},
    root::Vector{Float64};
    dedup_tol::Float64=1e-3,
)
    any(r -> norm(r .- root, Inf) <= dedup_tol, roots) && return false
    push!(roots, root)
    return true
end

function _qcat_root_seeds(rgs, flux, feasible; max_norm_seeds::Integer=80)
    seeds = Vector{Vector{Float64}}()
    if length(rgs) == 1
        xs = collect(rgs[1])
        for i in eachindex(xs)
            feasible[i] && push!(seeds, [xs[i]])
        end
        return seeds
    end

    xs = collect(rgs[1])
    ys = collect(rgs[2])
    for i in 1:length(xs)-1, j in 1:length(ys)-1
        all(feasible[i + di, j + dj] for di in 0:1, dj in 0:1) || continue
        f1s = [flux[i + di, j + dj][1] for di in 0:1, dj in 0:1]
        f2s = [flux[i + di, j + dj][2] for di in 0:1, dj in 0:1]
        minimum(f1s) <= 0 <= maximum(f1s) || continue
        minimum(f2s) <= 0 <= maximum(f2s) || continue
        push!(seeds, [(xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2])
    end

    norms = Tuple{Float64,Int,Int}[]
    for i in eachindex(xs), j in eachindex(ys)
        feasible[i, j] || continue
        push!(norms, (norm(flux[i, j], Inf), i, j))
    end
    sort!(norms; by=first)
    for (_, i, j) in Iterators.take(norms, max_norm_seeds)
        push!(seeds, [xs[i], ys[j]])
    end
    return seeds
end

function _numeric_fixed_points_qcat(
    model::Bnc,
    logwKk,
    rgs,
    flux,
    feasible;
    method::Union{Symbol,Nothing}=nothing,
    tol::Float64=1e-6,
    root_tol::Float64=1e-7,
)
    bounds = [(Float64(first(rg)), Float64(last(rg))) for rg in rgs]
    roots = Vector{Vector{Float64}}()
    if length(rgs) == 1
        xs = collect(rgs[1])
        fvals = [feasible[i] ? flux[i][1] : NaN for i in eachindex(xs)]
        for i in 1:length(xs)-1
            isfinite(fvals[i]) && abs(fvals[i]) <= root_tol && _push_unique_qcat_root!(roots, [xs[i]])
            (isfinite(fvals[i]) && isfinite(fvals[i + 1])) || continue
            fvals[i] == 0 && continue
            sign(fvals[i]) == sign(fvals[i + 1]) && continue
            lo, hi = xs[i], xs[i + 1]
            flo, fhi = fvals[i], fvals[i + 1]
            for _ in 1:60
                mid = (lo + hi) / 2
                out = _qcat_flux_at(model, [mid], logwKk; method=method, tol=tol)
                isnothing(out) && break
                fmid = out.qdot_scaled[1]
                abs(fmid) <= root_tol && (lo = hi = mid; break)
                if sign(flo) == sign(fmid)
                    lo, flo = mid, fmid
                else
                    hi, fhi = mid, fmid
                end
                abs(hi - lo) <= 1e-8 && break
            end
            _push_unique_qcat_root!(roots, [(lo + hi) / 2])
        end
    end

    for seed in _qcat_root_seeds(rgs, flux, feasible)
        root = _newton_refine_qcat_root(
            model,
            seed,
            logwKk,
            bounds;
            method=method,
            tol=tol,
            root_tol=root_tol,
        )
        isnothing(root) && continue
        _push_unique_qcat_root!(roots, root)
    end

    sort!(roots; by=x -> Tuple(x))
    pts = NamedTuple[]
    for (i, root) in pairs(roots)
        stable = _is_numerically_stable_qcat_root(model, root, logwKk; method=method, tol=tol)
        push!(pts, (; idx=i, logqcat=root, stable, source=:numeric))
    end
    return pts
end

function _draw_qcat_nullclines!(ax, rgs, flux, feasible; colors=(:red, :blue), linewidth::Real=2.5)
    length(rgs) == 2 || return nothing
    xs = collect(rgs[1])
    ys = collect(rgs[2])
    for component in 1:2
        field = [feasible[i, j] ? flux[i, j][component] : NaN for i in eachindex(xs), j in eachindex(ys)]
        contour!(
            ax,
            xs,
            ys,
            field;
            levels=[0.0],
            color=colors[component],
            linewidth=linewidth,
        )
    end
    return nothing
end

function _draw_qcat_fixed_points!(
    ax,
    fixed_points;
    label_prefix::AbstractString="FP",
    labels::Bool=true,
    markersize::Real=14,
)
    isempty(fixed_points) && return nothing
    if length(first(fixed_points).logqcat) == 1
        xs = [pt.logqcat[1] for pt in fixed_points]
        ys = zeros(length(xs))
        colors = [pt.stable === true ? :black : :white for pt in fixed_points]
        scatter!(ax, xs, ys; color=colors, strokecolor=:black, strokewidth=1.5, markersize=markersize)
        labels && text!(
            ax,
            xs,
            ys;
            text=["$label_prefix$(pt.idx)" for pt in fixed_points],
            align=(:left, :bottom),
            offset=(6, 6),
            color=:black,
            fontsize=12,
        )
    else
        xs = [pt.logqcat[1] for pt in fixed_points]
        ys = [pt.logqcat[2] for pt in fixed_points]
        colors = [pt.stable === true ? :black : :white for pt in fixed_points]
        scatter!(ax, xs, ys; color=colors, strokecolor=:black, strokewidth=1.5, markersize=markersize)
        labels && text!(
            ax,
            xs,
            ys;
            text=["$label_prefix$(pt.idx)" for pt in fixed_points],
            align=(:left, :bottom),
            offset=(6, 6),
            color=:black,
            fontsize=12,
        )
    end
    return nothing
end

"""
    plot_qcat_slice_with_flux(model; wKk, ranges=(-3,3), n=45, chart=:qK)

For fixed `(w,K,k)`, plot the regime partition over `q_cat` coordinates,
overlay the catalysis flux direction, and mark fixed points found by numerical
root solving of the catalysis flux. Supports one or two `q_cat` dimensions for
compact plots. In 2D, `show_nullclines=true` overlays the two zero-flux
nullclines before fixed-point markers and labels are drawn.
"""
function plot_qcat_slice_with_flux(
    model::Bnc;
    wKk,
    ranges=(-3.0, 3.0),
    n::Integer=45,
    chart::Symbol=:qK,
    method::Union{Symbol,Nothing}=nothing,
    input_logspace::Bool=true,
    colormap=:Pastel1_9,
    arrow_stride::Integer=5,
    tol::Float64=1e-6,
    show_nullclines::Bool=false,
    nullcline_colors=(:red, :blue),
    fixed_point_labels::Bool=true,
    fixed_point_label_prefix::AbstractString="FP",
    fixed_point_root_tol::Float64=1e-7,
)
    method = _resolve_qK2x_method(model, method)
    cn = model.catalysis
    cn.r_v <= 2 || throw(ArgumentError("plot_qcat_slice_with_flux currently keeps plots readable for q_cat dimension <= 2."))
    logwKk = _fixed_log_values(wKk_sym(model), wKk; input_logspace=input_logspace)
    qcat_syms = q_cat_sym(model)
    qK_base = zeros(Float64, model.d + model.r)
    qK_base[cn.r_v + 1:model.d] .= logwKk[1:cn.d_w]
    qK_base[model.d + 1:end] .= logwKk[cn.d_w + 1:cn.d_w + model.r]
    logk_old = _old_logk_from_logwKk(model, logwKk)

    rgs = _axis_ranges(cn.r_v; ranges=ranges, n=n, input_logspace=input_logspace)
    vals = Array{Int}(undef, length.(rgs)...)
    flux = Array{Vector{Float64}}(undef, size(vals))
    feasible = falses(size(vals))

    Threads.@threads for linear_idx in eachindex(vals)
        I = CartesianIndices(vals)[linear_idx]
        logqK = copy(qK_base)
        for ax_i in 1:cn.r_v
            logqK[ax_i] = rgs[ax_i][I[ax_i]]
        end
        try
            logx = solve_logx_checked(model, logqK; method=method, tol=tol)
            isnothing(logx) && error("No feasible qK point.")
            vals[I] = chart === :x ?
                assign_regime_x(model, logx; input_logspace=true, asymptotic_only=false, return_idx=true) :
                assign_regime_qK(model, logqK; input_logspace=true, asymptotic_only=false, return_idx=true)
            logv = cn.Π * logx .+ logk_old
            vshift = maximum(logv)
            vscaled = exp10.(logv .- vshift)
            flux[I] = Vector{Float64}(cn.S * vscaled)
            feasible[I] = true
        catch
            vals[I] = 0
            flux[I] = zeros(Float64, cn.r_v)
        end
    end

    labels = _sym_text.(qcat_syms)
    if cn.r_v == 1
        fig = Figure(size=(760, 420))
        ax = Axis(fig[1, 1], xlabel=labels[1], ylabel="flux", title="q_cat slice")
        xs = collect(rgs[1])
        ys = [flux[i][1] for i in eachindex(xs)]
        lines!(ax, xs, ys; color=:black)
        hlines!(ax, [0.0]; color=(:gray, 0.5), linestyle=:dash)
        fixed_points = _numeric_fixed_points_qcat(
            model,
            logwKk,
            rgs,
            flux,
            feasible;
            method=method,
            tol=tol,
            root_tol=fixed_point_root_tol,
        )
        _draw_qcat_fixed_points!(
            ax,
            fixed_points;
            labels=fixed_point_labels,
            label_prefix=fixed_point_label_prefix,
            markersize=12,
        )
        return fig, ax, (; ranges=rgs, values=vals, flux=flux, feasible=feasible, fixed_points)
    end

    fig, ax = _draw_partition_2d(rgs[1], rgs[2], vals; xlabel=labels[1], ylabel=labels[2], title="q_cat regime and flux slice", colormap=colormap)
    xs = collect(rgs[1])
    ys = collect(rgs[2])
    arrow_points = Point2f[]
    arrow_dirs = Vec2f[]
    for i in 1:arrow_stride:length(xs), j in 1:arrow_stride:length(ys)
        feasible[i, j] || continue
        f = flux[i, j]
        scale = norm(f)
        scale <= tol && continue
        push!(arrow_points, Point2f(xs[i], ys[j]))
        push!(arrow_dirs, Vec2f(f[1], f[2]) / scale * 0.15)
    end
    arrows2d!(ax, arrow_points, arrow_dirs; color=(:black, 0.65), shaftwidth=2, tipwidth=8, tiplength=10)
    show_nullclines && _draw_qcat_nullclines!(ax, rgs, flux, feasible; colors=nullcline_colors)
    fixed_points = _numeric_fixed_points_qcat(
        model,
        logwKk,
        rgs,
        flux,
        feasible;
        method=method,
        tol=tol,
        root_tol=fixed_point_root_tol,
    )
    fixed_points = filter(fixed_points) do pt
        first(rgs[1]) <= pt.logqcat[1] <= last(rgs[1]) &&
            first(rgs[2]) <= pt.logqcat[2] <= last(rgs[2])
    end
    _draw_qcat_fixed_points!(
        ax,
        fixed_points;
        labels=fixed_point_labels,
        label_prefix=fixed_point_label_prefix,
        markersize=14,
    )
    return fig, ax, (; ranges=rgs, values=vals, flux=flux, feasible=feasible, fixed_points)
end
