"""
    SIMO_plot(pths::SIMOPaths, pth_idx; rand_line=false, rand_ray=false, extend=4, kwargs...) -> Figure

Plot a SIMO path trajectory in x space colored by dominant regime.
"""
function SIMO_plot(pths::SIMOPaths, pth_idx; rand_line=false, rand_ray=false, extend=4, kwargs...)
    pth_idx = get_idx(pths, pth_idx)
    parameters = get_one_inner_point(get_polyhedron(pths, pth_idx); rand_line=rand_line, rand_ray=rand_ray, extend=extend)
    @show parameters
    return SIMO_plot(pths.bn, parameters, pths.change_qK_idx; kwargs...)
end

"""
    SIMO_plot(model::Bnc, parameters, change_idx; npoints=1000, start=-6, stop=6, colormap=:rainbow,
        size=(800, 600), observe_x=nothing, add_archeatype_lines=false, asymptotic_only=false) -> Figure

Plot x trajectories for a single changing qK coordinate.
"""
function SIMO_plot(
    model::Bnc,
    parameters,
    change_idx;
    npoints=1000,
    start=-6,
    stop=6,
    colormap=:rainbow,
    size=(800, 600),
    observe_x=nothing,
    fx=nothing,
    farchx=nothing,
    add_archeatype_lines::Bool=false,
    asymptotic_only::Bool=false,
)
    change_idx = locate_sym_qK(model, change_idx)
    change_sym = "log" * repr(qK_sym(model)[change_idx])
    change_S = range(start, stop, npoints)

    start_logqK = copy(parameters) |> x -> insert!(x, change_idx, start)
    end_logqK = copy(parameters) |> x -> insert!(x, change_idx, stop)
    logx = x_traj_with_qK_change(
        model,
        start_logqK,
        end_logqK;
        input_logspace=true,
        output_logspace=true,
        npoints=npoints,
        ensure_manifold=true,
    )[2]

    logx_arch = if add_archeatype_lines
        [qK2x(model, logqK; input_logspace=true, use_vtx=true, output_logspace=true) for logqK in range(start_logqK, end_logqK, npoints)]
    else
        nothing
    end

    rgms = logx .|> x -> assign_regime_x(model, x; input_logspace=true, asymptotic_only=asymptotic_only, return_idx=true)
    cmap = get_color_map(rgms; colormap=colormap)
    colors = getindex.(Ref(cmap), rgms)

    @info "Change in $(change_sym)"
    @info "parameters: $([i => j for (i, j) in zip([model.q_sym; model.K_sym] |> x -> deleteat!(x, change_idx), parameters)])"

    F = if isnothing(fx)
        draw_idx = isnothing(observe_x) ? (1:model.n) : locate_sym_x(model, observe_x)
        F = Figure(size=size)
        axes = Axis[]
        for (i, j) in enumerate(draw_idx)
            target_sym = "log" * repr(model.x_sym[j])
            @info "Target syms contains: $(target_sym) "
            ax = Axis(F[i, 1]; xlabel=change_sym, ylabel=target_sym, aspect=DataAspect())
            push!(axes, ax)

            y = getindex.(logx, j)
            lines!(ax, change_S, y; color=colors)

            if add_archeatype_lines
                yarch = getindex.(logx_arch, j)
                lines!(ax, change_S, yarch; color=:black, linestyle=:dash)
            end
        end
        linkxaxes!(axes...)
        F
    else
        F = Figure(size=size)
        ax = Axis(F[1, 1]; xlabel=change_sym, aspect=DataAspect())
        y = fx.(logx)
        lines!(ax, change_S, y; color=colors)

        if add_archeatype_lines
            archf = isnothing(farchx) ? fx : farchx
            isnothing(farchx) && @warn "No farchx provided, using fx as for archetype line."
            yarch = archf.(logx_arch)
            lines!(ax, change_S, yarch; color=:black, linestyle=:dash)
        end
        F
    end

    add_rgm_colorbar!(F, cmap)
    return F
end
