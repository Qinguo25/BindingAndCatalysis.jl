function show_regime_path(grh::SIMOPaths, pth)
    pth_idx = get_idx(grh, pth)
    path = get_path(grh, pth_idx; return_idx=true)
    volume = grh.path_volume_is_calc[pth_idx] ? grh.path_volume[pth_idx] : nothing
    print_path(path; prefix="#", id=pth_idx, volume=volume)
    return nothing
end

function summary(
    grh::SIMOPaths; show_volume::Bool=true, prefix::AbstractString="#", kwargs...
)
    paths = grh.rgm_paths
    if show_volume
        vols = get_volumes(grh; kwargs...)
        print_paths(paths; prefix=prefix, volumes=vols, ids=1:length(paths))
    else
        print_paths(paths; prefix=prefix, ids=1:length(paths))
    end
    return nothing
end

function summary_RO_path(
    grh::SIMOPaths;
    observe_x,
    show_volume::Bool=true,
    deduplicate::Bool=true,
    keep_singular::Bool=true,
    keep_nonasymptotic::Bool=true,
    kwargs...,
)
    ord_pth = get_RO_paths(
        grh;
        observe_x=observe_x,
        deduplicate=deduplicate,
        keep_singular=keep_singular,
        keep_nonasymptotic=keep_nonasymptotic,
    )
    !isempty(ord_pth) &&
        ord_pth[1] isa AbstractMatrix &&
        error(
            "summary_RO_path currently supports a single observed x. Use get_RO_paths(...) directly for multi-x data.",
        )

    volumes =
        show_volume ? get_volumes(grh; kwargs...) : fill(nothing, length(grh.rgm_paths))
    rsts = group_sum(ord_pth, volumes)
    ids = getindex.(rsts, 1)
    ords = getindex.(rsts, 2)
    vols = getindex.(rsts, 3)
    print_paths(ords; prefix="", ids=ids, volumes=vols)
    return nothing
end

function Base.display(grh::SIMOPaths)
    println(
        "SIMOPaths object with $(length(grh.rgm_paths)) paths for qK coordinate index $(grh.change_qK_idx)",
    )
    return nothing
end

function Base.show(io::IO, grh::SIMOPaths)
    return print(
        io,
        "SIMOPaths object with $(length(grh.rgm_paths)) paths for qK coordinate index $(grh.change_qK_idx)",
    )
end
