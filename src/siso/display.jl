function show_regime_path(grh::SISOPaths, pth)
    pth_idx = get_idx(grh, pth)
    path = get_path(grh, pth_idx; return_idx=true)
    volume = grh.path_volume_is_calc[pth_idx] ? grh.path_volume[pth_idx] : nothing
    print_path(path; prefix="#", id=pth_idx, volume=volume)
    return nothing
end

function summary(grh::SISOPaths; show_volume::Bool=true, prefix::AbstractString="#", kwargs...)
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
    grh::SISOPaths;
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

    volumes = show_volume ? get_volumes(grh; kwargs...) : fill(nothing, length(grh.rgm_paths))
    rsts = group_sum(ord_pth, volumes)
    ids = getindex.(rsts, 1)
    ords = getindex.(rsts, 2)
    vols = getindex.(rsts, 3)
    print_paths(ords; prefix="", ids=ids, volumes=vols)
    return nothing
end

function Base.display(grh::SISOPaths)
    println("SISOPaths object with $(length(grh.rgm_paths)) paths for qK coordinate index $(grh.change_qK_idx)")
    return nothing
end

Base.show(io::IO, grh::SISOPaths) = print(io, "SISOPaths object with $(length(grh.rgm_paths)) paths for qK coordinate index $(grh.change_qK_idx)")
