function sym_direction(Bnc::Bnc, dir)::String
    rst = ""
    for i in 1:Bnc.d
        if dir[i] > 1e-6
            rst *= "+" * repr(Bnc.q_sym[i]) * " "
        elseif dir[i] < -1e-6
            rst *= "-" * repr(Bnc.q_sym[i]) * " "
        end
    end
    rst *= "; "
    for j in 1:Bnc.r
        if dir[j + Bnc.d] > 1e-6
            rst *= "+" * repr(Bnc.K_sym[j]) * " "
        elseif dir[j + Bnc.d] < -1e-6
            rst *= "-" * repr(Bnc.K_sym[j]) * " "
        end
    end
    return rst
end

function sym_direction(dir; syms::AbstractArray{Num})::String
    rst = ""
    for i in eachindex(dir)
        if dir[i] > 1e-6
            rst *= "+" * repr(syms[i]) * " "
        elseif dir[i] < -1e-6
            rst *= "-" * repr(syms[i]) * " "
        end
    end
    return rst
end

@inline function _fmt_elem(x; digits::Int=3)::String
    if x isa Real
        if isnan(x)
            return "NaN"
        end
        xr = round(Float64(x); digits=digits)
        if isfinite(xr) && isapprox(xr, round(xr); atol=10.0^(-digits), rtol=0)
            return string(Int(round(xr)))
        else
            return string(xr)
        end
    else
        return repr(x)
    end
end

function format_arrow(path::AbstractVector; prefix::AbstractString="", digits::Int=3)::String
    isempty(path) && return ""
    parts = Vector{String}(undef, length(path))
    @inbounds for i in eachindex(path)
        parts[i] = prefix * _fmt_elem(path[i]; digits=digits)
    end
    return join(parts, " → ")
end

struct PathRow{I,P,V}
    id::I
    path::P
    volume::V
end

function _normalize_rows(paths::AbstractVector{<:AbstractVector}; ids=nothing, volumes=nothing)
    n = length(paths)
    ids === nothing && (ids = collect(1:n))
    volumes === nothing && (volumes = fill(nothing, n))
    @assert length(ids) == n "ids length must match paths length"
    @assert length(volumes) == n "volumes length must match paths length"
    rows = Vector{PathRow}(undef, n)
    @inbounds for i in 1:n
        rows[i] = PathRow(ids[i], paths[i], volumes[i])
    end
    return rows
end

function print_paths(rows::AbstractVector{<:PathRow}; prefix::AbstractString="", digits::Int=3, io::IO=stdout)
    isempty(rows) && return nothing
    id_strs = [repr(r.id) for r in rows]
    path_strs = [format_arrow(r.path; prefix=prefix, digits=digits) for r in rows]
    id_width = max(8, maximum(length.(id_strs)))
    path_width = max(10, maximum(length.(path_strs)))
    for (r, id_s, path_s) in zip(rows, id_strs, path_strs)
        if r.volume === nothing
            Printf.@printf(io, "Path %-*s  %-*s\n", id_width, id_s, path_width, path_s)
        else
            @assert typeof(r.volume) <: Volume
            v = r.volume.mean
            e = sqrt(r.volume.var)
            Printf.@printf(io, "Path %-*s  %-*s  Volume: %.4f ± %.4f\n", id_width, id_s, path_width, path_s, v, e)
        end
    end
    return nothing
end

print_paths(paths::AbstractVector{<:AbstractVector}; volumes=nothing, ids=nothing, kwargs...) =
    print_paths(_normalize_rows(paths; volumes=volumes, ids=ids); kwargs...)

print_path(path::AbstractVector; id=nothing, volume=nothing, kwargs...) =
    print_paths(_normalize_rows([path]; ids=id === nothing ? nothing : [id], volumes=volume === nothing ? nothing : [volume]); kwargs...)

function show_expression_path(grh::SISOPaths, pth; observe_x=nothing, kwargs...)
    pth_idx = get_idx(grh, pth)
    bn = get_binding_network(grh)
    observe_x_idx = isnothing(observe_x) ? (1:bn.n) : locate_sym_x.(Ref(bn), observe_x)
    change_qK_idx = grh.change_qK_idx
    xsym = x_sym(bn)[observe_x_idx]
    qKsym = qK_sym(bn)
    change_sym = qKsym[change_qK_idx]

    continuous, upward, downward = :→, :↑, :↓
    vars = @variables $continuous, $upward, $downward

    H_H0, rgm_interface = get_expression_path(grh, pth_idx; observe_x=observe_x_idx)

    expr_sym = let
        exprs = Vector{Any}(undef, length(H_H0))
        for (i, (H_row, H0_val)) in enumerate(H_H0)
            exprs[i] = if isnothing(H0_val)
                let
                    a = Vector{Num}(undef, size(H_row, 1))
                    for j in eachindex(a)
                        a[j] = if abs(H_row[j]) < 1e-6
                            vars[1]
                        else
                            H_row[j] > 0 ? vars[2] : vars[3]
                        end
                    end
                    a
                end
            else
                show_expression_mapping(H_row, H0_val, xsym, qKsym; kwargs...)
            end
        end
        exprs
    end

    interface = rgm_interface .|> x -> solve_sym_expr(x..., qKsym, change_qK_idx; kwargs...)
    for i in eachindex(expr_sym)
        if i == 1
            display(change_sym < interface[1].rhs)
        elseif i == length(expr_sym)
            display(change_sym > interface[end].rhs)
        else
            display((change_sym > interface[i - 1].rhs) & (change_sym < interface[i].rhs))
        end
        display(expr_sym[i])
    end
    return nothing
end

function show_expression_path(model::Bnc, rgm_path, change_qK_idx, observe_x_idx; kwargs...)::Tuple{Vector,Vector}
    change_qK_idx = locate_sym([model.q_sym; model.K_sym], change_qK_idx)
    observe_x_idx = locate_sym(model.x_sym, observe_x_idx)
    have_volume_mask = _get_regimes_mask(model, rgm_path; singular=false)
    idx = findall(have_volume_mask)
    exprs = map(idx) do id
        show_expression_x(model, rgm_path[id]; kwargs...)[observe_x_idx].rhs
    end
    edges = map(@view idx[1:end-1]) do i
        rgm_from = rgm_path[i]
        rgm_to = rgm_path[i + 1]
        show_interface(model, rgm_from, rgm_to; lhs_idx=change_qK_idx, kwargs...).rhs
    end
    return (exprs, edges)
end

show_expression_path(grh::SISOPaths, pth_idx, observe_x; kwargs...) =
    show_expression_path(get_binding_network(grh), grh.rgm_paths[pth_idx], grh.change_qK_idx, observe_x; kwargs...)
