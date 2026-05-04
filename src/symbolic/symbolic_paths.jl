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

@inline function _simo_marker_vars()
    continuous, upward, downward = :→, :↑, :↓
    return @variables $continuous, $upward, $downward
end

@inline function _simo_singular_marker(coeff::Real, vars)
    if abs(coeff) < 1e-6
        return vars[1]
    else
        return coeff > 0 ? vars[2] : vars[3]
    end
end

struct ExpressionPathView{P,E,B}
    rgm_path::P
    expr_rows::E
    boundary_exprs::B
end

Base.length(::ExpressionPathView) = 2
Base.getindex(v::ExpressionPathView, i::Integer) = i == 1 ? v.expr_rows : i == 2 ? v.boundary_exprs : throw(BoundsError(v, i))
Base.iterate(v::ExpressionPathView, state::Int=1) = state == 1 ? (v.expr_rows, 2) : state == 2 ? (v.boundary_exprs, 3) : nothing

@inline _path_block(x) = x isa AbstractVector ? x : Any[x]
@inline _path_line(x) = replace(sprint(show, MIME"text/plain"(), x), '\n' => ' ')
@inline _same_expr_row(a, b) = begin
    aa, bb = _path_block(a), _path_block(b)
    length(aa) == length(bb) && all(isequal.(aa, bb))
end

function _merged_expression_path_blocks(view::ExpressionPathView)
    isempty(view.rgm_path) && return NamedTuple[]
    starts = Int[1]
    for i in 2:length(view.rgm_path)
        _same_expr_row(view.expr_rows[i - 1], view.expr_rows[i]) || push!(starts, i)
    end
    blocks = NamedTuple[]
    for (k, start) in enumerate(starts)
        stop = k == length(starts) ? length(view.rgm_path) : starts[k + 1] - 1
        push!(blocks, (
            rgms=view.rgm_path[start:stop],
            exprs=_path_block(view.expr_rows[start]),
            boundary=stop < length(view.rgm_path) ? view.boundary_exprs[stop] : nothing,
            boundary_from=stop < length(view.rgm_path) ? view.rgm_path[stop] : nothing,
            boundary_to=stop < length(view.rgm_path) ? view.rgm_path[stop + 1] : nothing,
        ))
    end
    return blocks
end

function Base.show(io::IO, ::MIME"text/plain", view::ExpressionPathView)
    println(io, "Expression path along regimes: ", format_arrow(view.rgm_path))
    for block in _merged_expression_path_blocks(view)
        println(io, "Regime ", join(string.(block.rgms), ", "))
        for expr in block.exprs
            println(io, "  ", _path_line(expr))
        end
        if !isnothing(block.boundary)
            println(io, "  cross ", block.boundary_from, " → ", block.boundary_to, " when")
            for expr in _path_block(block.boundary)
                println(io, "    ", _path_line(expr))
            end
        end
    end
end

Base.show(io::IO, view::ExpressionPathView) = show(io, MIME"text/plain"(), view)

function _path_expression_rows(
    model::Bnc,
    rgm_path::AbstractVector{<:Integer},
    change_qK_idx::Integer,
    observe_x_idx::AbstractVector{<:Integer};
    log_space::Bool=false,
)
    change_qK_idx = locate_sym_qK(model, change_qK_idx)
    observe_x_idx = Int.(locate_sym_x.(Ref(model), observe_x_idx))
    xsym = x_sym(model)[observe_x_idx]
    vars = _simo_marker_vars()

    expr_rows = Vector{Vector{Any}}(undef, length(rgm_path))
    for (i, rgm_idx_raw) in enumerate(rgm_path)
        rgm_idx = get_idx(model, rgm_idx_raw)
        expr_rows[i] = if get_nullity(model, rgm_idx) == 0
            collect(show_expression_x(model, rgm_idx; log_space=log_space)[observe_x_idx])
        elseif get_nullity(model, rgm_idx) == 1
            H = get_H(model, rgm_idx)
            row = Vector{Any}(undef, length(observe_x_idx))
            for (j, x_idx) in enumerate(observe_x_idx)
                lhs = log_space ? log10(xsym[j]) : xsym[j]
                row[j] = lhs ~ _simo_singular_marker(H[x_idx, change_qK_idx], vars)
            end
            row
        else
            error("Nullity > 1 is not supported for expression path.")
        end
    end

    boundary_exprs = Vector{Any}(undef, max(length(rgm_path) - 1, 0))
    for i in eachindex(boundary_exprs)
        boundary_exprs[i] = show_interface(
            model,
            rgm_path[i],
            rgm_path[i + 1];
            lhs_idx=change_qK_idx,
            log_space=log_space,
        )
    end

    return expr_rows, boundary_exprs
end

function get_expression_path(
    model::Bnc,
    rgm_path::AbstractVector,
    change_qK_idx;
    observe_x=nothing,
    log_space::Bool=false,
)
    observe_x_idx, _, scalar_observe = _normalize_simo_observe_x(model, observe_x)
    expr_rows, boundary_exprs = _path_expression_rows(
        model,
        Int.(get_idx.(Ref(model), rgm_path)),
        change_qK_idx,
        observe_x_idx;
        log_space=log_space,
    )
    return scalar_observe ? (first.(expr_rows), boundary_exprs) : (expr_rows, boundary_exprs)
end

function get_expression_path(grh::SIMOPaths, pth; observe_x=nothing, kwargs...)
    pth_idx = get_idx(grh, pth)
    rgm_path = get_path(grh, pth_idx; return_idx=true)
    return get_expression_path(
        get_binding_network(grh),
        rgm_path,
        grh.change_qK_idx;
        observe_x=observe_x,
        kwargs...,
    )
end

function show_expression_path(grh::SIMOPaths, pth; observe_x=nothing, kwargs...)
    pth_idx = get_idx(grh, pth)
    rgm_path = get_path(grh, pth_idx; return_idx=true)
    expr_rows, boundary_exprs = get_expression_path(grh, pth; observe_x=observe_x, kwargs...)
    return ExpressionPathView(rgm_path, expr_rows, boundary_exprs)
end

function show_expression_path(model::Bnc, rgm_path, change_qK_idx, observe_x; kwargs...)
    expr_rows, boundary_exprs = get_expression_path(model, rgm_path, change_qK_idx; observe_x=observe_x, kwargs...)
    return ExpressionPathView(Int.(get_idx.(Ref(model), rgm_path)), expr_rows, boundary_exprs)
end

show_expression_path(grh::SIMOPaths, pth_idx, observe_x; kwargs...) =
    show_expression_path(grh, pth_idx; observe_x=observe_x, kwargs...)
