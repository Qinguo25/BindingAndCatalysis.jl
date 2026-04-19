struct SingularXRange
    observe_x_idx::Int
    observe_x_sym::Num
    equalities::Vector{Any}
    lower_bounds::Vector{Any}
    upper_bounds::Vector{Any}
    consistency::Vector{Any}
    lower_regimes::Vector{Int}
    upper_regimes::Vector{Int}
    projected_C::Any
    projected_C0::Any
    projected_nullity::Int
end

@inline _singular_range_text(x) = replace(sprint(show, MIME"text/plain"(), x), '\n' => ' ')
@inline _singular_range_key(x) = _singular_range_text(x)
@inline _relation_text(x) = _singular_range_text(x)

function _split_relation_text(text::AbstractString, op::AbstractString)
    parts = split(text, " $op "; limit=2)
    length(parts) == 2 || error("Could not parse relation `$text` with operator `$op`.")
    return parts[1], parts[2]
end

function _pick_preferred_equality(equalities::Vector{Any})
    texts = _relation_text.(equalities)
    idx = argmin(map(texts) do text
        _, rhs = _split_relation_text(text, "~")
        return (length(rhs), length(text), text)
    end)
    return texts[idx]
end

function _format_singular_interval(xr::SingularXRange)
    !isempty(xr.equalities) && return _pick_preferred_equality(xr.equalities)

    lower_parts = [_split_relation_text(_relation_text(expr), ">") for expr in xr.lower_bounds]
    upper_parts = [_split_relation_text(_relation_text(expr), "<") for expr in xr.upper_bounds]
    lhs = if !isempty(lower_parts)
        first(first(lower_parts))
    elseif !isempty(upper_parts)
        first(first(upper_parts))
    else
        _singular_range_text(xr.observe_x_sym)
    end

    lower_text = isempty(lower_parts) ? nothing : "max(" * join(last.(lower_parts), ", ") * ")"
    upper_text = isempty(upper_parts) ? nothing : "min(" * join(last.(upper_parts), ", ") * ")"

    if !isnothing(lower_text) && !isnothing(upper_text)
        return "$lower_text < $lhs < $upper_text"
    elseif !isnothing(lower_text)
        return "$lower_text < $lhs"
    elseif !isnothing(upper_text)
        return "$lhs < $upper_text"
    elseif !isempty(xr.consistency)
        return "$lhs is only constrained by singular-fiber consistency"
    else
        return "$lhs is unconstrained on the singular fiber"
    end
end

function Base.show(io::IO, xr::SingularXRange)
    print(
        io,
        "SingularXRange(",
        _singular_range_text(xr.observe_x_sym),
        ", eq=",
        length(xr.equalities),
        ", lower=",
        length(xr.lower_bounds),
        ", upper=",
        length(xr.upper_bounds),
        ", consistency=",
        length(xr.consistency),
        ", projected_nullity=",
        xr.projected_nullity,
        ")",
    )
end

@inline _iszero_coeff(x; tol::Float64=1e-10) = x isa AbstractFloat ? abs(x) <= tol : iszero(x)
@inline _coeff_sign(x; tol::Float64=1e-10) = _iszero_coeff(x; tol=tol) ? 0 : (Float64(x) > 0 ? 1 : -1)

function _dedup_symbolic_exprs(exprs::Vector{Any})
    seen = Set{String}()
    out = Any[]
    for expr in exprs
        key = _singular_range_key(expr)
        key in seen && continue
        push!(seen, key)
        push!(out, expr)
    end
    return out
end

function _project_singular_x_range_hrep(
    model::Bnc,
    rgm_idx::Integer,
    observe_x_idx::Integer,
)
    M, M0 = get_M_M0(model, rgm_idx)
    Cx, C0x = get_C_C0_x(model, rgm_idx)
    n = model.n

    # Build the joint system on variables [logx; logqK]:
    #   M * logx + M0 - logqK = 0
    #   Cx * logx + C0x > 0
    C_eq = hcat(sparse(M), sparse(-I, n, n))
    C_ineq = hcat(sparse(Cx), spzeros(eltype(Cx), size(Cx, 1), n))
    C_full = [C_eq; C_ineq]
    C0_full = vcat(M0, C0x)

    keep = BitSet(vcat([observe_x_idx], n .+ collect(1:n)))
    delset = BitSet(setdiff(1:(2n), collect(keep)))
    return backend_project_hrep(C_full, C0_full, n, delset)
end

function _render_qk_only_condition(
    model::Bnc,
    row::AbstractVector{<:Real},
    c0;
    equality::Bool=false,
    log_space::Bool=false,
)
    C = reshape(collect(row), 1, :)
    nlt = equality ? 1 : 0
    return only(show_condition_poly(C, [c0], nlt; syms=qK_sym(model), log_space=log_space))
end

function _render_singular_x_range_row(
    model::Bnc,
    observe_x_idx::Integer,
    row::AbstractVector{<:Real},
    c0;
    equality::Bool=false,
    log_space::Bool=false,
    tol::Float64=1e-10,
)
    a = row[1]
    if _iszero_coeff(a; tol=tol)
        return :consistency, _render_qk_only_condition(model, row[2:end], c0; equality=equality, log_space=log_space)
    end

    syms = [x_sym(model)[observe_x_idx]; qK_sym(model)]
    eq = solve_sym_expr(row, c0, syms, 1; log_space=log_space)
    if equality
        return :equality, eq
    elseif _coeff_sign(a; tol=tol) > 0
        return :lower, eq.lhs > eq.rhs
    else
        return :upper, eq.lhs < eq.rhs
    end
end

function _build_singular_x_range_projection(
    model::Bnc,
    rgm_idx::Integer,
    observe_x_idx::Integer;
    log_space::Bool=false,
    tol::Float64=1e-10,
)
    Cproj, C0proj, nlt_proj = _project_singular_x_range_hrep(model, rgm_idx, observe_x_idx)

    equalities = Any[]
    lower_bounds = Any[]
    upper_bounds = Any[]
    consistency = Any[]

    for row_idx in 1:size(Cproj, 1)
        row = vec(Array(Cproj[row_idx, :]))
        bucket, expr = _render_singular_x_range_row(
            model,
            observe_x_idx,
            row,
            C0proj[row_idx];
            equality=row_idx <= nlt_proj,
            log_space=log_space,
            tol=tol,
        )
        if bucket === :equality
            push!(equalities, expr)
        elseif bucket === :lower
            push!(lower_bounds, expr)
        elseif bucket === :upper
            push!(upper_bounds, expr)
        else
            push!(consistency, expr)
        end
    end

    return SingularXRange(
        observe_x_idx,
        x_sym(model)[observe_x_idx],
        _dedup_symbolic_exprs(equalities),
        _dedup_symbolic_exprs(lower_bounds),
        _dedup_symbolic_exprs(upper_bounds),
        _dedup_symbolic_exprs(consistency),
        Int[],
        Int[],
        Cproj,
        collect(C0proj),
        Int(nlt_proj),
    )
end

function get_singular_x_range(model::Bnc, rgm_idx; observe_x=nothing, log_space::Bool=false)
    rgm_idx = get_idx(model, rgm_idx)
    nlt = get_nullity(model, rgm_idx)
    nlt > 0 || error("get_singular_x_range only applies to singular regimes.")

    observe_x_idx, _, scalar_observe = _normalize_simo_observe_x(model, observe_x)
    ranges = [_build_singular_x_range_projection(model, rgm_idx, x_idx; log_space=log_space) for x_idx in observe_x_idx]
    return scalar_observe ? only(ranges) : ranges
end

get_singular_x_range(rgm::BindRegime; kwargs...) = get_singular_x_range(get_binding_network(rgm), get_idx(rgm); kwargs...)

function show_singular_x_range(args...; kwargs...)
    ranges = get_singular_x_range(args...; kwargs...)
    if ranges isa SingularXRange
        return vcat(ranges.consistency, ranges.equalities, ranges.lower_bounds, ranges.upper_bounds)
    else
        return [vcat(r.consistency, r.equalities, r.lower_bounds, r.upper_bounds) for r in ranges]
    end
end

function show_expression_x_range(args...; observe_x=nothing, kwargs...)
    bn = get_binding_network(args...)
    if get_nullity(args...) == 0
        observe_x_idx, _, scalar_observe = _normalize_simo_observe_x(bn, observe_x)
        exprs = show_expression_x(args...; kwargs...)
        out = exprs[observe_x_idx]
        return scalar_observe ? only(out) : out
    end
    ranges = get_singular_x_range(args...; observe_x=observe_x, kwargs...)
    if ranges isa SingularXRange
        return _format_singular_interval(ranges)
    else
        return _format_singular_interval.(ranges)
    end
end

@inline _q_totals_containing_x(model::Bnc, observe_x_idx::Integer) =
    findall(q_idx -> model.L[q_idx, observe_x_idx] > 0, 1:model.d)

# We keep direct projection as the source of truth for singular x-ranges.
# Replacing it by "regular-regime graph bounds + x_i < q_j for every q_j containing x_i"
# is not equivalent in general.
#
# Deterministic counterexample:
#   N =
#   [1 2 1 -1 0 0 0;
#    1 1 1  0 -1 0 0;
#    0 0 1  1  0 -1 0;
#    2 1 0  0  1  0 -1]
# and Lnew[1, :] = L[1, :] + L[2, :].
#
# For the nullity-1 singular regime with permutation [2, 2, 7], observe_x = x₁,
# one interior log(qK) point gives:
#   direct projection:  log10(x₁) in [-0.0822358769, 0.1326612673]
#   graph + q upper:    log10(x₁) <= 0.1326612673
#   graph + q lower:    log10(x₁) >= -0.3077875990
#
# So graph + q misses a stronger feasibility lower bound that is visible only after
# projecting the full singular fiber constraints.

function _evaluate_singular_x_range(
    xr::SingularXRange,
    qK::AbstractVector{<:Real},
    model::Bnc;
    input_logspace::Bool=false,
    tol::Float64=1e-8,
)
    C = xr.projected_C
    C0 = xr.projected_C0
    nlt = xr.projected_nullity
    qKlog = input_logspace ? Float64.(qK) : log10.(Float64.(qK))

    lower = -Inf
    upper = Inf
    fixed = nothing
    consistent = true

    for row_idx in 1:size(C, 1)
        row = vec(Array(C[row_idx, :]))
        a = Float64(row[1])
        offset = dot(Float64.(row[2:end]), qKlog) + Float64(C0[row_idx])

        if abs(a) <= tol
            if row_idx <= nlt
                consistent &= abs(offset) <= tol
            else
                consistent &= offset > -tol
            end
            continue
        end

        bound = -offset / a
        if row_idx <= nlt
            if isnothing(fixed)
                fixed = bound
            else
                consistent &= isapprox(bound, fixed; atol=tol, rtol=tol)
            end
        elseif a > 0
            lower = max(lower, bound)
        else
            upper = min(upper, bound)
        end
    end

    if !isnothing(fixed)
        lower = max(lower, fixed)
        upper = min(upper, fixed)
    end

    consistent &= lower <= upper + tol
    return (lower=lower, upper=upper, fixed=fixed, consistent=consistent)
end
