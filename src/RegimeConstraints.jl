export ParameterChart, ParameterConstraints, RestrictedRegime
export parameter_chart
export parameter_constraints, restrict_polyhedron, restrict_regime, restrict_regimes
export is_full_dimensional, stable_regime_intersections, multistability_profile
export multistability_R_index

"""
    ParameterChart

Affine reparameterization for an analysis chart.

The original chart variable `z` is represented by reduced coordinates `y` as

```text
z = F * y + F0
```

Use `chart.basis` and `chart.offset` as aliases for `F` and `F0` when working
with existing constraint internals.
"""
struct ParameterChart
    model::Bnc
    chart::Symbol
    original_symbols::Vector{Symbol}
    reduced_symbols::Vector{Symbol}
    F::Matrix{Float64}
    F0::Vector{Float64}
    basis_kind::Symbol
    notes::Vector{String}
end

function Base.getproperty(chart::ParameterChart, sym::Symbol)
    if sym === :basis
        return getfield(chart, :F)
    elseif sym === :offset
        return getfield(chart, :F0)
    elseif sym === :symbols
        return getfield(chart, :original_symbols)
    end
    return getfield(chart, sym)
end

"""
    ParameterConstraints

Analysis-time constraints in a named chart such as `:qK` or `:wKk`.

The original chart variable `z` is parameterized as `z = offset + basis * y`.
Equality constraints are absorbed into this affine parameterization. Inequality
constraints are stored in the reduced coordinate `y`.
"""
struct ParameterConstraints
    model::Bnc
    chart::Symbol
    symbols::Vector{Symbol}
    C::Matrix{Float64}
    C0::Vector{Float64}
    nullity::Int
    equality_C::Matrix{Float64}
    equality_C0::Vector{Float64}
    inequality_C::Matrix{Float64}
    inequality_C0::Vector{Float64}
    basis::Matrix{Float64}
    offset::Vector{Float64}
    reduced_symbols::Vector{Symbol}
    reduced_inequality_C::Matrix{Float64}
    reduced_inequality_C0::Vector{Float64}
    compatible::Bool
    residual::Float64
    notes::Vector{String}
    parameter_chart::ParameterChart
    basis_kind::Symbol
end

"""
    RestrictedRegime

Diagnostic result for a regime restricted by `ParameterConstraints`.
"""
struct RestrictedRegime
    regime::Any
    constraints::ParameterConstraints
    chart::Symbol
    C::Matrix{Float64}
    C0::Vector{Float64}
    nullity::Int
    poly::Union{Polyhedron, Nothing}
    feasible::Bool
    dim::Int
    ambient_dim::Int
    full_dim::Bool
    reason::Symbol
end

function _has_catalysis(model::Bnc)
    return !isnothing(model.catalysis)
end

function _resolve_constraint_chart(obj, chart::Symbol)
    chart !== :auto && return chart
    obj isa BncRegime && return :wKk
    obj isa CatalysisRegime && return :xk
    obj isa BindRegime && return :qK
    model = get_binding_network(obj)
    return _has_catalysis(model) ? :wKk : :qK
end

function _chart_symbols(obj, chart::Symbol)
    if chart === :qK
        return qK_symbol(obj)
    elseif chart === :qKk
        return qKk_symbol(obj)
    elseif chart === :wKk
        return wKk_symbol(obj)
    elseif chart === :xk
        return xk_symbol(obj)
    end
    throw(ArgumentError("Unsupported constraint chart $(repr(chart))."))
end

_constraint_symbol(x::Symbol) = x
_constraint_symbol(x::Num) = Symbol(x.val.name)
_constraint_symbol(x) = Symbol(x)

function _constraint_index(symbols::AbstractVector{Symbol}, item)
    if item isa Integer
        idx = Int(item)
    else
        target = _constraint_symbol(item)
        idx = findfirst(==(target), symbols)
        isnothing(idx) && throw(
            ArgumentError(
                "Unknown constraint symbol $(repr(target)). Available symbols are $(repr(symbols)).",
            ),
        )
    end
    1 <= idx <= length(symbols) ||
        throw(ArgumentError("Constraint index $idx is out of range 1:$(length(symbols))."))
    return idx
end

function _empty_constraint_matrix(n::Int)
    return zeros(Float64, 0, n), Float64[]
end

function _constraint_items(items)
    isnothing(items) && return Any[]
    items isa AbstractVector && return collect(items)
    items isa Tuple && return collect(items)
    return [items]
end

function _row_difference(symbols::AbstractVector{Symbol}, left, right; sign::Float64=1.0)
    row = zeros(Float64, length(symbols))
    row[_constraint_index(symbols, left)] += sign
    row[_constraint_index(symbols, right)] -= sign
    return row
end

function _constraint_equalities(symbols::AbstractVector{Symbol}, equalities)
    rows = Vector{Vector{Float64}}()
    biases = Float64[]
    for item in _constraint_items(equalities)
        if item isa Pair
            push!(rows, _row_difference(symbols, item.first, item.second))
            push!(biases, 0.0)
        elseif item isa Tuple && length(item) == 3
            op_symbol = item[2] isa Symbol ? item[2] : Symbol(item[2])
            op_symbol in (Symbol("=="), Symbol("=")) ||
                throw(ArgumentError("Equality tuple operator must be :==."))
            push!(rows, _row_difference(symbols, item[1], item[3]))
            push!(biases, 0.0)
        else
            throw(
                ArgumentError(
                    "Equality constraints must be `left => right` or `(left, :==, right)`."
                ),
            )
        end
    end
    isempty(rows) && return _empty_constraint_matrix(length(symbols))
    return reduce(vcat, transpose.(rows)), biases
end

function _constraint_inequalities(symbols::AbstractVector{Symbol}, inequalities)
    rows = Vector{Vector{Float64}}()
    biases = Float64[]
    notes = String[]
    for item in _constraint_items(inequalities)
        item isa Tuple && length(item) == 3 ||
            throw(ArgumentError("Inequality constraints must be `(left, op, right)`."))

        left, op, right = item
        op_symbol = op isa Symbol ? op : Symbol(op)
        row = if op_symbol in (:<, :<=)
            _row_difference(symbols, right, left)
        elseif op_symbol in (:>, :>=)
            _row_difference(symbols, left, right)
        else
            throw(
                ArgumentError(
                    "Unsupported inequality operator $(repr(op)). Use :<, :<=, :>, or :>=.",
                ),
            )
        end
        push!(rows, row)
        push!(biases, 0.0)
        op_symbol in (:<, :>) && push!(
            notes,
            "Strict inequality $(repr(item)) is represented as a closed halfspace for volume calculations.",
        )
    end
    isempty(rows) && return (_empty_constraint_matrix(length(symbols))..., notes)
    return reduce(vcat, transpose.(rows)), biases, notes
end

function _constraint_map_pairs(map)
    isnothing(map) && return Pair[]
    if map isa AbstractDict
        return collect(pairs(map))
    elseif map isa AbstractVector || map isa Tuple
        return collect(map)
    end
    return [map]
end

function _constraint_groups_pairs(groups)
    isnothing(groups) && return Pair[]
    groups isa AbstractDict && return collect(pairs(groups))
    groups isa AbstractVector && return collect(groups)
    groups isa Tuple && return collect(groups)
    return [groups]
end

function _identified_parameter_map(symbols::AbstractVector{Symbol}, map, groups)
    mapping = Dict(sym => sym for sym in symbols)

    for item in _constraint_groups_pairs(groups)
        item isa Pair || throw(
            ArgumentError("Constraint groups must be `new_symbol => old_symbols` pairs."),
        )
        reduced = _constraint_symbol(item.first)
        for old in _constraint_items(item.second)
            old_symbol = _constraint_symbol(old)
            old_symbol in symbols || throw(
                ArgumentError(
                    "Unknown grouped constraint symbol $(repr(old_symbol)). Available symbols are $(repr(symbols)).",
                ),
            )
            mapping[old_symbol] = reduced
        end
    end

    for item in _constraint_map_pairs(map)
        item isa Pair || throw(
            ArgumentError("Constraint map must use `old_symbol => new_symbol` pairs.")
        )
        old_symbol = _constraint_symbol(item.first)
        old_symbol in symbols || throw(
            ArgumentError(
                "Unknown mapped constraint symbol $(repr(old_symbol)). Available symbols are $(repr(symbols)).",
            ),
        )
        mapping[old_symbol] = _constraint_symbol(item.second)
    end

    return mapping
end

function _unique_mapped_symbols(symbols::AbstractVector{Symbol}, mapping)
    reduced = Symbol[]
    for sym in symbols
        target = mapping[sym]
        target in reduced || push!(reduced, target)
    end
    return reduced
end

function _has_parameter_identification(map, groups)
    return !isempty(_constraint_map_pairs(map)) ||
           !isempty(_constraint_groups_pairs(groups))
end

function _parameter_chart_from_map(
    model::Bnc, chart::Symbol, symbols::AbstractVector{Symbol}, map, groups, reduced_symbols
)
    mapping = _identified_parameter_map(symbols, map, groups)
    resolved_reduced_symbols = if isnothing(reduced_symbols)
        _unique_mapped_symbols(symbols, mapping)
    else
        _constraint_symbol.(collect(reduced_symbols))
    end

    F = zeros(Float64, length(symbols), length(resolved_reduced_symbols))
    for (i, sym) in pairs(symbols)
        target = mapping[sym]
        j = findfirst(==(target), resolved_reduced_symbols)
        isnothing(j) && throw(
            ArgumentError(
                "Mapped symbol $(repr(target)) is not present in reduced_symbols $(repr(resolved_reduced_symbols)).",
            ),
        )
        F[i, j] = 1.0
    end

    return ParameterChart(
        model,
        chart,
        collect(symbols),
        resolved_reduced_symbols,
        F,
        zeros(Float64, length(symbols)),
        :identified_parameters,
        String[],
    )
end

function _parameter_chart_from_matrix(
    model::Bnc,
    chart::Symbol,
    symbols::AbstractVector{Symbol};
    F,
    F0=nothing,
    reduced_symbols=nothing,
    basis::Symbol=:provided,
)
    F_mat = Matrix{Float64}(F)
    size(F_mat, 1) == length(symbols) || throw(
        ArgumentError(
            "F must have $(length(symbols)) rows for chart $(repr(chart)), got $(size(F_mat, 1)).",
        ),
    )

    F0_vec = if isnothing(F0)
        zeros(Float64, length(symbols))
    else
        Float64.(vec(F0))
    end
    length(F0_vec) == length(symbols) ||
        throw(ArgumentError("F0 length must match the number of original chart symbols."))

    resolved_reduced_symbols = if isnothing(reduced_symbols)
        [Symbol(:theta_, i) for i in 1:size(F_mat, 2)]
    else
        _constraint_symbol.(collect(reduced_symbols))
    end
    length(resolved_reduced_symbols) == size(F_mat, 2) ||
        throw(ArgumentError("reduced_symbols length must match the number of F columns."))

    return ParameterChart(
        model,
        chart,
        collect(symbols),
        resolved_reduced_symbols,
        F_mat,
        F0_vec,
        basis,
        String[],
    )
end

"""
    parameter_chart(model; chart=:auto, map=nothing, groups=nothing)
    parameter_chart(model; chart=:auto, F, F0=zeros(...), reduced_symbols)

Build an affine parameter chart `old = F * new + F0`.

`map` uses `old_symbol => new_symbol` pairs. `groups` uses
`new_symbol => old_symbols` pairs. Unmapped original symbols are kept as
independent reduced symbols.
"""
function parameter_chart(
    obj;
    chart::Symbol=:auto,
    map=nothing,
    groups=nothing,
    F=nothing,
    F0=nothing,
    reduced_symbols=nothing,
    basis::Symbol=:identified_parameters,
)
    model = get_binding_network(obj)
    resolved_chart = _resolve_constraint_chart(obj, chart)
    symbols = _chart_symbols(obj, resolved_chart)

    if !isnothing(F)
        return _parameter_chart_from_matrix(
            model,
            resolved_chart,
            symbols;
            F=F,
            F0=F0,
            reduced_symbols=reduced_symbols,
            basis=basis === :identified_parameters ? :provided : basis,
        )
    end

    if !_has_parameter_identification(map, groups)
        eye = Matrix{Float64}(I, length(symbols), length(symbols))
        return ParameterChart(
            model,
            resolved_chart,
            collect(symbols),
            collect(symbols),
            eye,
            zeros(Float64, length(symbols)),
            :identity,
            String[],
        )
    end

    return _parameter_chart_from_map(
        model, resolved_chart, symbols, map, groups, reduced_symbols
    )
end

function _matrix_constraints(C, C0, nullity::Integer, n::Int)
    if isnothing(C)
        isnothing(C0) || throw(ArgumentError("C0 was provided without C."))
        return zeros(Float64, 0, n), Float64[], 0
    end

    C_mat = Matrix{Float64}(C)
    size(C_mat, 2) == n || throw(
        ArgumentError(
            "C must have $n columns for the selected chart, got $(size(C_mat, 2))."
        ),
    )

    C0_vec = if isnothing(C0)
        zeros(Float64, size(C_mat, 1))
    else
        Float64.(vec(C0))
    end
    length(C0_vec) == size(C_mat, 1) ||
        throw(ArgumentError("C0 length must match the number of rows in C."))
    0 <= nullity <= size(C_mat, 1) ||
        throw(ArgumentError("nullity must be between 0 and the number of rows in C."))
    return C_mat, C0_vec, Int(nullity)
end

function _affine_subspace(
    E::AbstractMatrix{<:Real}, e0::AbstractVector{<:Real}; atol::Real=1e-10
)
    n = size(E, 2)
    if size(E, 1) == 0
        return zeros(Float64, n), Matrix{Float64}(I, n, n), true, 0.0
    end

    E64 = Matrix{Float64}(E)
    e064 = Float64.(vec(e0))
    F = svd(E64; full=true)
    scale = isempty(F.S) ? 0.0 : maximum(F.S)
    rank_tol = max(Float64(atol), eps(Float64) * max(size(E64)...) * scale)
    rnk = count(>(rank_tol), F.S)

    offset = pinv(E64) * (-e064)
    residual = norm(E64 * offset + e064)
    compatible = residual <= max(Float64(atol), 100 * eps(Float64) * max(1.0, norm(e064)))
    basis = Matrix(F.Vt')[1:n, (rnk + 1):n]
    return offset, basis, compatible, residual
end

function _reduce_constraint_rows(C, C0, chart::ParameterChart, symbols::Symbol)
    if symbols === :reduced
        return C, C0
    elseif symbols === :original
        return C * chart.F, C * chart.F0 + C0
    end
    throw(ArgumentError("Constraint symbols must be `:reduced` or `:original`."))
end

function _composed_basis_kind(parent::Symbol, equality_rows::Int, basis::Symbol)
    equality_rows == 0 && return parent
    parent === :identity && return basis
    return :composed
end

function _parameter_constraints_from_chart(
    chart_obj::ParameterChart;
    C=nothing,
    C0=nothing,
    nullity::Integer=0,
    equalities=nothing,
    inequalities=nothing,
    symbols::Symbol=:reduced,
    basis::Symbol=:orthonormal,
    atol::Real=1e-10,
)
    source_symbols = if symbols === :reduced
        chart_obj.reduced_symbols
    elseif symbols === :original
        chart_obj.original_symbols
    else
        throw(ArgumentError("Constraint symbols must be `:reduced` or `:original`."))
    end

    eq_C, eq_C0 = _constraint_equalities(source_symbols, equalities)
    ineq_C, ineq_C0, notes = _constraint_inequalities(source_symbols, inequalities)
    mat_C, mat_C0, mat_nullity = _matrix_constraints(C, C0, nullity, length(source_symbols))

    mat_eq_C = mat_C[1:mat_nullity, :]
    mat_eq_C0 = mat_C0[1:mat_nullity]
    mat_ineq_C = mat_C[(mat_nullity + 1):end, :]
    mat_ineq_C0 = mat_C0[(mat_nullity + 1):end]

    equality_C_source = vcat(eq_C, mat_eq_C)
    equality_C0_source = vcat(eq_C0, mat_eq_C0)
    inequality_C_source = vcat(ineq_C, mat_ineq_C)
    inequality_C0_source = vcat(ineq_C0, mat_ineq_C0)

    equality_C_y, equality_C0_y = _reduce_constraint_rows(
        equality_C_source, equality_C0_source, chart_obj, symbols
    )
    inequality_C_y, inequality_C0_y = _reduce_constraint_rows(
        inequality_C_source, inequality_C0_source, chart_obj, symbols
    )

    y_offset, y_basis, compatible, residual = _affine_subspace(
        equality_C_y, equality_C0_y; atol=atol
    )

    final_basis = chart_obj.F * y_basis
    final_offset = chart_obj.F * y_offset + chart_obj.F0
    reduced_inequality_C = inequality_C_y * y_basis
    reduced_inequality_C0 = inequality_C_y * y_offset + inequality_C0_y
    final_reduced_symbols = if size(equality_C_y, 1) == 0
        chart_obj.reduced_symbols
    else
        [Symbol(:theta_, i) for i in 1:size(final_basis, 2)]
    end
    basis_kind = _composed_basis_kind(chart_obj.basis_kind, size(equality_C_y, 1), basis)
    final_chart = ParameterChart(
        chart_obj.model,
        chart_obj.chart,
        chart_obj.original_symbols,
        final_reduced_symbols,
        final_basis,
        final_offset,
        basis_kind,
        vcat(chart_obj.notes, notes),
    )

    C_full = vcat(equality_C_source, inequality_C_source)
    C0_full = vcat(equality_C0_source, inequality_C0_source)

    return ParameterConstraints(
        chart_obj.model,
        chart_obj.chart,
        chart_obj.original_symbols,
        C_full,
        C0_full,
        size(equality_C_y, 1),
        equality_C_source,
        equality_C0_source,
        inequality_C_source,
        inequality_C0_source,
        final_basis,
        final_offset,
        final_reduced_symbols,
        reduced_inequality_C,
        reduced_inequality_C0,
        compatible,
        residual,
        vcat(chart_obj.notes, notes),
        final_chart,
        basis_kind,
    )
end

"""
    parameter_constraints(chart::ParameterChart; inequalities=nothing, symbols=:reduced)
    parameter_constraints(model; chart=:auto, C=nothing, C0=nothing, nullity=0,
                          equalities=nothing, inequalities=nothing)

Build an analysis-time constraint object. Matrix constraints use the package
condition convention: the first `nullity` rows are equalities and later rows are
inequalities.

When the first argument is a `ParameterChart`, constraints are interpreted in
the reduced chart by default. Use `symbols=:original` to write them in the
original chart and pull them back through the affine map.
"""
function parameter_constraints(
    chart_obj::ParameterChart;
    C=nothing,
    C0=nothing,
    nullity::Integer=0,
    equalities=nothing,
    inequalities=nothing,
    symbols::Symbol=:reduced,
    basis::Symbol=:orthonormal,
    atol::Real=1e-10,
)
    return _parameter_constraints_from_chart(
        chart_obj;
        C=C,
        C0=C0,
        nullity=nullity,
        equalities=equalities,
        inequalities=inequalities,
        symbols=symbols,
        basis=basis,
        atol=atol,
    )
end

function parameter_constraints(
    obj;
    chart::Symbol=:auto,
    C=nothing,
    C0=nothing,
    nullity::Integer=0,
    equalities=nothing,
    inequalities=nothing,
    map=nothing,
    groups=nothing,
    F=nothing,
    F0=nothing,
    reduced_symbols=nothing,
    constraint_symbols::Symbol=:original,
    basis::Symbol=:orthonormal,
    atol::Real=1e-10,
)
    if !isnothing(map) || !isnothing(groups) || !isnothing(F)
        chart_obj = parameter_chart(
            obj;
            chart=chart,
            map=map,
            groups=groups,
            F=F,
            F0=F0,
            reduced_symbols=reduced_symbols,
            basis=isnothing(F) ? :identified_parameters : :provided,
        )
        return parameter_constraints(
            chart_obj;
            C=C,
            C0=C0,
            nullity=nullity,
            equalities=equalities,
            inequalities=inequalities,
            symbols=constraint_symbols,
            basis=basis,
            atol=atol,
        )
    end

    model = get_binding_network(obj)
    resolved_chart = _resolve_constraint_chart(obj, chart)
    symbols = _chart_symbols(obj, resolved_chart)
    n = length(symbols)

    eq_C, eq_C0 = _constraint_equalities(symbols, equalities)
    ineq_C, ineq_C0, notes = _constraint_inequalities(symbols, inequalities)
    mat_C, mat_C0, mat_nullity = _matrix_constraints(C, C0, nullity, n)

    mat_eq_C = mat_C[1:mat_nullity, :]
    mat_eq_C0 = mat_C0[1:mat_nullity]
    mat_ineq_C = mat_C[(mat_nullity + 1):end, :]
    mat_ineq_C0 = mat_C0[(mat_nullity + 1):end]

    equality_C = vcat(eq_C, mat_eq_C)
    equality_C0 = vcat(eq_C0, mat_eq_C0)
    inequality_C = vcat(ineq_C, mat_ineq_C)
    inequality_C0 = vcat(ineq_C0, mat_ineq_C0)

    offset, basis_matrix, compatible, residual = _affine_subspace(
        equality_C, equality_C0; atol=atol
    )

    reduced_inequality_C = inequality_C * basis_matrix
    reduced_inequality_C0 = inequality_C * offset + inequality_C0
    reduced_symbols = [Symbol(:theta_, i) for i in 1:size(basis_matrix, 2)]
    basis_kind = size(equality_C, 1) == 0 ? :identity : basis
    chart_obj = ParameterChart(
        model,
        resolved_chart,
        collect(symbols),
        reduced_symbols,
        basis_matrix,
        offset,
        basis_kind,
        notes,
    )

    C_full = vcat(equality_C, inequality_C)
    C0_full = vcat(equality_C0, inequality_C0)

    return ParameterConstraints(
        model,
        resolved_chart,
        symbols,
        C_full,
        C0_full,
        size(equality_C, 1),
        equality_C,
        equality_C0,
        inequality_C,
        inequality_C0,
        basis_matrix,
        offset,
        reduced_symbols,
        reduced_inequality_C,
        reduced_inequality_C0,
        compatible,
        residual,
        notes,
        chart_obj,
        basis_kind,
    )
end

function _regime_C_C0_nullity(rgm::BindRegime, chart::Symbol)
    resolved = chart === :auto ? :qK : chart
    resolved === :qK || throw(
        ArgumentError("Binding regimes currently support `chart=:qK`, got $(repr(chart))."),
    )
    return get_C_C0_nullity_qK(rgm)
end

function _regime_C_C0_nullity(rgm::BncRegime, chart::Symbol)
    resolved = chart === :auto ? :wKk : chart
    if resolved === :wKk
        return get_C_C0_nullity_wKk(rgm)
    elseif resolved === :qKk
        return get_C_C0_nullity_qKk(rgm)
    end
    throw(
        ArgumentError(
            "BNC regimes currently support `chart=:wKk` or `chart=:qKk`, got $(repr(chart)).",
        ),
    )
end

function get_polyhedron(rgm::BindRegime; chart::Symbol=:qK, canonicalize::Bool=true)
    C, C0, nlt = _regime_C_C0_nullity(rgm, chart)
    return get_polyhedron(C, C0, nlt; canonicalize=canonicalize)
end

function _pullback_constraints(C, C0, nullity::Integer, constraints::ParameterConstraints)
    C_mat = Matrix{Float64}(C)
    C0_vec = Float64.(vec(C0))
    nlt = Int(nullity)

    eq_C = C_mat[1:nlt, :] * constraints.basis
    eq_C0 = C_mat[1:nlt, :] * constraints.offset + C0_vec[1:nlt]
    ineq_C = C_mat[(nlt + 1):end, :] * constraints.basis
    ineq_C0 = C_mat[(nlt + 1):end, :] * constraints.offset + C0_vec[(nlt + 1):end]

    C_reduced = vcat(eq_C, ineq_C, constraints.reduced_inequality_C)
    C0_reduced = vcat(eq_C0, ineq_C0, constraints.reduced_inequality_C0)
    return C_reduced, C0_reduced, nlt
end

function _restriction_result(
    regime, constraints, chart, C, C0, nullity, poly, reason::Symbol
)
    ambient_dim = size(constraints.basis, 2)
    feasible = !isnothing(poly) && !isempty(poly)
    dim_val = feasible ? dim(poly) : -1
    full_dim = feasible && dim_val == ambient_dim
    return RestrictedRegime(
        regime,
        constraints,
        chart,
        Matrix{Float64}(C),
        Float64.(vec(C0)),
        Int(nullity),
        poly,
        feasible,
        dim_val,
        ambient_dim,
        full_dim,
        reason,
    )
end

"""
    restrict_polyhedron(poly, constraints; canonicalize=true)

Pull a polyhedron back to the reduced coordinate chart defined by
`ParameterConstraints`.
"""
function restrict_polyhedron(
    poly::Polyhedron, constraints::ParameterConstraints; canonicalize::Bool=true
)
    if !constraints.compatible
        C_empty, C0_empty = _empty_constraint_matrix(size(constraints.basis, 2))
        return _restriction_result(
            poly,
            constraints,
            constraints.chart,
            C_empty,
            C0_empty,
            0,
            nothing,
            :incompatible_constraints,
        )
    end

    C, C0, nlt = get_C_C0_nullity(poly)
    C_reduced, C0_reduced, nlt_reduced = _pullback_constraints(C, C0, nlt, constraints)
    reduced_poly = get_polyhedron(
        C_reduced, C0_reduced, nlt_reduced; canonicalize=canonicalize
    )
    reason = isempty(reduced_poly) ? :empty : :ok
    return _restriction_result(
        poly,
        constraints,
        constraints.chart,
        C_reduced,
        C0_reduced,
        nlt_reduced,
        reduced_poly,
        reason,
    )
end

"""
    restrict_regime(rgm, constraints; chart=:auto, canonicalize=true)

Restrict a binding or BNC regime by analysis-time parameter constraints.
"""
function restrict_regime(
    rgm::Union{BindRegime, BncRegime},
    constraints::ParameterConstraints;
    chart::Symbol=:auto,
    canonicalize::Bool=true,
)
    resolved_chart = _resolve_constraint_chart(rgm, chart)
    resolved_chart === constraints.chart || throw(
        ArgumentError(
            "Regime chart $(repr(resolved_chart)) does not match constraints chart $(repr(constraints.chart)).",
        ),
    )

    if !constraints.compatible
        C_empty, C0_empty = _empty_constraint_matrix(size(constraints.basis, 2))
        return _restriction_result(
            rgm,
            constraints,
            resolved_chart,
            C_empty,
            C0_empty,
            0,
            nothing,
            :incompatible_constraints,
        )
    end

    C, C0, nlt = _regime_C_C0_nullity(rgm, resolved_chart)
    C_reduced, C0_reduced, nlt_reduced = _pullback_constraints(C, C0, nlt, constraints)
    poly = get_polyhedron(C_reduced, C0_reduced, nlt_reduced; canonicalize=canonicalize)
    reason = isempty(poly) ? :empty : :ok
    return _restriction_result(
        rgm, constraints, resolved_chart, C_reduced, C0_reduced, nlt_reduced, poly, reason
    )
end

function _matches_filter(value, filter)
    return isnothing(filter) || value == filter
end

function _passes_regime_filters(rgm; stable=nothing, singular=nothing, feasible=nothing)
    _matches_filter(is_singular(rgm), singular) || return false
    _matches_filter(is_feasible(rgm), feasible) || return false
    if !isnothing(stable)
        is_stable(rgm) === stable || return false
    end
    return true
end

"""
    restrict_regimes(rgms, constraints; stable=nothing, singular=nothing,
                     feasible=true, full_dim=true)

Return restricted regime diagnostics after optional regime-level filters.
"""
function restrict_regimes(
    rgms::AbstractVector{<:Union{BindRegime, BncRegime}},
    constraints::ParameterConstraints;
    stable=nothing,
    singular=nothing,
    feasible=true,
    full_dim=true,
    kwargs...,
)
    out = RestrictedRegime[]
    for rgm in rgms
        _passes_regime_filters(rgm; stable=stable, singular=singular, feasible=feasible) ||
            continue
        restricted = restrict_regime(rgm, constraints; kwargs...)
        _matches_filter(restricted.full_dim, full_dim) || continue
        push!(out, restricted)
    end
    return out
end

function is_full_dimensional(poly::Polyhedron; ambient_dim=nothing, canonicalize::Bool=true)
    p = canonicalize ? polyhedron(hrep(poly)) : poly
    canonicalize && detecthlinearity!(p)
    resolved_dim = isnothing(ambient_dim) ? fulldim(p) : ambient_dim
    return !isempty(p) && dim(p) == resolved_dim
end

function _intersect_restricted(
    a::RestrictedRegime, b::RestrictedRegime; canonicalize::Bool=true
)
    a.constraints === b.constraints || throw(
        ArgumentError("Restricted regimes must use the same ParameterConstraints object."),
    )
    if isnothing(a.poly) || isnothing(b.poly)
        return nothing, -1, false
    end
    poly = intersect(a.poly, b.poly)
    if canonicalize
        detecthlinearity!(poly)
        removehredundancy!(poly)
    end
    feasible = !isempty(poly)
    dim_val = feasible ? dim(poly) : -1
    return poly, dim_val, feasible && dim_val == a.ambient_dim
end

"""
    stable_regime_intersections(rgms; constraints, full_dim=true)

Compute pair intersections among stable restricted BNC regimes.
"""
function stable_regime_intersections(
    rgms::AbstractVector{<:BncRegime};
    constraints::ParameterConstraints,
    full_dim=true,
    singular=false,
    feasible=true,
    canonicalize::Bool=true,
)
    restricted = restrict_regimes(
        rgms,
        constraints;
        stable=true,
        singular=singular,
        feasible=feasible,
        full_dim=true,
        canonicalize=canonicalize,
    )
    return stable_regime_intersections(
        restricted; full_dim=full_dim, canonicalize=canonicalize
    )
end

function stable_regime_intersections(
    restricted::AbstractVector{<:RestrictedRegime}; full_dim=true, canonicalize::Bool=true
)
    rows = NamedTuple[]
    n = length(restricted)
    for i in 1:(n - 1)
        for j in (i + 1):n
            poly, dim_val, is_fulldim = _intersect_restricted(
                restricted[i], restricted[j]; canonicalize=canonicalize
            )
            _matches_filter(is_fulldim, full_dim) || continue
            push!(
                rows,
                (;
                    regime_i=get_idx(restricted[i].regime),
                    regime_j=get_idx(restricted[j].regime),
                    binding_i=get_idx(get_binding_regime(restricted[i].regime)),
                    binding_j=get_idx(get_binding_regime(restricted[j].regime)),
                    catalysis_i=get_idx(get_catalysis_regime(restricted[i].regime)),
                    catalysis_j=get_idx(get_catalysis_regime(restricted[j].regime)),
                    poly=poly,
                    dim=dim_val,
                    ambient_dim=restricted[i].ambient_dim,
                    full_dim=is_fulldim,
                ),
            )
        end
    end
    return rows
end

function _constraints_satisfied(y, C, C0; tol::Real=0.0, asymptotic::Bool=false)
    isempty(C0) && return true
    vals = if asymptotic
        C * y
    else
        C * y + C0
    end
    return all(vals .>= -abs(Float64(tol)))
end

function _restricted_hit(
    y, restricted::RestrictedRegime; tol::Real=0.0, asymptotic::Bool=false
)
    restricted.feasible || return false
    nlt = restricted.nullity
    if nlt > 0
        eq_vals = if asymptotic
            restricted.C[1:nlt, :] * y
        else
            restricted.C[1:nlt, :] * y + restricted.C0[1:nlt]
        end
        all(abs.(eq_vals) .<= abs(Float64(tol))) || return false
    end
    C_ineq = restricted.C[(nlt + 1):end, :]
    C0_ineq = restricted.C0[(nlt + 1):end]
    return _constraints_satisfied(y, C_ineq, C0_ineq; tol=tol, asymptotic=asymptotic)
end

function _draw_reduced_sample!(y, rng, sampler::Symbol, log_lower, log_upper)
    if sampler === :gaussian
        Random.randn!(rng, y)
    elseif sampler === :uniform_box
        if log_lower isa AbstractVector
            lower = Float64.(log_lower)
            upper = Float64.(log_upper)
            length(lower) == length(y) && length(upper) == length(y) || throw(
                ArgumentError("log_lower/log_upper vectors must match reduced dimension."),
            )
            for i in eachindex(y)
                y[i] = lower[i] + Random.rand(rng) * (upper[i] - lower[i])
            end
        else
            lo = Float64(log_lower)
            hi = Float64(log_upper)
            for i in eachindex(y)
                y[i] = lo + Random.rand(rng) * (hi - lo)
            end
        end
    else
        throw(ArgumentError("sampler must be `:gaussian` or `:uniform_box`."))
    end
    return y
end

function _combination_table(counts::Dict{Tuple, Int}, total::Int)
    rows = [
        (; regimes=collect(k), count=v, fraction=total == 0 ? 0.0 : v / total) for
        (k, v) in counts
    ]
    sort!(rows; by=x -> (-x.count, x.regimes))
    return rows
end

function _multistability_mode(mode::Symbol)
    mode in (:finite_region, :asymptotic_R) ||
        throw(ArgumentError("mode must be one of `:finite_region` or `:asymptotic_R`."))
    return mode
end

"""
    multistability_profile(model; constraints=parameter_constraints(model), samples=100_000, mode=:finite_region)

Estimate constrained R-index summaries by sampling points that satisfy the
analysis constraints and counting how many stable restricted BNC regimes contain
each sampled point.

`mode=:finite_region` uses the full restricted inequalities, including offsets.
`mode=:asymptotic_R` strips offsets and samples the recession-cone membership
used by asymptotic solid-angle R-index calculations.
"""
function multistability_profile(
    model::Bnc;
    constraints::ParameterConstraints=parameter_constraints(model; chart=:auto),
    regimes=get_bnc_regimes(model; feasible=true),
    samples::Integer=100_000,
    max_draws::Integer=max(10 * samples, samples + 1),
    sampler::Symbol=:gaussian,
    log_lower=-6.0,
    log_upper=6.0,
    rng_seed::Integer=0x12345678,
    stable::Bool=true,
    singular::Bool=false,
    feasible::Bool=true,
    full_dim::Bool=true,
    hit_tol::Real=0.0,
    constraint_tol::Real=0.0,
    pair_intersections::Bool=true,
    mode::Symbol=:finite_region,
)
    resolved_mode = _multistability_mode(mode)
    asymptotic = resolved_mode === :asymptotic_R
    restricted = restrict_regimes(
        regimes,
        constraints;
        stable=stable,
        singular=singular,
        feasible=feasible,
        full_dim=full_dim,
    )

    rng = Random.MersenneTwister(rng_seed)
    y = zeros(Float64, size(constraints.basis, 2))
    accepted = 0
    draws = 0
    hit_hist = Dict{Int, Int}()
    combination_counts = Dict{Tuple, Int}()
    at_least_counts = Dict{Int, Int}()
    max_hit_count = 0

    while accepted < samples && draws < max_draws
        draws += 1
        _draw_reduced_sample!(y, rng, sampler, log_lower, log_upper)
        _constraints_satisfied(
            y,
            constraints.reduced_inequality_C,
            constraints.reduced_inequality_C0;
            tol=constraint_tol,
            asymptotic=asymptotic,
        ) || continue

        accepted += 1
        hits = Int[]
        for rr in restricted
            _restricted_hit(y, rr; tol=hit_tol, asymptotic=asymptotic) &&
                push!(hits, get_idx(rr.regime))
        end
        nhit = length(hits)
        max_hit_count = max(max_hit_count, nhit)
        hit_hist[nhit] = get(hit_hist, nhit, 0) + 1

        if nhit > 0
            key = Tuple(sort!(hits))
            combination_counts[key] = get(combination_counts, key, 0) + 1
        end
        for k in 1:nhit
            at_least_counts[k] = get(at_least_counts, k, 0) + 1
        end
    end

    R_atleast = Dict(k => v / max(accepted, 1) for (k, v) in at_least_counts)
    pairs = if pair_intersections
        stable_regime_intersections(restricted; full_dim=true)
    else
        NamedTuple[]
    end

    return (;
        constraints=constraints,
        mode=resolved_mode,
        denominator=asymptotic ? :constraint_cone : :constraint_region,
        basis_kind=constraints.basis_kind,
        reduced_symbols=constraints.reduced_symbols,
        requested_samples=Int(samples),
        accepted_samples=accepted,
        draws=draws,
        stable_regimes=[get_idx(rr.regime) for rr in restricted],
        restricted_regimes=restricted,
        pair_table=pairs,
        hit_histogram=Dict(k => v for (k, v) in sort(collect(hit_hist))),
        combination_counts=_combination_table(combination_counts, accepted),
        max_hit_count=max_hit_count,
        R_atleast=R_atleast,
        R_atleast_1=get(R_atleast, 1, 0.0),
        R_atleast_2=get(R_atleast, 2, 0.0),
        R_atleast_3=get(R_atleast, 3, 0.0),
    )
end

"""
    multistability_R_index(model; constraints=parameter_constraints(model), mode=:asymptotic_R, samples=100_000)

Report-oriented constrained multistability summary. This wraps
`multistability_profile` and returns deterministic regime counts together with
the conditional `R_multistability = R_atleast_2`.

`full_dim_regimes` counts all feasible full-dimensional restricted BNC regimes.
`stable_full_dim_regimes`, `pair_intersections`, and `R_multistability` use the
candidate filter controlled by `singular`, which defaults to nonsingular
regimes.
"""
function multistability_R_index(
    model::Bnc;
    constraints::ParameterConstraints=parameter_constraints(model; chart=:auto),
    regimes=get_bnc_regimes(model; feasible=true),
    samples::Integer=100_000,
    max_draws::Integer=max(10 * samples, samples + 1),
    sampler::Symbol=:gaussian,
    log_lower=-6.0,
    log_upper=6.0,
    rng_seed::Integer=0x12345678,
    singular::Bool=false,
    feasible::Bool=true,
    full_dim::Bool=true,
    hit_tol::Real=0.0,
    constraint_tol::Real=0.0,
    mode::Symbol=:asymptotic_R,
)
    full_dim_restricted = restrict_regimes(
        regimes,
        constraints;
        stable=nothing,
        singular=nothing,
        feasible=feasible,
        full_dim=full_dim,
    )
    profile = multistability_profile(
        model;
        constraints=constraints,
        regimes=regimes,
        samples=samples,
        max_draws=max_draws,
        sampler=sampler,
        log_lower=log_lower,
        log_upper=log_upper,
        rng_seed=rng_seed,
        stable=true,
        singular=singular,
        feasible=feasible,
        full_dim=full_dim,
        hit_tol=hit_tol,
        constraint_tol=constraint_tol,
        pair_intersections=true,
        mode=mode,
    )
    stable_full_dim_restricted = profile.restricted_regimes

    p = profile.R_atleast_2
    stderr =
        profile.accepted_samples == 0 ? NaN : sqrt(p * (1 - p) / profile.accepted_samples)

    return (;
        constraints=constraints,
        profile=profile,
        mode=profile.mode,
        denominator=profile.denominator,
        basis_kind=constraints.basis_kind,
        reduced_symbols=constraints.reduced_symbols,
        total_bnc_regimes=length(regimes),
        full_dim_regimes=length(full_dim_restricted),
        stable_full_dim_regimes=length(stable_full_dim_restricted),
        pair_intersections=length(profile.pair_table),
        R_multistability=profile.R_atleast_2,
        stderr=stderr,
        samples=profile.accepted_samples,
        requested_samples=profile.requested_samples,
        draws=profile.draws,
        pair_table=profile.pair_table,
        combination_counts=profile.combination_counts,
        notes=constraints.notes,
    )
end
