export BncLinearControlModel
export hbd_source, get_H_bd_info
export linear_control_model, control_metrics
export controllability_matrix, output_controllability_matrix, output_controllability_row
export markov_coefficients, controllability_gramian, output_energy
export dynamic_steady_state_gain, affine_steady_state_gain, steady_state_gain
export steady_state_invariance, is_steady_state_invariant
export input_drive, input_responsiveness, input_responsive, compare_input_responsiveness

"""
    BncLinearControlModel

Linearized Binding-Catalysis control data for a `BncRegime`.

The state is `log(q_cat)`. Inputs are selected from `wKk_symbol(rgm)`,
and outputs are selected from either `x_symbol(rgm)` or `q_cat_symbol(rgm)`.
The matrices follow the standard state-space convention: `A` is the state or
system matrix, `B` is the input matrix, `C` is the output matrix, and `D` is
the direct-feedthrough matrix.
"""
struct BncLinearControlModel
    rgm::BncRegime
    A::Matrix{Float64}
    B::Matrix{Float64}
    C::Matrix{Float64}
    D::Matrix{Float64}
    H_bd::Matrix{Float64}
    B_full::Matrix{Float64}
    input::Vector{Symbol}
    output::Vector{Symbol}
    input_indices::Vector{Int}
    output_indices::Vector{Int}
    output_space::Symbol
    qcat_symbols::Vector{Symbol}
    wKk_symbols::Vector{Symbol}
    x_symbols::Vector{Symbol}
    hbd_source::Symbol
    timescale_source::Symbol
    eigvals::Vector{ComplexF64}
    stable::Bool
end

"""
    hbd_source(rgm::BncRegime) -> Symbol

Return the provenance of `H_bd` for a Binding-Catalysis regime.

Regular binding regimes use `:exact_regime_derivative`. Singular binding
regimes use `:numerical_binding_derivative`, matching the current
Binding-Catalysis construction path.
"""
function hbd_source(rgm::BncRegime)
    return if is_singular(get_binding_regime(rgm))
        :numerical_binding_derivative
    else
        :exact_regime_derivative
    end
end

"""
    get_H_bd_info(rgm::BncRegime) -> NamedTuple

Return `(; H, source, binding_nullity)` for the Binding-Catalysis dynamic
matrix derivative.
"""
function get_H_bd_info(rgm::BncRegime)
    binding_rgm = get_binding_regime(rgm)
    return (;
        H=Matrix{Float64}(get_H_bd(rgm)),
        source=hbd_source(rgm),
        binding_nullity=get_nullity(binding_rgm),
    )
end

function _dense_float_matrix(A)
    return Matrix{Float64}(Float64.(A))
end

function _selector_items(selector)
    selector isa AbstractVector && return collect(selector)
    selector isa Tuple && return collect(selector)
    return [selector]
end

function _selector_indices(
    selector, symbols::AbstractVector{Symbol}; what::AbstractString, default_all::Bool=true
)
    if isnothing(selector) || selector === :all
        default_all || throw(ArgumentError("$what selector cannot be `nothing` or `:all`."))
        return collect(eachindex(symbols))
    end

    idxs = Int[]
    for item in _selector_items(selector)
        idx = if item isa Integer
            Int(item)
        else
            target = Symbol(item)
            found = findfirst(==(target), symbols)
            isnothing(found) && throw(
                ArgumentError(
                    "Unknown $what symbol $(repr(target)). Available $what symbols are $(repr(symbols)).",
                ),
            )
            found
        end

        1 <= idx <= length(symbols) ||
            throw(ArgumentError("$what index $idx is out of range 1:$(length(symbols))."))
        push!(idxs, idx)
    end
    return idxs
end

function _resolve_output_indices(
    output,
    x_symbols::AbstractVector{Symbol},
    qcat_symbols::AbstractVector{Symbol};
    output_space::Symbol=:auto,
)
    if output_space === :x
        idxs = if output === :x || output === :all || isnothing(output)
            collect(eachindex(x_symbols))
        else
            _selector_indices(output, x_symbols; what="output")
        end
        return idxs, :x
    elseif output_space === :qcat
        idxs = if output === :qcat || output === :all || isnothing(output)
            collect(eachindex(qcat_symbols))
        else
            _selector_indices(output, qcat_symbols; what="output")
        end
        return idxs, :qcat
    elseif output_space !== :auto
        throw(ArgumentError("output_space must be one of `:auto`, `:x`, or `:qcat`."))
    end

    if output === :qcat
        return collect(eachindex(qcat_symbols)), :qcat
    elseif output === :x || output === :all || isnothing(output)
        return collect(eachindex(x_symbols)), :x
    end

    items = _selector_items(output)
    if all(item -> !(item isa Integer) && Symbol(item) in x_symbols, items)
        return _selector_indices(output, x_symbols; what="output"), :x
    elseif all(item -> !(item isa Integer) && Symbol(item) in qcat_symbols, items)
        return _selector_indices(output, qcat_symbols; what="output"), :qcat
    elseif all(item -> item isa Integer, items)
        return _selector_indices(output, x_symbols; what="output"), :x
    end

    throw(
        ArgumentError(
            "Could not infer output_space for $(repr(output)). Use `output_space=:x` or `output_space=:qcat`.",
        ),
    )
end

function _timescale_scaling(timescale, n::Int)
    if timescale === :identity || isnothing(timescale)
        return Matrix{Float64}(I, n, n), :identity
    end

    tau = Float64.(vec(timescale))
    length(tau) == n ||
        throw(ArgumentError("timescale length must be $n, got $(length(tau))."))
    all(isfinite, tau) || throw(ArgumentError("timescale entries must be finite."))
    all(>(0), tau) || throw(ArgumentError("timescale entries must be positive."))
    return Matrix{Float64}(Diagonal(inv.(tau))), :provided
end

function _binding_control_blocks(rgm::BncRegime)
    binding_rgm = get_binding_regime(rgm)
    catalysis_rgm = get_catalysis_regime(rgm)
    bn = get_binding_network(rgm)
    cn = get_catalysis_network(rgm)

    H_bind = if is_singular(binding_rgm)
        get_H_numerically(binding_rgm)
    else
        get_affine_qK2x(binding_rgm)[1]
    end
    H_bind = _dense_float_matrix(H_bind)
    PΠ = _dense_float_matrix(get_PΠ(catalysis_rgm))
    Pk = _dense_float_matrix(get_P(catalysis_rgm) * cn.F)

    r_v = cn.r_v
    d_w = cn.d_w
    r = bn.r
    n_k = cn.n_k

    qcat_cols = 1:r_v
    w_cols = (r_v + 1):(r_v + d_w)
    K_cols = (bn.d + 1):(bn.d + r)

    H_qcat = H_bind[:, qcat_cols]
    H_w = H_bind[:, w_cols]
    H_K = H_bind[:, K_cols]

    H_bd = PΠ * H_qcat
    B_full = hcat(PΠ * H_w, PΠ * H_K, Pk)
    D_x_full = hcat(H_w, H_K, zeros(Float64, bn.n, n_k))

    return H_bd, B_full, D_x_full, H_qcat
end

"""
    linear_control_model(rgm::BncRegime; input=:all, output=:x, output_space=:auto, timescale=:identity)

Build the linearized BNC control model

```text
d log(q_cat) / dt = A log(q_cat) + B u
y                = C log(q_cat) + D u
```

`input` selects symbols from `wKk_symbol(rgm)`. `output=:x` selects all
species, `output=:qcat` selects all catalytic state variables, and specific
symbols are inferred from `x_symbol(rgm)` or `q_cat_symbol(rgm)`.

`timescale=:identity` uses the regime derivative directly. Passing a positive
vector applies row scaling by `1 ./ timescale`.
"""
function linear_control_model(
    rgm::BncRegime; input=:all, output=:x, output_space::Symbol=:auto, timescale=:identity
)
    H_bd, B_full, D_x_full, H_qcat = _binding_control_blocks(rgm)
    n_state = size(H_bd, 1)
    scale, timescale_source = _timescale_scaling(timescale, n_state)

    wKk_symbols = wKk_symbol(rgm)
    x_symbols = x_symbol(rgm)
    qcat_symbols = q_cat_symbol(rgm)

    input_indices = _selector_indices(input, wKk_symbols; what="input")
    output_indices, resolved_output_space = _resolve_output_indices(
        output, x_symbols, qcat_symbols; output_space=output_space
    )

    A = scale * H_bd
    B = scale * B_full[:, input_indices]

    C, D = if resolved_output_space === :x
        H_qcat[output_indices, :], D_x_full[output_indices, input_indices]
    else
        selector = Matrix{Float64}(I, n_state, n_state)[output_indices, :]
        selector, zeros(Float64, length(output_indices), length(input_indices))
    end

    lambda = ComplexF64.(eigvals(A))
    return BncLinearControlModel(
        rgm,
        A,
        B,
        Matrix{Float64}(C),
        Matrix{Float64}(D),
        Matrix{Float64}(H_bd),
        Matrix{Float64}(B_full),
        wKk_symbols[input_indices],
        if resolved_output_space === :x
            x_symbols[output_indices]
        else
            qcat_symbols[output_indices]
        end,
        input_indices,
        output_indices,
        resolved_output_space,
        qcat_symbols,
        wKk_symbols,
        x_symbols,
        hbd_source(rgm),
        timescale_source,
        lambda,
        all(real.(lambda) .< 0),
    )
end

"""
    control_metrics(rgm::BncRegime; kwargs...) -> NamedTuple
    control_metrics(ctrl::BncLinearControlModel) -> NamedTuple

Return the common linear control matrices and metadata for a BNC regime:
`A`, `B`, `C`, `D`, `eigvals`, selected `input`/`output`, and provenance.
"""
function control_metrics(ctrl::BncLinearControlModel)
    return (;
        A=ctrl.A,
        B=ctrl.B,
        C=ctrl.C,
        D=ctrl.D,
        eigvals=ctrl.eigvals,
        stable=ctrl.stable,
        input=ctrl.input,
        output=ctrl.output,
        output_space=ctrl.output_space,
        qcat_symbols=ctrl.qcat_symbols,
        wKk_symbols=ctrl.wKk_symbols,
        hbd_source=ctrl.hbd_source,
        timescale_source=ctrl.timescale_source,
    )
end

function control_metrics(rgm::BncRegime; kwargs...)
    return control_metrics(linear_control_model(rgm; kwargs...))
end

"""
    controllability_matrix(ctrl; order=size(ctrl.A, 1) - 1)

Return `[B A*B ... A^order*B]` for the state equation.
"""
function controllability_matrix(
    ctrl::BncLinearControlModel; order::Integer=size(ctrl.A, 1) - 1
)
    order >= 0 || throw(ArgumentError("order must be non-negative."))
    blocks = Matrix{Float64}[]
    Apow = Matrix{Float64}(I, size(ctrl.A, 1), size(ctrl.A, 1))
    for _ in 0:order
        push!(blocks, Apow * ctrl.B)
        Apow = Apow * ctrl.A
    end
    return hcat(blocks...)
end

function controllability_matrix(rgm::BncRegime; order=nothing, kwargs...)
    ctrl = linear_control_model(rgm; kwargs...)
    return if isnothing(order)
        controllability_matrix(ctrl)
    else
        controllability_matrix(ctrl; order=order)
    end
end

"""
    markov_coefficients(ctrl; order=size(ctrl.A, 1) - 1)

Return `[D, C*B, C*A*B, ...]` through the requested order.
"""
function markov_coefficients(
    ctrl::BncLinearControlModel; order::Integer=size(ctrl.A, 1) - 1
)
    order >= 0 || throw(ArgumentError("order must be non-negative."))
    coeffs = Matrix{Float64}[ctrl.D]
    Apow = Matrix{Float64}(I, size(ctrl.A, 1), size(ctrl.A, 1))
    for _ in 0:order
        push!(coeffs, ctrl.C * Apow * ctrl.B)
        Apow = Apow * ctrl.A
    end
    return coeffs
end

function markov_coefficients(rgm::BncRegime; order=nothing, kwargs...)
    ctrl = linear_control_model(rgm; kwargs...)
    return if isnothing(order)
        markov_coefficients(ctrl)
    else
        markov_coefficients(ctrl; order=order)
    end
end

"""
    output_controllability_matrix(ctrl; order=size(ctrl.A, 1) - 1)

Return `[D C*B C*A*B ... C*A^order*B]`.
"""
function output_controllability_matrix(
    ctrl::BncLinearControlModel; order::Integer=size(ctrl.A, 1) - 1
)
    coeffs = markov_coefficients(ctrl; order=order)
    return hcat(coeffs...)
end

function output_controllability_matrix(rgm::BncRegime; order=nothing, kwargs...)
    ctrl = linear_control_model(rgm; kwargs...)
    return if isnothing(order)
        output_controllability_matrix(ctrl)
    else
        output_controllability_matrix(ctrl; order=order)
    end
end

"""
    output_controllability_row(args...; kwargs...)

Alias for `output_controllability_matrix`. For a single-output model this is
the row used in many report-level controllability checks.
"""
output_controllability_row(args...; kwargs...) =
    output_controllability_matrix(args...; kwargs...)

"""
    controllability_gramian(ctrl; horizon=:infinite)

Return the infinite-horizon controllability Gramian by solving
`A*W + W*A' + B*B' = 0`. The model must be Hurwitz stable.
"""
function controllability_gramian(ctrl::BncLinearControlModel; horizon::Symbol=:infinite)
    horizon === :infinite ||
        throw(ArgumentError("Only `horizon=:infinite` is currently supported."))
    all(real.(ctrl.eigvals) .< 0) || throw(
        ArgumentError("The infinite-horizon Gramian requires a Hurwitz-stable A matrix."),
    )

    n = size(ctrl.A, 1)
    eye = Matrix{Float64}(I, n, n)
    Q = ctrl.B * transpose(ctrl.B)
    L = kron(eye, ctrl.A) + kron(ctrl.A, eye)
    W = reshape(-(L \ vec(Q)), n, n)
    return Matrix{Float64}((W + transpose(W)) ./ 2)
end

function controllability_gramian(rgm::BncRegime; horizon::Symbol=:infinite, kwargs...)
    return controllability_gramian(linear_control_model(rgm; kwargs...); horizon=horizon)
end

"""
    dynamic_steady_state_gain(ctrl)
    steady_state_gain(ctrl)

Return the dynamic DC gain `D - C * inv(A) * B`. `steady_state_gain` is kept as
the short alias for this dynamic state-space quantity.
"""
function dynamic_steady_state_gain(ctrl::BncLinearControlModel)
    return ctrl.D - ctrl.C * (ctrl.A \ ctrl.B)
end

steady_state_gain(ctrl::BncLinearControlModel) = dynamic_steady_state_gain(ctrl)

function dynamic_steady_state_gain(rgm::BncRegime; kwargs...)
    return dynamic_steady_state_gain(linear_control_model(rgm; kwargs...))
end

function steady_state_gain(rgm::BncRegime; kwargs...)
    return dynamic_steady_state_gain(rgm; kwargs...)
end

"""
    affine_steady_state_gain(ctrl)
    affine_steady_state_gain(rgm::BncRegime; kwargs...)

Return the exact affine steady-state coefficient from the regime map.

For `output_space=:x`, this uses `get_affine_wKk2x(rgm)`. For
`output_space=:qcat`, this uses `get_qcat_F_F0(rgm)`.
"""
function affine_steady_state_gain(ctrl::BncLinearControlModel)
    H = if ctrl.output_space === :x
        get_affine_wKk2x(ctrl.rgm)[1]
    elseif ctrl.output_space === :qcat
        get_qcat_F_F0(ctrl.rgm)[1]
    else
        throw(ArgumentError("Unsupported output_space $(repr(ctrl.output_space))."))
    end
    return _dense_float_matrix(H)[ctrl.output_indices, ctrl.input_indices]
end

function affine_steady_state_gain(rgm::BncRegime; kwargs...)
    return affine_steady_state_gain(linear_control_model(rgm; kwargs...))
end

"""
    output_energy(ctrl; horizon=:infinite, include_direct=false)

Return the output energy matrix `C*W*C'`, where `W` is the controllability
Gramian. If `include_direct=true`, add the direct term `D*D'`.
"""
function output_energy(
    ctrl::BncLinearControlModel; horizon::Symbol=:infinite, include_direct::Bool=false
)
    W = controllability_gramian(ctrl; horizon=horizon)
    energy = ctrl.C * W * transpose(ctrl.C)
    include_direct && (energy += ctrl.D * transpose(ctrl.D))
    return Matrix{Float64}((energy + transpose(energy)) ./ 2)
end

function output_energy(
    rgm::BncRegime; horizon::Symbol=:infinite, include_direct::Bool=false, kwargs...
)
    return output_energy(
        linear_control_model(rgm; kwargs...); horizon=horizon, include_direct=include_direct
    )
end

function _maxabs(A)
    isempty(A) && return 0.0
    return Float64(maximum(abs, A))
end

function _directional_score(A, direction::Symbol)
    if direction === :any
        return _maxabs(A)
    elseif direction === :positive
        isempty(A) && return 0.0
        return Float64(maximum(A))
    elseif direction === :negative
        isempty(A) && return 0.0
        return Float64(maximum(-A))
    end

    throw(ArgumentError("direction must be one of `:any`, `:positive`, or `:negative`."))
end

function _first_signed_markov_coefficients(
    ctrl::BncLinearControlModel; order::Integer, coefficient_atol::Real
)
    coeffs = markov_coefficients(ctrl; order=order)
    first_coeffs = fill(NaN, size(ctrl.D))
    found = falses(size(ctrl.D))

    for coeff in coeffs
        for idx in CartesianIndices(coeff)
            if !found[idx] && abs(coeff[idx]) > coefficient_atol
                first_coeffs[idx] = coeff[idx]
                found[idx] = true
            end
        end
    end

    return first_coeffs, found
end

function _output_reachability_score(
    ctrl::BncLinearControlModel; order::Integer, direction::Symbol, coefficient_atol::Real
)
    if direction === :any
        return Float64(norm(output_controllability_matrix(ctrl; order=order)))
    end

    first_coeffs, found = _first_signed_markov_coefficients(
        ctrl; order=order, coefficient_atol=coefficient_atol
    )
    any(found) || return 0.0
    return _directional_score(first_coeffs[found], direction)
end

function _effective_threshold(standard::Symbol, threshold::Real, threshold_scale::Symbol)
    if standard !== :gramian
        return Float64(threshold)
    elseif threshold_scale === :energy
        return Float64(threshold)
    elseif threshold_scale === :amplitude
        return Float64(threshold)^2
    end

    throw(ArgumentError("threshold_scale must be one of `:energy` or `:amplitude`."))
end

function _resolve_include_direct(ctrl::BncLinearControlModel, include_direct)
    isnothing(include_direct) && return ctrl.output_space === :x
    include_direct isa Bool ||
        throw(ArgumentError("include_direct must be `true`, `false`, or `nothing`."))
    return include_direct
end

"""
    steady_state_invariance(ctrl; standard=:affine, atol=1e-8)
    steady_state_invariance(rgm::BncRegime; standard=:affine, atol=1e-8, kwargs...)

Return a diagnostic named tuple for the steady-state input-output gain.

`standard=:affine` checks the exact affine steady-state map of the BNC regime.
`standard=:dynamic_dc_gain` checks the local dynamic DC gain
`D - C * (A \\ B)`. If the gain cannot be computed, `invariant=false` and
`error` records the reason.
"""
function steady_state_invariance(
    ctrl::BncLinearControlModel; standard::Symbol=:affine, atol::Real=1e-8
)
    try
        gain, source = if standard === :affine
            affine_steady_state_gain(ctrl), :affine_steady_state_map
        elseif standard === :dynamic_dc_gain || standard === :dc_gain
            dynamic_steady_state_gain(ctrl), :dynamic_dc_gain
        else
            throw(
                ArgumentError(
                    "Unknown invariance standard $(repr(standard)). Supported standards are `:affine` and `:dynamic_dc_gain`.",
                ),
            )
        end
        residual = _maxabs(gain)
        return (;
            standard=standard,
            invariant=residual <= atol,
            residual=residual,
            atol=Float64(atol),
            gain=gain,
            source=source,
            input=ctrl.input,
            output=ctrl.output,
            output_space=ctrl.output_space,
            stable=ctrl.stable,
            error=nothing,
        )
    catch err
        return (;
            standard=standard,
            invariant=false,
            residual=Inf,
            atol=Float64(atol),
            gain=nothing,
            source=missing,
            input=ctrl.input,
            output=ctrl.output,
            output_space=ctrl.output_space,
            stable=ctrl.stable,
            error=sprint(showerror, err),
        )
    end
end

function steady_state_invariance(
    rgm::BncRegime; standard::Symbol=:affine, atol::Real=1e-8, kwargs...
)
    return steady_state_invariance(
        linear_control_model(rgm; kwargs...); standard=standard, atol=atol
    )
end

"""
    is_steady_state_invariant(args...; kwargs...) -> Bool

Boolean convenience wrapper around `steady_state_invariance`.
"""
function is_steady_state_invariant(args...; kwargs...)
    return steady_state_invariance(args...; kwargs...).invariant
end

"""
    input_drive(rgm::BncRegime; input=:all, positive, negative=nothing, target_space=:x, direction=:positive)

Return a signed direct input-drive diagnostic. `positive` and `negative` select
rows in `target_space`; the reported drive is `sum(positive rows) - sum(negative rows)`.

Use `target_space=:x` for binding-species drive terms and `target_space=:qcat`
for direct drive into catalytic state variables.
"""
function input_drive(
    rgm::BncRegime;
    input=:all,
    positive,
    negative=nothing,
    target_space::Symbol=:x,
    direction::Symbol=:positive,
    timescale=:identity,
)
    wKk_symbols = wKk_symbol(rgm)
    input_indices = _selector_indices(input, wKk_symbols; what="input")

    target_symbols, drive_matrix, source = if target_space === :x
        _, _, D_x_full, _ = _binding_control_blocks(rgm)
        x_symbol(rgm), D_x_full[:, input_indices], :binding_direct_input_drive
    elseif target_space === :qcat
        ctrl = linear_control_model(
            rgm; input=input, output=:qcat, output_space=:qcat, timescale=timescale
        )
        ctrl.qcat_symbols, ctrl.B, :qcat_input_drive
    else
        throw(ArgumentError("target_space must be one of `:x` or `:qcat`."))
    end

    positive_indices = _selector_indices(positive, target_symbols; what="positive")
    negative_indices = if isnothing(negative)
        Int[]
    else
        _selector_indices(negative, target_symbols; what="negative")
    end

    positive_drive = sum(drive_matrix[positive_indices, :]; dims=1)
    negative_drive = if isempty(negative_indices)
        zeros(Float64, 1, length(input_indices))
    else
        sum(drive_matrix[negative_indices, :]; dims=1)
    end
    drive = Matrix{Float64}(positive_drive - negative_drive)
    score = _directional_score(drive, direction)

    return (;
        score=score,
        drive=drive,
        direction=direction,
        input=wKk_symbols[input_indices],
        positive=target_symbols[positive_indices],
        negative=target_symbols[negative_indices],
        target_space=target_space,
        source=source,
    )
end

function input_drive(ctrl::BncLinearControlModel; input=nothing, kwargs...)
    selected_input = isnothing(input) ? ctrl.input : input
    return input_drive(ctrl.rgm; input=selected_input, kwargs...)
end

function _responsiveness_score(
    ctrl::BncLinearControlModel,
    standard::Symbol;
    order::Integer,
    horizon::Symbol,
    direction::Symbol,
    coefficient_atol::Real,
    threshold_scale::Symbol,
    include_direct,
    positive_flux,
    negative_flux,
    flux_space::Symbol,
)
    if standard === :direct_feedthrough
        return _maxabs(ctrl.D)
    elseif standard === :direct_flux
        isnothing(positive_flux) && throw(
            ArgumentError(
                "`standard=:direct_flux` requires `positive_flux`; pass `negative_flux` when the drive is a signed difference. Use `standard=:direct_feedthrough` for the output-equation `D` term.",
            ),
        )
        return input_drive(
            ctrl;
            positive=positive_flux,
            negative=negative_flux,
            target_space=flux_space,
            direction=direction,
        ).score
    elseif standard === :output_controllability
        return _maxabs(output_controllability_matrix(ctrl; order=order))
    elseif standard === :output_reachability
        return _output_reachability_score(
            ctrl; order=order, direction=direction, coefficient_atol=coefficient_atol
        )
    elseif standard === :gramian
        resolved_include_direct = _resolve_include_direct(ctrl, include_direct)
        return Float64(
            norm(
                output_energy(ctrl; horizon=horizon, include_direct=resolved_include_direct)
            ),
        )
    elseif standard === :steady_state_gain
        return _maxabs(dynamic_steady_state_gain(ctrl))
    end

    throw(
        ArgumentError(
            "Unknown responsiveness standard $(repr(standard)). Supported standards are `:direct_feedthrough`, `:direct_flux`, `:output_controllability`, `:output_reachability`, `:gramian`, and `:steady_state_gain`.",
        ),
    )
end

"""
    input_responsiveness(ctrl; standard=:output_reachability, threshold=1e-8)
    input_responsiveness(rgm::BncRegime; input=:all, output=:x, standard=:output_reachability, threshold=1e-8)

Return a diagnostic named tuple for a selected input-output responsiveness
standard.

Supported standards are:

  - `:direct_feedthrough`: direct `D` term magnitude.
  - `:direct_flux`: signed input drive from explicit `positive_flux` and optional `negative_flux` terms.
  - `:output_controllability`: max absolute entry of `[D C*B C*A*B ...]`.
  - `:output_reachability`: norm of `[D C*B C*A*B ...]` for `direction=:any`,
    or the first signed Markov coefficient for `direction=:positive`/`:negative`.
  - `:gramian`: norm of output energy. `threshold_scale=:amplitude` compares
    energy against `threshold^2`; `threshold_scale=:energy` compares directly.
  - `:steady_state_gain`: max absolute DC gain.
"""
function input_responsiveness(
    ctrl::BncLinearControlModel;
    standard::Symbol=:output_reachability,
    threshold::Real=1e-8,
    order::Integer=size(ctrl.A, 1) - 1,
    horizon::Symbol=:infinite,
    direction::Symbol=:any,
    coefficient_atol::Real=1e-12,
    threshold_scale::Symbol=:energy,
    include_direct=nothing,
    positive_flux=nothing,
    negative_flux=nothing,
    flux_space::Symbol=:x,
)
    score = _responsiveness_score(
        ctrl,
        standard;
        order=order,
        horizon=horizon,
        direction=direction,
        coefficient_atol=coefficient_atol,
        threshold_scale=threshold_scale,
        include_direct=include_direct,
        positive_flux=positive_flux,
        negative_flux=negative_flux,
        flux_space=flux_space,
    )
    effective_threshold = _effective_threshold(standard, threshold, threshold_scale)
    resolved_include_direct =
        standard === :gramian ? _resolve_include_direct(ctrl, include_direct) : false
    return (;
        standard=standard,
        responsive=score > effective_threshold,
        score=score,
        threshold=Float64(threshold),
        effective_threshold=effective_threshold,
        threshold_scale=standard === :gramian ? threshold_scale : missing,
        order=Int(order),
        direction=if standard === :output_reachability || standard === :direct_flux
            direction
        else
            missing
        end,
        coefficient_atol=Float64(coefficient_atol),
        include_direct=standard === :gramian ? resolved_include_direct : missing,
        positive_flux=standard === :direct_flux ? positive_flux : missing,
        negative_flux=standard === :direct_flux ? negative_flux : missing,
        flux_space=standard === :direct_flux ? flux_space : missing,
        input=ctrl.input,
        output=ctrl.output,
        output_space=ctrl.output_space,
        stable=ctrl.stable,
        hbd_source=ctrl.hbd_source,
        error=nothing,
    )
end

function input_responsiveness(
    rgm::BncRegime;
    standard::Symbol=:output_reachability,
    threshold::Real=1e-8,
    order=nothing,
    horizon::Symbol=:infinite,
    direction::Symbol=:any,
    coefficient_atol::Real=1e-12,
    threshold_scale::Symbol=:energy,
    include_direct=nothing,
    positive_flux=nothing,
    negative_flux=nothing,
    flux_space::Symbol=:x,
    kwargs...,
)
    ctrl = linear_control_model(rgm; kwargs...)
    resolved_order = isnothing(order) ? size(ctrl.A, 1) - 1 : order
    return input_responsiveness(
        ctrl;
        standard=standard,
        threshold=threshold,
        order=resolved_order,
        horizon=horizon,
        direction=direction,
        coefficient_atol=coefficient_atol,
        threshold_scale=threshold_scale,
        include_direct=include_direct,
        positive_flux=positive_flux,
        negative_flux=negative_flux,
        flux_space=flux_space,
    )
end

"""
    input_responsive(args...; kwargs...) -> Bool

Boolean convenience wrapper around `input_responsiveness`.
"""
function input_responsive(args...; kwargs...)
    return input_responsiveness(args...; kwargs...).responsive
end

function _comparison_regime_id(rgm::BncRegime)
    return get_idx(rgm)
end

"""
    compare_input_responsiveness(rgms; input=:all, outputs=:x, standards=(...), threshold=1e-8)

Return a vector of named tuples comparing responsiveness standards across BNC
regimes and outputs.
"""
function compare_input_responsiveness(
    rgms::AbstractVector{<:BncRegime};
    input=:all,
    outputs=:x,
    standards=(:direct_feedthrough, :output_controllability, :output_reachability),
    threshold::Real=1e-8,
    order=nothing,
    kwargs...,
)
    rows = NamedTuple[]
    for rgm in rgms
        for output in _selector_items(outputs)
            for standard in _selector_items(standards)
                try
                    result = input_responsiveness(
                        rgm;
                        input=input,
                        output=output,
                        standard=Symbol(standard),
                        threshold=threshold,
                        order=order,
                        kwargs...,
                    )
                    push!(
                        rows,
                        (;
                            regime=_comparison_regime_id(rgm),
                            binding_regime=get_idx(get_binding_regime(rgm)),
                            catalysis_regime=get_idx(get_catalysis_regime(rgm)),
                            result...,
                        ),
                    )
                catch err
                    push!(
                        rows,
                        (;
                            regime=_comparison_regime_id(rgm),
                            binding_regime=get_idx(get_binding_regime(rgm)),
                            catalysis_regime=get_idx(get_catalysis_regime(rgm)),
                            standard=Symbol(standard),
                            responsive=missing,
                            score=NaN,
                            threshold=Float64(threshold),
                            effective_threshold=NaN,
                            threshold_scale=missing,
                            order=isnothing(order) ? missing : order,
                            direction=missing,
                            coefficient_atol=missing,
                            include_direct=missing,
                            positive_flux=missing,
                            negative_flux=missing,
                            flux_space=missing,
                            input=_selector_items(input),
                            output=_selector_items(output),
                            output_space=missing,
                            stable=missing,
                            hbd_source=hbd_source(rgm),
                            error=sprint(showerror, err),
                        ),
                    )
                end
            end
        end
    end
    return rows
end
