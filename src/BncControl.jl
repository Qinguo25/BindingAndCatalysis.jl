export BncLinearControlModel
export hbd_source, get_H_bd_info
export linear_control_model, control_metrics
export controllability_matrix, output_controllability_matrix, output_controllability_row
export markov_coefficients, controllability_gramian, steady_state_gain

"""
    BncLinearControlModel

Linearized Binding-Catalysis control data for a `BncRegime`.

The state is `log(q_cat)`. Inputs are selected from `wKk_symbol(rgm)`,
and outputs are selected from either `x_symbol(rgm)` or `q_cat_symbol(rgm)`.
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
    steady_state_gain(ctrl)

Return the DC gain `D - C * inv(A) * B`.
"""
function steady_state_gain(ctrl::BncLinearControlModel)
    return ctrl.D - ctrl.C * (ctrl.A \ ctrl.B)
end

function steady_state_gain(rgm::BncRegime; kwargs...)
    return steady_state_gain(linear_control_model(rgm; kwargs...))
end
