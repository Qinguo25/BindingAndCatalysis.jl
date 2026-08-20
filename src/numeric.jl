export logder_x_qK, logder_qK_x, ∂logx_∂logqK, ∂logqK_∂logx, get_H_numerically

#----------------Functions for calculates the derivative of log(x) with respect to log(qK) and vice versa----------------------

"""
    ∂logqK_∂logx(bnc::Bnc; x=nothing, qK=nothing, input=:linear) -> Matrix

Compute the Jacobian of `log(q,K)` with respect to `log(x)` at a given point.

# Keyword Arguments

  - `x`: Species concentrations in linear space.
  - `qK`: Totals/binding constants in linear space.

# Returns

  - Jacobian matrix of `logqK` with respect to `logx`.
"""
function ∂logqK_∂logx(
    Bnc::Bnc;
    x::Union{AbstractVector{<:Real}, Nothing}=nothing,
    qK::Union{AbstractVector{<:Real}, Nothing}=nothing,
    input::Symbol=:linear,
    input_logspace::Union{Bool, Nothing}=nothing,
)::Matrix{Float64}
    input = _resolve_space_mode(input, input_logspace, :input_logspace)

    x = if isnothing(x)
        if isnothing(qK)
            error("Either x or qK must be provided")
        else
            qK2x(Bnc, qK; input=input, output=:linear)
        end
    elseif input === :log
        exp10.(x) # Convert from log space to linear space
    else
        x
    end

    q = if isnothing(qK)
        Bnc.L * x
    elseif input === :log
        exp10.(qK[1:(Bnc.d)])
    else
        qK[1:(Bnc.d)]
    end

    return vcat(x' .* Matrix{Float64}(Bnc.L) ./ q, Matrix{Float64}(Bnc.N))
end
"""
    ∂logx_∂logqK(bnc::Bnc; x=nothing, qK=nothing, q=nothing) -> Matrix

Compute the Jacobian of `log(x)` with respect to `log(q,K)`.
"""
∂logx_∂logqK(args...; kwargs...) = inv(∂logqK_∂logx(args...; kwargs...))

"""
    logder_x_qK(args...; kwargs...) -> Matrix

Alias for `∂logx_∂logqK`.
"""
logder_x_qK(args...; kwargs...) = ∂logx_∂logqK(args...; kwargs...)
"""
    logder_qK_x(args...; kwargs...) -> Matrix

Alias for `∂logqK_∂logx`.
"""
logder_qK_x(args...; kwargs...) = ∂logqK_∂logx(args...; kwargs...)

function get_H_numerically(rgm::BindRegime)
    bn = get_binding_network(rgm)
    C, C0 = get_C_C0_x(rgm)
    poly = get_polyhedron(C, C0, 0; canonicalize=true)
    logx = get_one_inner_point(poly; rand_line=false, rand_ray=false, extend=4)
    H = logder_x_qK(bn; x=logx, input=:log)
    return H
end
