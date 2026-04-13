function show_condition_poly(
    C::AbstractMatrix{<:Real},
    C0::AbstractVector{<:Real},
    nullity::Integer=0;
    syms::AbstractVector{Num},
    log_space::Bool=false,
    asymptotic::Bool=false,
)
    make_expr(Crow, C0v) = if log_space
        expr = Crow * log10.(syms)
        asymptotic ? expr : expr .+ C0v
    else
        asymptotic ? handle_log_weighted_sum(Crow, syms) : handle_log_weighted_sum(Crow, syms, C0v)
    end

    make_cond(expr, op) = begin
        if log_space
            op == :eq ? (expr .~ 0) : (expr .> 0)
        else
            expr .|> x -> begin
                num, den = numerator(x), denominator(x)
                op == :eq ? (num ~ den) : (num > den)
            end
        end
    end

    if nullity == 0
        return make_cond(make_expr(C, C0), :uneq)
    else
        eq_expr = make_expr(C[1:nullity, :], C0[1:nullity])
        uneq_expr = make_expr(C[nullity + 1:end, :], C0[nullity + 1:end])
        return vcat(make_cond(eq_expr, :eq), make_cond(uneq_expr, :uneq))
    end
end

show_condition_poly(poly::Polyhedron; kwargs...) = show_condition_poly(get_C_C0_nullity(poly)...; kwargs...)
show_condition_poly(C_qK::AbstractVector{<:Real}, C0_qK::Real, args...; kwargs...) =
    show_condition_poly(C_qK', [C0_qK], args...; kwargs...)[1]

function show_expression_mapping(
    C::AbstractMatrix{<:Real},
    C0::AbstractVector{<:Real},
    y,
    x;
    log_space::Bool=false,
    asymptotic::Bool=false,
)::Vector{Equation}
    if log_space
        return asymptotic ? log10.(y) .~ C * log10.(x) : log10.(y) .~ C * log10.(x) .+ C0
    else
        return asymptotic ? y .~ handle_log_weighted_sum(C, x) : y .~ handle_log_weighted_sum(C, x, C0)
    end
end

show_expression_mapping(C::AbstractVector{<:Real}, C0::Real, args...; kwargs...) =
    show_expression_mapping(C', [C0], args...; kwargs...)[1]

@inline _render_condition_from(data::Tuple, syms; kwargs...) = show_condition_poly(data...; syms=syms, kwargs...)
@inline _render_expression_from(data::Tuple, y, x; kwargs...) = show_expression_mapping(data..., y, x; kwargs...)

function _exact_integer_power(base::Integer, exp::Rational{<:Integer})
    denominator(exp) == 1 || return nothing
    e = Int(numerator(exp))
    b = big(base)
    return e >= 0 ? (b ^ e) // 1 : 1 // (b ^ (-e))
end

function _exact_exp10_factor(b::ExactLogExpr)
    out = one(BigInt) // one(BigInt)
    if !iszero(b.constant)
        term = _exact_integer_power(10, b.constant)
        isnothing(term) && return 10 ^ Float64(b)
        out *= term
    end
    for (p, c) in sort!(collect(b.coeffs); by=first)
        term = _exact_integer_power(p, c)
        isnothing(term) && return 10 ^ Float64(b)
        out *= term
    end
    return out
end

_exp10_factor(b::ExactLogExpr) = _exact_exp10_factor(b)
_exp10_factor(b::Integer) = _exact_integer_power(10, Int(b)//1)
function _exp10_factor(b::Rational{<:Integer})
    term = _exact_integer_power(10, Int(numerator(b)) // Int(denominator(b)))
    return isnothing(term) ? 10^Float64(b) : term
end
_exp10_factor(b::AbstractFloat) = 10^b
_exp10_factor(b::Real) = 10^Float64(b)

function handle_log_weighted_sum(A::AbstractMatrix{<:Real}, x, b::Union{Nothing,AbstractVector{<:Real}}=nothing)::Vector{Num}
    rows = size(A, 1)
    rst = Vector{Num}(undef, rows)
    b = isnothing(b) ? zeros(Int, rows) : b
    for i in 1:rows
        rst[i] = x .^ A[i, :] |> prod |> (u -> u * _exp10_factor(b[i]))
    end
    return rst
end

function solve_sym_expr(a::AbstractVector{<:Real}, b::Real, x, idx; log_space::Bool=false)
    a = copy(collect(a))
    x = copy(x)
    ai = popat!(a, idx)
    target_x = popat!(x, idx)
    @assert abs(ai) > 1e-10 "Cannot solve for the variable at index $idx since its coefficient is zero."
    a ./= -ai
    b /= -ai
    target = log_space ? log10(target_x) : target_x
    expr = log_space ? a' * log10.(x) .+ b : handle_log_weighted_sum(a', x, [b])[1]
    return target ~ expr
end
