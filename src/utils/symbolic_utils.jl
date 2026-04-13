"""
    name_converter(name::Vector) -> Vector{Num}

Convert a vector of symbols or numbers into Symbolics variables.
"""
function name_converter(name::Vector{<:T})::Vector{Num} where T
    if T <: Num
        return name
    else
        return [Symbolics.variable(x; T=Real) for x in name]
    end
end

"""
    log10_sym(x) -> Num

Symbolic `log10` wrapper that preserves `Num(0)` for unity.
"""
log10_sym(x) = x == 1 ? Num(0) : Symbolics.wrap(Symbolics.Term(log10, [x, ]))

"""
    exp10_sym(x) -> Num

Symbolic `exp10` wrapper.
"""
exp10_sym(x) = Symbolics.wrap(Symbolics.Term(exp10, [x, ]))

"""
    render_array(M, empty_posi_subs=nothing) -> String

Render an array as a formatted string.
"""
function render_array(M::AbstractArray, empty_posi_subs=nothing)
    A = Array{Any}(M)
    f(x) = begin
        a = try
            Int(round(x; digits=3))
        catch
            round(x; digits=5)
        end
        a == 0 ? empty_posi_subs : a
    end
    A = f.(A)
    return latexify(A)
end

"""
    strip_before_bracket(s::AbstractString) -> String

Remove everything before the first `[` character, including the bracket.
"""
strip_before_bracket(s::AbstractString) = replace(s, r"^[^\[]*" => "")
