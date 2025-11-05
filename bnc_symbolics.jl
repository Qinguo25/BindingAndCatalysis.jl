#----------------------------------------------------------Symbolics calculation fucntions-----------------------------------------------------------



"""
Symbolicly calculate ∂logqK/∂logx
"""
function ∂logqK_∂logx_sym(Bnc::Bnc; show_x_space::Bool=false)::Matrix{Num}

    if show_x_space
        q = Bnc.L * Bnc.x_sym
    else
        q = Bnc.q_sym
    end

    return [
        transpose(Bnc.x_sym) .* Bnc.L ./ q
        Bnc.N
    ]
end
"""
Symbolicly calculate ∂logx/∂logqK
"""
function ∂logx_∂logqK_sym(Bnc::Bnc;show_x_space::Bool=false)::Matrix{Num}
    # Calculate the symbolic derivative of log(qK) with respect to log(x)
    # This function is used for symbolic calculations, not numerical ones.
    return inv(∂logqK_∂logx_sym(Bnc; show_x_space=show_x_space)).|> simplify
end

#---------------------------------------------------------
#   Below are regimes associtaed symbolic functions
#---------------------------------------------------------

"""
handle C log(sym) + C0 >= 0
"""
function show_sym_conds(C::AbstractMatrix{<:Real},
                        C0::AbstractVector{<:Real},
                        syms::AbstractVector{Num},
                        nullity::Integer = 0;
                        log_space::Bool = true,
                        asymptotic::Bool = false)::Vector{Num}

    # Helper: generate symbolic expression per row
    make_expr(Crow, C0v) = if log_space
        expr = Crow * log10.(syms)
        asymptotic ? expr : expr .+ C0v
    else
        asymptotic ?
            handle_log_weighted_sum(Crow, syms) :
            handle_log_weighted_sum(Crow, syms, C0v)
    end

    # Helper: generate symbolic comparison
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

    # Handle two cases: nullity == 0 vs >0
    if nullity == 0
        expr = make_expr(C, C0)
        conds = make_cond(expr, :uneq)
        return conds .|> Num
    else
        eq_expr   = make_expr(C[1:nullity, :], C0[1:nullity])
        uneq_expr = make_expr(C[nullity+1:end, :], C0[nullity+1:end])

        eq   = make_cond(eq_expr, :eq)
        uneq = make_cond(uneq_expr, :uneq)

        return vcat(eq, uneq) .|> Num
    end
end
function show_sym_conds(C_qK::AbstractVector{<:Real},
                        C0_qK::Real,
                        args...;
                        kwargs...)
    show_sym_conds(C_qK', [C0_qK], args...; kwargs...)
end


"""
handle log(sym2) = C log(sym1) + C0
"""
function show_sym_expr(C::AbstractMatrix{<:Real}, C0::AbstractVector{<:Real}, sym2, sym1; log_space::Bool=true,asymptotic::Bool=false)::Vector{Num}
    if log_space
        expr =  asymptotic ?   log10.(sym2) .~ C * log10.(sym1) : log10.(sym2) .~ C * log10.(sym1) .+ C0
    else
        expr =  asymptotic ? sym2 .~ handle_log_weighted_sum(C, sym1) : sym2 .~ handle_log_weighted_sum(C, sym1,C0)
    end
    return expr .|> Num
end
function show_sym_expr(C::AbstractVector{<:Real}, C0::Real, sym2, sym1; log_space::Bool=true,asymptotic::Bool=false)
    show_sym_expr(C', [C0], sym2, sym1; log_space=log_space, asymptotic=asymptotic)
end



function show_expression_x(Bnc::Bnc, perm;kwargs...)
    H,H0 = get_H_H0!(Bnc, perm)
    qK_syms = [Bnc.q_sym; Bnc.K_sym]
    show_sym_expr(H, H0, qK_syms, Bnc.x_sym; kwargs...)
end
function show_expression_qK(Bnc::Bnc, perm;kwargs...)
    M,M0 = get_M_M0!(Bnc, perm)
    qK_syms = [Bnc.q_sym; Bnc.K_sym]
    show_sym_expr(M, M0, qK_syms, Bnc.x_sym; kwargs...)
end
function show_dominant_condition(Bnc::Bnc, perm;kwargs...)
    P,P_0 = get_P_P0!(Bnc, perm)
    show_sym_conds(P, P_0, Bnc.q_sym; kwargs...)
end




function show_condition_x(Bnc::Bnc, perm; kwargs...)
    C_x, C_0 =  get_C_C0_x!(Bnc,perm)
    # Show the conditions for the x space for the given regime.
    show_sym_conds(C_x, C_0, Bnc.x_sym; kwargs...)
end

function show_condition_qK(Bnc::Bnc, perm; kwargs...)
    C_qK, C0_qK = get_C_C0_qK!(Bnc, perm)
    syms = [Bnc.q_sym; Bnc.K_sym]
    nullity = get_nullity!(Bnc, perm)
    show_sym_conds(C_qK, C0_qK, syms, nullity; kwargs...)
end

function show_condition_poly(Bnc::Bnc, poly::Polyhedron,change_dir_idx=nothing; kwargs...)
    C,C0,nullity= get_C_C0(poly)
    syms = [Bnc.q_sym; Bnc.K_sym]
    isnothing(change_dir_idx) || popat!(syms, change_dir_idx)
    @show syms
    nullity = isempty(nullity) ? 0 : maximum(nullity)
    show_sym_conds(C, C0, syms, maximum(nullity); kwargs...)
end

function show_conservation(Bnc::Bnc)
    return Bnc.q_sym .~ Bnc._L_sparse * Bnc.x_sym .|> Num
end

function show_equilibrium(Bnc::Bnc;log_space::Bool=true)
    sym2 = Bnc.K_sym
    sym1 = Bnc.x_sym
    return show_sym_expr(Bnc.N, zeros(Int,Bnc.r), sym2, sym1; log_space=log_space)
end

"""
Symbolic helper function to convert a sum of log10 terms into a product form.
from ∑a log b to log ∏b^a

The final expression contains ∏b^a term.
"""
function handle_log_weighted_sum(C::AbstractMatrix{<:Real}, syms , C0::Union{Nothing,AbstractVector{<:Real}}=nothing)::Vector{Num}
    rows = size(C,1)
    rst = Vector{Num}(undef, rows)
    C0 = isnothing(C0) ? zeros(Int, rows) : C0
    for i in 1:rows
        rst[i] = syms .^ C[i,:] |> prod |> (x-> x*10^C0[i])
    end
    return rst
end



function sym_direction(Bnc::Bnc,dir)
    rst = ""
    for i in 1:Bnc.d
        if dir[i] > 0
            rst *= "+"*repr(Bnc.q_sym[i])*" "
        elseif dir[i] < 0
            rst *= "-"*repr(Bnc.q_sym[i])*" "
        end
    end
    rst*="; "
    for j in 1:Bnc.r
        if dir[j+Bnc.d] > 0
            rst *= "+"*repr(Bnc.K_sym[j])*" "
        elseif dir[j+Bnc.d] < 0
            rst *= "-"*repr(Bnc.K_sym[j])*" "
        end
    end
    return rst
end


function render_arrow(a::Vector, appendix="")
    v = Vector{Any}(undef, length(a))
    for i in eachindex(a)
        try 
            v[i] = Int(a[i])
        catch
            v[i] = a[i]
        end
    end
    # @show v
    s = appendix*repr(v[1])
    for x in v[2:end]
        s *= " → " *appendix* repr(x)
    end
    return s
end

# function sym_direction(Bnc::Bnc,dir; show_val::Bool=true)
#     dir = show_val ? dir : sign.(dir)
#     rst = ""
#     for i in 1:Bnc.d
#         if dir[i] > 0
#             rst *= "+"*repr(Bnc.q_sym[i]^dir[i])*" "
#         elseif dir[i] < 0
#             rst *= "-"*repr(Bnc.q_sym[i]^(-dir[i]))*" "
#         end
#     end
#     rst*="; "
#     for j in 1:Bnc.r
#         if dir[j+Bnc.d] > 0
#             rst *= "+"*repr(Bnc.K_sym[j]^dir[j+Bnc.d])*" "
#         elseif dir[j+Bnc.d] < 0
#             rst *= "-"*repr(Bnc.K_sym[j]^(-dir[j+Bnc.d]))*" "
#         end
#     end
#     return rst
# end