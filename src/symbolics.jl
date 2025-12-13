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
    return inv(∂logqK_∂logx_sym(Bnc; show_x_space=show_x_space)).|> Symbolics.simplify
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
                        asymptotic::Bool = false
)::Vector{Num}

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
function show_sym_expr(C::AbstractMatrix{<:Real}, C0::AbstractVector{<:Real}, sym2, sym1; log_space::Bool=true,asymptotic::Bool=false)::Vector{Equation}
    if log_space
        expr =  asymptotic ?   log10.(sym2) .~ C * log10.(sym1) : log10.(sym2) .~ C * log10.(sym1) .+ C0
    else
        expr =  asymptotic ? sym2 .~ handle_log_weighted_sum(C, sym1) : sym2 .~ handle_log_weighted_sum(C, sym1,C0)
    end
    return expr 
end
function show_sym_expr(C::AbstractVector{<:Real}, C0::Real, sym2, sym1; log_space::Bool=true,asymptotic::Bool=false)
    show_sym_expr(C', [C0], sym2, sym1; log_space=log_space, asymptotic=asymptotic)
end


"""
 for a^T x +b = 0 =>  xi = -(1/ai)(b + ∑j≠i aj xj)
"""
function solve_sym_expr(a::AbstractVector{<:Real}, b::Real, x, idx; log_space::Bool=true)
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


function show_expression_x(Bnc::Bnc, perm;kwargs...)
    H,H0 = get_H_H0!(Bnc, perm)
    qK_syms = [Bnc.q_sym; Bnc.K_sym]
    show_sym_expr(H, H0, Bnc.x_sym, qK_syms; kwargs...)
end
function show_expression_qK(Bnc::Bnc, perm;kwargs...)
    M,M0 = get_M_M0!(Bnc, perm)
    qK_syms = [Bnc.q_sym; Bnc.K_sym]
    show_sym_expr(M, M0, qK_syms, Bnc.x_sym; kwargs...)
end
function show_dominant_condition(Bnc::Bnc, perm;kwargs...)
    P,P_0 = get_P_P0!(Bnc, perm)
    show_sym_expr(P, P_0, Bnc.q_sym, Bnc.x_sym; kwargs...)
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

function show_condition_qK(grh::SISO_graph, pth_idx; kwargs...)
    poly = get_polyhedra!(grh, pth_idx)[1]
    syms = [grh.bn.q_sym; grh.bn.K_sym]
    popat!(syms, grh.change_qK_idx)
    return show_condition_poly(poly, syms; kwargs...)
end


function show_condition_poly(poly::Polyhedron, syms; kwargs...)
    C,C0,nullity= get_C_C0(poly)
    nullity = isempty(nullity) ? 0 : maximum(nullity)
    show_sym_conds(C, C0, syms, maximum(nullity); kwargs...)
end

function show_condition_path(Bnc::Bnc, path::AbstractVector{<:Integer}, change_qK; kwargs...)
    # directly calculate the polyhedron for the path, may not useful.
    poly = _calc_polyhedra_for_path(Bnc, path,change_qK)
    show_condition_path(Bnc, poly, change_qK; kwargs...)
end






function show_conservation(Bnc::Bnc)::Vector{Equation}
    eq  = Bnc.q_sym .~ Bnc._L_sparse * Bnc.x_sym
    return eq 
end

function show_equilibrium(Bnc::Bnc;log_space::Bool=true)::Vector{Equation}
    sym2 = Bnc.K_sym
    sym1 = Bnc.x_sym
    return show_sym_expr(Bnc.N, zeros(Int,Bnc.r), sym2, sym1; log_space=log_space)
end



function show_interface(Bnc::Bnc, from,to, change_idx::Union{Nothing,Integer}=nothing;kwargs...)
    C, C0 = get_interface(Bnc,from,to) # C' log qK + C0 =0
    if isnothing(change_idx)
        return show_sym_conds(C, C0, [Bnc.q_sym; Bnc.K_sym], 1;kwargs...)
    else
        return solve_sym_expr(C,C0, [Bnc.q_sym; Bnc.K_sym], change_idx;kwargs...)
    end
end






"""
Symbolic helper function to convert a sum of log10 terms into a product form.
from A*log x + b  to x^A*10^b

The final expression contains ∏b^a term.
"""
function handle_log_weighted_sum(A::AbstractMatrix{<:Real}, x , b::Union{Nothing,AbstractVector{<:Real}}=nothing)::Vector{Num}
    rows = size(A,1)
    rst = Vector{Num}(undef, rows)
    b = isnothing(b) ? zeros(Int, rows) : b
    for i in 1:rows
        rst[i] = x .^ A[i,:] |> prod |> (x-> x*10^b[i])
    end
    return rst
end


"""
Create a symbolic representation of the direction change in qK-space.
eg: +q1 -q2 +K2 from [1,-1,0,1]
"""
function sym_direction(Bnc::Bnc,dir)::String
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




"""
Render an arrow representation of a vector, with an optional appendix before each element.
eg: render_arrow([1,0,-1],"x") -> "x1 → x0 → x-1"
"""
function render_arrow(a::Vector, appendix="")::String
    v = Vector{Any}(undef, length(a))
    for i in eachindex(a)
        try 
            v[i] = Int(round(a[i];digits=3))
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








function render_path(pths::AbstractVector{<:Tuple};kwargs...)
    if length(pths[1]) == 2
        return render_path(1:length(pths), getindex.(pths,1), getindex.(pths,2); kwargs...)
    elseif length(pths[1]) == 1
        return render_path(1:length(pths), getindex.(pths,1); kwargs...)
    else
        return render_path(getindex.(pths,1), getindex.(pths,2),getindex.(pths,3); kwargs...)
    end
end
render_path(groups::AbstractVector{Vector{<:Real}}; kwargs...) = render_path(eachindex(groups), groups, nothing; kwargs...)
function render_path(groups, pths, volumes=nothing; appendix="")
    path_width = 15  # "Path" 列的宽度
    arrow_width = 30  # render_arrow 的列宽度
    volume_width = 6  # Volume 列的宽度
    if isnothing(volumes)
        for (i, pth) in zip(groups, pths)
            # 格式化输出路径编号、箭头路径和 volume 列（无 volume）
            println(Printf.@sprintf("Path %-*s  %-*s", path_width, repr(i), arrow_width, render_arrow(pth, appendix)))
        end
    elseif length(volumes[1]) == 2
        for (i, pth, vals) in zip(groups, pths, volumes)
            # 格式化输出路径编号、箭头路径以及体积和误差
            println(Printf.@sprintf("Path %-*s  %-*s\t  Volume: %-*s ± %-*s", path_width, repr(i), arrow_width, render_arrow(pth, appendix), volume_width, string(round(vals[1], digits=4)), volume_width, string(round(vals[2], digits=4))))
        end
    else # volumes[1] is a single value
        for (i, pth, vals) in zip(groups, pths, volumes)
            # 格式化输出路径编号、箭头路径以及体积
            println(Printf.@sprintf("Path %-*s  %-*s\t  Volume: %-*s", path_width, repr(i), arrow_width, render_arrow(pth, appendix), volume_width, string(round(vals, digits=4))))
        end
    end
    return nothing
end
function show_path(grh::SISO_graph; show_volume::Bool=true,kwargs...)
    pths = grh.rgm_paths
    if show_volume 
        val =  get_volume!(grh,kwargs...)
        render_path(1:length(pths), pths, val; appendix ="#")
    else
        render_path(1:length(pths), pths; appendix ="#")
    end
    return nothing
end




"""
Given a regime path, change_qK_idx, and observe_x_idx, return the symbolic expressions for the path in the form
[expression1, edge1, expression2, edge2,...]
"""
function show_expression_path(model::Bnc, rgm_path, change_qK_idx, observe_x_idx;log_space::Bool=false)::Tuple{Vector,Vector}
    change_qK_idx = locate_sym([model.q_sym;model.K_sym],change_qK_idx)
    observe_x_idx = locate_sym(model.x_sym, observe_x_idx)
    have_volume_mask = _get_vertices_mask(model, rgm_path; singular=false)
    idx = findall(have_volume_mask)
    exprs = map(idx) do id
        show_expression_x(model, rgm_path[id];log_space=log_space)[observe_x_idx].rhs
    end
    edges = map(@view idx[1:end-1]) do i
        rgm_from = rgm_path[i]
        rgm_to   = rgm_path[i+1]
        edge = show_interface(model, rgm_from, rgm_to,change_qK_idx;log_space=log_space).rhs
        return edge
    end
    return (exprs, edges)
end

function show_expression_path(grh::SISO_graph, pth_idx, observe_x; kwargs...)
    return show_expression_path(grh.bn, grh.rgm_paths[pth_idx], grh.change_qK_idx, observe_x; kwargs...)
end



function render_array(M::AbstractArray)
    A = Array{Any}(M)
    f(x) = begin
            a = try 
                    Int(round(x;digits=3))
                catch
                    round(x;digits=5)
                end
            a == 0 ? nothing : a
        end
    A = f.(A)
    return latexify(A)
end
