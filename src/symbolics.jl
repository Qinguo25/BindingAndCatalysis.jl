#----------------------------------------------------------Symbolics calculation fucntions-----------------------------------------------------------

# General method for binding network
x_sym(args...)=get_binding_network(args...).x_sym
q_sym(args...)=get_binding_network(args...).q_sym
K_sym(args...)=get_binding_network(args...).K_sym
qK_sym(args...)= [q_sym(args...); K_sym(args...)]

#special api for SISOPaths
q_sym(grh::SISOPaths,args...)= begin
    bn = get_binding_network(grh)
    q_sym = if grh.change_qK_idx <= bn.d
        deleteat!(copy(bn.q_sym), grh.change_qK_idx)
    else
        bn.q_sym
    end
    return q_sym
end
K_sym(grh::SISOPaths,args...)= begin
    bn = get_binding_network(grh)
    K_sym = if grh.change_qK_idx > bn.d
        deleteat!(copy(bn.K_sym), grh.change_qK_idx - bn.d)
    else
        bn.K_sym
    end
    return K_sym
end

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
logder_qK_x_sym(args...;kwargs...) = ∂logqK_∂logx_sym(args...;kwargs...)

"""
Symbolicly calculate ∂logx/∂logqK
"""
function ∂logx_∂logqK_sym(Bnc::Bnc;show_x_space::Bool=false)::Matrix{Num}
    # Calculate the symbolic derivative of log(qK) with respect to log(x)
    # This function is used for symbolic calculations, not numerical ones.
    return inv(∂logqK_∂logx_sym(Bnc; show_x_space=show_x_space)).|> Symbolics.simplify
end
logder_x_qK_sym(args...;kwargs...) = ∂logx_∂logqK_sym(args...;kwargs...)


#---------------------------------------------------------
#   Below are regimes associtaed symbolic functions
#---------------------------------------------------------

"""
handle C log(sym) + C0 >= 0 , polyhedron in log space
"""
function show_condition_poly(C::AbstractMatrix{<:Real},
                        C0::AbstractVector{<:Real},
                        nullity::Integer = 0;
                        syms::AbstractVector{Num},
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
show_condition_poly(poly::Polyhedron; kwargs...)=show_condition_poly(get_C_C0_nullity(poly)...; kwargs...)
show_condition_poly(C_qK::AbstractVector{<:Real},C0_qK::Real,args...;kwargs...)=show_condition_poly(C_qK', [C0_qK], args...; kwargs...)[1]


show_condition_x(args...; kwargs...)= show_condition_poly(get_C_C0_x(args...)...; syms=x_sym(args...), kwargs...)
show_condition_qK(args...; kwargs...)= show_condition_poly(get_C_C0_nullity_qK(args...)...; syms=qK_sym(args...), kwargs...)
show_condition(args...; kwargs...)= show_condition_qK(args...; kwargs...)



"""
Show the symbolic conditions for a given regime path.
"""
function show_condition_path(Bnc::Bnc, path::AbstractVector{<:Integer}, change_qK; kwargs...)
    # we couldn't name it as "show_condition" as "path" will be confused with perms
    # directly calculate the polyhedron for the path, may not useful.
    poly = _calc_polyhedra_for_path(Bnc, path,change_qK)
    syms = copy(qK_sym(Bnc)) |> x->deleteat!(x,locate_sym_qK(Bnc, change_qK)) 
    show_condition_poly(poly; syms=syms, kwargs...)
end
function show_condition_path(grh::SISOPaths, pth_idx; kwargs...)
    bn = get_binding_network(grh)
    poly = get_polyhedron(grh, pth_idx)
    path = get_path(grh, pth_idx; return_idx=true)
    change_qK = grh.change_qK_idx     
    show_condition_path(bn, path, change_qK; kwargs...)
end





"""
handle log(y) = C log(x) + C0
"""
function show_expression_mapping(C::AbstractMatrix{<:Real}, C0::AbstractVector{<:Real}, y, x; log_space::Bool=true,asymptotic::Bool=false)::Vector{Equation}
    if log_space
        expr =  asymptotic ?   log10.(y) .~ C * log10.(x) : log10.(y) .~ C * log10.(x) .+ C0
    else
        expr =  asymptotic ? y .~ handle_log_weighted_sum(C, x) : y .~ handle_log_weighted_sum(C, x,C0)
    end
    return expr 
end
show_expression_mapping(C::AbstractVector{<:Real}, C0::Real, args...;kwargs...)=show_expression_mapping(C', [C0], args...;kwargs...)[1]

show_expression_x(args...;kwargs...)= begin
    bn = get_binding_network(args...)
    y = x_sym(bn)
    x = qK_sym(bn)
    show_expression_mapping(get_H_H0(args...), y,x; kwargs...)
end

show_expression_qK(args...;kwargs...)= begin
    bn = get_binding_network(args...)
    y = qK_sym(bn)
    x = x_sym(bn)
    show_expression_mapping(get_M_M0(args...), y,x; kwargs...)
end


show_dominant_condition(args...;log_space=false, kwargs...)= begin
    bn = get_binding_network(args...)
    y = q_sym(bn)
    x = x_sym(bn)
    show_expression_mapping(get_P_P0(args...)..., y,x; log_space=log_space,kwargs...)
end
show_conservation(Bnc::Bnc)=Bnc.q_sym .~ Bnc._L_sparse * Bnc.x_sym
show_equilibrium(Bnc::Bnc;log_space::Bool=true) = show_expression_mapping(Bnc.N, zeros(Int,Bnc.r), Bnc.K_sym, Bnc.x_sym; log_space=log_space)










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
Format a single path element for display.

- If it's a Real and close to an integer (after rounding), show as Int.
- Otherwise show rounded Real (digits).
- Non-real values fall back to `repr`.
"""
@inline function _fmt_elem(x; digits::Int=3)::String
    if x isa Real
        if isnan(x)
            return "NaN"
        end
        xr = round(Float64(x); digits=digits)
        # “接近整数就显示整数”，避免 try/catch
        if isfinite(xr) && isapprox(xr, round(xr); atol=10.0^(-digits), rtol=0)
            return string(Int(round(xr)))
        else
            return string(xr)
        end
    else
        return repr(x)
    end
end

"""
Format a path vector as an arrow-separated string.

Example:
    format_arrow([1,0,-1]; prefix="x") == "x1 → x0 → x-1"
"""
function format_arrow(path::AbstractVector; prefix::AbstractString="", digits::Int=3)::String
    isempty(path) && return ""
    parts = Vector{String}(undef, length(path))
    @inbounds for i in eachindex(path)
        parts[i] = prefix * _fmt_elem(path[i]; digits=digits)
    end
    return join(parts, " → ")
end



"""
A normalized row for printing:
- id:     group/path id (any displayable object)
- path:   the actual path vector Could be either regime path or Reaction order path
- volume: nothing | Float64 | Tuple(value, err)
"""
struct PathRow{I,P,V}
    id::I
    path::P
    volume::V
end

# 4.1 用户直接给 paths（无 volume），id 自动用 1:N
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



"""
Print rows of paths in aligned columns.

Keyword args:
- prefix: prefix before each node in arrow (e.g. "#")
- digits: rounding digits for real display
- io: output stream
"""
function print_paths(rows::AbstractVector{<:PathRow};
    prefix::AbstractString="",
    digits::Int=3,
    io::IO=stdout,
)
    isempty(rows) && return nothing

    # 动态列宽：比写死 15/30 更不容易“错位”
    id_strs    = [repr(r.id) for r in rows]
    path_strs  = [format_arrow(r.path; prefix=prefix, digits=digits) for r in rows]

    id_width   = max(8, maximum(length.(id_strs)))     # 至少 8
    path_width = max(10, maximum(length.(path_strs)))  # 至少 10

    for (r, id_s, path_s) in zip(rows, id_strs, path_strs)
        if r.volume === nothing
            Printf.@printf(io, "Path %-*s  %-*s\n", id_width, id_s, path_width, path_s)
        else#if  r.volume isa Volume
            @assert typeof(r.volume) <: Volume 
            v = r.volume.mean
            e = sqrt(r.volume.var)
            Printf.@printf(io, "Path %-*s  %-*s  Volume: %.4f ± %.4f\n", id_width, id_s, path_width, path_s, v, e)
        end
    end
    return nothing
end
# 6.1 直接给 paths



"""
print_paths(paths; prefix, ids, volumes, digits, io)
"""

print_paths(paths::AbstractVector{<:AbstractVector}; volumes, ids, kwargs...) =
    print_paths(_normalize_rows(paths; volumes=volumes, ids=ids); kwargs...)


print_path(path::AbstractVector; id = nothing, volume = nothing, kwargs...) =
    print_paths(
        _normalize_rows(
            [path]; 
            ids = id === nothing ? nothing : [id], 
            volumes = volume === nothing ? nothing : [volume]
        ); kwargs...)




"""
show Reaction Order for a path.
"""



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

"""
Normally display the expression of the interface, when denoting the idx, whill express the interface in terms of that qK idx.
"""
function show_interface(Bnc::Bnc, from,to, change_idx::Union{Nothing,Integer}=nothing; kwargs...)
    C, C0 = get_interface(Bnc,from,to) # C' log qK + C0 =0
    if isnothing(change_idx)
        return show_condition_poly(C, C0, 1 , qK_sym(Bnc) ;kwargs...)
    else
        return solve_sym_expr(C,C0, qK_sym(Bnc), change_idx;kwargs...)
    end
end





function show_expression_path(grh::SISOPaths, pth_idx::Integer; observe_x=nothing, kwargs...)
    bn = get_binding_network(grh)

    observe_x_idx = isnothing(observe_x) ? (1:bn.n) : locate_sym_x.(Ref(bn), observe_x)
    change_qK_idx = grh.change_qK_idx

    xsym = x_sym(bn)[observe_x_idx]
    qKsym = qK_sym(bn)
    change_sym = qKsym[change_qK_idx]

    
    H_H0, rgm_interface = get_expression_path(grh, pth_idx; observe_x = observe_x_idx)

    expr_sym = let 
        exprs = Vector{Any}(undef, length(H_H0))
        for (i, (H_row, H0_val)) in enumerate(H_H0)
            if isnothing(H0_val)# singular regime, expression is just H_row * log(qK)
                exprs[i] = map(H_row[:,change_qK_idx]) do i 
                        if abs(i) < 1e-6
                            nothing
                        else
                            if i > 0
                                :↑
                            else
                                :↓
                            end
                        end
                    end
            else
                # @show H_row, H0_val,qKsym, xsym
                exprs[i] = show_expression_mapping(H_row, H0_val, xsym, qKsym  ; kwargs...)
            end
        end
        exprs
    end
                
    interface = rgm_interface .|> x -> solve_sym_expr(x..., qKsym, change_qK_idx; kwargs...)
    
    for i in eachindex(expr_sym)
        if i == 1
            display(change_sym < interface[1].rhs)
        elseif i == length(expr_sym)
            display(change_sym > interface[end].rhs)
        else
            display((change_sym > interface[i-1].rhs) & (change_sym < interface[i].rhs))
        end
        display(expr_sym[i])
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

show_expression_path(grh::SISOPaths, pth_idx, observe_x; kwargs...)=show_expression_path(get_binding_network(grh), grh.rgm_paths[pth_idx], grh.change_qK_idx, observe_x; kwargs...)
