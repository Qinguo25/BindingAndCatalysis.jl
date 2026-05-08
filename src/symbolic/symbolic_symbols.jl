x_sym(args...) = get_binding_network(args...).x_sym
q_sym(args...) = get_binding_network(args...).q_sym
K_sym(args...) = get_binding_network(args...).K_sym
qK_sym(args...) = [q_sym(args...); K_sym(args...)]

k_sym(args...) = get_catalysis_network(args...).k_sym
v_sym(args...) = get_catalysis_network(args...).v_sym

function q_cat_sym(args...)
    bn = get_binding_network(args...)
    cn = get_catalysis_network(args...)
    return bn.q_sym[1:cn.r_v]
end

function w_sym(args...)
    bn = get_binding_network(args...)
    cn = get_catalysis_network(args...)
    return bn.q_sym[cn.r_v + 1:bn.d]
end

@inline xk_sym(args...) = [x_sym(args...); k_sym(args...)]
@inline qKk_sym(args...) = [q_sym(args...); K_sym(args...); k_sym(args...)]
@inline wKk_sym(args...) = [w_sym(args...); K_sym(args...); k_sym(args...)]

_to_plain_symbol(x::Symbol) = x
_to_plain_symbol(x::Num) = Symbol(x.val.name)
_to_plain_symbol(x) = Symbol(x)

x_symbol(args...) = _to_plain_symbol.(x_sym(args...))
q_symbol(args...) = _to_plain_symbol.(q_sym(args...))
K_symbol(args...) = _to_plain_symbol.(K_sym(args...))
qK_symbol(args...) = _to_plain_symbol.(qK_sym(args...))
k_symbol(args...) = _to_plain_symbol.(k_sym(args...))
v_symbol(args...) = _to_plain_symbol.(v_sym(args...))
q_cat_symbol(args...) = _to_plain_symbol.(q_cat_sym(args...))
w_symbol(args...) = _to_plain_symbol.(w_sym(args...))
xk_symbol(args...) = _to_plain_symbol.(xk_sym(args...))
qKk_symbol(args...) = _to_plain_symbol.(qKk_sym(args...))
wKk_symbol(args...) = _to_plain_symbol.(wKk_sym(args...))

@inline _time_sym() = Symbolics.variable(:t)
@inline _d_dt(syms) = Symbolics.Differential(_time_sym()).(syms)

function _flux_sym(args...)
    cn = get_catalysis_network(args...)
    flux_monomials = handle_log_weighted_sum(cn.Π, x_sym(args...))
    return k_sym(args...) .* flux_monomials
end

q_sym(grh::SIMOPaths, args...) = begin
    bn = get_binding_network(grh)
    if grh.change_qK_idx <= bn.d
        deleteat!(copy(bn.q_sym), grh.change_qK_idx)
    else
        bn.q_sym
    end
end

K_sym(grh::SIMOPaths, args...) = begin
    bn = get_binding_network(grh)
    if grh.change_qK_idx > bn.d
        deleteat!(copy(bn.K_sym), grh.change_qK_idx - bn.d)
    else
        bn.K_sym
    end
end

function ∂logqK_∂logx_sym(Bnc::Bnc; show_x_space::Bool=false)::Matrix{Num}
    q = show_x_space ? Bnc.L * Bnc.x_sym : Bnc.q_sym
    return [
        transpose(Bnc.x_sym) .* Matrix(Bnc.L) ./ q
        Matrix(Bnc.N)
    ]
end

logder_qK_x_sym(args...; kwargs...) = ∂logqK_∂logx_sym(args...; kwargs...)

function ∂logx_∂logqK_sym(Bnc::Bnc; show_x_space::Bool=false)::Matrix{Num}
    return inv(∂logqK_∂logx_sym(Bnc; show_x_space=show_x_space)) .|> Symbolics.simplify
end

logder_x_qK_sym(args...; kwargs...) = ∂logx_∂logqK_sym(args...; kwargs...)
