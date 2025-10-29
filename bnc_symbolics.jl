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
function show_x_expression(Bnc::Bnc, perm; log_space::Bool=true,asymptotic::Bool=false)
    H,H0 = get_H_H0!(Bnc, perm)
    qK_syms = [Bnc.q_sym; Bnc.K_sym]
    if log_space
        return asymptotic ? log10.(Bnc.x_sym) .~ H * log10.(qK_syms) : log10.(Bnc.x_sym) .~ H * (log10.(qK_syms) .+ H0)
    else
        return asymptotic ? Bnc.x_sym .~ handle_log_weighted_sum(H, qK_syms) : Bnc.x_sym .~ handle_log_weighted_sum(H, qK_syms, H0)
    end
end

function show_dominance_condition(Bnc::Bnc, perm;log_space::Bool=true,asymptotic::Bool=false)
    P,P_0 = get_P_P0!(Bnc, perm)
    # Show the dominance conditions for the given regime.
    if log_space
        return asymptotic ?   log10.(Bnc.q_sym) .~ P * log10.(Bnc.x_sym) : log10.(Bnc.q_sym) .~ P * log10.(Bnc.x_sym) .+ P_0
    else
        return asymptotic ? Bnc.q_sym .~ handle_log_weighted_sum(P, Bnc.x_sym) : Bnc.q_sym .~ handle_log_weighted_sum(P, Bnc.x_sym,P_0)
    end
end

function show_x_space_condition(Bnc::Bnc, perm;log_space::Bool=true,asymptotic::Bool=false)
    C_x, C_0 =  get_C_C0_x!(Bnc,perm)
    # Show the conditions for the x space for the given regime.
    if log_space
        return asymptotic ? C_x * log10.(Bnc.x_sym) .>0 : C_x * log10.(Bnc.x_sym).+C_0 .> 0
    else
        return asymptotic ? handle_log_weighted_sum(C_x, Bnc.x_sym).>1 : handle_log_weighted_sum(C_x, Bnc.x_sym,C0) .> 1
    end
end

function show_qK_space_condition(Bnc::Bnc, perm;log_space::Bool=true,asymptotic::Bool=false)
    C_qK, C0_qK = get_C_C0_qK!(Bnc, perm)
    syms = [Bnc.q_sym; Bnc.K_sym]
    nullity = get_nullity!(Bnc, perm)
    # @show nullity
    if nullity == 0
        if log_space
            # eq = asymptotic ? C_qK[1:nullity,:] * log10.(syms) .~ 0 : C_qK * log10.(syms) .+ C0_qK[1:nullity,:] .~ 0
            return asymptotic ? C_qK * log10.(syms) .> 0 : C_qK * log10.(syms) .+ C0_qK .> 0
        else
            return  asymptotic ? handle_log_weighted_sum(C_qK, syms) .> 1 : handle_log_weighted_sum(C_qK, syms,C0_qK) .> 1
        end
    else
        if log_space
            eq = asymptotic ? C_qK[1:nullity,:] * log10.(syms) .~ 0 : C_qK[1:nullity,:] * log10.(syms) .+ C0_qK[1:nullity,:] .~ 0
            uneq = asymptotic ? C_qK[nullity+1:end,:] * log10.(syms) .> 0 : C_qK[nullity+1:end,:] * log10.(syms) .+ C0_qK[nullity+1:end,:] .> 0
        else
            eq = asymptotic ? handle_log_weighted_sum(C_qK[1:nullity,:], syms) .~ 1 : handle_log_weighted_sum(C_qK[1:nullity,:], syms,C0_qK[1:nullity,:]) .~ 1
            uneq = asymptotic ? handle_log_weighted_sum(C_qK[nullity+1:end,:], syms) .> 1 : handle_log_weighted_sum(C_qK[nullity+1:end,:], syms,C0_qK[nullity+1:end,:]) .> 1
        end
        return vec([eq; uneq])
    end
end

function show_condition_poly(Bnc::Bnc, poly::Polyhedron,change_dir_idx=nothing; log_space::Bool=true,asymptotic::Bool=false)
    nullity = nhyperplanes(poly)
    p = MixedMatHRep(hrep(poly))
    syms = [Bnc.q_sym; Bnc.K_sym]
    isnothing(change_dir_idx) || popat!(syms, change_dir_idx)
    C_qK,C0_qK = -p.A, p.b
    if nullity == 0
        if log_space
            # eq = asymptotic ? C_qK[1:nullity,:] * log10.(syms) .~ 0 : C_qK * log10.(syms) .+ C0_qK[1:nullity,:] .~ 0
            return asymptotic ? C_qK * log10.(syms) .> 0 : C_qK * log10.(syms) .+ C0_qK .> 0
        else
            return  asymptotic ? handle_log_weighted_sum(C_qK, syms) .> 1 : handle_log_weighted_sum(C_qK, syms,C0_qK) .> 1
        end
    else
        if log_space
            eq = asymptotic ? C_qK[1:nullity,:] * log10.(syms) .~ 0 : C_qK[1:nullity,:] * log10.(syms) .+ C0_qK[1:nullity,:] .~ 0
            uneq = asymptotic ? C_qK[nullity+1:end,:] * log10.(syms) .> 0 : C_qK[nullity+1:end,:] * log10.(syms) .+ C0_qK[nullity+1:end,:] .> 0
        else
            eq = asymptotic ? handle_log_weighted_sum(C_qK[1:nullity,:], syms) .~ 1 : handle_log_weighted_sum(C_qK[1:nullity,:], syms,C0_qK[1:nullity,:]) .~ 1
            uneq = asymptotic ? handle_log_weighted_sum(C_qK[nullity+1:end,:], syms) .> 1 : handle_log_weighted_sum(C_qK[nullity+1:end,:], syms,C0_qK[nullity+1:end,:]) .> 1
        end
        return vec([eq; uneq])
    end
end

"""
Symbolic helper function to convert a sum of log10 terms into a product form.
from ∑a log b to log ∏b^a

The final expression contains ∏b^a term.
"""
function handle_log_weighted_sum(C, syms , C0 = nothing)
    rows = size(C,1)
    rst = Vector{Num}(undef, rows)
    C0 = isnothing(C0) ? zeros(Int, rows) : C0
    for i in 1:rows
        rst[i] = syms .^ C[i,:] |> prod |> (x-> x*10^C0[i])
    end
    return rst
end