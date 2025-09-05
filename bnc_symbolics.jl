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


function show_x_space_constrains(Bnc::Bnc, perm;log_space::Bool=true,asymptotic::Bool=false)
    C_x, C_0 =  get_C_C0_x!(Bnc,perm)
    # Show the conditions for the x space for the given regime.
    if log_space
        return asymptotic ? C_x * log10.(Bnc.x_sym) .>0 : C_x * log10.(Bnc.x_sym).+C_0 .> 0
    else
        return handle_log_weighted_sum.(C_x * log10.(Bnc.x_sym).+C_0) .> 1
    end
end

function show_qK_space_constrains(Bnc::Bnc, perm;log_space::Bool=true,asymptotic::Bool=false)
    C_qK, C0_qK = get_C_C0_qK!(Bnc, perm)
    if log_space
        return asymptotic ? C_qK * log10.([Bnc.q_sym; Bnc.K_sym]) .> 0 : C_qK * log10.([Bnc.q_sym; Bnc.K_sym]) .+ C0_qK .> 0
    else
        return handle_log_weighted_sum.(C_qK * log10.([Bnc.q_sym; Bnc.K_sym]) .+ C0_qK) .> 1
    end
end

