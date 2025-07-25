#----------------------------------------------------------Symbolics calculation fucntions-----------------------------------------------------------
function ∂logqK_∂logx_sym(Bnc::Bnc; show_x_space::Bool=false)::Matrix{Num}
    
    # if any(isnothing, [Bnc.x_sym, Bnc.q_sym])
    #     fill_bnc_symbolic!(Bnc) # Ensure all symbolic variables are defined
    # end

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

function ∂logx_∂logqK_sym(Bnc::Bnc;show_x_space::Bool=false)::Matrix{Num}
    # Calculate the symbolic derivative of log(qK) with respect to log(x)
    # This function is used for symbolic calculations, not numerical ones.
    return inv(∂logqK_∂logx_sym(Bnc; show_x_space=show_x_space)).|> simplify
end




show_x_space_conditions(Bnc::Bnc; regime::Regime) = show_x_space_conditions(Bnc; regime=regime.regime)
function show_x_space_conditions(Bnc::Bnc; regime::Vector{Int})
    # Show the conditions for the x space for the given regime.
    return x_ineq_mtx(Bnc; regime=regime) * log10.(Bnc.x_sym) .< 0
end

function show_qK_space_conditions(Bnc::Bnc; regime::Regime)
    regime.singularity != 0 ? @error("Regime is singular, cannot show qK space conditions") : nothing
    A = x_ineq_mtx(Bnc; regime=regime)
    Mtd_N = regime.logder_qK_x
    return simplify.(A * (Mtd_N \ log10.([Bnc.q_sym ./ a; Bnc.K_sym])) .< 0)
end

function show_qK_space_conditions(Bnc::Bnc; regime::Vector{Int})
    A = x_ineq_mtx(Bnc; regime=regime)
    Mtd, a = Mtd_a_from_regime(Bnc, regime; check=true)
    Mtd_N = [Mtd; Bnc.N]
    # @show Mtd
    # return simplify.(A / Mtd_N * log10.([Bnc.q_sym ./ a; Bnc.K_sym]) .< 0)
    return simplify.(A * (Mtd_N \ log10.([Bnc.q_sym ./ a; Bnc.K_sym])) .< 0)
end