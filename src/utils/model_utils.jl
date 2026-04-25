"""
    randomize(n::Int, size; kwargs...) -> Array{Vector{Float64}}

Generate an array of random vectors in log space.
"""
function randomize(n::Int, size; kwargs...)::Array{Vector{Float64}}
    N = Array{Vector{Float64}}(undef, size...)
    Threads.@threads for i in eachindex(N)
        N[i] = randomize(n; kwargs...)
    end
    return N
end

"""
    randomize(n::Int; log_lower=-6, log_upper=6, output_logspace=true) -> Vector{Float64}

Generate a random vector with entries sampled uniformly in log10 space.
"""
function randomize(n::Int; log_lower=-6, log_upper=6, output_logspace::Bool=true)::Vector{Float64}
    if !output_logspace
        exp10.(rand(n) .* (log_upper - log_lower) .+ log_lower)
    else
        rand(n) .* (log_upper - log_lower) .+ log_lower
    end
end
randomize(Bnc::Bnc, size; kwargs...) = randomize(Bnc.n, size; kwargs...)

"""
    N_generator(r::Int, n::Int; min_binder=2, max_binder=2) -> Matrix{Int}

Generate a random stoichiometry matrix.
"""
function N_generator(r::Int, n::Int; min_binder::Int=2, max_binder::Int=2)::Matrix{Int}
    @assert n > r "n must be greater than r"
    @assert min_binder >= 1 && max_binder >= min_binder "min_binder and max_binder must be at least 1"
    @assert min_binder <= n - r "min_binder must be smaller than n-r"
    d = n - r
    N = [zeros(r, d) -I(r)]
    Threads.@threads for i in 1:r
        idx = sample(1:d + i - 1, rand(min_binder:max_binder); replace=true)
        for j in idx
            N[i, j] += 1
        end
    end
    return N
end

"""
    L_generator(d::Int, n::Int; kwargs...) -> Matrix{Int}

Generate a random conservation matrix `L`.
"""
function L_generator(d::Int, n::Int; kwargs...)::Matrix{Int}
    N = N_generator(n - d, n; kwargs...)
    L = L_from_N(N)
    return L
end

"""
    locate_sym(syms, target_sym) -> Int

Locate a symbol in a list of Symbolics variables.
"""
function locate_sym(syms, target_sym)
    target_sym = Symbol(target_sym)
    idx = findfirst(x -> x.val.name == target_sym, syms)
    isnothing(idx) && throw(ArgumentError("Unknown symbol $(repr(target_sym)). Available symbols are $(string.(syms))."))
    return idx
end

"""
    locate_sym(syms, target_sym::Integer) -> Integer

Return the provided index directly for convenience.
"""
locate_sym(syms, target_sym::Integer) = target_sym
locate_sym_x(model::Bnc, target_sym) = locate_sym(x_sym(model), target_sym)
locate_sym_qK(model::Bnc, target_sym) = locate_sym(qK_sym(model), target_sym)
