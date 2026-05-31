#========================================================================================#
# Binding network construction
#========================================================================================#

"""
    Bnc(; N=nothing, L=nothing, x_sym=nothing, q_sym=nothing, K_sym=nothing,
        Γ=nothing, Π=nothing, F=nothing, F0=nothing, k_sym=nothing,
        v_sym=nothing, w_sym=nothing, x_picked=nothing, q_picked=nothing) -> Bnc

Construct a binding network model from stoichiometry (`N`) or conservation (`L`)
matrices and optional symbol metadata. Catalysis data can be attached through
`Γ`, `Π`, `k_sym`, `v_sym`, `w_sym`, `x_picked`, and `q_picked`.

# Keyword Arguments
- `N`: Stoichiometry matrix (reactions × species).
- `L`: Conservation matrix (totals × species).
- `x_sym`: Symbols for species concentrations.
- `q_sym`: Symbols for total concentrations.
- `K_sym`: Symbols for binding constants.
- `Γ`: Catalysis change matrix in qK space.
- `Π`: Catalysis index and coefficient matrix. If omitted while `Γ` and
  `x_picked` are supplied, defaults to the identity on the picked species.
- `F`, `F0`: Optional affine constraints on old catalysis rate constants,
  `log k_old = F * log k + F0`.
- `k_sym`: Symbols for catalysis rate constants.
- `v_sym`: Symbols for catalysis fluxes.
- `w_sym`: Symbols for the new conservation quantities induced by catalysis.
- `x_picked`: Picked species for catalysis flux monomials.
- `q_picked`: Picked qK coordinates affected by catalysis.

# Returns
- A `Bnc` model with derived matrices and caches initialized.
"""
function Bnc(;
    N=nothing,
    L=nothing,
    x_sym=nothing,
    q_sym=nothing,
    K_sym=nothing,
    kwargs...
)::Bnc
    # If N is not provided, derive it from L; otherwise validate row rank.
    N = isnothing(N) ? N_from_L(L) : N
    row_idx = independent_row_idx(N)
    old_r = size(N, 1)
    r = length(row_idx)

    if isnothing(L)
        if r != old_r
            @warn "N has been reduced from $old_r to $r rows because rows were linearly dependent."
            N = N[row_idx, :] # reduce N to independent rows
            if !isnothing(K_sym) && length(K_sym) == old_r
                K_sym = K_sym[row_idx]
            end
        end
        L = L_from_N(N)
    else # L is provided
        if r != old_r && old_r + size(L, 1) == size(N, 2)
            @warn "N is not full row rank and can't be reduced; numerical issues could happen."
        end
    end

    r, n = size(N)
    d = size(L, 1)

    x_sym = isnothing(x_sym) ? Symbolics.variables(:x, 1:n) : name_converter(x_sym)
    q_sym = isnothing(q_sym) ? Symbolics.variables(:q, 1:d) : name_converter(q_sym)
    K_sym = isnothing(K_sym) ? Symbolics.variables(:K, 1:r) : name_converter(K_sym)

    model = Bnc{Int}(N, L, x_sym, q_sym, K_sym, nothing)
    _warn_if_free_energy_qK2x_disabled(model)
    update_catalysis!(model; kwargs...)
    return model
end


#========================================================================================#
# qK2x method selection
#========================================================================================#

function _free_energy_qK2x_allowed(model::Bnc; tol::Real=0)
    residual = dropzeros(model.L * transpose(model.N))
    if tol <= 0
        return nnz(residual) == 0
    end
    nz = nonzeros(residual)
    return isempty(nz) || maximum(abs.(nz)) <= tol
end

_default_method(model::Bnc) = _free_energy_qK2x_allowed(model) ? :free_energy : :homotopy

function _resolve_qK2x_method(model::Bnc, method::Union{Symbol,Nothing})
    resolved = isnothing(method) ? _default_method(model) : method
    return (resolved === :free_energy && !_free_energy_qK2x_allowed(model)) ? :homotopy : resolved
end

function _warn_if_free_energy_qK2x_disabled(model::Bnc)
    _free_energy_qK2x_allowed(model) && return nothing
    @warn "L * N' is nonzero; qK2x defaults to :homotopy and :free_energy requests are redirected to :homotopy for this model."
    return nothing
end


#========================================================================================#
# Catalysis initialization
#========================================================================================#

"""
    update_catalysis!(bnc::Bnc; Γ=nothing, Π=nothing, F=nothing, F0=nothing, k_sym=nothing, v_sym=nothing, w_sym=nothing, x_picked=nothing, q_picked=nothing) -> nothing

Attach or update catalysis data on a `Bnc` model in-place.

# Arguments
- `bnc`: Binding network model to update.

# Keyword Arguments
- `Γ`: Catalysis change matrix in qK space.
- `Π`: Catalysis index and coefficient matrix. If omitted while `Γ` and
  `x_picked` are supplied, defaults to the identity on the picked species.
- `F`, `F0`: Optional affine constraints on old catalysis rate constants,
  `log k_old = F * log k + F0`.
- `k_sym`: Symbols for catalysis rate constants.
- `v_sym`: Symbols for catalysis fluxes, where `log v = Π log x + F log k + F0`.
- `w_sym`: Symbols for the new conservation quantities induced by catalysis.
- `x_picked`: Picked species for catalysis flux monomials.
- `q_picked`: Picked qK coordinates affected by catalysis.

# Returns
- `nothing`. The supplied `bnc` is updated in-place.
"""
function update_catalysis!(
    model::Bnc;
    Γ::Union{<:AbstractMatrix{Int},Nothing}=nothing,
    Π::Union{<:AbstractMatrix{Int},Nothing}=nothing,
    F::Union{<:AbstractMatrix{<:Real},Nothing}=nothing,
    F0::Union{<:AbstractVector{<:Real},Nothing}=nothing,
    k_sym::Union{<:AbstractVector,Nothing}=nothing,
    v_sym::Union{<:AbstractVector,Nothing}=nothing,
    w_sym::Union{<:AbstractVector,Nothing}=nothing,
    x_picked::Union{<:AbstractVector,Nothing}=nothing,
    q_picked::Union{<:AbstractVector,Nothing}=nothing,
)
    if isnothing(Γ) && isnothing(Π)
        return nothing
    elseif isnothing(Γ)
        throw(ArgumentError("You shall provide Γ when providing Π."))
    elseif isnothing(Π)
        isnothing(x_picked) && throw(ArgumentError("You shall provide Π, or provide x_picked so Π can default to the identity on picked species."))
        n_flux = size(Γ, 2)
        length(x_picked) == n_flux || throw(ArgumentError("When Π is omitted, length(x_picked) must match the number of catalysis fluxes, got $(length(x_picked)) and $n_flux."))
        Π = Matrix{Int}(I, n_flux, n_flux)
    end

    Π = if isnothing(x_picked)
        Π
    else
        x_idx = locate_sym_x.(Ref(model), x_picked)
        Π2 = zeros(Int, size(Π, 1), model.n)
        for (i, x) in enumerate(x_idx)
            Π2[:, x] .= Π[:, i]
        end
        Π2
    end

    if !isnothing(q_picked)
        q_idx = locate_sym_qK.(Ref(model), q_picked)
        new_order = vcat(q_idx, setdiff(1:model.d, q_idx))
        _change_q_L_order!(model, new_order)
        _remove_regime_data!(model)
    else
        @info "q_cat is not picked, the catalysis will involve the first r_v q by default"
    end

    n_old_k = size(Π, 1)
    n_independent_k = isnothing(F) ? n_old_k : size(F, 2)
    k_sym = isnothing(k_sym) ? Symbolics.variables(:k, 1:n_independent_k) : name_converter(k_sym)
    v_sym = isnothing(v_sym) ? Symbolics.variables(:v, 1:size(Π, 1)) : name_converter(v_sym)
    w_sym = isnothing(w_sym) ? nothing : name_converter(w_sym)
    model.catalysis = CatalysisData(model, Γ, Π, k_sym, w_sym, v_sym, F, F0)
    _warn_if_free_energy_qK2x_disabled(model)
    return nothing
end


#========================================================================================#
# Catalysis-induced conservation basis repair
#========================================================================================#

function fix_bn_catalysis!(bn::Bnc, new_ord::Vector{Int}, L_Γ::AbstractMatrix{Int}, w_sym)
    d_dep = size(L_Γ, 2)
    d_cat_full = length(new_ord)
    d_cat = d_cat_full - d_dep

    if new_ord != collect(1:length(new_ord))
        _change_q_L_order!(bn, new_ord)

        @info "q is reordered to make catalysis-involving species first"
    end

    if d_dep > 0
        L_w = L_Γ' * bn.L[1:d_cat_full, :]
        if any(L_w .< 0)
            repaired = _nonnegative_conservation_basis(L_Γ)
            if !isnothing(repaired)
                repaired_L_w = repaired' * bn.L[1:d_cat_full, :]
                if all(repaired_L_w .>= 0)
                    L_Γ = repaired
                    L_w = repaired_L_w
                end
            end
        end

        @info "New conservation forms as catalysis involves"
        old_q_sym = copy(bn.q_sym[1:d_cat_full])
        w_names = isnothing(w_sym) ? Symbolics.variables(:w, 1:d_dep) : w_sym
        for j in 1:d_dep
            terms = String[]
            for i in 1:d_cat_full
                coeff = L_Γ[i, j]
                coeff == 0 && continue
                qname = string(old_q_sym[i])
                term = if coeff == 1
                    qname
                elseif coeff == -1
                    "-" * qname
                else
                    string(coeff) * "*" * qname
                end
                push!(terms, term)
            end
            relation = isempty(terms) ? "0" : join(terms, " + ")
            @info "$(w_names[j]) = $relation"
        end
        # Rename the dependent q block and replace its conservation rows with L_w.
        bn.q_sym[(d_cat + 1):d_cat_full] = w_names
        @assert all(L_w .>= 0) "L_w should be non-negative"
        bn.L[(d_cat + 1):d_cat_full, :] = L_w
    end

    dropzeros!(bn.L)
    _remove_regime_data!(bn) # remove the cached regime data, since the regimes will be changed.
    _rebuild_helper!(bn) # rebuild the helper parameters since L has been changed.
    return nothing
end


#========================================================================================#
# Cache invalidation and lazy helpers
#========================================================================================#

@inline function _change_q_L_order!(bn::Bnc, new_ord::Vector{Int})
    bn.q_sym[1:length(new_ord)] = bn.q_sym[new_ord]
    bn.L[1:length(new_ord), :] = bn.L[new_ord, :]
end

function _primitive_int_vector(v::AbstractVector{<:Integer})
    g = foldl(gcd, abs.(v); init=0)
    g == 0 && return collect(v)
    return collect(div.(v, g))
end

function _row_components_from_basis(B::AbstractMatrix{<:Integer})
    n = size(B, 1)
    parent = collect(1:n)
    find_root(i) = begin
        while parent[i] != i
            parent[i] = parent[parent[i]]
            i = parent[i]
        end
        i
    end
    union_root!(a, b) = begin
        ra = find_root(a)
        rb = find_root(b)
        ra == rb || (parent[rb] = ra)
        nothing
    end

    for j in 1:size(B, 2)
        rows = findall(!iszero, B[:, j])
        isempty(rows) && continue
        first_row = first(rows)
        for row in rows[2:end]
            union_root!(first_row, row)
        end
    end

    groups = Dict{Int, Vector{Int}}()
    for i in 1:n
        push!(get!(groups, find_root(i), Int[]), i)
    end
    return collect(values(groups))
end

function _nonnegative_conservation_basis(L_Γ::AbstractMatrix{<:Integer}; max_coeff::Int=3)
    n_rows, n_basis = size(L_Γ)
    n_basis == 0 && return Matrix{Int}(undef, n_rows, 0)

    candidates = Vector{Vector{Int}}()
    seen = Set{Tuple{Vararg{Int}}}()
    components = _row_components_from_basis(L_Γ)

    for radius in 1:max_coeff
        ranges = ntuple(_ -> -radius:radius, n_basis)
        for coeff_tuple in Iterators.product(ranges...)
            all(iszero, coeff_tuple) && continue
            y = _primitive_int_vector(vec(Matrix{Int}(L_Γ) * collect(Int, coeff_tuple)))
            any(!iszero, y) || continue
            all(>=(0), y) || continue
            support = findall(!iszero, y)
            any(component -> all(in(component), support), components) || continue
            key = Tuple(y)
            key in seen && continue
            push!(seen, key)
            push!(candidates, y)
        end

        component_rank(y) = begin
            support = findall(!iszero, y)
            idx = findfirst(component -> all(in(component), support), components)
            isnothing(idx) ? typemax(Int) : minimum(components[idx])
        end
        sort!(candidates; by = y -> (component_rank(y), findfirst(!iszero, y), count(!iszero, y), sum(abs, y), y))
        selected = Vector{Vector{Int}}()
        for y in candidates
            old_basis = isempty(selected) ? zeros(Int, n_rows, 0) : hcat(selected...)
            new_basis = hcat(old_basis, y)
            rank(Matrix{Float64}(new_basis)) > length(selected) || continue
            push!(selected, y)
            length(selected) == n_basis && return hcat(selected...)
        end
    end

    return nothing
end

@inline function _rebuild_helper!(bn::Bnc)
    bn.direction = sign(det([bn.L; bn.N])) # recalculate the direction, since L has been changed.
    bn.IntegrationHelper = nothing # lazily rebuild integration helper on first numerical integration.
    bn._L_helper = _build_matrix_helper(bn.L)
    return nothing
end

@inline function _integration_helper!(bn::Bnc)
    helper = bn.IntegrationHelper
    if !isnothing(helper)
        return helper
    end

    lock(bn._integration_helper_lock)
    try
        helper = bn.IntegrationHelper
        if isnothing(helper)
            helper = calc_integration_helper(bn.L, bn.N)
            bn.IntegrationHelper = helper
        end
        return helper
    finally
        unlock(bn._integration_helper_lock)
    end
end

@inline function _remove_regime_data!(bn::Bnc{T}) where T
    bn.BindRegimes = nothing
    bn.BncRegimes = nothing
    bn.vertices_graph = nothing
    bn._vertices_Nρ_inv_dict = Dict{Vector{T}, Tuple{SparseMatrixCSC{Float64, Int},T}}()
    bn._regimes_affine_ready = false
    return nothing
end


#========================================================================================#
# Display helpers
#========================================================================================#

function _print_bnc_summary(io::IO, model::Bnc; final_newline::Bool=true)
    println(io, "----------Binding Network Summary:-------------")
    println(io, "Number of species (n): ", model.n)
    println(io, "Number of conserved quantities (d): ", model.d)
    println(io, "Number of reactions (r): ", model.r)
    println(io, "L matrix: ", model.L)
    println(io, "N matrix: ", model.N)
    println(io, "Direction of binding reactions: ", model.direction > 0 ? "forward" : "backward")
    println(io, "Catalysis involved: ", isnothing(model.catalysis) ? "No" : "Yes")
    println(io, "Regimes constructed: ", is_bind_regimes_built(model) ? "Yes" : "No")

    if is_bind_regimes_built(model)
        regimes = _bind_regimes_data(model)
        regime_counts = countmap((rgm.is_asymptotic, rgm.nullity > 0) for rgm in regimes)
        println(io, "Number of regimes: ", length(regimes))
        println(io, "  - Invertible + Asymptotic: ", get(regime_counts, (true, false), 0))
        println(io, "  - Singular +  Asymptotic: ", get(regime_counts, (true, true), 0))
        println(io, "  - Invertible +  Non-Asymptotic: ", get(regime_counts, (false, false), 0))
        println(io, "  - Singular +  Non-Asymptotic: ", get(regime_counts, (false, true), 0))
    end

    final_newline ? println(io, "-----------------------------------------------") :
                    print(io, "-----------------------------------------------")
    return nothing
end

"""
    summary(bnc::Bnc) -> String

Print a summary of a binding network model to standard output.
"""
function summary(model::Bnc)
    _print_bnc_summary(stdout, model; final_newline=true)
    return nothing
end

"""
    show(io::IO, ::MIME"text/plain", bnc::Bnc)

Pretty-print a `Bnc` model in plain text contexts.
"""
function show(io::IO, ::MIME"text/plain", bnc::Bnc)
    _print_bnc_summary(io, bnc; final_newline=false)
end
