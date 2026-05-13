"""
    Bnc(; N=nothing, L=nothing, x_sym=nothing, q_sym=nothing, K_sym=nothing,
        Γ=nothing, Π=nothing, k_sym=nothing, v_sym=nothing, w_sym=nothing, cat_x_idx=nothing) -> Bnc

Construct a binding network model from stoichiometry (`N`) or conservation (`L`)
matrices and optional symbol metadata. Catalysis data can be attached through
`Γ`, `Π`, `k_sym`, `v_sym`, and `w_sym`.

# Keyword Arguments
- `N`: Stoichiometry matrix (reactions × species).
- `L`: Conservation matrix (totals × species).
- `x_sym`: Symbols for species concentrations.
- `q_sym`: Symbols for total concentrations.
- `K_sym`: Symbols for binding constants.
- `Γ`: Catalysis change matrix in qK space.
- `Π`: Catalysis index and coefficient matrix.
- `k_sym`: Symbols for catalysis rate constants.
- `v_sym`: Symbols for catalysis fluxes.
- `w_sym`: Symbols for the new conservation quantities induced by catalysis.
- `cat_x_idx`: Index of catalytic species.

# Returns
- A `Bnc` model with derived matrices and caches initialized.
"""
function Bnc(;N=nothing,L=nothing,
    x_sym=nothing,q_sym=nothing,K_sym=nothing,
    kwargs...
)::Bnc
    # if N is not provided, derive it from L, if provided, check its linear indenpendency
    
    N = isnothing(N) ? N_from_L(L) : N
    row_idx = independent_row_idx(N)
    r = length(row_idx)

    if isnothing(L)
        if r != size(N,1) @warn("N has been reduced from $r to $r_new rows, for linear dependent.") : nothing
            N = N[row_idx, :] # reduce N to independent rows
            if !isnothing(K_sym) && length(K_sym) == r
                K_sym = K_sym[row_idx] # reduce K_sym to independent rows 
            end
        end
        L = L_from_N(N)
    else # L is provided
        if r!= size(N,1) && size(N,1) +size(L,1) ==size(N,2)
            @warn "N is not full row rank and can't be reduced, numerical issures could happen"
        end
    end

    r,n = size(N)
    d = size(L,1)
    

    # Call the inner constructor
    # Number of variables in the binding network
    x_sym = isnothing(x_sym) ? Symbolics.variables(:x, 1:n) : name_converter(x_sym) # convert x_sym to a vector of symbols
    q_sym = isnothing(q_sym) ? Symbolics.variables(:q, 1:d) : name_converter(q_sym) # convert q_sym to a vector of symbols
    K_sym = isnothing(K_sym) ? Symbolics.variables(:K, 1:r) : name_converter(K_sym) # convert K_sym to a vector of symbols

    model = Bnc{Int}(N, L, x_sym, q_sym, K_sym, nothing)
    update_catalysis!(model; kwargs...)
    return model
end



"""
    update_catalysis!(bnc::Bnc; Γ=nothing, Π=nothing, k_sym=nothing, v_sym=nothing, w_sym=nothing, cat_x_idx=nothing) -> Bnc

Attach or update catalysis data on a `Bnc` model in-place.

# Arguments
- `bnc`: Binding network model to update.

# Keyword Arguments
- `Γ`: Catalysis change matrix in qK space.
- `Π`: Catalysis index and coefficient matrix.
- `k_sym`: Symbols for catalysis rate constants.
- `v_sym`: Symbols for catalysis fluxes, where `log v = Π log x + log k`.
- `w_sym`: Symbols for the new conservation quantities induced by catalysis.
- `cat_x_idx`: Index of catalytic species.

# Returns
- The updated `bnc`.
"""
function update_catalysis!(model::Bnc;
    Γ::Union{<:AbstractMatrix{Int},Nothing}=nothing,
    Π::Union{<:AbstractMatrix{Int},Nothing}=nothing,
    k_sym::Union{<:AbstractVector,Nothing}=nothing,
    v_sym::Union{<:AbstractVector,Nothing}=nothing,
    w_sym::Union{<:AbstractVector,Nothing}=nothing,
    x_picked::Union{<:AbstractVector,Nothing}=nothing,
    q_picked::Union{<:AbstractVector,Nothing}=nothing,
    )
    if isnothing(Γ) && isnothing(Π)
        return nothing
    else 
        @assert !isnothing(Γ) && !isnothing(Π) "You shall provide both Γ and Π"
    end

    Π = if isnothing(x_picked)
            Π
        else
            x_idx = locate_sym_x.(Ref(model), x_picked)
            Π2 = zeros(Int, (size(Π,1),model.n))
            for (i,x) in enumerate(x_idx)
                Π2[:,x] .= Π[:, i]
            end
            Π2
        end

    if !isnothing(q_picked)
        q_idx = locate_sym_qK.(Ref(model), q_picked)
        new_order = vcat(q_idx, setdiff(1:model.d, q_idx))
        _change_q_L_order!(model, new_order) # reorder the q and L in the model to make the picked q first, since the catalysis will involve the first r_v q.
        _remove_regime_data!(model) # remove the cached regime data, since the regimes will be changed after reordering q and L.
    else
        @info "q_cat is not picked, the catalysis will involve the first r_v q by default"
    end

    k_sym = isnothing(k_sym) ? Symbolics.variables(:k, 1:size(Π,1)) : name_converter(k_sym)
    v_sym = isnothing(v_sym) ? Symbolics.variables(:v, 1:size(Π,1)) : name_converter(v_sym)
    w_sym = isnothing(w_sym) ? nothing : name_converter(w_sym)
    model.catalysis = CatalysisData(model, Γ, Π, k_sym, w_sym, v_sym)
    return nothing
end

function fix_bn_catalysis!(bn::Bnc, new_ord::Vector{Int}, L_Γ::AbstractMatrix{Int}, w_sym)
    d_dep = size(L_Γ,2)
    d_cat_full = length(new_ord)
    d_cat = d_cat_full - d_dep

    if new_ord !== collect(1:length(new_ord)) # no reording should be made
        _change_q_L_order!(bn, new_ord)

        @info "q is reordered to make catalysis-involving species first"
    end

    if d_dep >0
        L_w = L_Γ' * bn.L[1:d_cat_full,:]
        if any(L_w .< 0)
            repaired = _nonnegative_conservation_basis(L_Γ)
            if !isnothing(repaired)
                repaired_L_w = repaired' * bn.L[1:d_cat_full,:]
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
        #update the name of q_sym to make the first d_cat are q_cat, and the rest are q_dep
        bn.q_sym[(d_cat+1):d_cat_full] = w_names
        @assert all(L_w .>=0) "L_w should be non-negative"
        #update L_w to replace L_dep
        bn.L[(d_cat+1):d_cat_full,:] = L_w
    end

    dropzeros!(bn.L)
    _remove_regime_data!(bn) # remove the cached regime data, since the regimes will be changed.
    _rebuild_helper!(bn) # rebuild the helper parameters since L has been changed.
    return nothing
end


@inline function _change_q_L_order!(bn::Bnc, new_ord::Vector{Int})
    bn.q_sym[1:length(new_ord)] = bn.q_sym[new_ord]
    bn.L[1:length(new_ord),:] = bn.L[new_ord, :]
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
    bn.direction = sign(det([bn.L;bn.N])) # recalculate the direction, since L has been changed.
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


"""
    summary(bnc::Bnc) -> String

Print a summary of a binding network model to standard output.
"""
function summary(model::Bnc)
    println("----------Binding Network Summary:-------------")
    println("Number of species (n): ", model.n)
    println("Number of conserved quantities (d): ", model.d)
    println("Number of reactions (r): ", model.r)
    println("L matrix: ", model.L)
    println("N matrix: ", model.N)
    println("Direction of binding reactions: ", model.direction > 0 ? "forward" : "backward")
    catalysis_str = isnothing(model.catalysis) ? "No" : "Yes"
    println("Catalysis involved: ", catalysis_str)
    is_regimes_built = is_bind_regimes_built(model) ? "Yes" : "No"
    println("Regimes constructed: ", is_regimes_built)
    if is_bind_regimes_built(model)
        vertices = _bind_regimes_data(model)
        map = countmap((vtx.is_asymptotic, vtx.nullity > 0) for vtx in vertices)
        println("Number of regimes: ", length(vertices))
        println("  - Invertible + Asymptotic: ", get(map, (true, false), 0))
        println("  - Singular +  Asymptotic: ", get(map, (true, true), 0))
        println("  - Invertible +  Non-Asymptotic: ", get(map, (false, false), 0))
        println("  - Singular +  Non-Asymptotic: ", get(map, (false, true), 0))
    end
    println("-----------------------------------------------")
end

"""
    show(io::IO, ::MIME"text/plain", bnc::Bnc)

Pretty-print a `Bnc` model in plain text contexts.
"""
function show(io::IO, ::MIME"text/plain", bnc::Bnc)
    println(io, "----------Binding Network Summary:-------------")
    println(io, "Number of species (n): ", bnc.n)
    println(io, "Number of conserved quantities (d): ", bnc.d)
    println(io, "Number of reactions (r): ", bnc.r)
    println(io, "L matrix: ", bnc.L)
    println(io, "N matrix: ", bnc.N)
    println(io, "Direction of binding reactions: ", bnc.direction > 0 ? "forward" : "backward")
    catalysis_str = isnothing(bnc.catalysis) ? "No" : "Yes"
    println(io, "Catalysis involved: ", catalysis_str)
    is_regimes_built = is_bind_regimes_built(bnc) ? "Yes" : "No"
    println(io, "Regimes constructed: ", is_regimes_built)
    if is_bind_regimes_built(bnc)
        vertices = _bind_regimes_data(bnc)
        map = countmap((vtx.is_asymptotic, vtx.nullity > 0) for vtx in vertices)
        println(io, "Number of regimes: ", length(vertices))
        println(io, "  - Invertible + Asymptotic: ", get(map, (true, false), 0))
        println(io, "  - Singular +  Asymptotic: ", get(map, (true, true), 0))
        println(io, "  - Invertible +  Non-Asymptotic: ", get(map, (false, false), 0))
        println(io, "  - Singular +  Non-Asymptotic: ", get(map, (false, true), 0))
    end
    print(io, "-----------------------------------------------") # 最后一行可用 print 避免额外空行
end
