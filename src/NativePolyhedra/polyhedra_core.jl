struct HalfSpace{TA<:Real,TB<:Real}
    a::Vector{TA}
    β::TB
end

struct HyperPlane{TA<:Real,TB<:Real}
    a::Vector{TA}
    β::TB
end

struct HRep{TA<:Real,TB<:Real}
    A::SparseMatrixCSC{TA,Int}
    b::Vector{TB}
    linset::BitSet
end

const MixedMatHRep = HRep

mutable struct Polyhedron{TA<:Real,TB<:Real}
    A::SparseMatrixCSC{TA,Int}
    b::Vector{TB}
    linset::BitSet
    empty::Bool
    normalized::Bool
end

function hrep(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real}, linset::BitSet=BitSet())
    size(A, 1) == length(b) || throw(DimensionMismatch("size(A,1) must match length(b)."))
    TA = promote_type(eltype(A), Int)
    TB = eltype(b)
    return HRep(sparse(TA.(A)), TB.(collect(b)), copy(linset))
end

hrep(poly::Polyhedron) = (_canonicalize!(poly); HRep(copy(poly.A), copy(poly.b), copy(poly.linset)))

function polyhedron(rep::HRep, _backend=nothing)
    poly = Polyhedron(copy(rep.A), copy(rep.b), copy(rep.linset), false, false)
    removehredundancy!(poly)
    return poly
end

fulldim(poly::Polyhedron) = size(poly.A, 2)

function _row_sparse(A::SparseMatrixCSC, i::Int)
    row = vec(Array(A[i:i, :]))
    idxs = findall(x -> !iszero(x), row)
    return sparsevec(idxs, row[idxs], length(row))
end

function _row_entries(A::SparseMatrixCSC, i::Int)
    row = A[i, :]
    I, V = findnz(row)
    return Int.(I), collect(V)
end

function _is_rational_row(vals)
    return all(v -> v isa Rational || v isa Integer, vals)
end

function _normalize_signed_row(vals::AbstractVector, β)
    if isempty(vals)
        return Any[], β
    end
    if _is_rational_row(vals)
        rats = Rational{Int}[Int(numerator(v)) // Int(denominator(v)) for v in vals]
        lcm_den = foldl(lcm, (denominator(v) for v in rats); init=1)
        ints = [Int(numerator(v * lcm_den)) for v in rats]
        g = foldl(gcd, abs.(ints); init=0)
        g == 0 && (g = 1)
        scale = (lcm_den // g)
        ints = Int.(ints ./ g)
        return ints, scale * β
    else
        nz = findfirst(!iszero, vals)
        nz === nothing && return Float64[], Float64(β)
        scale = maximum(abs, Float64.(vals))
        norm_vals = round.(Float64.(vals) ./ scale; digits=12)
        norm_β = round(Float64(β) / scale; digits=12)
        return collect(norm_vals), norm_β
    end
end

function _row_signature(A::SparseMatrixCSC, i::Int)
    I, V = _row_entries(A, i)
    vals, β = _normalize_signed_row(V, nothing)
    return I, vals
end

function _signed_signature(A::SparseMatrixCSC, b::AbstractVector, i::Int)
    I, V = _row_entries(A, i)
    vals, β = _normalize_signed_row(V, b[i])
    return (Tuple(I), Tuple(vals), β)
end

function _negate_signature(sig)
    I, vals, β = sig
    vals_neg = map(v -> -v, vals)
    return (I, Tuple(vals_neg), -β)
end

function _unsigned_signature(sig)
    I, vals, β = sig
    if isempty(vals)
        return sig
    end
    first_nz = findfirst(v -> !iszero(v), vals)
    first_nz === nothing && return sig
    if vals[first_nz] isa Real && vals[first_nz] < 0
        return _negate_signature(sig)
    end
    return sig
end

function _rebuild_polyhedron(
    rows::AbstractVector{<:SparseVector},
    bs::AbstractVector,
    linrows::AbstractVector{Bool},
    nvars::Int,
    normalized::Bool=false,
) 
    TA = isempty(rows) ? Int : foldl(promote_type, map(eltype, rows); init=Int)
    TB = isempty(bs) ? Int : foldl(promote_type, map(typeof, bs); init=Int)
    nnz_total = isempty(rows) ? 0 : sum(nnz, rows)
    I = Vector{Int}(undef, nnz_total)
    J = Vector{Int}(undef, nnz_total)
    V = Vector{TA}(undef, nnz_total)
    ptr = 1
    for (i, row) in enumerate(rows)
        idxs, vals = findnz(row)
        for k in eachindex(idxs)
            I[ptr] = i
            J[ptr] = idxs[k]
            V[ptr] = vals[k]
            ptr += 1
        end
    end
    A = sparse(I, J, V, length(rows), nvars)
    linset = BitSet(findall(identity, linrows))
    return Polyhedron(A, TB.(collect(bs)), linset, false, normalized)
end

function detecthlinearity!(poly::Polyhedron)
    poly.empty && return poly
    poly.normalized && return poly
    _invalidate_vrep_cache!(poly)
    nrows = size(poly.A, 1)
    nvars = fulldim(poly)
    paired = falses(nrows)
    eq_rows = SparseVector[]
    eq_bs = eltype(poly.b)[]
    ineq_rows = SparseVector[]
    ineq_bs = eltype(poly.b)[]

    eq_seen = Dict{Any,Int}()
    signed_seen = Dict{Any,Int}()

    for i in 1:nrows
        sig = _signed_signature(poly.A, poly.b, i)
        if i in poly.linset
            usig = _unsigned_signature(sig)
            haskey(eq_seen, usig) && continue
            eq_seen[usig] = i
            push!(eq_rows, poly.A[i, :])
            push!(eq_bs, poly.b[i])
            continue
        end

        negsig = _negate_signature(sig)
        if haskey(signed_seen, negsig)
            j = signed_seen[negsig]
            paired[j] = true
            paired[i] = true
            usig = _unsigned_signature(sig)
            if !haskey(eq_seen, usig)
                eq_seen[usig] = i
                push!(eq_rows, poly.A[i, :])
                push!(eq_bs, poly.b[i])
            end
        else
            signed_seen[sig] = i
        end
    end

    for i in 1:nrows
        (i in poly.linset || paired[i]) && continue
        push!(ineq_rows, poly.A[i, :])
        push!(ineq_bs, poly.b[i])
    end

    rows = vcat(eq_rows, ineq_rows)
    bs = vcat(eq_bs, ineq_bs)
    linrows = vcat(fill(true, length(eq_rows)), fill(false, length(ineq_rows)))
    rebuilt = _rebuild_polyhedron(rows, bs, linrows, nvars, false)
    poly.A = rebuilt.A
    poly.b = rebuilt.b
    poly.linset = rebuilt.linset
    poly.normalized = false
    return poly
end

function _violates_zero_row(is_eq::Bool, β)
    rhs = Float64(β)
    return is_eq ? abs(rhs) > 1e-9 : rhs < -1e-9
end

function _row_dense(poly::Polyhedron, i::Int)
    return vec(Array(poly.A[i:i, :]))
end

function _subpoly_without_row(poly::Polyhedron, skip::Int)
    keep = [i for i in 1:size(poly.A, 1) if i != skip]
    linset = BitSet()
    for (new_i, old_i) in enumerate(keep)
        old_i in poly.linset && push!(linset, new_i)
    end
    return Polyhedron(poly.A[keep, :], poly.b[keep], linset, false, true)
end

function _prune_equality_basis!(poly::Polyhedron; tol::Float64=1e-9)
    eq_idxs = sort!(collect(poly.linset))
    length(eq_idxs) <= 1 && return poly

    A = Matrix(poly.A)
    selected = Int[]
    selected_rank = 0

    for idx in eq_idxs
        trial = isempty(selected) ? [idx] : vcat(selected, idx)
        aug = hcat(A[trial, :], poly.b[trial])
        trial_rank = _matrix_rank(aug; tol=tol)
        if trial_rank > selected_rank
            push!(selected, idx)
            selected_rank = trial_rank
        end
    end

    keep_eq = Set(selected)
    rows = SparseVector[]
    bs = eltype(poly.b)[]
    linrows = Bool[]
    for i in 1:size(poly.A, 1)
        if i in poly.linset
            i in keep_eq || continue
            push!(rows, poly.A[i, :])
            push!(bs, poly.b[i])
            push!(linrows, true)
        else
            push!(rows, poly.A[i, :])
            push!(bs, poly.b[i])
            push!(linrows, false)
        end
    end

    rebuilt = _rebuild_polyhedron(rows, bs, linrows, fulldim(poly), false)
    poly.A = rebuilt.A
    poly.b = rebuilt.b
    poly.linset = rebuilt.linset
    poly.empty = false
    poly.normalized = false
    return poly
end

function _strong_reduce_constraints(poly::Polyhedron; tol::Float64=1e-7)
    nrows = size(poly.A, 1)
    nrows == 0 && return poly

    keep = trues(nrows)
    make_eq = falses(nrows)

    for i in 1:nrows
        row = _row_dense(poly, i)
        β = poly.b[i]
        subpoly = _subpoly_without_row(poly, i)

        if i in poly.linset
            status_pos, val_pos = _optimize_linear(subpoly, row, β)
            status_neg, val_neg = _optimize_linear(subpoly, -row, -β)
            if (status_pos == :empty || (status_pos == :ok && val_pos <= tol)) &&
               (status_neg == :empty || (status_neg == :ok && val_neg <= tol))
                keep[i] = false
            end
            continue
        end

        status, val = _optimize_linear(subpoly, row, β)
        if status == :empty || (status == :ok && val <= tol)
            keep[i] = false
            continue
        end

        status_rev, val_rev = _optimize_linear(poly, -row, -β)
        if status_rev == :ok && val_rev <= tol
            make_eq[i] = true
        end
    end

    rows = SparseVector[]
    bs = eltype(poly.b)[]
    linrows = Bool[]
    for i in 1:nrows
        keep[i] || continue
        push!(rows, poly.A[i, :])
        push!(bs, poly.b[i])
        push!(linrows, (i in poly.linset) || make_eq[i])
    end

    rebuilt = _rebuild_polyhedron(rows, bs, linrows, fulldim(poly), false)
    poly.A = rebuilt.A
    poly.b = rebuilt.b
    poly.linset = rebuilt.linset
    poly.empty = false
    poly.normalized = false
    _prune_equality_basis!(poly; tol=tol)
    return poly
end

function removehredundancy!(poly::Polyhedron; strong::Bool=true)
    poly.empty && return poly
    poly.normalized && return poly
    _invalidate_vrep_cache!(poly)
    detecthlinearity!(poly)

    nrows = size(poly.A, 1)
    nvars = fulldim(poly)
    eq_rows = SparseVector[]
    eq_bs = eltype(poly.b)[]
    ineq_rows = SparseVector[]
    ineq_bs = eltype(poly.b)[]
    eq_seen = Set{Any}()
    ineq_seen = Set{Any}()

    for i in 1:nrows
        row = poly.A[i, :]
        idxs, vals = findnz(row)
        is_eq = i in poly.linset
        if isempty(idxs)
            if _violates_zero_row(is_eq, poly.b[i])
                poly.empty = true
                return poly
            end
            continue
        end

        sig = _signed_signature(poly.A, poly.b, i)
        key = is_eq ? _unsigned_signature(sig) : sig
        target = is_eq ? eq_seen : ineq_seen
        key in target && continue
        push!(target, key)
        if is_eq
            push!(eq_rows, row)
            push!(eq_bs, poly.b[i])
        else
            push!(ineq_rows, row)
            push!(ineq_bs, poly.b[i])
        end
    end

    rows = vcat(eq_rows, ineq_rows)
    bs = vcat(eq_bs, ineq_bs)
    linrows = vcat(fill(true, length(eq_rows)), fill(false, length(ineq_rows)))
    rebuilt = _rebuild_polyhedron(rows, bs, linrows, nvars, true)
    poly.A = rebuilt.A
    poly.b = rebuilt.b
    poly.linset = rebuilt.linset
    poly.empty = false
    poly.normalized = true

    if strong && size(poly.A, 1) <= 128 && fulldim(poly) <= 12
        poly.normalized = false
        _strong_reduce_constraints(poly)
        detecthlinearity!(poly)
        _strong_reduce_constraints(poly)
        detecthlinearity!(poly)
        rebuilt = _rebuild_polyhedron(
            [poly.A[i, :] for i in 1:size(poly.A, 1)],
            poly.b,
            [i in poly.linset for i in 1:size(poly.A, 1)],
            nvars,
            true,
        )
        poly.A = rebuilt.A
        poly.b = rebuilt.b
        poly.linset = rebuilt.linset
        poly.empty = false
        poly.normalized = true
    end

    return poly
end

_canonicalize!(poly::Polyhedron) = removehredundancy!(poly)

function _constraint_vectors(poly::Polyhedron)
    rows = Vector{Vector{Float64}}(undef, size(poly.A, 1))
    for i in 1:size(poly.A, 1)
        rows[i] = vec(Float64.(Array(poly.A[i:i, :])))
    end
    return rows, Float64.(poly.b)
end

function _new_optimizer()
    model = JuMP.Model(Clarabel.Optimizer)
    set_silent(model)
    return model
end

function feasible_point(poly::Polyhedron; tol::Float64=1e-8)
    poly.empty && return nothing
    n = fulldim(poly)
    rows, rhs = _constraint_vectors(poly)
    model = _new_optimizer()
    @variable(model, x[1:n])
    @variable(model, t >= 0)
    @objective(model, Min, t)

    for i in 1:length(rows)
        expr = sum(rows[i][j] * x[j] for j in 1:n)
        if i in poly.linset
            @constraint(model, expr <= rhs[i] + t)
            @constraint(model, expr >= rhs[i] - t)
        else
            @constraint(model, expr <= rhs[i] + t)
        end
    end

    optimize!(model)
    term = termination_status(model)
    if !(term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL))
        return nothing
    end
    value(t) <= tol || return nothing
    return value.(x)
end

function interior_point(poly::Polyhedron; tol::Float64=1e-8)
    pt = feasible_point(poly; tol=tol)
    isnothing(pt) && return nothing

    n = fulldim(poly)
    rows, rhs = _constraint_vectors(poly)
    model = _new_optimizer()
    @variable(model, x[1:n])
    @variable(model, t)
    @objective(model, Max, t)
    @constraint(model, t >= -1.0)

    for i in 1:length(rows)
        expr = sum(rows[i][j] * x[j] for j in 1:n)
        if i in poly.linset
            @constraint(model, expr == rhs[i])
        else
            @constraint(model, expr <= rhs[i] - t)
        end
    end

    for j in 1:n
        set_start_value(x[j], pt[j])
    end
    set_start_value(t, 0.0)

    optimize!(model)
    term = termination_status(model)
    if term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL)
        return value.(x)
    end
    return pt
end

function isempty(poly::Polyhedron)
    poly.empty && return true
    pt = feasible_point(poly)
    if isnothing(pt)
        poly.empty = true
        return true
    end
    return false
end

dim(poly::Polyhedron) = (isempty(poly) ? -1 : (_canonicalize!(poly); fulldim(poly) - length(poly.linset)))
hashyperplanes(poly::Polyhedron) = (_canonicalize!(poly); !isempty(poly.linset))

function hyperplanes(poly::Polyhedron)
    _canonicalize!(poly)
    out = HyperPlane[]
    for i in sort!(collect(poly.linset))
        push!(out, HyperPlane(vec(Array(poly.A[i:i, :])), poly.b[i]))
    end
    return out
end

function allhalfspaces(rep::Union{HRep,Polyhedron})
    rep isa Polyhedron && _canonicalize!(rep)
    out = HalfSpace[]
    for i in 1:size(rep.A, 1)
        i in rep.linset && continue
        push!(out, HalfSpace(vec(Array(rep.A[i:i, :])), rep.b[i]))
    end
    return out
end

function _append_constraint!(
    rows::Vector,
    bs::Vector,
    linrows::Vector{Bool},
    c::HalfSpace,
)
    idxs = findall(x -> !iszero(x), c.a)
    push!(rows, sparsevec(idxs, c.a[idxs], length(c.a)))
    push!(bs, c.β)
    push!(linrows, false)
    return nothing
end

function _append_constraint!(
    rows::Vector,
    bs::Vector,
    linrows::Vector{Bool},
    c::HyperPlane,
)
    idxs = findall(x -> !iszero(x), c.a)
    push!(rows, sparsevec(idxs, c.a[idxs], length(c.a)))
    push!(bs, c.β)
    push!(linrows, true)
    return nothing
end

function _stack_polyhedra(polys::AbstractVector{<:Polyhedron}; canonicalize::Bool=true)
    isempty(polys) && throw(ArgumentError("Need at least one polyhedron to intersect."))
    nvars = fulldim(polys[1])
    all(p -> fulldim(p) == nvars, polys) || throw(DimensionMismatch("Polyhedra must live in the same dimension."))

    Atype = foldl(promote_type, (eltype(p.A) for p in polys); init=Int)
    Btype = foldl(promote_type, (eltype(p.b) for p in polys); init=Int)
    mats = Vector{SparseMatrixCSC{Atype,Int}}(undef, length(polys))
    bs_parts = Vector{Vector{Btype}}(undef, length(polys))
    linset = BitSet()
    row_offset = 0

    for (i, poly) in enumerate(polys)
        mats[i] = eltype(poly.A) === Atype ? poly.A : SparseMatrixCSC{Atype,Int}(poly.A)
        bs_parts[i] = poly.b isa Vector{Btype} ? poly.b : Btype.(poly.b)
        for row in poly.linset
            push!(linset, row + row_offset)
        end
        row_offset += size(poly.A, 1)
    end

    A = reduce(vcat, mats)
    b = reduce(vcat, bs_parts)
    poly_new = Polyhedron(A, b, linset, false, false)
    canonicalize && removehredundancy!(poly_new)
    return poly_new
end

function intersect(poly::Polyhedron)
    return poly
end

function intersect(poly::Polyhedron, others...; canonicalize::Bool=true)
    if !isempty(others) && all(other -> other isa Polyhedron, others)
        polys = Polyhedron[poly, others...]
        return _stack_polyhedra(polys; canonicalize=canonicalize)
    end

    nvars = fulldim(poly)
    rows = SparseVector[]
    bs = Any[]
    linrows = Bool[]

    for i in 1:size(poly.A, 1)
        push!(rows, poly.A[i, :])
        push!(bs, poly.b[i])
        push!(linrows, i in poly.linset)
    end

    for other in others
        if other isa Polyhedron
            fulldim(other) == nvars || throw(DimensionMismatch("Polyhedra must live in the same dimension."))
            for i in 1:size(other.A, 1)
                push!(rows, other.A[i, :])
                push!(bs, other.b[i])
                push!(linrows, i in other.linset)
            end
        elseif other isa Union{HalfSpace,HyperPlane}
            length(other.a) == nvars || throw(DimensionMismatch("Constraint dimension mismatch."))
            _append_constraint!(rows, bs, linrows, other)
        else
            throw(ArgumentError("Unsupported constraint type $(typeof(other))."))
        end
    end

    Atype = foldl(promote_type, map(eltype, rows); init=Int)
    Btype = foldl(promote_type, map(typeof, bs); init=Int)
    rows_typed = SparseVector{Atype,Int}[SparseVector{Atype,Int}(r) for r in rows]
    bs_typed = Btype[b for b in bs]
    poly_new = _rebuild_polyhedron(rows_typed, bs_typed, linrows, nvars, false)
    canonicalize && removehredundancy!(poly_new)
    return poly_new
end

function _scale_bound(c, β)
    return c * β
end

function _scaled_row(row::SparseVector{T,Int}, c) where {T<:Real}
    idxs, vals = findnz(row)
    new_vals = similar(vals, promote_type(T, typeof(c)))
    for i in eachindex(vals)
        new_vals[i] = vals[i] * c
    end
    return sparsevec(idxs, new_vals, length(row))
end

function _expand_rows(poly::Polyhedron)
    rows = SparseVector[]
    bs = Any[]
    for i in 1:size(poly.A, 1)
        row = poly.A[i, :]
        if i in poly.linset
            push!(rows, row)
            push!(bs, poly.b[i])
            push!(rows, -row)
            push!(bs, -poly.b[i])
        else
            push!(rows, row)
            push!(bs, poly.b[i])
        end
    end
    return rows, bs
end

function _drop_axis(row::SparseVector, axis::Int, nvars_new::Int)
    idxs, vals = findnz(row)
    kept_idxs = Int[]
    kept_vals_any = Any[]
    for k in eachindex(idxs)
        idxs[k] == axis && continue
        push!(kept_idxs, idxs[k] > axis ? idxs[k] - 1 : idxs[k])
        push!(kept_vals_any, vals[k])
    end
    T = isempty(kept_vals_any) ? Int : foldl(promote_type, map(typeof, kept_vals_any); init=Int)
    kept_vals = T[v for v in kept_vals_any]
    return sparsevec(kept_idxs, kept_vals, nvars_new)
end

function _eliminate_one(poly::Polyhedron, axis::Int; canonicalize::Bool=true)
    axis in 1:fulldim(poly) || throw(ArgumentError("Invalid elimination axis $axis."))
    isempty(poly) && return polyhedron(hrep(spzeros(Int, 0, fulldim(poly) - 1), Int[]))

    rows, bs = _expand_rows(poly)
    pos_rows = Int[]
    neg_rows = Int[]
    zero_rows = Int[]

    coeffs_axis = Any[]
    for i in eachindex(rows)
        coeff = rows[i][axis]
        push!(coeffs_axis, coeff)
        if Float64(coeff) > 1e-12
            push!(pos_rows, i)
        elseif Float64(coeff) < -1e-12
            push!(neg_rows, i)
        else
            push!(zero_rows, i)
        end
    end

    out_rows = SparseVector[]
    out_bs = Any[]

    for i in zero_rows
        push!(out_rows, _drop_axis(rows[i], axis, fulldim(poly) - 1))
        push!(out_bs, bs[i])
    end

    for ip in pos_rows, ineg in neg_rows
        ap = coeffs_axis[ip]
        an = coeffs_axis[ineg]
        row = _scaled_row(rows[ip], -an) + _scaled_row(rows[ineg], ap)
        push!(out_rows, _drop_axis(row, axis, fulldim(poly) - 1))
        push!(out_bs, _scale_bound(-an, bs[ip]) + _scale_bound(ap, bs[ineg]))
    end

    Atype = isempty(out_rows) ? Int : foldl(promote_type, map(eltype, out_rows); init=Int)
    Btype = isempty(out_bs) ? Int : foldl(promote_type, map(typeof, out_bs); init=Int)
    rows_typed = SparseVector{Atype,Int}[SparseVector{Atype,Int}(r) for r in out_rows]
    bs_typed = Btype[b for b in out_bs]
    poly_new = _rebuild_polyhedron(rows_typed, bs_typed, falses(length(rows_typed)), fulldim(poly) - 1, false)
    canonicalize && removehredundancy!(poly_new)
    return poly_new
end

function eliminate(poly::Polyhedron, axis::Integer; canonicalize::Bool=true, method::Symbol=:fourier)
    method in (:fourier, :block, :auto) || throw(ArgumentError("Unknown elimination method $method."))
    return _eliminate_one(poly, Int(axis); canonicalize=canonicalize)
end

function eliminate(poly::Polyhedron, axes::BitSet; canonicalize::Bool=true, method::Symbol=:auto)
    method in (:fourier, :block, :auto) || throw(ArgumentError("Unknown elimination method $method."))
    isempty(axes) && return poly
    if method !== :fourier && length(axes) > 1
        return _block_eliminate(poly, axes; canonicalize=canonicalize)
    end
    out = poly
    for axis in sort!(collect(axes); rev=true)
        out = _eliminate_one(out, axis; canonicalize=canonicalize)
    end
    return out
end

function in(x::AbstractVector{<:Real}, poly::Polyhedron)
    length(x) == fulldim(poly) || return false
    for i in 1:size(poly.A, 1)
        lhs = dot(vec(Float64.(Array(poly.A[i:i, :]))), Float64.(x))
        rhs = Float64(poly.b[i])
        if i in poly.linset
            abs(lhs - rhs) <= 1e-7 || return false
        else
            lhs <= rhs + 1e-7 || return false
        end
    end
    return true
end

function _optimize_linear(poly::Polyhedron, a::AbstractVector{<:Real}, β::Real)
    isempty(poly) && return :empty, -Inf
    n = fulldim(poly)
    rows, rhs = _constraint_vectors(poly)
    model = _new_optimizer()
    @variable(model, x[1:n])
    @objective(model, Max, sum(Float64(a[j]) * x[j] for j in 1:n) - Float64(β))

    for i in 1:length(rows)
        expr = sum(rows[i][j] * x[j] for j in 1:n)
        if i in poly.linset
            @constraint(model, expr == rhs[i])
        else
            @constraint(model, expr <= rhs[i])
        end
    end

    optimize!(model)
    term = termination_status(model)
    if term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL)
        return :ok, objective_value(model)
    elseif term in (MOI.DUAL_INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
        return :unbounded, Inf
    else
        return :failed, Inf
    end
end

function _issubset_generators(poly::Polyhedron, a::AbstractVector{<:Real}, β; tol::Float64=1e-7)
    rep = vrep(poly)

    for pt in points(rep)
        Float64(dot(a, pt) - β) <= tol || return false
    end

    for ray in rays(rep)
        Float64(dot(a, ray)) <= tol || return false
    end

    for line in lines(rep)
        abs(Float64(dot(a, line))) <= tol || return false
    end

    return true
end

function issubset(poly::Polyhedron, h::HalfSpace; tol::Float64=1e-7)
    isempty(poly) && return true
    try
        return _issubset_generators(poly, h.a, h.β; tol=tol)
    catch
        status, val = _optimize_linear(poly, h.a, h.β)
        return status == :empty || (status == :ok && val <= tol)
    end
end

function issubset(poly::Polyhedron, h::HyperPlane; tol::Float64=1e-7)
    isempty(poly) && return true
    try
        rep = vrep(poly)

        for pt in points(rep)
            abs(Float64(dot(h.a, pt) - h.β)) <= tol || return false
        end

        for ray in rays(rep)
            abs(Float64(dot(h.a, ray))) <= tol || return false
        end

        for line in lines(rep)
            abs(Float64(dot(h.a, line))) <= tol || return false
        end

        return true
    catch
        return issubset(poly, HalfSpace(h.a, h.β); tol=tol) &&
               issubset(poly, HalfSpace(-h.a, -h.β); tol=tol)
    end
end
