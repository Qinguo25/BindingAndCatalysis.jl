struct HyperPlane{TA<:Real,TB<:Real}
    a::Vector{TA}
    β::TB
end

struct HalfSpace{TA<:Real,TB<:Real}
    p::HyperPlane{TA,TB}
    sign::Int8

    function HalfSpace(p::HyperPlane{TA,TB}, sign::Integer=1) where {TA<:Real,TB<:Real}
        s = Int8(sign)
        s in (-1, 0, 1) || throw(ArgumentError("HalfSpace sign must be -1, 0, or 1."))
        return new{TA,TB}(p, s)
    end
end

HalfSpace(a::AbstractVector{TA}, β::TB, sign::Integer=1) where {TA<:Real,TB<:Real} =
    HalfSpace(HyperPlane(collect(a), β), sign)

struct HRep{TA<:Real,TB<:Real}
    halfspaces::Vector{HalfSpace{TA,TB}}
    ambient_dim::Int
end

const MixedMatHRep = HRep

mutable struct Polyhedron{TA<:Real,TB<:Real}
    halfspaces::Vector{HalfSpace{TA,TB}}
    ambient_dim::Int
    empty::Bool
    normalized::Bool
end

const _MATRIX_REP_CACHE = IdDict{Any,Tuple{SparseMatrixCSC,Vector,BitSet}}()

_invalidate_matrix_cache!(obj) = pop!(_MATRIX_REP_CACHE, obj, nothing)

function _oriented_halfspace(h::HalfSpace)
    if h.sign >= 0
        return h.p
    end
    return HyperPlane(-h.p.a, -h.p.β)
end

function Base.getproperty(h::HalfSpace, sym::Symbol)
    if sym === :a
        return getfield(_oriented_halfspace(h), :a)
    elseif sym === :β
        return getfield(_oriented_halfspace(h), :β)
    end
    return getfield(h, sym)
end

function Base.propertynames(::HalfSpace, private::Bool=false)
    names = (:p, :sign, :a, :β)
    return private ? names : names
end

function _constraint_from_row(row::SparseVector{TA,Int}, β::TB, is_eq::Bool) where {TA<:Real,TB<:Real}
    hp = HyperPlane(collect(row), β)
    return HalfSpace(hp, is_eq ? 0 : 1)
end

function _matrix_from_halfspaces(halfspaces::AbstractVector, ambient_dim::Int)
    rows = SparseVector[]
    bs = Any[]
    linset = BitSet()

    for (i, hs) in enumerate(halfspaces)
        hp = _oriented_halfspace(hs)
        idxs = findall(!iszero, hp.a)
        push!(rows, sparsevec(idxs, hp.a[idxs], ambient_dim))
        push!(bs, hp.β)
        hs.sign == 0 && push!(linset, i)
    end

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
    A = sparse(I, J, V, length(rows), ambient_dim)
    return A, TB.(collect(bs)), linset
end

function _matrix_rep(obj::Polyhedron)
    return get!(_MATRIX_REP_CACHE, obj) do
        _matrix_from_halfspaces(getfield(obj, :halfspaces), getfield(obj, :ambient_dim))
    end
end

_matrix_rep(obj::HRep) = _matrix_from_halfspaces(getfield(obj, :halfspaces), getfield(obj, :ambient_dim))

function Base.getproperty(rep::Union{HRep,Polyhedron}, sym::Symbol)
    if sym === :A
        return _matrix_rep(rep)[1]
    elseif sym === :b
        return _matrix_rep(rep)[2]
    elseif sym === :linset
        return _matrix_rep(rep)[3]
    end
    return getfield(rep, sym)
end

function Base.propertynames(::Union{HRep,Polyhedron}, private::Bool=false)
    names = (:halfspaces, :ambient_dim, :A, :b, :linset)
    return private ? names : names
end

function HRep(
    A::SparseMatrixCSC{TA,Int},
    b::AbstractVector{TB},
    linset::BitSet=BitSet(),
) where {TA<:Real,TB<:Real}
    size(A, 1) == length(b) || throw(DimensionMismatch("size(A,1) must match length(b)."))
    halfspaces = HalfSpace{TA,TB}[]
    sizehint!(halfspaces, size(A, 1))
    for i in 1:size(A, 1)
        push!(halfspaces, _constraint_from_row(A[i, :], b[i], i in linset))
    end
    return HRep{TA,TB}(halfspaces, size(A, 2))
end

function HRep(halfspaces::AbstractVector{<:HalfSpace}, ambient_dim::Integer)
    Atype = isempty(halfspaces) ? Int : foldl(promote_type, (eltype(h.p.a) for h in halfspaces); init=Int)
    Btype = isempty(halfspaces) ? Int : foldl(promote_type, (typeof(h.p.β) for h in halfspaces); init=Int)
    typed = HalfSpace{Atype,Btype}[_typed_halfspace(Atype, Btype, h) for h in halfspaces]
    return HRep{Atype,Btype}(typed, Int(ambient_dim))
end

function Polyhedron(
    A::SparseMatrixCSC{TA,Int},
    b::AbstractVector{TB},
    linset::BitSet,
    empty::Bool=false,
    normalized::Bool=false,
) where {TA<:Real,TB<:Real}
    rep = HRep(A, b, linset)
    return Polyhedron{TA,TB}(rep.halfspaces, rep.ambient_dim, empty, normalized)
end

function Polyhedron(
    halfspaces::AbstractVector{<:HalfSpace{TA,TB}},
    ambient_dim::Integer,
    empty::Bool=false,
    normalized::Bool=false,
) where {TA<:Real,TB<:Real}
    return Polyhedron{TA,TB}(collect(halfspaces), Int(ambient_dim), empty, normalized)
end

function Polyhedron(
    halfspaces::AbstractVector{<:HalfSpace},
    ambient_dim::Integer,
    empty::Bool=false,
    normalized::Bool=false,
)
    Atype = isempty(halfspaces) ? Int : foldl(promote_type, (eltype(h.p.a) for h in halfspaces); init=Int)
    Btype = isempty(halfspaces) ? Int : foldl(promote_type, (typeof(h.p.β) for h in halfspaces); init=Int)
    typed = HalfSpace{Atype,Btype}[_typed_halfspace(Atype, Btype, h) for h in halfspaces]
    return Polyhedron{Atype,Btype}(typed, Int(ambient_dim), empty, normalized)
end

function _replace_polyhedron!(poly::Polyhedron, rebuilt::Polyhedron)
    poly.halfspaces = copy(rebuilt.halfspaces)
    poly.ambient_dim = rebuilt.ambient_dim
    poly.empty = rebuilt.empty
    poly.normalized = rebuilt.normalized
    _invalidate_matrix_cache!(poly)
    _invalidate_vrep_cache!(poly)
    return poly
end

function hrep(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real}, linset::BitSet=BitSet())
    size(A, 1) == length(b) || throw(DimensionMismatch("size(A,1) must match length(b)."))
    TA = promote_type(eltype(A), Int)
    TB = eltype(b)
    return HRep(sparse(TA.(A)), TB.(collect(b)), copy(linset))
end

hrep(poly::Polyhedron) = (removehredundancy!(poly; strong=false); HRep(copy(poly.halfspaces), poly.ambient_dim))

function polyhedron(rep::HRep, _backend=nothing; strong::Bool=false)
    poly = Polyhedron(copy(rep.halfspaces), rep.ambient_dim, false, false)
    removehredundancy!(poly; strong=strong)
    return poly
end

fulldim(poly::Polyhedron) = poly.ambient_dim

function _rhs_type(obj)
    isempty(obj.halfspaces) && return Int
    return foldl(promote_type, (typeof(h.p.β) for h in obj.halfspaces); init=Int)
end

_nconstraints(obj::Union{HRep,Polyhedron}) = length(obj.halfspaces)
_isequality(h::HalfSpace) = h.sign == 0
_isequality(obj::Union{HRep,Polyhedron}, i::Int) = _isequality(obj.halfspaces[i])

@inline _constraint_rhs(h::HalfSpace) = h.sign >= 0 ? h.p.β : -h.p.β

function _constraint_vector(h::HalfSpace)
    if h.sign >= 0
        return copy(h.p.a)
    end
    return [-x for x in h.p.a]
end

function _constraint_entries(h::HalfSpace)
    idxs = Int[]
    vals = Any[]
    sgn = h.sign >= 0 ? 1 : -1
    for j in eachindex(h.p.a)
        v = h.p.a[j]
        iszero(v) && continue
        push!(idxs, j)
        push!(vals, sgn == 1 ? v : -v)
    end
    return idxs, vals
end

function _constraint_sparsevec(h::HalfSpace, ambient_dim::Int=length(h.p.a))
    idxs, vals_any = _constraint_entries(h)
    T = isempty(vals_any) ? Int : foldl(promote_type, map(typeof, vals_any); init=Int)
    return sparsevec(idxs, T[v for v in vals_any], ambient_dim)
end

_equality_indices(obj::Union{HRep,Polyhedron}) = findall(_isequality, obj.halfspaces)
_inequality_indices(obj::Union{HRep,Polyhedron}) = findall(h -> !_isequality(h), obj.halfspaces)

function _constraint_rhs_vector(obj::Union{HRep,Polyhedron}, idxs::AbstractVector{<:Integer}=collect(1:_nconstraints(obj)))
    T = isempty(idxs) ? Int : foldl(promote_type, (typeof(_constraint_rhs(obj.halfspaces[i])) for i in idxs); init=Int)
    return T[_constraint_rhs(obj.halfspaces[i]) for i in idxs]
end

function _constraint_dense_matrix(obj::Union{HRep,Polyhedron}, idxs::AbstractVector{<:Integer}=collect(1:_nconstraints(obj)))
    nrows = length(idxs)
    ncols = obj.ambient_dim
    T = isempty(idxs) ? Int : foldl(promote_type, (eltype(obj.halfspaces[i].p.a) for i in idxs); init=Int)
    A = Matrix{T}(undef, nrows, ncols)
    for (r, i) in enumerate(idxs)
        row = _constraint_vector(obj.halfspaces[i])
        @inbounds for c in 1:ncols
            A[r, c] = convert(T, row[c])
        end
    end
    return A
end

function _constraint_dot(h::HalfSpace, x::AbstractVector)
    idxs, vals = _constraint_entries(h)
    acc = nothing
    @inbounds for k in eachindex(idxs)
        term = vals[k] * x[idxs[k]]
        acc = isnothing(acc) ? term : acc + term
    end
    if !isnothing(acc)
        return acc
    end
    if isempty(h.p.a)
        return 0
    end
    return zero(promote_type(eltype(h.p.a), typeof(first(x))))
end

function _typed_halfspace(::Type{TA}, ::Type{TB}, h::HalfSpace) where {TA<:Real,TB<:Real}
    hp = HyperPlane(TA[x for x in h.p.a], convert(TB, h.p.β))
    return HalfSpace(hp, h.sign)
end

function _copy_halfspace(h::HalfSpace)
    return _typed_halfspace(eltype(h.p.a), typeof(h.p.β), h)
end

function _constraint_signature(h::HalfSpace)
    idxs, vals = _constraint_entries(h)
    vals_norm, β_norm = _normalize_signed_row(vals, _constraint_rhs(h))
    return (Tuple(idxs), Tuple(vals_norm), β_norm)
end

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
    nrows = _nconstraints(poly)
    nvars = fulldim(poly)
    paired = falses(nrows)
    eq_halfspaces = HalfSpace[]
    ineq_halfspaces = HalfSpace[]

    eq_seen = Dict{Any,Int}()
    signed_seen = Dict{Any,Int}()

    for i in 1:nrows
        hs = poly.halfspaces[i]
        sig = _constraint_signature(hs)
        if _isequality(hs)
            usig = _unsigned_signature(sig)
            haskey(eq_seen, usig) && continue
            eq_seen[usig] = i
            push!(eq_halfspaces, HalfSpace(HyperPlane(_constraint_vector(hs), _constraint_rhs(hs)), 0))
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
                push!(eq_halfspaces, HalfSpace(HyperPlane(_constraint_vector(hs), _constraint_rhs(hs)), 0))
            end
        else
            signed_seen[sig] = i
        end
    end

    for i in 1:nrows
        (_isequality(poly, i) || paired[i]) && continue
        push!(ineq_halfspaces, _copy_halfspace(poly.halfspaces[i]))
    end

    rebuilt = Polyhedron(vcat(eq_halfspaces, ineq_halfspaces), nvars, false, false)
    _replace_polyhedron!(poly, rebuilt)
    poly.normalized = false
    return poly
end

function _violates_zero_row(is_eq::Bool, β)
    rhs = Float64(β)
    return is_eq ? abs(rhs) > 1e-9 : rhs < -1e-9
end

function _light_reduce_halfspaces(
    halfspaces::AbstractVector{<:HalfSpace},
    nvars::Int,
)
    eq_halfspaces = HalfSpace[]
    ineq_halfspaces = HalfSpace[]
    eq_seen = Set{Any}()
    ineq_seen = Set{Any}()

    for hs in halfspaces
        idxs, _ = _constraint_entries(hs)
        is_eq = _isequality(hs)
        if isempty(idxs)
            if _violates_zero_row(is_eq, _constraint_rhs(hs))
                return nothing
            end
            continue
        end

        sig = _constraint_signature(hs)
        key = is_eq ? _unsigned_signature(sig) : sig
        target = is_eq ? eq_seen : ineq_seen
        key in target && continue
        push!(target, key)

        if is_eq
            push!(eq_halfspaces, HalfSpace(HyperPlane(_constraint_vector(hs), _constraint_rhs(hs)), 0))
        else
            push!(ineq_halfspaces, _copy_halfspace(hs))
        end
    end

    return Polyhedron(vcat(eq_halfspaces, ineq_halfspaces), nvars, false, false)
end

function _light_reduce_polyhedron!(poly::Polyhedron)
    reduced = _light_reduce_halfspaces(poly.halfspaces, fulldim(poly))
    if isnothing(reduced)
        rebuilt = Polyhedron(HalfSpace[], fulldim(poly), true, true)
        _replace_polyhedron!(poly, rebuilt)
        poly.empty = true
        poly.normalized = true
        return poly
    end

    _replace_polyhedron!(poly, reduced)
    poly.empty = false
    poly.normalized = false
    return poly
end

function _row_dense(poly::Polyhedron, i::Int)
    return _constraint_vector(poly.halfspaces[i])
end

function _subpoly_without_row(poly::Polyhedron, skip::Int)
    keep = [_copy_halfspace(poly.halfspaces[i]) for i in 1:_nconstraints(poly) if i != skip]
    return Polyhedron(keep, poly.ambient_dim, false, true)
end

function _prune_equality_basis!(poly::Polyhedron; tol::Float64=1e-9)
    eq_idxs = _equality_indices(poly)
    length(eq_idxs) <= 1 && return poly

    A = _constraint_dense_matrix(poly)
    b = _constraint_rhs_vector(poly)
    selected = Int[]
    selected_rank = 0

    for idx in eq_idxs
        trial = isempty(selected) ? [idx] : vcat(selected, idx)
        aug = hcat(A[trial, :], b[trial])
        trial_rank = _matrix_rank(aug; tol=tol)
        if trial_rank > selected_rank
            push!(selected, idx)
            selected_rank = trial_rank
        end
    end

    keep_eq = Set(selected)
    kept_halfspaces = HalfSpace[]
    for i in 1:_nconstraints(poly)
        if _isequality(poly, i)
            i in keep_eq || continue
            push!(kept_halfspaces, _copy_halfspace(poly.halfspaces[i]))
        else
            push!(kept_halfspaces, _copy_halfspace(poly.halfspaces[i]))
        end
    end

    rebuilt = Polyhedron(kept_halfspaces, fulldim(poly), false, false)
    _replace_polyhedron!(poly, rebuilt)
    poly.empty = false
    poly.normalized = false
    return poly
end

function _strong_reduce_constraints(poly::Polyhedron; tol::Float64=1e-7)
    nrows = _nconstraints(poly)
    nrows == 0 && return poly

    keep = trues(nrows)
    make_eq = falses(nrows)

    for i in 1:nrows
        hs = poly.halfspaces[i]
        row = _constraint_vector(hs)
        β = _constraint_rhs(hs)
        subpoly = _subpoly_without_row(poly, i)

        if _isequality(hs)
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

    halfspaces = HalfSpace[]
    for i in 1:nrows
        keep[i] || continue
        hs = poly.halfspaces[i]
        sign = (_isequality(hs) || make_eq[i]) ? Int8(0) : hs.sign
        push!(halfspaces, HalfSpace(HyperPlane(copy(hs.p.a), hs.p.β), sign))
    end

    rebuilt = Polyhedron(halfspaces, fulldim(poly), false, false)
    _replace_polyhedron!(poly, rebuilt)
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

    nvars = fulldim(poly)
    rebuilt = _light_reduce_halfspaces(poly.halfspaces, nvars)
    if isnothing(rebuilt)
        poly.empty = true
        return poly
    end
    rebuilt.normalized = true
    _replace_polyhedron!(poly, rebuilt)
    poly.empty = false
    poly.normalized = true
    _prune_equality_basis!(poly)
    poly.normalized = true

    if strong && _nconstraints(poly) <= 128 && fulldim(poly) <= 12
        poly.normalized = false
        _strong_reduce_constraints(poly)
        detecthlinearity!(poly)
        _strong_reduce_constraints(poly)
        detecthlinearity!(poly)
        rebuilt = Polyhedron([_copy_halfspace(h) for h in poly.halfspaces], nvars, false, true)
        _replace_polyhedron!(poly, rebuilt)
        poly.empty = false
        poly.normalized = true
    end

    return poly
end

_canonicalize!(poly::Polyhedron) = removehredundancy!(poly)

function _constraint_vectors(poly::Polyhedron)
    rows = Vector{Vector{Float64}}(undef, _nconstraints(poly))
    rhs = Vector{Float64}(undef, _nconstraints(poly))
    for i in 1:_nconstraints(poly)
        hs = poly.halfspaces[i]
        rows[i] = Float64.(_constraint_vector(hs))
        rhs[i] = Float64(_constraint_rhs(hs))
    end
    return rows, rhs
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

    for i in eachindex(poly.halfspaces)
        expr = sum(rows[i][j] * x[j] for j in 1:n)
        if _isequality(poly, i)
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

    for i in eachindex(poly.halfspaces)
        expr = sum(rows[i][j] * x[j] for j in 1:n)
        if _isequality(poly, i)
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

dim(poly::Polyhedron) = (isempty(poly) ? -1 : (removehredundancy!(poly; strong=false); fulldim(poly) - length(_equality_indices(poly))))
hashyperplanes(poly::Polyhedron) = (removehredundancy!(poly; strong=false); !isempty(_equality_indices(poly)))

function hyperplanes(poly::Polyhedron)
    removehredundancy!(poly; strong=false)
    out = HyperPlane[]
    for hs in poly.halfspaces
        _isequality(hs) || continue
        push!(out, HyperPlane(copy(hs.p.a), hs.p.β))
    end
    return out
end

function hyperplanes(rep::HRep)
    out = HyperPlane[]
    for hs in rep.halfspaces
        _isequality(hs) || continue
        push!(out, HyperPlane(copy(hs.p.a), hs.p.β))
    end
    return out
end

function allhalfspaces(rep::Union{HRep,Polyhedron})
    rep isa Polyhedron && removehredundancy!(rep; strong=false)
    out = HalfSpace[]
    for hs in rep.halfspaces
        _isequality(hs) && continue
        push!(out, _copy_halfspace(hs))
    end
    return out
end

function _append_constraint!(halfspaces::Vector, c::HalfSpace)
    push!(halfspaces, _copy_halfspace(c))
    return nothing
end

function _append_constraint!(halfspaces::Vector, c::HyperPlane)
    push!(halfspaces, HalfSpace(HyperPlane(copy(c.a), c.β), 0))
    return nothing
end

function _stack_polyhedra(polys::AbstractVector{<:Polyhedron}; canonicalize::Bool=true)
    isempty(polys) && throw(ArgumentError("Need at least one polyhedron to intersect."))
    nvars = fulldim(polys[1])
    all(p -> fulldim(p) == nvars, polys) || throw(DimensionMismatch("Polyhedra must live in the same dimension."))

    Atype = foldl(promote_type, (eltype(h.p.a) for p in polys for h in p.halfspaces); init=Int)
    Btype = foldl(promote_type, (typeof(h.p.β) for p in polys for h in p.halfspaces); init=Int)
    halfspaces = HalfSpace{Atype,Btype}[]
    for poly in polys
        append!(halfspaces, (_typed_halfspace(Atype, Btype, h) for h in poly.halfspaces))
    end

    poly_new = _light_reduce_halfspaces(halfspaces, nvars)
    isnothing(poly_new) && return Polyhedron(HalfSpace[], nvars, true, true)
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
    raw_halfspaces = Any[_copy_halfspace(h) for h in poly.halfspaces]

    for other in others
        if other isa Polyhedron
            fulldim(other) == nvars || throw(DimensionMismatch("Polyhedra must live in the same dimension."))
            append!(raw_halfspaces, (_copy_halfspace(h) for h in other.halfspaces))
        elseif other isa Union{HalfSpace,HyperPlane}
            length(other.a) == nvars || throw(DimensionMismatch("Constraint dimension mismatch."))
            _append_constraint!(raw_halfspaces, other)
        else
            throw(ArgumentError("Unsupported constraint type $(typeof(other))."))
        end
    end

    Atype = foldl(promote_type, (eltype(h.p.a) for h in raw_halfspaces); init=Int)
    Btype = foldl(promote_type, (typeof(h.p.β) for h in raw_halfspaces); init=Int)
    halfspaces = HalfSpace{Atype,Btype}[_typed_halfspace(Atype, Btype, h) for h in raw_halfspaces]
    poly_new = Polyhedron(halfspaces, nvars, false, false)
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
    for hs in poly.halfspaces
        row = _constraint_sparsevec(hs, fulldim(poly))
        β = _constraint_rhs(hs)
        if _isequality(hs)
            push!(rows, row)
            push!(bs, β)
            push!(rows, -row)
            push!(bs, -β)
        else
            push!(rows, row)
            push!(bs, β)
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
    if canonicalize
        removehredundancy!(poly_new)
    else
        _light_reduce_polyhedron!(poly_new)
    end
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

function in(x::AbstractVector, poly::Polyhedron)
    all(v -> v isa Real, x) || return false
    length(x) == fulldim(poly) || return false
    for hs in poly.halfspaces
        lhs = Float64(_constraint_dot(hs, x))
        rhs = Float64(_constraint_rhs(hs))
        if _isequality(hs)
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

    for i in eachindex(poly.halfspaces)
        expr = sum(rows[i][j] * x[j] for j in 1:n)
        if _isequality(poly, i)
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
