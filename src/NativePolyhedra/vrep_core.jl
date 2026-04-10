struct VRep{T<:Real}
    points::Vector{Vector{T}}
    rays::Vector{Vector{T}}
    lines::Vector{Vector{T}}
    anchor::Union{Nothing,Vector{T}}
    source::Any
end

const _VREP_CACHE = IdDict{Polyhedron,Any}()
_invalidate_vrep_cache!(poly::Polyhedron) = pop!(_VREP_CACHE, poly, nothing)

points(rep::VRep) = rep.points
rays(rep::VRep) = rep.rays
lines(rep::VRep) = rep.lines

@inline _is_exact_scalar(x) = x isa Integer || x isa Rational
@inline _is_exact_point_scalar(x) = x isa Integer || x isa Rational || x isa ExactLogExpr
@inline _as_rational(x::Integer) = Int(x) // 1
@inline _as_rational(x::Rational{<:Integer}) = Int(numerator(x)) // Int(denominator(x))

function _matrix_is_exact(A::AbstractMatrix)
    return all(_is_exact_scalar, A)
end

function _vector_is_exact(v::AbstractVector)
    return all(_is_exact_point_scalar, v)
end

function _rref_exact(Ain::AbstractMatrix)
    A = _as_rational.(Ain)
    m, n = size(A)
    pivots = Int[]
    row = 1
    col = 1
    while row <= m && col <= n
        pivot = nothing
        for i in row:m
            if !iszero(A[i, col])
                pivot = i
                break
            end
        end
        if isnothing(pivot)
            col += 1
            continue
        end
        pivot != row && ((A[row, :], A[pivot, :]) = (A[pivot, :], A[row, :]))
        piv = A[row, col]
        for j in col:n
            A[row, j] /= piv
        end
        for i in 1:m
            i == row && continue
            fac = A[i, col]
            iszero(fac) && continue
            for j in col:n
                A[i, j] -= fac * A[row, j]
            end
        end
        push!(pivots, col)
        row += 1
        col += 1
    end
    return A, pivots
end

function _matrix_rank(A::AbstractMatrix; tol::Float64=1e-9)
    if _matrix_is_exact(A)
        _, pivots = _rref_exact(A)
        return length(pivots)
    end
    return rank(Matrix{Float64}(A); atol=tol, rtol=tol)
end

function _solve_square_exact(Ain::AbstractMatrix, b::AbstractVector)
    A = _as_rational.(Ain)
    m, n = size(A)
    m == n || return nothing
    rhs = collect(b)
    row = 1
    col = 1
    while row <= m && col <= n
        pivot = nothing
        for i in row:m
            if !iszero(A[i, col])
                pivot = i
                break
            end
        end
        isnothing(pivot) && return nothing
        if pivot != row
            (A[row, :], A[pivot, :]) = (A[pivot, :], A[row, :])
            (rhs[row], rhs[pivot]) = (rhs[pivot], rhs[row])
        end
        piv = A[row, col]
        for j in col:n
            A[row, j] /= piv
        end
        rhs[row] /= piv
        for i in 1:m
            i == row && continue
            fac = A[i, col]
            iszero(fac) && continue
            for j in col:n
                A[i, j] -= fac * A[row, j]
            end
            rhs[i] -= fac * rhs[row]
        end
        row += 1
        col += 1
    end
    return rhs
end

function _solve_square_system(A::AbstractMatrix, b::AbstractVector; tol::Float64=1e-9)
    m, n = size(A)
    m == n || return nothing
    if _matrix_is_exact(A)
        return _solve_square_exact(A, b)
    end
    Af = Matrix{Float64}(A)
    rank(Af; atol=tol, rtol=tol) == n || return nothing
    return Af \ Float64.(b)
end

function _nullspace_basis_exact(Ain::AbstractMatrix)
    A, pivots = _rref_exact(Ain)
    n = size(A, 2)
    pivotset = Set(pivots)
    freecols = [j for j in 1:n if !(j in pivotset)]
    basis = Vector{Vector{Rational{Int}}}()
    for freecol in freecols
        v = zeros(Rational{Int}, n)
        v[freecol] = 1 // 1
        for (row, pcol) in enumerate(pivots)
            v[pcol] = -A[row, freecol]
        end
        push!(basis, v)
    end
    return basis
end

function _nullspace_basis(A::AbstractMatrix; tol::Float64=1e-9)
    if _matrix_is_exact(A)
        return _nullspace_basis_exact(A)
    end
    F = svd(Matrix{Float64}(A))
    basis = Vector{Vector{Float64}}()
    for (i, s) in enumerate(F.S)
        s <= tol || continue
        push!(basis, copy(F.V[:, i]))
    end
    if length(F.S) < size(F.V, 2)
        for i in (length(F.S) + 1):size(F.V, 2)
            push!(basis, copy(F.V[:, i]))
        end
    end
    return basis
end

function _for_each_combination(items::Vector{Int}, k::Int, f::Function)
    if k < 0 || k > length(items)
        return nothing
    elseif k == 0
        f(Int[])
        return nothing
    end
    combo = Vector{Int}(undef, k)
    function rec(start::Int, depth::Int)
        remaining = k - depth + 1
        stop = length(items) - remaining + 1
        for pos in start:stop
            combo[depth] = items[pos]
            if depth == k
                f(copy(combo))
            else
                rec(pos + 1, depth + 1)
            end
        end
    end
    rec(1, 1)
    return nothing
end

function _point_key(x::AbstractVector; digits::Int=10)
    if _vector_is_exact(x)
        return Tuple(x)
    end
    return Tuple(round.(Float64.(x); digits=digits))
end

function _normalize_exact_direction(v::AbstractVector)
    rats = _as_rational.(v)
    idx = findfirst(v -> !iszero(v), rats)
    idx === nothing && return Tuple(zeros(Int, length(rats)))
    sgn = rats[idx] < 0 ? -1 : 1
    vals = rats .* sgn
    lcm_den = foldl(lcm, (denominator(val) for val in vals if !iszero(val)); init=1)
    ints = [Int(numerator(val * lcm_den)) for val in vals]
    g = foldl(gcd, abs.(ints); init=0)
    g == 0 && (g = 1)
    ints = Int.(ints ./ g)
    return Tuple(ints)
end

function _normalize_float_direction(v::AbstractVector; digits::Int=10)
    vf = Float64.(v)
    idx = findfirst(x -> abs(x) > 1e-10, vf)
    idx === nothing && return Tuple(fill(0.0, length(vf)))
    if vf[idx] < 0
        vf .*= -1
    end
    scale = maximum(abs, vf)
    scale > 0 || return Tuple(fill(0.0, length(vf)))
    return Tuple(round.(vf ./ scale; digits=digits))
end

function _direction_key(v::AbstractVector; digits::Int=10)
    if all(_is_exact_scalar, v)
        return _normalize_exact_direction(v)
    end
    return _normalize_float_direction(v; digits=digits)
end

function _normalize_direction(v::AbstractVector)
    if all(_is_exact_scalar, v)
        key = _normalize_exact_direction(v)
        return collect(key)
    end
    key = _normalize_float_direction(v)
    return collect(key)
end

function _split_constraints(poly::Polyhedron; strong::Bool=false)
    removehredundancy!(poly; strong=strong)
    m = size(poly.A, 1)
    eq_idxs = sort!(collect(poly.linset))
    eq_mask = falses(m)
    for i in eq_idxs
        eq_mask[i] = true
    end
    ineq_idxs = [i for i in 1:m if !eq_mask[i]]
    return eq_idxs, ineq_idxs
end

function _enumerate_vertices(poly::Polyhedron; tol::Float64=1e-9)
    isempty(poly) && return Vector{Vector{Float64}}()
    eq_idxs, ineq_idxs = _split_constraints(poly; strong=false)
    A = Matrix(poly.A)
    b = poly.b
    n = fulldim(poly)
    rank_eq = isempty(eq_idxs) ? 0 : _matrix_rank(A[eq_idxs, :]; tol=tol)
    needed = n - rank_eq
    points_out = Vector{Vector}()
    seen = Set{Any}()

    function handle_subset(extra::Vector{Int})
        idxs = vcat(eq_idxs, extra)
        B = A[idxs, :]
        _matrix_rank(B; tol=tol) == n || return
        x = _solve_square_system(B, b[idxs]; tol=tol)
        isnothing(x) && return
        x in poly || return
        key = _point_key(x)
        key in seen && return
        push!(seen, key)
        push!(points_out, collect(x))
    end

    if needed == 0
        handle_subset(Int[])
    else
        _for_each_combination(ineq_idxs, needed, handle_subset)
    end
    return points_out
end

function _independent_row_subset(A::AbstractMatrix, idxs::Vector{Int}; tol::Float64=1e-9)
    selected = Int[]
    selected_rank = 0
    for idx in idxs
        trial = isempty(selected) ? A[idx:idx, :] : A[[selected; idx], :]
        trial_rank = _matrix_rank(trial; tol=tol)
        if trial_rank > selected_rank
            push!(selected, idx)
            selected_rank = trial_rank
        end
    end
    return selected
end

function _solve_basis_vertex(A::AbstractMatrix, b::AbstractVector, basis_rows::Vector{Int}; tol::Float64=1e-9)
    B = A[basis_rows, :]
    size(B, 1) == size(B, 2) || return nothing
    _matrix_rank(B; tol=tol) == size(B, 2) || return nothing
    return _solve_square_system(B, b[basis_rows]; tol=tol)
end

function _find_start_basis(poly::Polyhedron; tol::Float64=1e-9)
    eq_idxs, ineq_idxs = _split_constraints(poly; strong=false)
    A = Matrix(poly.A)
    b = poly.b
    n = fulldim(poly)
    eq_basis = _independent_row_subset(A, eq_idxs; tol=tol)
    extra_needed = n - length(eq_basis)
    extra_needed < 0 && return nothing

    if extra_needed == 0
        x = _solve_basis_vertex(A, b, eq_basis; tol=tol)
        (!isnothing(x) && x in poly) || return nothing
        return eq_basis, Int[], collect(x)
    end

    combo = Vector{Int}(undef, extra_needed)
    found = Ref(false)
    found_extra = Vector{Int}()
    found_x = Ref{Any}(nothing)

    function rec(start::Int, depth::Int)
        found[] && return
        remaining = extra_needed - depth + 1
        stop = length(ineq_idxs) - remaining + 1
        for pos in start:stop
            combo[depth] = ineq_idxs[pos]
            if depth == extra_needed
                basis_rows = sort!(vcat(copy(eq_basis), combo))
                x = _solve_basis_vertex(A, b, basis_rows; tol=tol)
                (!isnothing(x) && x in poly) || continue
                empty!(found_extra)
                append!(found_extra, combo)
                sort!(found_extra)
                found_x[] = collect(x)
                found[] = true
                return
            else
                rec(pos + 1, depth + 1)
            end
            found[] && return
        end
    end

    rec(1, 1)
    found[] || return nothing
    return eq_basis, found_extra, found_x[]
end

function _basis_edge_direction(
    poly::Polyhedron,
    A::AbstractMatrix,
    basis_rows::Vector{Int},
    leaving_row::Int;
    tol::Float64=1e-9,
)
    keep_rows = [idx for idx in basis_rows if idx != leaving_row]
    _matrix_rank(A[keep_rows, :]; tol=tol) == fulldim(poly) - 1 || return nothing
    dirs = _nullspace_basis(A[keep_rows, :]; tol=tol)
    length(dirs) == 1 || return nothing
    dir = collect(dirs[1])
    coeff = _generic_dot(poly.A[leaving_row, :], dir)
    abs(Float64(coeff)) <= tol && return nothing
    Float64(coeff) > 0 && (dir = -dir)
    return dir
end

function _enumerate_pointed_generators_reverse_search(poly::Polyhedron; tol::Float64=1e-9)
    isempty(poly) && return Vector{Vector{Float64}}(), Vector{Vector{Float64}}()
    start = _find_start_basis(poly; tol=tol)
    isnothing(start) && return nothing

    eq_basis, start_extra, _ = start
    eq_idxs, ineq_idxs = _split_constraints(poly; strong=false)
    A = Matrix(poly.A)
    b = poly.b
    n = fulldim(poly)

    points_out = Vector{Vector}()
    rays_out = Vector{Vector}()
    seen_points = Set{Any}()
    seen_rays = Set{Any}()
    seen_bases = Set{Any}()
    stack = Vector{Vector{Int}}()
    push!(stack, start_extra)

    while !isempty(stack)
        extra = pop!(stack)
        basis_rows = sort!(vcat(copy(eq_basis), extra))
        basis_key = Tuple(basis_rows)
        basis_key in seen_bases && continue
        push!(seen_bases, basis_key)

        x = _solve_basis_vertex(A, b, basis_rows; tol=tol)
        (!isnothing(x) && x in poly) || continue

        pkey = _point_key(x)
        if !(pkey in seen_points)
            push!(seen_points, pkey)
            push!(points_out, collect(x))
        end

        isempty(extra) && continue
        extra_set = Set(extra)

        for leaving_row in extra
            dir = _basis_edge_direction(poly, A, basis_rows, leaving_row; tol=tol)
            isnothing(dir) && continue

            blockers = Int[]
            tmin = Inf
            for idx in ineq_idxs
                idx in extra_set && continue
                ajd = _generic_dot(poly.A[idx, :], dir)
                ajd_f = Float64(ajd)
                ajd_f > tol || continue
                slack = b[idx] - _generic_dot(poly.A[idx, :], x)
                slack_f = Float64(slack)
                slack_f >= -tol || continue
                t = slack_f / ajd_f
                t >= -tol || continue
                if t < tmin - tol
                    tmin = t
                    empty!(blockers)
                    push!(blockers, idx)
                elseif abs(t - tmin) <= tol
                    push!(blockers, idx)
                end
            end

            if isempty(blockers)
                rkey = _direction_key(dir)
                if !(rkey in seen_rays)
                    push!(seen_rays, rkey)
                    push!(rays_out, _normalize_direction(dir))
                end
                continue
            end

            for entering_row in blockers
                entering_row == leaving_row && continue
                new_extra = [idx for idx in extra if idx != leaving_row]
                entering_row in new_extra && continue
                push!(new_extra, entering_row)
                sort!(new_extra)
                length(new_extra) == n - length(eq_basis) || continue
                new_rows = sort!(vcat(copy(eq_basis), new_extra))
                _matrix_rank(A[new_rows, :]; tol=tol) == n || continue
                push!(stack, new_extra)
            end
        end
    end

    return points_out, rays_out
end

function _rowspace_basis_matrix(A::AbstractMatrix; tol::Float64=1e-9)
    n = size(A, 2)
    if _matrix_is_exact(A)
        R, _ = _rref_exact(A)
        rows = Vector{Vector{Rational{Int}}}()
        for i in 1:size(R, 1)
            row = collect(R[i, :])
            any(!iszero, row) || continue
            push!(rows, row)
        end
        isempty(rows) && return zeros(Rational{Int}, n, 0)
        return hcat(rows...)
    end
    Af = Matrix{Float64}(A)
    r = rank(Af; atol=tol, rtol=tol)
    r == 0 && return zeros(Float64, n, 0)
    F = qr(transpose(Af))
    return Matrix(F.Q[:, 1:r])
end

function _enumerate_lineality(poly::Polyhedron; tol::Float64=1e-9)
    isempty(poly) && return Vector{Vector{Float64}}()
    Aall = Matrix(poly.A)
    dirs = _nullspace_basis(Aall; tol=tol)
    out = Vector{Vector}()
    seen = Set{Any}()
    for dir in dirs
        key = _direction_key(dir)
        key in seen && continue
        push!(seen, key)
        push!(out, _normalize_direction(dir))
    end
    return out
end

function _ray_orientation(poly::Polyhedron, dir::AbstractVector; tol::Float64=1e-9)
    eq_idxs, ineq_idxs = _split_constraints(poly; strong=false)
    A = poly.A
    for i in eq_idxs
        viol = Float64(_generic_dot(A[i, :], dir))
        abs(viol) <= tol || return nothing
    end
    vals = Float64[]
    sizehint!(vals, length(ineq_idxs))
    for i in ineq_idxs
        push!(vals, Float64(_generic_dot(A[i, :], dir)))
    end
    all(v -> v <= tol, vals) || return all(v -> v >= -tol, vals) ? -collect(dir) : nothing
    any(v -> v < -tol, vals) || return nothing
    return collect(dir)
end

function _enumerate_rays(poly::Polyhedron; tol::Float64=1e-9)
    isempty(poly) && return Vector{Vector{Float64}}()
    eq_idxs, ineq_idxs = _split_constraints(poly; strong=false)
    A = Matrix(poly.A)
    n = fulldim(poly)
    rank_eq = isempty(eq_idxs) ? 0 : _matrix_rank(A[eq_idxs, :]; tol=tol)
    needed = n - 1 - rank_eq
    needed < 0 && return Vector{Vector{Float64}}()
    out = Vector{Vector}()
    seen = Set{Any}()

    function handle_subset(extra::Vector{Int})
        idxs = vcat(eq_idxs, extra)
        B = A[idxs, :]
        _matrix_rank(B; tol=tol) == n - 1 || return
        basis = _nullspace_basis(B; tol=tol)
        length(basis) == 1 || return
        dir = _ray_orientation(poly, basis[1]; tol=tol)
        isnothing(dir) && return
        key = _direction_key(dir)
        key in seen && return
        push!(seen, key)
        push!(out, _normalize_direction(dir))
    end

    if needed == 0
        handle_subset(Int[])
    else
        _for_each_combination(ineq_idxs, needed, handle_subset)
    end
    return out
end

function _generic_dot(row::SparseVector, x::AbstractVector)
    s = 0
    I, V = findnz(row)
    @inbounds for k in eachindex(I)
        s += V[k] * x[I[k]]
    end
    return s
end

function _common_generator_type(points_in, rays_in, lines_in, anchor)
    Ts = DataType[]
    for coll in (points_in, rays_in, lines_in)
        for vec in coll
            for x in vec
                push!(Ts, typeof(x))
            end
        end
    end
    if !isnothing(anchor)
        append!(Ts, typeof.(anchor))
    end
    return isempty(Ts) ? Float64 : foldl(promote_type, Ts)
end

function VRep(
    points_in::AbstractVector{<:AbstractVector}=Vector{Vector{Float64}}(),
    rays_in::AbstractVector{<:AbstractVector}=Vector{Vector{Float64}}(),
    lines_in::AbstractVector{<:AbstractVector}=Vector{Vector{Float64}}();
    anchor=nothing,
    source=nothing,
)
    T = _common_generator_type(points_in, rays_in, lines_in, anchor)
    pts = [T[x for x in vec] for vec in points_in]
    rys = [T[x for x in vec] for vec in rays_in]
    lns = [T[x for x in vec] for vec in lines_in]
    anc = isnothing(anchor) ? nothing : T[x for x in anchor]
    return VRep{T}(pts, rys, lns, anc, source)
end

function _cast_generator_collection(::Type{T}, coll) where {T}
    out = Vector{Vector{T}}(undef, length(coll))
    for i in eachindex(coll)
        out[i] = T[x for x in coll[i]]
    end
    return out
end

function _vrep_dim(rep::VRep)
    for coll in (rep.points, rep.rays, rep.lines)
        for vec in coll
            return length(vec)
        end
    end
    return isnothing(rep.anchor) ? 0 : length(rep.anchor)
end

function vrep(poly::Polyhedron; tol::Float64=1e-9)
    removehredundancy!(poly; strong=false)
    haskey(_VREP_CACHE, poly) && return _VREP_CACHE[poly]
    if isempty(poly)
        rep = VRep()
        _VREP_CACHE[poly] = rep
        return rep
    end
    lines_out = _enumerate_lineality(poly; tol=tol)
    if isempty(lines_out)
        gens = _enumerate_pointed_generators_reverse_search(poly; tol=tol)
        if isnothing(gens)
            pts = _enumerate_vertices(poly; tol=tol)
            rays_out = _enumerate_rays(poly; tol=tol)
        else
            pts, rays_out = gens
        end
        anchor = isempty(pts) ? feasible_point(poly) : copy(first(pts))
        Trep = _common_generator_type(pts, rays_out, lines_out, anchor)
        pts_t = _cast_generator_collection(Trep, pts)
        rays_t = _cast_generator_collection(Trep, rays_out)
        lines_t = _cast_generator_collection(Trep, lines_out)
        anchor_t = isnothing(anchor) ? nothing : Trep[x for x in anchor]
        rep = VRep(pts_t, rays_t, lines_t; anchor=anchor_t, source=poly)
        _VREP_CACHE[poly] = rep
        return rep
    end

    Aall = Matrix(poly.A)
    Q = _rowspace_basis_matrix(Aall; tol=tol)
    k = size(Q, 2)
    anchor =
        if all(iszero, poly.b)
            T = eltype(poly.b)
            T[zero(T) for _ in 1:fulldim(poly)]
        else
            feasible_point(poly)
        end
    isnothing(anchor) && return VRep()

    Aq = Aall * Q
    bq = poly.b .- Aall * anchor
    poly_q = polyhedron(hrep(Aq, bq, copy(poly.linset)))
    gens_q = _enumerate_pointed_generators_reverse_search(poly_q; tol=tol)
    if isnothing(gens_q)
        pts_q = _enumerate_vertices(poly_q; tol=tol)
        rays_q = _enumerate_rays(poly_q; tol=tol)
    else
        pts_q, rays_q = gens_q
    end

    Tmap = promote_type(eltype(Q), eltype(anchor))
    anchor_t = Tmap[x for x in anchor]
    pts = Vector{Vector{Tmap}}(undef, length(pts_q))
    for i in eachindex(pts_q)
        pts[i] = Tmap[x for x in (anchor_t .+ Q * pts_q[i])]
    end
    rays_out = Vector{Vector{Tmap}}(undef, length(rays_q))
    for i in eachindex(rays_q)
        rays_out[i] = Tmap[x for x in (Q * rays_q[i])]
    end
    lines_t = _cast_generator_collection(Tmap, lines_out)
    rep = VRep(pts, rays_out, lines_t; anchor=anchor_t, source=poly)
    _VREP_CACHE[poly] = rep
    return rep
end

vrep(rep::HRep; tol::Float64=1e-9) = vrep(polyhedron(rep); tol=tol)

function _dual_polyhedron_from_vrep(rep::VRep)
    d = _vrep_dim(rep)
    rows = SparseVector[]
    bs = Any[]
    linrows = Bool[]
    Tgen = _common_generator_type(rep.points, rep.rays, rep.lines, rep.anchor)

    function push_row(generator::AbstractVector, is_point::Bool, is_line::Bool)
        vals = Tgen[is_point ? convert(Tgen, 1) : zero(Tgen)]
        append!(vals, collect(generator))
        idxs = findall(x -> !iszero(x), vals)
        coeffs = [vals[i] for i in idxs]
        push!(rows, sparsevec(idxs, coeffs, d + 1))
        push!(bs, zero(Tgen))
        push!(linrows, is_line)
    end

    for pt in rep.points
        push_row(pt, true, false)
    end
    for ray in rep.rays
        push_row(ray, false, false)
    end
    for line in rep.lines
        push_row(line, false, true)
    end

    if isempty(rows)
        return polyhedron(hrep(spzeros(Int, 0, d + 1), Int[]))
    end

    Atype = foldl(promote_type, map(eltype, rows); init=Int)
    Btype = isempty(bs) ? Int : foldl(promote_type, map(typeof, bs); init=Int)
    rows_typed = SparseVector{Atype,Int}[SparseVector{Atype,Int}(r) for r in rows]
    bs_typed = Btype[b for b in bs]
    A = _rebuild_polyhedron(rows_typed, bs_typed, linrows, d + 1, false).A
    linset = BitSet(findall(identity, linrows))
    return polyhedron(hrep(-A, fill(zero(Btype), length(bs_typed)), linset))
end

function _hrep_from_dual_vrep(rep::VRep)
    rows = SparseVector[]
    bs = Any[]
    linrows = Bool[]
    d = _vrep_dim(rep) - 1
    d >= 0 || throw(ArgumentError("Dual V-representation dimension must be at least 1."))

    function push_constraint(y::AbstractVector, is_line::Bool)
        length(y) == d + 1 || throw(DimensionMismatch("Dual generator has incompatible dimension."))
        β = y[1]
        a = [-y[j] for j in 2:length(y)]
        idxs = findall(x -> !iszero(x), a)
        coeffs = [a[i] for i in idxs]
        push!(rows, sparsevec(idxs, coeffs, d))
        push!(bs, β)
        push!(linrows, is_line)
    end

    for ray in rep.rays
        push_constraint(ray, false)
    end
    for line in rep.lines
        push_constraint(line, true)
    end

    if isempty(rows)
        return hrep(spzeros(Int, 0, d), Int[])
    end

    Atype = foldl(promote_type, map(eltype, rows); init=Int)
    Btype = foldl(promote_type, map(typeof, bs); init=Int)
    rows_typed = SparseVector{Atype,Int}[SparseVector{Atype,Int}(r) for r in rows]
    bs_typed = Btype[b for b in bs]
    A = _rebuild_polyhedron(rows_typed, bs_typed, linrows, d, false).A
    linset = BitSet(findall(identity, linrows))
    return hrep(A, bs_typed, linset)
end

function polyhedron(rep::VRep, _backend=nothing)
    if rep.source isa Polyhedron
        return polyhedron(hrep(rep.source))
    end
    d = _vrep_dim(rep)
    d > 0 || return polyhedron(hrep(spzeros(Int, 0, 0), Int[]))
    dual_poly = _dual_polyhedron_from_vrep(rep)
    dual_v = vrep(dual_poly)
    poly = polyhedron(_hrep_from_dual_vrep(dual_v))
    removehredundancy!(poly)
    return poly
end

hrep(rep::VRep) = hrep(polyhedron(rep))

function _dual_normalization_polyhedron(poly::Polyhedron, del_axes::Vector{Int})
    eq_idxs, ineq_idxs = _split_constraints(poly; strong=false)
    A = Matrix(poly.A)
    n_del = length(del_axes)
    Aineq_del = isempty(ineq_idxs) ? zeros(eltype(A), 0, n_del) : A[ineq_idxs, del_axes]
    Aeq_del = isempty(eq_idxs) ? zeros(eltype(A), 0, n_del) : A[eq_idxs, del_axes]

    n_u = length(ineq_idxs) + 2 * length(eq_idxs)
    n_u == 0 && return polyhedron(hrep(spzeros(Int, 0, 0), Int[]))
    coeff_T = _matrix_is_exact(A) ? Rational{Int} : Float64
    coeff_cast(x) = coeff_T === Rational{Int} ? _as_rational(x) : Float64(x)
    rows = SparseVector[]
    bs = coeff_T[]
    linrows = Bool[]

    for row in 1:n_del
        coeff = coeff_T[]
        idxs = Int[]
        for (j, val) in enumerate(Aineq_del[:, row])
            iszero(val) && continue
            push!(idxs, j)
            push!(coeff, coeff_cast(val))
        end
        for (j, val) in enumerate(Aeq_del[:, row])
            iszero(val) && continue
            push!(idxs, length(ineq_idxs) + j)
            push!(coeff, coeff_cast(val))
        end
        for (j, val) in enumerate(Aeq_del[:, row])
            iszero(val) && continue
            push!(idxs, length(ineq_idxs) + length(eq_idxs) + j)
            push!(coeff, -coeff_cast(val))
        end
        push!(rows, sparsevec(idxs, coeff, n_u))
        push!(bs, zero(coeff_T))
        push!(linrows, true)
    end

    for j in 1:n_u
        push!(rows, sparsevec([j], [one(coeff_T)], n_u))
        push!(bs, zero(coeff_T))
        push!(linrows, false)
    end

    push!(rows, sparsevec(collect(1:n_u), fill(one(coeff_T), n_u), n_u))
    push!(bs, one(coeff_T))
    push!(linrows, true)

    rows_typed = SparseVector{coeff_T,Int}[SparseVector{coeff_T,Int}(r) for r in rows]
    poly_dual = _rebuild_polyhedron(rows_typed, bs, linrows, n_u, false)
    removehredundancy!(poly_dual)
    return poly_dual
end

function _projected_constraint_from_dual_point(poly::Polyhedron, del_axes::Vector{Int}, u::AbstractVector)
    eq_idxs, ineq_idxs = _split_constraints(poly; strong=false)
    A = Matrix(poly.A)
    b = poly.b
    keep_axes = [j for j in 1:fulldim(poly) if !(j in del_axes)]

    n_ineq = length(ineq_idxs)
    n_eq = length(eq_idxs)
    z1 = u[1:n_ineq]
    z2p = u[(n_ineq + 1):(n_ineq + n_eq)]
    z2m = u[(n_ineq + n_eq + 1):(n_ineq + 2n_eq)]
    z2 = z2p .- z2m

    T = foldl(promote_type, (typeof(x) for x in vcat(collect(z1), collect(z2), collect(b))); init=Int)
    coeff = zeros(T, length(keep_axes))
    β = zero(T)

    for (loc, idx) in enumerate(ineq_idxs)
        iszero(z1[loc]) && continue
        coeff .+= z1[loc] .* A[idx, keep_axes]
        β += z1[loc] * b[idx]
    end
    for (loc, idx) in enumerate(eq_idxs)
        iszero(z2[loc]) && continue
        coeff .+= z2[loc] .* A[idx, keep_axes]
        β += z2[loc] * b[idx]
    end
    return coeff, β
end

function _block_eliminate(poly::Polyhedron, axes::BitSet; canonicalize::Bool=true, tol::Float64=1e-9)
    isempty(axes) && return poly
    del_axes = sort!(collect(axes))
    length(del_axes) == 1 && return _eliminate_one(poly, first(del_axes); canonicalize=canonicalize)
    isempty(poly) && return polyhedron(hrep(spzeros(Int, 0, fulldim(poly) - length(del_axes)), Int[]))

    dual_poly = _dual_normalization_polyhedron(poly, del_axes)
    dual_vrep = vrep(dual_poly; tol=tol)
    pts = points(dual_vrep)
    isempty(pts) && return eliminate(poly, axes; canonicalize=canonicalize, method=:fourier)

    keep_n = fulldim(poly) - length(del_axes)
    rows = SparseVector[]
    bs = Any[]
    linrows = Bool[]
    for u in pts
        coeff, β = _projected_constraint_from_dual_point(poly, del_axes, u)
        idxs = findall(x -> !iszero(x), coeff)
        push!(rows, sparsevec(idxs, coeff[idxs], keep_n))
        push!(bs, β)
        push!(linrows, false)
    end

    Atype = isempty(rows) ? Int : foldl(promote_type, map(eltype, rows); init=Int)
    Btype = isempty(bs) ? Int : foldl(promote_type, map(typeof, bs); init=Int)
    rows_typed = SparseVector{Atype,Int}[SparseVector{Atype,Int}(r) for r in rows]
    bs_typed = Btype[b for b in bs]
    poly_new = _rebuild_polyhedron(rows_typed, bs_typed, linrows, keep_n, false)
    canonicalize && removehredundancy!(poly_new)
    return poly_new
end
