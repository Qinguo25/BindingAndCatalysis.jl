export get_one_inner_point, polyhedra_hrep_library, polyhedra_regime_graph

function same_polyhedron(P, Q)
    fulldim(P) == fulldim(Q) || return false

    HP = hrep(P)
    HQ = hrep(Q)

    return all(h -> issubset(P, h), allhalfspaces(HQ)) &&
           all(h -> issubset(P, h), hyperplanes(HQ)) &&
           all(h -> issubset(Q, h), allhalfspaces(HP)) &&
           all(h -> issubset(Q, h), hyperplanes(HP))
end

function _poly_canonical_copy(poly::Polyhedron; canonicalize::Bool=true)
    return _poly_normalized_copy(poly; canonicalize=canonicalize, detect_linearities=true)
end

function _poly_exact_log_constant(x; tol::Float64=1.0e-10)
    x isa ExactLogExpr && return x
    x isa Integer && return ExactLogExpr(x)
    x isa Rational && return ExactLogExpr(x)
    return ExactLogExpr(rationalize(Int, Float64(x); tol=tol))
end

function _poly_sparse_rational_vec(v; tol::Float64=1.0e-10)
    vv = vec(v)
    I = Int[]
    V = Rational{Int}[]
    for i in eachindex(vv)
        val = vv[i]
        abs(Float64(val)) <= tol && continue
        push!(I, i)
        push!(V, val isa Rational{Int} ? val : rationalize(Int, Float64(val); tol=tol))
    end
    return sparsevec(I, V, length(vv))
end

function _poly_add_halfspace!(
    db::RegimeToHyperplanePool, c, c0, sign::Integer=1; tol::Float64=1.0e-10
)
    return add_halfspace!(
        db,
        _poly_sparse_rational_vec(c; tol=tol),
        _poly_exact_log_constant(c0; tol=tol),
        Int8(sign);
        canonicalize=true,
    )
end

function _poly_add_hrep_to_library!(
    db::RegimeToHyperplanePool,
    poly::Polyhedron,
    poly_idx::Int,
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{Int8};
    tol::Float64=1.0e-10,
)
    C, C0, _ = get_C_C0_nullity(poly)
    for row in axes(C, 1)
        hid, dir = _poly_add_halfspace!(db, @view(C[row, :]), C0[row], 1; tol=tol)
        hid == 0 && continue
        push!(I, poly_idx)
        push!(J, hid)
        push!(V, dir)
    end
    return db
end

function _poly_finalize_incidence!(
    db::RegimeToHyperplanePool,
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{Int8},
    n_polys::Int,
)
    M = sparse(I, J, V, n_polys, length(db.hyperplanes))
    MT = sparse(J, I, V, length(db.hyperplanes), n_polys)
    db.hp_to_poly = FacetIncidence(M, MT)
    return db
end

function _poly_validate_same_ambient_dim(polys::AbstractVector{<:Polyhedron})
    isempty(polys) && throw(ArgumentError("polys must not be empty"))
    ambient_dim = fulldim(first(polys))
    for (i, poly) in enumerate(polys)
        fulldim(poly) == ambient_dim || throw(
            ArgumentError("polys[$i] has fulldim=$(fulldim(poly)); expected $ambient_dim"),
        )
    end
    return ambient_dim
end

"""
    polyhedra_hrep_library(polys; canonicalize=true, tol=1e-10) -> RegimeToHyperplanePool

Build a hyperplane library from the H-representations of `polys`. Rows of
`library.hp_to_poly.M` correspond to polyhedron indices, so each input
polyhedron is represented as one indexed regime in the library.
"""
function polyhedra_hrep_library(
    polys::AbstractVector{<:Polyhedron}; canonicalize::Bool=true, tol::Float64=1.0e-10
)
    ambient_dim = _poly_validate_same_ambient_dim(polys)
    db = RegimeToHyperplanePool(ambient_dim)
    I = Int[]
    J = Int[]
    V = Int8[]

    for (poly_idx, poly) in enumerate(polys)
        canonical_poly = _poly_canonical_copy(poly; canonicalize=canonicalize)
        _poly_add_hrep_to_library!(db, canonical_poly, poly_idx, I, J, V; tol=tol)
    end

    return _poly_finalize_incidence!(db, I, J, V, length(polys))
end

function _poly_intersection_dim(poly1::Polyhedron, poly2::Polyhedron)
    status = _poly_intersection_status(
        poly1, poly2; canonicalize=false, detect_linearities=true
    )
    return status.dim, status.poly
end

function _poly_interface_from_intersection(ins::Polyhedron; tol::Float64=1.0e-10)
    hplanes = collect(hyperplanes(ins))
    isempty(hplanes) && return nothing
    hp = hplanes[end]
    c = droptol!(sparse(hp.a), tol)
    c0 = -hp.β
    return c, c0
end

"""
    polyhedra_regime_graph(polys; canonicalize=true, tol=1e-10, edge_space=:poly) -> RegimeGraph

Construct a `RegimeGraph` whose nodes are the input polyhedra. Two nodes are
connected when their intersection has dimension `fulldim(polys[1]) - 1`.
The graph carries a reconstructed H-rep hyperplane library so existing
`draw_graph(::RegimeGraph)` machinery can render nodes and edge labels.
"""
function polyhedra_regime_graph(
    polys::AbstractVector{<:Polyhedron};
    canonicalize::Bool=true,
    tol::Float64=1.0e-10,
    edge_space::Symbol=:poly,
)
    ambient_dim = _poly_validate_same_ambient_dim(polys)
    canonical_polys = [
        _poly_canonical_copy(poly; canonicalize=canonicalize) for poly in polys
    ]
    db = RegimeToHyperplanePool(ambient_dim)
    I = Int[]
    J = Int[]
    V = Int8[]

    for (poly_idx, poly) in enumerate(canonical_polys)
        _poly_add_hrep_to_library!(db, poly, poly_idx, I, J, V; tol=tol)
    end

    n = length(canonical_polys)
    neighbors = [RegimeEdge[] for _ in 1:n]
    target_dim = ambient_dim - 1

    for i in 1:(n - 1)
        for j in (i + 1):n
            ins_dim, ins = _poly_intersection_dim(canonical_polys[i], canonical_polys[j])
            ins_dim == target_dim || continue

            interface = _poly_interface_from_intersection(ins; tol=tol)
            isnothing(interface) && continue
            c, c0 = interface
            hid, dir = _poly_add_halfspace!(db, c, c0, 1; tol=tol)
            hid == 0 && continue

            push!(neighbors[i], RegimeEdge(j, 0, Tuple{Int, Int8}[(hid, dir)]))
            push!(neighbors[j], RegimeEdge(i, 0, Tuple{Int, Int8}[(hid, -dir)]))
        end
    end

    _poly_finalize_incidence!(db, I, J, V, n)
    return RegimeGraph(neighbors, Any[db]; bn=nothing, space_idx=Dict(edge_space => 1))
end

function get_neighbor_graph(polys::AbstractVector{<:Polyhedron}; kwargs...)
    return get_neighbor_graph(polyhedra_regime_graph(polys); kwargs...)
end

function draw_graph(polys::AbstractVector{<:Polyhedron}; kwargs...)
    return draw_graph(polyhedra_regime_graph(polys); kwargs...)
end

"""
        get_one_inner_point(poly::Polyhedron; rand_line=true, rand_ray=true, extend=3, normalize_to_extend=false) -> Vector

Return a point guaranteed to lie inside the polyhedron.

Options:

  - `rand_line`: include randomized contribution from linear rays (default: `true`).
  - `rand_ray`: randomize scaling of ray directions (default: `true`).
  - `extend`: scale factor for ray contributions (default: `3`).
  - `normalize_to_extend`: if `true`, the combined ray displacement is normalized
    so its Euclidean norm is approximately `extend` (useful when you want `extend`
    to correspond roughly to distance from `point`). Default: `false`.
"""
function get_one_inner_point(
    poly::T; rand_line=true, rand_ray=true, extend=3, normalize_to_extend=false
) where {T <: Polyhedron}
    vrep_poly = MixedMatVRep(vrep(poly))
    point = if size(vrep_poly.V, 1) == 0
        zeros(Float64, fulldim(poly))
    else
        [mean(col) for col in eachcol(vrep_poly.V)]
    end
    ray_avg = zeros(eltype(point), length(point))
    for (i, ray) in enumerate(eachrow(vrep_poly.R))
        if i ∉ vrep_poly.Rlinset
            norm_ray = norm(ray)
            sigma = rand_ray ? (rand() + 0.5) * extend : extend
            ray_avg .+= ray ./ norm_ray .* sigma
        elseif rand_line
            norm_ray = norm(ray)
            sigma = (rand() - 0.5) * extend
            ray_avg .+= ray ./ norm_ray .* sigma
        end
    end
    # Optionally normalize the total ray displacement so its norm ≈ `extend`.
    if normalize_to_extend && !(norm(ray_avg) ≈ 0)
        ray_avg .= ray_avg ./ norm(ray_avg) .* extend
    end

    return point .+ ray_avg
end
