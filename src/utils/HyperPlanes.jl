"""
    Hyperplane_perm

Canonical hyperplane stored with `u < v`:

    z_u - z_v + log10(num / den) = 0

where `(num, den)` is the reduced integer ratio.

One oriented inequality induced by choosing p in row i.
If `sign == +1`, use the canonical side:

    crow * z + c0 > 0

If `sign == -1`, use the opposite side:

    crow_neg * z - c0 > 0

`competitor` is the losing column k compared against the perm dominant p.
`oriented_c0 = log10(L[i,p] / L[i,k])`

so the actual inequality is:
z_p - z_k + oriented_c0 > 0
"""
struct Hyperplane_perm{Tv <: Integer, To <: Real} <: AbstractHyperPlane
    u::Int # fast access #j2 by default 
    v::Int # fast access #j1 by default 

    num::Tv # reduced positive integer L_{i,j2}
    den::Tv # reduced positive integer L_{i,j1}
    c0::To # pre-logarithm log10(num/den)
end

function Base.:*(hp::Hyperplane_perm, M::AbstractMatrix{<:Real})
    return M[hp.u, :] - M[hp.v, :]
end

function mul(hp::Hyperplane_perm, q::AbstractVector{<:Real}; with_c0::Bool=true)
    if with_c0
        return q[hp.u] - q[hp.v] .+ hp.c0
    else
        return q[hp.u] - q[hp.v]
    end
end

Base.:*(hp::Hyperplane_perm, q::AbstractVector{<:Real}) = mul(hp, q; with_c0=true)

# @inline _calc_c(hp::Hyperplane_perm,n::Int,sign::Int8) = let 
#     if sign > 0 
#         return sparsevec([hp.u, hp.v], Int8[1 -1], n)
#     else
#         return sparsevec([hp.u, hp.v], Int8[-1, 1], n)
#     end
# end

@inline _calc_c(hp::Hyperplane_perm, n::Int, sign::Int8) =
    let
        I = [hp.u, hp.v]
        J = [1, 1]
        V = Int8[sign, -sign]
        return sparse(I, J, V, n, 1)
    end

@inline _calc_c_c0(hp::Hyperplane_perm, n::Int, sign::Int8) =
    let
        if sign > 0
            return _calc_c(hp, n, sign), hp.c0
        else
            return _calc_c(hp, n, sign), -hp.c0
        end
    end

get_hp_key(hp::Hyperplane_perm) = (hp.u, hp.v)
get_hp_key(j1::Int, j2::Int) = j1 < j2 ? (j1, j2) : (j2, j1)

#=================================================================#
# General hyperplane
#=================================================================#

struct RegimeHyperplane <: AbstractHyperPlane
    change_dir_qK::SparseVector{Rational{Int}, Int}
    intersect_qK::ExactLogExpr
end

function _calc_c_c0(hp::RegimeHyperplane, dir::Int8)
    let
        if dir > 0
            return hp.change_dir_qK, hp.intersect_qK
        else
            return -hp.change_dir_qK, -hp.intersect_qK
        end
    end
end

# struct RgmPolyhedron
#     halfspaces::Vector{Tuple{Int,Int8}} # (hyperplane id, direction) pairs
# end

#=================================================================#
# General hyperplane database.
#=================================================================#

struct HyperplaneKey{C}
    nzind::Tuple{Vararg{Int}}
    nzval::Tuple{Vararg{Rational{Int}}}
    c0::C
end

function get_hp_key(c::SparseVector{<:Rational}, c0)
    return HyperplaneKey(Tuple(c.nzind), Tuple(c.nzval), c0)
end
get_hp_key(hp::RegimeHyperplane) = get_hp_key(hp.change_dir_qK, hp.intersect_qK)

struct FacetIncidence
    M::SparseMatrixCSC{Int8, Int}   # pid × hid
    MT::SparseMatrixCSC{Int8, Int}  # hid × pid
end

mutable struct RegimeToHyperplanePool
    dim::Int

    hyperplanes::Vector{RegimeHyperplane}
    # polytopes::Vector{RgmPolyhedron}
    hp_to_poly::FacetIncidence
    hp_dict::Dict{HyperplaneKey, Int}

    function RegimeToHyperplanePool(dim::Int)
        return new(
            dim,
            RegimeHyperplane[],
            # RgmPolyhedron[],
            FacetIncidence(spzeros(Int8, 0, 0), spzeros(Int8, 0, 0)),
            Dict{HyperplaneKey, Int}(),
        )
    end
end

get_hyperplane(db::RegimeToHyperplanePool, hid::Int) = db.hyperplanes[hid]

# Canonicalize the hyperplanes, temperal.

function _canonicalize_halfspace(c::SparseVector{<:Rational}, c0::ExactLogExpr)
    dir, scale = let
        v = nonzeros(c)[1]
        (v >= 0 ? Int8(1) : Int8(-1)), abs(v)
    end

    # normalize
    c.nzval .= (c.nzval .* dir) ./ scale
    c0 = (c0 * dir) / scale

    hp = RegimeHyperplane(c, c0)
    return hp, dir
end
function _canonicalize_hyperplane(args...; kwargs...)
    return _canonicalize_halfspace(args...; kwargs...)[1]
end

# You should make sure the hyperplane is already canonicalized before calling this function
# Will return the hid
function add_hyperplane!(
    db::RegimeToHyperplanePool,
    c::SparseVector{<:Rational},
    c0::ExactLogExpr;
    canonicalize::Bool=false,
)
    hp = if canonicalize
        _canonicalize_hyperplane(c, c0)
    else
        RegimeHyperplane(c, c0)
    end

    key = get_hp_key(hp)

    hid = get!(db.hp_dict, key) do
        push!(db.hyperplanes, hp)
        length(db.hyperplanes)
    end

    return hid
end
function add_hyperplane!(
    db::RegimeToHyperplanePool, hp::RegimeHyperplane; canonicalize::Bool=false
)
    return add_hyperplane!(db, hp.change_dir_qK, hp.intersect_qK; canonicalize=canonicalize)
end

# function add_hyperplane!(db::RegimeToHyperplanePool, c::SparseVector{<:Rational}, c0::ExactLogExpr; canonicalize::Bool=true)
function add_halfspace!(
    db::RegimeToHyperplanePool,
    c::SparseVector{<:Rational},
    c0::ExactLogExpr,
    dir::Int8;
    canonicalize::Bool=true,
)
    if isempty(c.nzind) # empty hyperplane, just return 0
        return 0, dir
    end

    hp, dir_inner = if canonicalize
        _canonicalize_halfspace(c, c0)
    else
        (RegimeHyperplane(c, c0), Int8(1))
    end

    hid = add_hyperplane!(db, hp; canonicalize=false)

    return hid, sign(dir * dir_inner)
end
function add_halfspace!(
    db::RegimeToHyperplanePool, hp::RegimeHyperplane, dir::Int8; canonicalize::Bool=true
)
    return add_halfspace!(
        db, hp.change_dir_qK, hp.intersect_qK, dir; canonicalize=canonicalize
    )
end
