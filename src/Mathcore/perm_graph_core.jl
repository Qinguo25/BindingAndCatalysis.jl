
"""
    RegimeEdge

Edge metadata connecting neighboring vertices in a regime graph.
"""
mutable struct RegimeEdge
    to::Int
    i ::Int # different row index
    c_c0_x_idx::Int
    c_c0_x_sign::Int8
    qK_interface_idx::Int
    qK_interface_sign::Int8

    function RegimeEdge(to::Int, i::Int, c_c0_x_idx::Int, c_c0_x_sign::Int8)
        return new(to, i, c_c0_x_idx, c_c0_x_sign, 0, 0)
    end

end



@inline _edge_has_qK_interface(edge::RegimeEdge) =
    edge.qK_interface_idx != 0



struct RegimeHyperplane
    change_dir_qK::SparseVector{Float64, Int}
    intersect_qK::Float64
end


# Adjacency list + optional caches
"""
    RegimeGraph

Adjacency structure for vertices with optional caches for change directions.
"""
mutable struct RegimeGraph{Tv}
    bn::Union{AbstractBnc, Nothing}
    neighbors::Vector{Vector{RegimeEdge}}

    edge_pos::Vector{Dict{Int, Int}}  # (u,v) -> (u,edge_pos[u][v]) to locate the RegimeEdge.

    qK_interface_pool::Vector{RegimeHyperplane}
    x_interface_pool::Vector{Hyperplane_perm{Tv}}
    change_dir_qK_computed::Bool
    qK_classifier_full::Any
    qK_classifier_asymptotic::Any

    function RegimeGraph(L_helper::MatrixHelper{Tv}, neighbors::Vector{Vector{RegimeEdge}}) where {Tv}
        
        edge_pos = let
            edge_pos = Vector{Dict{Int,Int}}(undef, length(neighbors))
                for i in eachindex(neighbors)
                    edges = neighbors[i]
                    d = Dict{Int,Int}()
                    sizehint!(d, length(edges))
                    for (k, e) in enumerate(edges)
                        d[e.to] = k
                    end
                    edge_pos[i] = d
                end
                edge_pos
            end

        return new{Tv}(
            nothing,
            neighbors,
            edge_pos,
            RegimeHyperplane[],
            L_helper.hyperplanes,
            false,
            nothing,
            nothing,
        )
    end
end



Base.display(io::IO, grh::RegimeGraph) = print(io, "RegimeGraph with $(length(grh.neighbors)) vertices and $(sum(length.(grh.neighbors))) edges")
Base.show(io::IO, grh::RegimeGraph) = print(io, "RegimeGraph with $(length(grh.neighbors)) vertices and $(sum(length.(grh.neighbors))) edges")


#-----------------------------------------------------------------------------------------------
#This is graph associated functions for Bnc models and archetyple behaviors associated code
#-----------------------------------------------------------------------------------------------
"""
    _calc_regimes_graph(bnc::Bnc, perms) -> RegimeGraph

Build a `RegimeGraph` from regime permutations, connecting regimes that differ
in exactly one row.
"""
function _calc_regimes_graph(helper::MatrixHelper, perms::Vector{<:AbstractVector{T}}) where {T<:Integer}
    # n = helper.n
    n_vtxs = length(perms)
    d = length(perms[1])
    thread_edges = [Vector{Tuple{Int, RegimeEdge}}() for _ in 1:Threads.maxthreadid()]

    # 按行分桶：key 为去掉该行后的签名（Tuple），值为该签名下的 (regime idx, row choice)
    @showprogress for i in 1:d
        buckets = Dict{Tuple{Vararg{T}}, Vector{Tuple{Int,T}}}()
        @inbounds for q in 1:n_vtxs
            v = perms[q]
            sig = if i == 1
                    Tuple(v[2:end])
                elseif i == d
                    Tuple(v[1:end-1])
                else
                    Tuple((v[1:i-1]..., v[i+1:end]...))
                end
            push!(get!(buckets, sig) do
                Vector{Tuple{Int,T}}()
            end, (q, v[i]))
        end

        groups = collect(values(buckets))

        # 同桶内两两相连：沿边方向表示“增加 target dominant term，减少 source dominant term”
        Threads.@threads for gi in eachindex(groups)
            tid = Threads.threadid()
            local_edges = thread_edges[tid]
            group = groups[gi]
            m = length(group)
            m <= 1 && continue

            @inbounds for a in 1:m-1
                from_idx, j_from = group[a]
                for b in a+1:m
                    to_idx, j_to = group[b]
                    j_from == j_to && continue

                    hp_id = choiceineq_between(helper, i, j_to, j_from)
                    push!(local_edges, (from_idx, RegimeEdge(to_idx, i, hp_id.hid, hp_id.sign)))
                    push!(local_edges, (to_idx, RegimeEdge(from_idx, i, hp_id.hid, -hp_id.sign)))
                end
            end
        end
    end

    all_edges = reduce(vcat, thread_edges; init=Tuple{Int, RegimeEdge}[])
    neighbors = [Vector{RegimeEdge}() for _ in 1:n_vtxs]
    for (from, e) in all_edges
        push!(neighbors[from], e)
    end
    return RegimeGraph(helper, neighbors)
end
#=============================================================================================#
#          Calc qK-space change directions for edges with nullity <= 1 regimes This part is pure AI
#=============================================================================================#

function _edge_qK_interface(grh::RegimeGraph, edge::RegimeEdge)
    edge.qK_interface_idx == 0 && return nothing

    hp = grh.qK_interface_pool[edge.qK_interface_idx]
    if edge.qK_interface_sign >= 0
        return hp.change_dir_qK, hp.intersect_qK
    else
        return -hp.change_dir_qK, -hp.intersect_qK
    end
end



# function _materialize_edge_qK_interface!(grh::RegimeGraph, edge::RegimeEdge)
#     return edge
# end

@inline function _dense_hyperplane_sign_and_scale(dir::SparseVector{Float64,Int}; atol::Float64=1e-10)
    I, V = findnz(dir)
    @inbounds for k in eachindex(V)
        if abs(V[k]) > atol
            sgn = V[k] >= 0 ? Int8(1) : Int8(-1)
            return sgn, abs(V[k])
        end
    end
    return Int8(1), 0.0
end

@inline function _dense_hyperplane_sign_and_scale(dir::SparseVector; atol::Float64=1e-10)
    I, V = findnz(dir)
    @inbounds for k in eachindex(V)
        if abs(Float64(V[k])) > atol
            sgn = V[k] >= 0 ? Int8(1) : Int8(-1)
            return sgn, abs(V[k])
        end
    end
    return Int8(1), zero(eltype(dir))
end

@inline _is_exact_hyperplane_scalar(x) = x isa Integer || x isa Rational || x isa ExactLogExpr

@inline function _exact_hyperplane_bias(x)
    if _is_exact_hyperplane_scalar(x)
        return x
    elseif x isa AbstractFloat && iszero(x)
        return 0//1
    else
        return nothing
    end
end

function _canonicalize_qK_key_exact(
    I::AbstractVector{<:Integer},
    V::AbstractVector,
    intersect,
    sign::Int8,
    scale,
)
    all(v -> v isa Integer || v isa Rational, V) || return nothing
    β = _exact_hyperplane_bias(intersect)
    isnothing(β) && return nothing
    coeffs = Tuple(((v * sign) / scale for v in V))
    βnorm = (β * sign) / scale
    return (Tuple(Int.(I)), coeffs, βnorm)
end

@inline function _canonicalize_qK_key_float(
    I::AbstractVector{<:Integer},
    Vnorm::AbstractVector{<:Real},
    bnorm::Real;
    round_digits::Int=10,
)
    return (
        Tuple(Int.(I)),
        Tuple(round.(Float64.(Vnorm); digits=round_digits)),
        round(Float64(bnorm); digits=round_digits),
    )
end

function _canonicalize_qK_interface(
    dir::SparseVector,
    intersect::Real;
    key_mode::Symbol=:float,
    atol::Float64=1e-10,
    round_digits::Int=10,
)
    droptol!(dir, atol)
    nnz(dir) == 0 && return nothing

    sign, scale = _dense_hyperplane_sign_and_scale(dir; atol=atol)
    scale <= atol && return nothing

    I, V = findnz(dir)
    Vnorm = (Float64.(V) .* sign) ./ scale
    dir_norm = SparseArrays.sparsevec(I, Vnorm, length(dir))
    droptol!(dir_norm, atol)
    I2, V2 = findnz(dir_norm)
    bnorm = Float64(intersect) * sign / scale
    key = if key_mode === :exact
        exact_key = _canonicalize_qK_key_exact(I, V, intersect, sign, scale)
        isnothing(exact_key) ? _canonicalize_qK_key_float(I2, V2, bnorm; round_digits=round_digits) : exact_key
    else
        _canonicalize_qK_key_float(I2, V2, bnorm; round_digits=round_digits)
    end
    return dir_norm, bnorm, sign, key
end

function _intern_qK_interface!(
    grh::RegimeGraph,
    key_to_id::Dict,
    dir::SparseVector,
    intersect::Real;
    key_mode::Symbol=:float,
    atol::Float64=1e-10,
    round_digits::Int=10,
)
    canon = _canonicalize_qK_interface(dir, intersect; key_mode=key_mode, atol=atol, round_digits=round_digits)
    canon === nothing && return 0, Int8(0)
    dir_norm, bnorm, sign, key = canon
    hid = get!(key_to_id, key) do
        push!(grh.qK_interface_pool, RegimeHyperplane(dir_norm, bnorm))
        length(grh.qK_interface_pool)
    end
    return hid, sign
end




"""
    _fulfill_regimes_graph!(vtx_graph::RegimeGraph) -> nothing

Compute qK-space change directions for edges in the vertex graph.
"""
function _fulfill_regimes_graph!(vtx_graph::RegimeGraph)
    Bnc = vtx_graph.bn
    regimes = _bind_regimes_data(Bnc)
    empty!(vtx_graph.qK_interface_pool)
    key_to_id = Dict{Any,Int}()
    key_mode = _affine_is_exact(Bnc) ? :exact : :float

    for edges in vtx_graph.neighbors
        for e in edges
            e.qK_interface_idx = 0
            e.qK_interface_sign = 0
        end
    end

    @showprogress for p1 in eachindex(vtx_graph.neighbors)
        edges = vtx_graph.neighbors[p1]
        for e in edges
            p2 = e.to
            p1 < p2 || continue

            rev_pos = get(vtx_graph.edge_pos[p2], p1, nothing)
            rev_pos === nothing && continue
            e_rev = vtx_graph.neighbors[p2][rev_pos]

            src_idx, src_edge, dst_edge = let
                nlt1 = regimes[p1].nullity
                nlt2 = regimes[p2].nullity
                if nlt1 <= 1
                    (p1, e, e_rev)
                elseif nlt2 <= 1
                    (p2, e_rev, e)
                else
                    (0, nothing, nothing)
                end
            end
            src_idx == 0 && continue

            src_rgm = regimes[src_idx]
            c_c0 = vtx_graph.x_interface_pool[src_edge.c_c0_x_idx]
            
            dir_qK, intersect_qK = _calc_dir(
                src_rgm.nullity,
                src_rgm.H,
                src_rgm.H0,
                c_c0,
                src_edge.c_c0_x_sign,
            )

            hid, sign = _intern_qK_interface!(vtx_graph, key_to_id, dir_qK, intersect_qK; key_mode=key_mode)
            hid == 0 && continue

            src_edge.qK_interface_idx = hid
            src_edge.qK_interface_sign = sign
            dst_edge.qK_interface_idx = hid
            dst_edge.qK_interface_sign = Int8(-sign)
        end
    end
    return nothing
end


@inline function _calc_dir(
    nlt::Int,
    H::SparseMatrixCSC{<:Real,Int},
    H0::AbstractVector{<:Real},
    c_c0::Hyperplane_perm,
    sign::Int8,
    drop_tol::Float64=1e-10,
)
    c_qK = c_c0 * H .* sign
    c0_qK = nlt ==0 ? c_c0 * H0 * sign : mul(c_c0, H0; with_c0=false) * sign 

    I, V = findnz(c_qK)
    c_qK = SparseArrays.sparsevec(I, Float64.(V), length(c_qK))
    if drop_tol > 0 
        droptol!(c_qK, drop_tol) 
        c0_qK = droptol!(Float64(c0_qK), drop_tol)
    else
        c0_qK = Float64(c0_qK)
    end

    return c_qK, c0_qK
end
