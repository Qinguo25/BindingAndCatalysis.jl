include(joinpath(@__DIR__,"get_prism.jl"))

mutable struct RegimePath
    path::Vector{Int} # the path of vertices, represented by their indices in the vertices array of Bnc
    condition::Polyhedra.Polyhedron # the path condition, represented as a polyhedron in H-representation

    # constructor
    function RegimePath(path::Vector{Int}, condition::Polyhedra.Polyhedron)
        new(path, condition)
    end
end


mutable struct RegimeGraph
    connectome::Matrix{Bool}    # the connectome of the graph, where connectome[i,j] is true if there is an edge from vertex i to vertex j
    upstream::Vector{Set{Int}}   # the achievable upstream vertices for each vertex, where upstream[i] is the list of indices of the vertices that can reach vertex i
    downstream::Vector{Set{Int}} # the achievable downstream vertices for each vertex, where downstream[i] is the list of indices of the vertices that can be reached from vertex i
    paths::Matrix{Union{Vector{RegimePath},Nothing}} # the paths of the graph, where paths[i,j] is the paths from vertex i to vertex j. Nothing if the paths have not been calculated yet.
    


end

"""

    _graph_init(
        bnc_sys::Bnc,
        v::Vector{Float64}
        )::Matrix{Bool}

based on the BnC system and the vector v, evaluate the direction of each edge in the graph, and generate the basic connectome to fill in.
"""
function _graph_init(
    bnc_sys::Bnc,
    v::Vector{Float64},
    )::Matrix{Bool}

    # ensure that the vertices have been calculated
    n_vtx = length(bnc_sys.vertices_perm)
    n_vtx == 0 && find_all_vertices!(bnc_sys)
    n_vtx = length(bnc_sys.vertices_perm)

    # ensure that the length of v is correct
    length(v) == bnc_sys.d || error("Length of v must be $(bnc_sys.d), got $(length(v)).")

    
    vtx_grh = get_vertices_graph!(bnc_sys; full=true)
    connectome = falses(n_vtx, n_vtx)

    tol = 1e-6
    for (i, edges) in enumerate(vtx_grh.neighbors)
        if get_nullity(bnc_sys, i) > 1
            continue
        end
        for e in edges
            # process each undirected pair once
            if isnothing(e.change_dir_qK) || e.to < i
                continue
            end

            # use only q-space components (first d coordinates in qK space)
            dir = dot(e.change_dir_qK[1:bnc_sys.d], v)
            if dir > tol
                connectome[i, e.to] = true
            elseif dir < -tol
                connectome[e.to, i] = true
            end
        end
    end

    return connectome
end

"""
    _path_tracing(
        connectome::Matrix{Bool},
    )::Tuple{Vector{Set{Int}}, Vector{Set{Int}}}

Based on the connectome of the graph, calculate the upstream and downstream vertices for each vertex.
"""
function _path_tracing(
    connectome::Matrix{Bool},
    )::Tuple{Vector{Set{Int}}, Vector{Set{Int}}}
    
end

function _better_path_finder!(
    bnc_sys::Bnc,
    vertex_idx_from::Int,
    vertex_idx_to::Int,
    v::Vector{Float64},
    )::nothing
    Nothing

end