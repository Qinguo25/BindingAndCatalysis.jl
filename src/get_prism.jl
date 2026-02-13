
using Polyhedra
using CDDLib
using LinearAlgebra
using SparseArrays

"""
    _get_interface_prism(
        bnc_sys::Bnc,
        vertex_idx_from::Vector{Int},
        vertex_idx_to::Vector{Int},
        axis_to_eliminate::Int,
        H::Matrix{Float64}
    )::Polyhedra.Polyhedron

Get the interface prism between two vertices, which is the intersection of the two vertices' polyhedra.
The interface prism is represented as a polyhedron in H-representation.
H is the Householder transformation matrix
"""
function _get_interface_prism(
    bnc_sys::Bnc,
    vertex_idx_from::Vector{Int},
    vertex_idx_to::Vector{Int},
    axis_to_eliminate::Int,
    H::Matrix{Float64} = nothing,
    )::Polyhedra.Polyhedron
    
    # 1. get the interface polyhedron of the two vertices
    p_from = get_polyhedron(bnc_sys, vertex_idx_from)
    p_to = get_polyhedron(bnc_sys, vertex_idx_to)
    p = intersect(p_from, p_to)
    detecthlinearity!(p)
    removehredundancy!(p)
    
    if isempty(p)
        return p
    end

    # if the vector v is not aligned with any coordinate axis
    if axis_to_eliminate == -1
        # 2. apply Householder transformation to the polyhedron to make v the last coordinate axis
        # this step enables the elimination of the dimension along V
        p = linear_map(H, p)
    end

    # 3. remove the corresponding coordinate to get the interface prism
    p = eliminate(p, axis_to_eliminate)
    removehredundancy!(p)

    # 4. apply inverse Householder transformation to the interface prism to get the final result
    if axis_to_eliminate == -1
        p = linear_map(inv(H), p)
    end
    return p
end

"""
    _get_axis_to_eliminate(v::Vector{Float64})::Int

Get the index of the coordinate axis to eliminate, which is the axis that is aligned with the vector v.
If v is not aligned with any coordinate axis, return -1.
"""
function _get_axis_to_eliminate(
    v::Vector{Float64}
    )::Int
    # if the vector v is aligned with any coordinate axis, return the index of that axis
    axis_to_eliminate = -1
    for i in 1:length(v)
        if v[i] != 0.0
            if axis_to_eliminate == -1
                axis_to_eliminate = i
            else
                # if v is not aligned with any coordinate axis, return -1
                return -1
            end
        end
    end
    return axis_to_eliminate
end

"""
    _get_Householder_transformation(v::Vector{Float64})::Matrix{Float64}

Get the Householder transformation matrix to make v the last coordinate axis.
"""
function _get_Householder_transformation(
    v::Vector{Float64},
    )::Matrix{Float64}

    n = length(v)
    H = Matrix{Float64}(I, n, n)  # Initialize the Householder matrix as the identity matrix

    v_norm = norm(v)  # Calculate the norm (length) of the vector v
    if v_norm == 0.0
        # Raise an error if v is the zero vector, as the transformation is undefined
        error("The vector v to calculate SISO graph is the zero vector.")
    end
    v = v / v_norm  # Normalize v to have unit length
    
    e_n = zeros(Float64, n)  # Create a zero vector of the same length as v
    e_n[end] = 1.0  # Set the last element to 1, representing the target direction

    u = v - e_n  # Compute the vector u, which is the difference between v and e_n
    u_norm = norm(u)  # Calculate the norm of u
    if u_norm == 0.0
        # If u is the zero vector, return the identity matrix as no transformation is needed
        return H
    end
    u = u / u_norm  # Normalize u to have unit length

    # Construct the Householder matrix H using the formula H = I - 2 * (u * u')
    H = I - 2 * (u * u')  # This transformation reflects points across the hyperplane orthogonal to u
    return H  # Return the Householder transformation matrix
end