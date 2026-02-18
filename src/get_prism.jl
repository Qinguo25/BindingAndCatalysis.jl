
using Polyhedra
using CDDLib
using LinearAlgebra
using SparseArrays

"""
    _apply_householder_in_hrep(
        p::Polyhedra.Polyhedron,
        H::Matrix{Float64}
    )::Polyhedra.Polyhedron

Apply the variable change `y = H*x` to an H-representation polyhedron `p`.
If `p` is `C*x <= C0`, then in `y`-space constraints are `C*H'*y <= C0`.
"""
function _apply_householder_in_hrep(
    p::Polyhedra.Polyhedron,
    H::Matrix{Float64}
    )::Polyhedra.Polyhedron

    C, C0, nullity = get_C_C0_nullity(p)
    n_vars = size(C, 2)
    d = size(H, 1)
    d == size(H, 2) || error("Householder matrix H must be square.")
    d <= n_vars || error("Householder matrix dimension cannot exceed polyhedron variable dimension.")

    # qK variables are ordered as [q; K]. Rotate only q-part and keep K-part unchanged.
    H_full = Matrix{Float64}(I, n_vars, n_vars)
    H_full[1:d, 1:d] = H

    C_t = Matrix(C) * transpose(H_full)
    p_t = get_polyhedron(C_t, C0, nullity)
    return p_t
end

"""
    _get_interface_prism(
        bnc_sys::Bnc,
        vertex_idx_from::Int,
        vertex_idx_to::Int,
        axis_to_eliminate::Int,
        H::Matrix{Float64}
    )::Polyhedra.Polyhedron

Get the interface prism between two vertices, which is the intersection of the two vertices' polyhedra.
The interface prism is represented as a polyhedron in H-representation.
H is the Householder transformation matrix
"""
function _get_interface_prism(
    bnc_sys::Bnc,
    vertex_idx_from::Int,
    vertex_idx_to::Int,
    axis_to_eliminate::Int,
    H::Union{Matrix{Float64}, Nothing} = nothing,
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
        isnothing(H) && error("Householder matrix H must be provided when axis_to_eliminate == -1.")
        size(H, 1) == size(H, 2) || error("Householder matrix H must be square.")
        # 2. apply Householder transformation to the polyhedron to make v the last coordinate axis
        # this step enables the elimination of the dimension along V
        p = _apply_householder_in_hrep(p, H)
    end

    # 3. remove the corresponding coordinate to get the interface prism
    eliminate_axis = axis_to_eliminate == -1 ? size(H, 1) : axis_to_eliminate
    p = eliminate(p, eliminate_axis)
    removehredundancy!(p)

    return p
end

"""
    _get_polyhedron_prism(
        bnc_sys::Bnc,
        vertex_idx::Int,
        axis_to_eliminate::Int,
        H::Matrix{Float64}
    )::Polyhedra.Polyhedron

Get the polyhedron prism of a vertex, which is the projection of the vertex's polyhedron onto the subspace orthogonal to the vector v.
The polyhedron prism is represented as a polyhedron in H-representation.
"""
function _get_polyhedron_prism(
    bnc_sys::Bnc,
    vertex_idx::Int,
    axis_to_eliminate::Int,
    H::Union{Matrix{Float64}, Nothing} = nothing,
    )::Polyhedra.Polyhedron

    # 1. get the polyhedron of the vertex and eliminate the corresponding coordinate to get the prism
    p = get_polyhedron(bnc_sys, vertex_idx)
    detecthlinearity!(p)
    removehredundancy!(p)

    if isempty(p)
        return p
    end

    # 2. if the vector v is not aligned with any coordinate axis
    if axis_to_eliminate == -1
        isnothing(H) && error("Householder matrix H must be provided when axis_to_eliminate == -1.")
        size(H, 1) == size(H, 2) || error("Householder matrix H must be square.")
        p = _apply_householder_in_hrep(p, H)
    end

    # 3. remove the corresponding coordinate to get the prism
    eliminate_axis = axis_to_eliminate == -1 ? size(H, 1) : axis_to_eliminate
    p = eliminate(p, eliminate_axis)

    # NOTE:
    # For some axis-aligned projections, `removehredundancy!` on the eliminated
    # polyhedron can trigger a CDD blow-up / process kill. We therefore skip
    # post-elimination redundancy removal in the axis-aligned branch.
    if axis_to_eliminate == -1
        removehredundancy!(p)
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
    tol = 1e-12
    for i in 1:length(v)
        if abs(v[i]) > tol
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