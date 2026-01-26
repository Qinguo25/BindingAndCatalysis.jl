using Polyhedra
using CDDLib
using LinearAlgebra

"""
    get_prism_along_vector(C, C0, v)

Returns the H-representation of the prism formed by projecting the 
polyhedron C*x + C0 >= 0 along vector v.
"""
function get_prism_along_vector(C, C0, v)
    dim = size(C, 2)
    v_norm = normalize(v)
    
    # 1. Define Input, construct Polyhedron
    h_in = HPolyhedron(-C, C0)
    poly_in = polyhedron(h_in, CDDLib.Library())

    # 2. Check Alignment: Is v parallel to a coordinate axis?
    # We look for a standard basis vector (e.g., [0, 1, 0])
    axis_idx = findfirst(x -> abs(x) > 1.0 - 1e-8, v_norm)
    is_parallel = !isnothing(axis_idx) && count(x -> abs(x) > 1e-8, v_norm) == 1

    if is_parallel
        println("Vector is parallel to dimension $axis_idx. Fast path.")
        
        # A. Eliminate the dimension (Projection)
        # This gives us the constraints in (n-1) dimensions
        poly_shadow = eliminate(poly_in, [axis_idx])
        h_shadow = hrep(poly_shadow)
        
        # B. Inflate back to n-dimensions (The Prism Step)
        # We insert a column of ZEROS at 'axis_idx'. 
        # A 0 coefficient means this dimension does not constrain the shape.
        A_small = h_shadow.A
        b_final = h_shadow.b
        
        # Splicing the zero column into A
        n_constraints = length(b_final)
        A_final = zeros(n_constraints, dim)
        
        # Copy columns: 1 to axis_idx-1  ->  Into 1 to axis_idx-1
        if axis_idx > 1
            A_final[:, 1:axis_idx-1] = A_small[:, 1:axis_idx-1]
        end
        # Copy columns: axis_idx to end  ->  Into axis_idx+1 to end
        if axis_idx < dim
            # Note: The shadow matrix has one less column, so we take from axis_idx to end
            A_final[:, axis_idx+1:end] = A_small[:, axis_idx:end]
        end
        
        # Convert back to your C format (C = -A)
        return -A_final, b_final

    else
        println("Vector is arbitrary. Rotating space.")
        
        # A. Construct Rotation to align v with the LAST dimension (Z-axis)
        e_n = zeros(dim); e_n[end] = 1.0
        
        # Rotation R such that R * v = e_n
        u = v_norm - e_n
        u = u / norm(u)
        R = I(dim) - 2 * (u * u') # Householder reflection
        
        # Rotate the input constraints
        poly_rot = R * poly_in
        
        # B. Eliminate the last dimension
        poly_shadow_rot = eliminate(poly_rot, [dim])
        h_shadow_rot = hrep(poly_shadow_rot)
        
        # C. Inflate to Prism in Rotated Space
        # Append a column of zeros at the end (the Z-dimension)
        A_small = h_shadow_rot.A
        b_final = h_shadow_rot.b
        
        A_prism_rot = hcat(A_small, zeros(length(b_final)))
        
        # D. Rotate BACK to Original Space
        # The constraints in rotated space are: A_prism_rot * y <= b
        # Substitute y = R * x  ->  A_prism_rot * R * x <= b
        # So A_final = A_prism_rot * R
        poly_prism_rot = HPolyhedron(A_prism_rot, b_final)
        
        # Since R is symmetric and orthogonal (Householder), R' = R
        # We can apply the transform directly to the polyhedron object
        poly_final = R' * poly_prism_rot # Rotate back
        
        h_final = hrep(poly_final)
        return -h_final.A, h_final.b
    end
end