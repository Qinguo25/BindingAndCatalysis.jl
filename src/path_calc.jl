using LinearAlgebra
using SparseArrays
using Polyhedra
using CDDLib

# ==============================================================================
# 1. Data Structures & Conversion
# ==============================================================================

# Your input struct (representing Cx + C0 >= 0)
struct Vertex
    C_qK::SparseMatrixCSC{Float64, Int} 
    C0_qK::Vector{Float64}              
end

# Wrapper for the result
struct SubRegime
    geometry::Vertex
    covered_by::Vector{Int}
end

# Choose the library. CDDLib is exact (Rational) which avoids floating point errors
# but can be slower. We use 'Float64' mode if possible, or fallback to default.
const LIB = CDDLib.Library() 

"""
    to_polyhedron(v::Vertex)

Converts your Vertex struct (Cx + C0 >= 0) to a Polyhedra.jl object.
Polyhedra.jl expects Ax <= b.
Transformation: Cx >= -C0  ==>  -Cx <= C0
So: A = -C, b = C0
"""
function to_polyhedron(v::Vertex)
    # CDDLib usually prefers dense matrices
    A = -Matrix(v.C_qK) 
    b = v.C0_qK
    
    # Create the H-representation
    h = HPolyhedron(A, b)
    
    # Create polyhedron object using the CDD library
    return polyhedron(h, LIB)
end

"""
    from_polyhedron(p::Polyhedron)

Converts a Polyhedra object back to your Vertex struct.
"""
function from_polyhedron(p::Polyhedron)
    # Ensure we have the H-representation
    h = hrep(p)
    
    # Extract A and b where Ax <= b
    # We mix halfspaces (inequalities) and hyperplanes (equalities) if any
    # But usually, regions are full-dimensional, so mostly halfspaces.
    
    # Use Polyhedra's mixed H-rep iterator
    # Note: accessors might vary slightly by version, getting A and b safely:
    A_list = Vector{Float64}[]
    b_list = Float64[]
    
    for hp in halfspaces(h)
        push!(A_list, hp.a)
        push!(b_list, hp.β)
    end
    
    # Combine into Matrix/Vector
    if isempty(A_list)
        return Vertex(spzeros(0,0), Float64[])
    else
        A_mat = reduce(vcat, transpose.(A_list))
        b_vec = b_list
        
        # Convert back to your Cx + C0 >= 0 format
        # -Cx <= C0  ==>  C = -A, C0 = b
        return Vertex(sparse(-A_mat), b_vec)
    end
end

# ==============================================================================
# 2. Geometric Logic (Replacing the JuMP Solver)
# ==============================================================================

function process_cell_polyhedra(
    current_poly,          # Polyhedron Object
    candidates::Vector{Int}, 
    all_regimes_poly       # Vector of Polyhedron Objects
)
    fully_covering = Int[]
    partially_overlapping = Int[]

    # --- PRUNING STEP ---
    for idx in candidates
        regime = all_regimes_poly[idx]
        
        # 1. Intersection Check (Disjoint?)
        # computing intersection geometry is expensive, so checking isempty is better
        # Note: CDDLib is fast at logical checks if set up correctly
        inter = intersect(current_poly, regime)
        
        if isempty(inter)
            continue # Disjoint
        end
        
        # 2. Containment Check (Inside?)
        # issubset(inner, outer)
        if issubset(current_poly, regime)
            push!(fully_covering, idx)
        else
            push!(partially_overlapping, idx)
        end
    end

    # --- BASE CASE ---
    if isempty(partially_overlapping)
        # Convert the geometry back to your struct
        geo_struct = from_polyhedron(current_poly)
        return [SubRegime(geo_struct, fully_covering)]
    end

    # --- SPLIT STEP ---
    # Pick the first facet of the first partial regime to split space
    splitter_idx = partially_overlapping[1]
    splitter_regime = all_regimes_poly[splitter_idx]
    
    # Extract the first halfspace (inequality) from the splitter
    # We need to access the raw H-rep
    splitter_hrep = hrep(splitter_regime)
    hp = first(halfspaces(splitter_hrep)) # This is a HalfSpace object (a⋅x ≤ β)

    # Left Child: Cell ∩ HalfSpace
    left_poly = intersect(current_poly, polyhedron(hp, LIB))
    
    # Right Child: Cell ∩ Complement(HalfSpace)
    # Complement of a⋅x ≤ β is a⋅x > β, which we approximate as -a⋅x ≤ -β
    # (Polyhedra.jl handles hyperplane logic, but for splitting we create the opposite halfspace)
    hp_opp = HalfSpace(-hp.a, -hp.β) 
    right_poly = intersect(current_poly, polyhedron(hp_opp, LIB))

    # --- RECURSION ---
    # Pass 'partially_overlapping' to children (coverage logic is same as before)
    
    # Optimization: Check if children are empty before recursing
    results = SubRegime[]
    
    if !isempty(left_poly)
        append!(results, process_cell_polyhedra(left_poly, partially_overlapping, all_regimes_poly))
    end
    
    if !isempty(right_poly)
        append!(results, process_cell_polyhedra(right_poly, partially_overlapping, all_regimes_poly))
    end
    
    # Update coverage
    final_results = SubRegime[]
    for res in results
        new_coverage = sort(unique(vcat(res.covered_by, fully_covering)))
        push!(final_results, SubRegime(res.geometry, new_coverage))
    end
    
    return final_results
end

# ==============================================================================
# 3. Main Function
# ==============================================================================

function decompose_regimes(regimes::Vector{Vertex})
    # 1. Convert all inputs to Polyhedra objects once
    polys = to_polyhedron.(regimes)
    
    # 2. Define Universe (Whole Space)
    # In Polyhedra, a universe can be empty constraints.
    dim = size(regimes[1].C_qK, 2)
    universe_h = HPolyhedron(zeros(0, dim), zeros(0))
    universe = polyhedron(universe_h, LIB)
    
    # 3. Run Recursion
    candidates = collect(1:length(regimes))
    return process_cell_polyhedra(universe, candidates, polys)
end