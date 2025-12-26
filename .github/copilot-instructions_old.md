# BNC (Binding and Catalysis Networks) Julia Package - Copilot Instructions

## Project Overview

This is a Julia package for analyzing **Binding and Catalysis Networks** (BNC). The package computes feasible equilibrium states (vertices/regimes) of biochemical networks with binding and catalytic reactions, analyzes their properties, and visualizes relationships between these regimes.

### Key Concepts
- **Vertices/Regimes**: Feasible equilibrium states of the system defined by regime vectors (permutation vectors of species ordering)
- **Binding Networks**: Networks where molecules bind together (E + X ↔ C type reactions)
- **Catalytic Networks**: Reactions with catalytic constants and rate laws
- **Conservation Laws**: Represented by the L matrix; define conserved quantities
- **Binding Reactions**: Represented by the N matrix; define binding equilibrium relationships
- **qK Space**: Logarithmic space of binding constants used for easier computation

## Main Data Structures

### Core Structs

#### `Bnc{T}` (Main Model)
- `N::Matrix{Int}` - Binding reaction matrix (each column is a reaction)
- `L::Matrix{Int}` - Conservation law matrix
- `n::Int` - Number of variables/species
- `r::Int` - Number of binding reactions
- `d::Int` - Number of conserved quantities (d = n - r)
- `x_sym::Vector{Num}` - Species symbols
- `q_sym::Vector{Num}` - Conserved quantity symbols
- `K_sym::Vector{Num}` - Binding constant symbols
- `vertices_perm::Vector{Vector{T}}` - All feasible regime vectors
- `vertices_data::Vector{Vertex}` - Detailed data for each vertex
- `vertices_graph::VertexGraph` - Graph connecting neighboring regimes

#### `Vertex{F,T}` (Individual Regime)
- `perm::Vector{T}` - Regime vector (ordering of species)
- `idx::Int` - Index in vertices list
- `asymptotic_flag::Bool` - Whether regime is physically realizable
- `nullity::Int` - Nullity of the jacobian at this vertex
- Computed properties: polyhedron, binding constants expressions, H0 values, etc.

#### `CatalysisData`
- `S::Matrix{Int}` - Catalysis change in qK space
- `aT::Matrix{Int}` - Catalysis index and coefficients (rate law exponents)
- `k::Vector` - Rate constants for catalytic reactions
- `cat_x_idx::Vector{Int}` - Indices of species that catalyze each reaction

## Main Workflow Functions

### 1. Model Initialization
```julia
# Create model from binding matrix N (automatically calculates L)
model = Bnc(N=N)

# Or provide conservation law matrix L explicitly
model = Bnc(L=L)

# With custom species/constant symbols
model = Bnc(N=N, x_sym=x_sym, q_sym=q_sym, K_sym=K_sym)
```

### 2. Finding Vertices (Feasible Regimes)
```julia
find_all_vertices!(model)  # Computes all possible regime vectors
summary(model)             # Print summary of found vertices

# Get specific vertices
get_vertices(model, singular=false)  # Non-singular vertices
get_vertices(model, real=true)       # Asymptotically real vertices
```

### 3. Building Vertex Graphs
```julia
get_vertices_graph!(model, full=true)  # Build graph connecting neighboring regimes
SISO_graph(model, component)           # Extract single-input-single-output subgraph
```

### 4. Querying Individual Vertex Properties
```julia
# For a specific vertex index
get_vertex!(model, idx)           # Retrieve full vertex data
get_idx(model, idx)               # Get regime vector
get_perm(model, idx)              # Get permutation
get_polyhedron(model, idx)        # Get feasibility polyhedron
get_nullity!(model, idx)          # Compute nullity
get_C_C0_qK!(model, idx)          # Get binding constants expressions
get_H!(model, idx)                # Get Hill coefficient
get_H0!(model, idx)               # Get reference Hill coefficient
show_condition_qK(model, idx)     # Display binding constant constraints
```

### 5. Visualization & Display
```julia
show_condition_qK(model, idx; log_space=false)    # Show qK constraints
show_expression_x(model, idx; log_space=false)    # Show species relationships
get_vertices_graph!(model, full=true)             # Build and display graph
render_array(matrix)                               # Pretty-print arrays
```

## Source Code Structure

### Key Modules (in `src/`)

- **initialize.jl** - Core data structures: `Bnc`, `Vertex`, `CatalysisData`, `VertexGraph`
- **regimes.jl** - Core regime computation and vertex finding algorithms
- **regime_enumerate.jl** - Enumeration of all feasible regimes
- **regime_assign.jl** - Assigning properties to discovered regimes
- **regime_graphs.jl** - Building and analyzing graphs between regimes
- **symbolics.jl** - Symbolic computation of binding constants and expressions
- **numeric.jl** - Numerical calculations (Hill coefficients, conservation laws)
- **qK_x_mapping.jl** - Mapping between x (species) and qK (binding constants) spaces
- **volume_calc.jl** - Computing feasibility polyhedron volumes
- **visualize.jl** - Visualization functions
- **helperfunctions.jl** - Utility functions (matrix conversions, helper methods)

## Common Development Tasks

### Adding New Properties to Vertices
1. Define getter function in appropriate module
2. Add computation logic considering regime vector `perm`
3. Cache results if expensive (use `Vertex.cached_*` fields)
4. Ensure dimension compatibility with N, L matrices

### Extending Graph Analysis
1. Modify `regime_graphs.jl` to add new edge types or properties
2. Ensure bidirectional edges are consistent
3. Test with examples in `Examples/` folder
4. Update visualization in `visualize.jl` if needed

### Adding Numerical Solvers
1. Update `numeric.jl` or create new solver module
2. Handle both invertible (nullity=0) and non-invertible regimes
3. Test convergence and stability
4. Consider cached results to avoid recomputation

## Best Practices

### Code Organization
- Keep symbolic computations in `symbolics.jl`
- Keep numerical computations in `numeric.jl`
- Use sparse matrices for efficiency (L_sparse, N_sparse, etc.)
- Cache expensive computations (nullity, polyhedron, etc.)

### Performance Considerations
- Use `@views` for matrix slicing to avoid copies
- Leverage sparse matrices (SparseMatrixCSC) throughout
- Use `Threads.@threads` for parallelizable loops (as in `regime_enumerate.jl`)
- Use `ThreadsX` for higher-level parallel operations

### Testing & Validation
- Test with simple examples (2-species, 1-reaction systems)
- Validate Hill coefficient calculations against known models
- Check rank and nullity computations
- Verify polyhedron constraints are satisfied

### Documentation
- Use Julia docstrings with """ """ for public functions
- Explain mathematical concepts in comments for complex algorithms
- Include type signatures for clarity
- Reference papers/theory where applicable

## Example Workflows

### Workflow 1: Simple Binding System
```julia
# E + X ↔ C with one binding reaction
N = [1 1 -1]
model = Bnc(N=N)
find_all_vertices!(model)
get_vertices(model)  # Get all feasible regimes
```

### Workflow 2: Multi-Species Competition
```julia
# E + X1 ↔ C1, E + X2 ↔ C2, X1 + X2 ↔ C3 (competitive binding)
N = [1 1 0 -1 0 0;
     1 0 1 0 -1 0;
     0 1 1 0 0 -1]
model = Bnc(N=N)
find_all_vertices!(model)
get_vertices_graph!(model, full=true)
```

### Workflow 3: Complex Analysis with Catalysis
```julia
# Use N and S matrices for binding + catalytic reactions
N = [...]  # Binding matrix
S = [...]  # Catalysis change matrix
aT = [...]  # Catalysis exponents
model = Bnc(N=N)
model.catalysis = CatalysisData(model.n, S, aT, nothing, nothing)
find_all_vertices!(model)
```

## Common Issues & Solutions

### Issue: Vertices not being found
- Check N matrix has full row rank (rank should equal r = n - d)
- Verify L matrix construction (use `L_from_N` if needed)
- Ensure conservation law compatibility

### Issue: Nullity calculations incorrect
- Verify regime vector `perm` is valid (within expected ranges)
- Check Jacobian computation in `numeric.jl`
- Consider numerical precision (use higher precision if needed)

### Issue: Graph construction fails
- Ensure all vertices are properly initialized
- Check that edges connect vertices differing in exactly one regime component
- Verify sparse matrix operations preserve sparsity

## Important Functions to Know

- `rank(N)` - Get rank of binding matrix
- `nullspace(N')` - Get nullspace (conservation laws)
- `svd(N)` - Singular value decomposition for basis analysis
- `lu()`, `inv()` - For matrix inversion in regime calculations
- `Polyhedra.vrep()`, `hrep()` - For polyhedron representation
- `eliminate()` - Project polyhedra to subspaces
