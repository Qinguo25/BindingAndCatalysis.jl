# GitHub Copilot instructions for BindingAndCatalysis.jl

## Project context
- This repository implements binding-network and catalysis modeling in Julia.
- Core usage patterns are in `Examples/Minimal_example.ipynb` and should stay in sync with the public API.
- The main module is `BindingAndCatalysis` in `src/BindingAndCatalysis.jl`.

## Code style
- Use Julia docstrings with a consistent structure:
  - signature line
  - short description
  - optional `# Arguments`, `# Keyword Arguments`, and `# Returns`
- Prefer clear, descriptive variable names (`bnc`, `perm`, `qK`, `logx`).
- Keep numerical routines allocation-conscious and favor `SparseMatrixCSC` when appropriate.
- Never wrap imports in `try/catch` blocks.

## Domain conventions
- `N` is the stoichiometry matrix; `L` is the conservation matrix with `N * L' = 0`.
- `x` denotes species concentrations, `q` denotes totals, and `K` denotes binding constants.
- Functions frequently accept log10-space inputs/outputs; honor `input_logspace` and `output_logspace` flags.
- Regimes/vertices are identified by permutations; many APIs accept either a permutation or its index.

## Testing and examples
- Update `Examples/Minimal_example.ipynb` when adding new public APIs or altering behavior.
- Document any new functionality in the README when user-facing.

## File navigation
- Mapping and numerical solvers: `src/qK_x_mapping.jl`, `src/numeric.jl`
- Regimes and graphs: `src/regimes.jl`, `src/regime_graphs.jl`, `src/regime_assign.jl`
- Visualization: `src/visualize.jl`
- Symbolic utilities: `src/symbolics.jl`
