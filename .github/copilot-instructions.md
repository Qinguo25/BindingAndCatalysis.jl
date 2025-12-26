## Copilot Instructions for `bnc_julia`

### Purpose
- Julia toolkit for binding/catalysis network regimes (vertices), feasibility polyhedra, and Hill/coefficient analysis.
- Source lives under `src/`; runnable examples live under `Examples/` (notably `multi_binding_hill.ipynb`).

### Quick start (notebook or REPL)
- From `Examples/`: `using Pkg; Pkg.activate("../")`; `include("../src/initialize.jl")`.
- Build model: `model = Bnc(N=N)` (or `Bnc(L=L)` if L provided) → `find_all_vertices!(model)`.
- Explore: `get_vertices_graph!(model, full=true)`; `show_condition_qK(model, i; log_space=false)`; `show_expression_x(model, i; log_space=false)`; `get_H!(model, i)` / `get_H0!(model, i)`.

### Key files under `src/`
- `initialize.jl`: types (`Bnc`, `Vertex`, `VertexGraph`, `CatalysisData`) and constructors.
- `regimes.jl`, `regime_enumerate.jl`, `regime_assign.jl`: regime discovery, enumeration, and property attachment.
- `regime_graphs.jl`: graph construction, SISO extraction, neighbor logic.
- `symbolics.jl`, `qK_x_mapping.jl`: symbolic expressions and x↔qK mappings.
- `numeric.jl`, `volume_calc.jl`: numerical/Hill calculations, polyhedron volumes.
- `visualize.jl`, `helperfunctions.jl`: plotting/display helpers and utilities.

### How to ask Copilot
- “Locate `get_C_C0_qK!` definition and summarize what it returns.”
- “Add an edge-label helper in `regime_graphs.jl` that tags edges with Hill coefficient differences.”
- “Suggest a new example cell in `Examples/multi_binding_hill.ipynb` that sweeps N variants and plots `show_condition_qK`.”
- “Profile `find_all_vertices!` for a 3x3 system and point out threading hotspots.”

### Common development tasks
- **New vertex observable**: implement in the relevant module (often `regimes.jl`/`numeric.jl`), cache on `Vertex` if expensive, expose via `get_*`, and wire into summaries/graphs when useful.
- **Graph tweaks**: edit `regime_graphs.jl`; keep edges between regimes differing by a single component; maintain symmetry.
- **Symbolic vs numeric**: keep symbolic manipulations in `symbolics.jl`; do numerical evaluation/stability checks in `numeric.jl`.
- **Performance**: prefer sparse matrices, use `@views` in hot loops, avoid recomputing polyhedra/nullity (cache), keep threaded loops thread-safe.

### Using `Examples/multi_binding_hill.ipynb`
- Shows minimal setup, vertex finding, graph extraction, and condition display; extend with additional N matrices, `get_H!`/`get_H0!` calls, and regime graph visualizations.

### Style guidelines
- Add docstrings to public APIs; keep comments brief and for non-obvious logic.
- Place functions in the file matching their concern (enumeration vs graph vs visualization).
- Prefer pure functions and avoid global mutable state; cache on model/vertex structs when needed.

### If Copilot is unsure
- First search in `src/` for similar helpers before proposing new ones.
- Surface definitions/usages of a symbol before editing or refactoring.
