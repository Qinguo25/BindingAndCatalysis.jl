# BindingAndCatalysis.jl Architecture

This document describes the current architecture of `BindingAndCatalysis.jl`.
It is the canonical maintainer-facing architecture reference for the package.

## Overview

`BindingAndCatalysis.jl` analyzes biochemical systems with a fast binding
equilibrium layer and an optional slower catalysis layer. The central object is
`Bnc`.

`Bnc` stores:

- model matrices and symbols: `L`, `N`, `x_sym`, `q_sym`, `K_sym`;
- optional catalysis data attached by `update_catalysis!`;
- lazily built binding, catalysis, and Bnc-regime caches;
- regime graphs, compiled classifiers, and numerical integration helpers.

At a high level:

```text
Bnc model
  ├─ Binding layer: q = Lx, N log x = log K
  │   ├─ BindRegime
  │   ├─ binding RegimeGraph
  │   └─ qK <-> x numerical solvers
  ├─ Catalysis layer: log v = Π log x + log k, qdot = Γv
  │   ├─ CatalysisData
  │   ├─ CatalysisRegime
  │   └─ catalysis RegimeGraph
  ├─ Binding-Catalysis fixed-point layer
  │   ├─ BncRegime = BindRegime × CatalysisRegime
  │   ├─ Bnc RegimeGraph
  │   └─ stability / fixed-point conditions
  ├─ Linear BNC control layer
  │   ├─ BncLinearControlModel
  │   ├─ A/B/C/D regime linearization
  │   └─ invariance / responsiveness metrics
  ├─ Constrained regime analysis layer
  │   ├─ ParameterConstraints
  │   ├─ restricted regime diagnostics
  │   └─ constrained multistability/R-index sampling
  └─ Higher-level tools
      ├─ volume estimation
      ├─ symbolic rendering
      ├─ SIMO path workflows
      └─ visualization extension
```

## Core Types

### `Bnc`

`Bnc` is the main binding-network model type. After `update_catalysis!`, it
also owns the catalysis data.

Responsibilities:

- Store sparse `L` and `N` matrices.
- Store `x_sym`, `q_sym`, `K_sym`, and catalysis-related symbols.
- Store `catalysis::Union{Nothing,CatalysisData}`.
- Cache `BindRegimes`, `BncRegimes`, and regime graphs.
- Cache numerical integration helpers and low-rank matrix helpers.

Some fields still use legacy names, such as `vertices_graph` and
`_vertices_Nρ_inv_dict`. Their current meaning is regime graph / regime inverse
cache, but the field names are preserved for compatibility.

### `Regimes`

`Regimes` is the collection wrapper for regime objects:

```julia
Regimes(regimes_perm_dict, regimes_data)
```

- `regimes_data` stores regime objects.
- `regimes_perm_dict` maps a permutation to its index.
- `vertices_data` and `vertices_perm_dict` remain readable as legacy property
  aliases.

### `BindRegime`

`BindRegime` represents a binding dominance regime.

Key fields:

- `perm`: dominant species for each conservation-total row.
- `idx`: index in the binding-regime collection.
- `is_asymptotic`: whether the regime is asymptotic.
- `P, P0`: dominance affine approximation from `log x` to `log q`.
- `M, M0`: affine map from `log x` to `log(q,K)`.
- `C_x, C0_x`: x-space dominance conditions.
- `H, H0`: affine map from `log(q,K)` to `log x` for regular regimes, or
  directional/fallback data for singular regimes.
- `C_qK, C0_qK`: qK-space conditions.
- `nullity`: regular, singular, or higher-codimension status.
- `volume`: optional cached `Volume`.

### `CatalysisData` and `CatalysisRegime`

`CatalysisData` is attached to `Bnc` by `update_catalysis!`. It describes the
slow reaction layer:

```math
\log v = \Pi\log x + \log k,\qquad \dot q = \Gamma v.
```

`CatalysisRegime` represents a flux-dominance and fixed-point-balance regime.

Key fields:

- `perm`: flux dominance permutation.
- `P, P0`: positive/negative flux balance expression.
- `C, C0`: v-space dominance conditions.
- `PΠ, CΠ`: the same data pulled back through
  `log v = Π log x + log k`.

### `BncRegime`

`BncRegime` pairs one binding regime with one catalysis regime:

```julia
BncRegime(bind_rgm, catalysis_rgm)
```

It describes a Binding-Catalysis fixed-point regime.

Key fields:

- `bind_rgm`, `catalysis_rgm`;
- `perm = (binding_perm, catalysis_perm)`;
- `is_feasible`;
- `nlt`: Bnc-regime nullity;
- `H, H0`: regular Bnc-regime affine map from `log(w,K,k)` to `log x`;
- `C_qKk_cat, C0_qKk_cat`: qKk-space catalysis consistency conditions;
- `C_wKk, C0_wKk`: wKk-space fixed-point conditions;
- `H_bd`: matrix used by the stability classifier;
- `is_stable`: cached stability flag, where `1` means stable, `-1` unstable,
  `2` unknown, and `0` not computed.

## Mathematical Layers

### Binding Layer

The binding model is

```math
q = Lx,\qquad N\log x = \log K.
```

Each binding regime fixes the dominant species in each conservation-total row.
This gives a piecewise affine chart:

```math
\log q = P\log x + P_0
```

and

```math
\log(q,K)=
\begin{bmatrix}P\\N\end{bmatrix}\log x+
\begin{bmatrix}P_0\\0\end{bmatrix}.
```

Regular regimes have a direct affine expression `H, H0` for
`log(q,K) -> log x`. Singular regimes keep conditions and available directional
data instead.

### Catalysis Layer

The catalysis layer is

```math
\log v = \Pi\log x + \log k,\qquad \dot q = \Gamma v.
```

`update_catalysis!` partitions q variables into:

- `q_cat`: totals participating in slow dynamics;
- `w`: totals conserved by the slow dynamics.

The reduced dynamics are

```math
\dot q_{cat}=Sv,\qquad \dot w=0.
```

### Binding-Catalysis Fixed-Point Layer

A Binding-Catalysis fixed-point regime is determined by both a binding chart
and a catalysis chart. Regular Bnc regimes produce `log(w,K,k) -> log x`, and
then an expression for `q_cat`. Singular and high-nullity Bnc regimes retain
consistency conditions and any computable local data.

Stability is determined from `H_bd` with `judge_dstable`:

```math
H^{bd}=P\Pi H_{q_{cat}}.
```

`is_stable(rgm)` returns `true`, `false`, or `missing`. `missing` means the
algorithm could not determine stability. Use `stability_code(rgm)` to retrieve
the underlying integer code.

### Linear BNC Control Layer

`src/BncControl.jl` provides public control-oriented summaries built on top of a
`BncRegime`. The state is `log(q_cat)`, inputs are selected from
`wKk_symbol(rgm)`, and outputs are selected from either `x_symbol(rgm)` or
`q_cat_symbol(rgm)`.

The local model is:

```math
\delta\dot q_{cat}=A\delta q_{cat}+B\delta u,\qquad
\delta y=C\delta q_{cat}+D\delta u.
```

The matrix names follow standard state-space control terminology:

- `A`: state/system/dynamics matrix,
- `B`: input/control matrix,
- `C`: output/observation matrix,
- `D`: direct-feedthrough/direct-transmission matrix.

The public constructor is:

```julia
linear_control_model(rgm; input=:all, output=:x, timescale=:identity)
```

`timescale=:identity` uses the regime derivative directly. A positive vector can
be supplied to row-scale `A` and `B`; physical timescales are not inferred from a
regime object alone.

The layer also exposes:

```julia
control_metrics
controllability_matrix
output_controllability_matrix
output_controllability_row
markov_coefficients
controllability_gramian
output_energy
dynamic_steady_state_gain
affine_steady_state_gain
steady_state_gain
steady_state_invariance
is_steady_state_invariant
input_drive
input_responsiveness
input_responsive
compare_input_responsiveness
```

`steady_state_invariance` has two explicit standards. `standard=:affine`
checks the exact affine steady-state map of the regime and is the appropriate
standard for Xiao-style steady-state invariance. `standard=:dynamic_dc_gain`
checks the state-space DC gain `D - C*(A \ B)`.

Input responsiveness separates generic control quantities from
project-specific biological criteria. `standard=:direct_feedthrough` uses the
`D` matrix. `standard=:direct_flux` requires explicit `positive_flux` and
optional `negative_flux` selectors and is implemented through `input_drive`;
the package does not infer circuit-specific flux signs from species names.
`standard=:output_reachability` supports `direction=:any`, `:positive`, and
`:negative`. `standard=:gramian` supports energy-like and amplitude-like
threshold semantics through `threshold_scale`.

`hbd_source(rgm)` and `get_H_bd_info(rgm)` expose whether the BNC dynamic
derivative came from exact regime derivatives or numerical binding derivatives.
This provenance is important for singular binding regimes. Note that
`get_bnc_regimes(...; singular=false)` filters BNC nullity; it does not imply
that the underlying binding derivative was exact.

### Constrained Regime Analysis Layer

`src/RegimeConstraints.jl` provides analysis-time parameter constraints and
constrained multistability summaries. This layer does not mutate `Bnc` or
replace the model's regime caches. A constraint family is passed explicitly to
restriction, overlap, and sampling functions.

The constrained analysis layer has two central objects:

```julia
ParameterChart
ParameterConstraints
```

`ParameterChart` stores the affine relation from reduced analysis coordinates
to original chart coordinates:

```math
z = F y + F_0.
```

This is the package-level representation of biological parameter
identification such as "these degradation rates are the same reduced
parameter." It preserves the measure convention used by constrained R-index
calculations. `map`/`groups` are user-facing constructors; direct `F,F0` input is
available for advanced affine charts.

`ParameterConstraints` stores additional equality/inequality rows after the
chart has been chosen. By default, constraints attached to a `ParameterChart`
are interpreted in the reduced coordinates. Use `symbols=:original` to write
constraints in the original chart and pull them back through `F,F0`.

The older matrix constraint convention remains supported: the first `nullity`
rows are equalities and later rows are inequalities.

Default chart selection:

- binding-only analysis uses `:qK`;
- BNC/catalysis analysis uses `:wKk`.

The public functions are:

```julia
parameter_chart
parameter_constraints
restrict_polyhedron
restrict_regime
restrict_regimes
stable_regime_intersections
multistability_profile
multistability_R_index
is_full_dimensional
```

`multistability_profile` samples the constraint region first, then counts how
many stable restricted BNC regimes contain each accepted sample. This makes the
R-index denominator explicit: `denominator=:constraint_region`.

`multistability_profile(...; mode=:asymptotic_R)` strips offsets and samples
recession-cone membership, returning `denominator=:constraint_cone`.
`multistability_R_index` is the report-oriented wrapper that defaults to
`mode=:asymptotic_R` and returns deterministic regime counts together with the
conditional `R_multistability`.

## Public API Conventions

The maintained API favors explicit function names and clear keyword semantics.

### Regime Queries

New code should prefer full names:

```julia
get_binding_regime(model, index_or_perm)
get_binding_regimes(model; singular=nothing, asymptotic=nothing)
get_binding_indices(model; singular=nothing, asymptotic=nothing)
get_binding_perms(model; singular=nothing, asymptotic=nothing)

get_catalysis_regime(model, index_or_perm)
get_catalysis_regimes(model; asymptotic=nothing)
get_catalysis_indices(model; asymptotic=nothing)
get_catalysis_perms(model; asymptotic=nothing)

get_bnc_regime(model, bind, cat)
get_bnc_regimes(model; feasible=true, stable=nothing)
get_bnc_indices(model; feasible=true, stable=nothing)
get_bnc_perms(model; feasible=true, stable=nothing)
```

`get_regime(...)` is only a binding-regime convenience alias. It is not a
generic dispatch point for catalysis or Bnc regimes. Use `get_catalysis_regime`
and `get_bnc_regime` explicitly.

### Return Shape

New APIs should express return shape in the function name:

```julia
filter_regimes(...)              # selected indices
filter_regimes_mask(...)         # mask
filter_regimes_with_mask(...)    # selected, mask

assign_regime(...)               # permutation
assign_regime_index(...)         # index
assign_regime_qK_index(...)      # qK input, index
assign_regime_x_index(...)       # x input, index
```

`return_idx` is a legacy compatibility keyword. New internal code should not
propagate it.

### Coordinate-Space Keywords

Numerical input/output spaces use Symbol modes:

```julia
qK2x(model, qK; input=:log, output=:linear)
x2qK(model, x; input=:linear, output=:log)
assign_regime_index(model, qK; input=:log)
```

Supported modes are:

- `:linear`
- `:log`

`input_logspace` and `output_logspace` are compatibility keywords. Core entry
points translate them immediately to `input` and `output`; internal code should
use the Symbol mode.

Symbolic and display functions continue to use `log_space::Bool`, because this
is a display toggle rather than a coordinate-space mode.

### Filtering and Search

Filtering APIs use tri-state filters:

```julia
singular=true/false/nothing
asymptotic=true/false/nothing
feasible=true/false/nothing
stable=true/false/nothing
```

Assignment and search APIs use:

```julia
asymptotic_only=true/false
```

A single function should not expose both `asymptotic` and `asymptotic_only`.

### Recalculation and Tolerances

Cache refresh uses:

```julia
recompute=false
```

Numerical tolerances follow SciML naming:

```julia
reltol
abstol
```

Do not add new `rel_tol` or `abs_tol` keywords.

## Caches and Initialization

The explicit cache-building entry points are:

```julia
ensure_binding_regimes!(model)
ensure_catalysis_regimes!(model)
ensure_bnc_regimes!(model)
ensure_regime_data!(rgm)
```

They define the cache boundaries behind public getters:

- `ensure_binding_regimes!` calls `find_all_regimes!`.
- `ensure_catalysis_regimes!` calls `find_catalysis_regimes!`.
- `ensure_bnc_regimes!` calls `match_regimes!`.
- `ensure_regime_data!` materializes affine maps and conditions.

Maintenance rules:

- Getters may trigger lazy construction.
- Expensive recomputation should be guarded by `recompute` or explicit function
  calls.
- Regime objects cache expensive affine and condition fields.
- Graph and classifier caches live on the model or graph object.

## Regime Graphs

`RegimeGraph` stores adjacency between regimes. Each node is a regime; each edge
stores one or more separating hyperplanes in named coordinate charts.

Common charts:

- binding graph: `:x`, `:qK`;
- catalysis graph: `:v`, `:xk`;
- Bnc graph: `:xk`, `:qKk`, `:wKk`.

Callers use Symbol chart names rather than fixed integer slots.

Relevant files:

- `src/BindingRegimeGraph.jl`
- `src/CatalysisRegimeGraph.jl`
- `src/BncRegimeGraph.jl`
- `src/Mathcore/perm_graph_core.jl`

## Compiled Classifiers and Assignment

`src/regime_assign.jl` compiles qK hyperplane incidence data into a
`CompiledClassifier`.

Important fields:

- `regime_ids`: regime indices covered by the classifier;
- `dirs`, `bias`: hyperplane data;
- `allow_pos`, `allow_neg`: `BitVector` masks for fast candidate pruning.

Assignment flow:

1. `get_regimes_graph!(model; full=true)` builds the qK graph and hyperplane
   data.
2. `_build_qK_hyperplane_classifier` builds the classifier from incidence data.
3. `assign_regime_qK_index` classifies a logqK point.
4. If the classifier finds no candidate, `_assign_regime_qK_fallback_index`
   falls back to condition checking.

`assign_regime_x_index` uses x-space dominance logic. `assign_regime_qK_index`
with `x=...` first maps through `x2qK`.

## qK <-> x Numerical Layer

`src/qK_x_mapping.jl` provides:

- `x2qK`: direct map from x to qK;
- `qK2x`: numerical inverse map from qK to x;
- `qK2x_residual`: log-coordinate residual check;
- homotopy trajectories and catalysis simulations.

`qK2x` supports:

- `method=:homotopy`;
- `method=:free_energy`;
- `method=:newton_nullspace`;
- `method=:nlsolve`;
- `method=:regime`.

The default method is selected by `_default_method` and
`_resolve_qK2x_method`. For models where the free-energy assumptions do not
hold, `:free_energy` requests are redirected to `:homotopy`.

Solver-specific kwargs should document their forwarding target. SciML-style
kwargs should use `reltol`, `abstol`, `alg`, and `maxiters`.

## Volume Layer

`Volume` is defined in `src/volume_calc.jl`:

```julia
Volume(mean, var)
```

Volume estimation lives in `src/volume_calc_impl.jl`.

Main routes:

- binding regime volume can use the qK classifier route or the polyhedra route;
- Bnc regime volume uses wKk-space constraints;
- raw polyhedron volume uses Monte Carlo sampling.

Sampling keywords:

```julia
sampler=:gaussian
sampler=:uniform_box
batch_size
time_limit
reltol
abstol
show_progress
```

`get_volume` and `get_volumes` use cached `volume` fields. Force a refresh with
`recompute=true`.

## Symbolic Layer

`src/symbolics.jl` includes the symbolic implementation:

- `symbolic_symbols.jl`: symbol accessors;
- `symbolic_renderers.jl`: expression and condition rendering;
- `symbolic_api.jl`: public symbolic functions;
- `symbolic_paths.jl`: SIMO path symbolic helpers.

Common public APIs:

```julia
show_condition_x(...)
show_condition_qK(...)
show_condition_xk(...)
show_condition_qKk(...)
show_condition_wKk(...)
show_expression_x(...)
show_expression_qK(...)
show_expression_qcat(...)
show_catalysis_dynamics(...)
```

Symbolic rendering uses `log_space::Bool` for display mode.

## SIMO Path Workflow

`src/SIMO.jl` includes the SIMO implementation under `src/simo/`:

- `core.jl`: `SIMOPaths`, path enumeration, graph helpers;
- `polyhedra.jl`: node, edge, and path polyhedron construction;
- `reaction_order.jl`: reaction-order summaries along paths;
- `display.jl`: printing and path formatting.

`SIMOPaths` caches:

- qK graph;
- source and sink nodes;
- regime paths;
- node, edge, and path polyhedra;
- path volumes.

Path polyhedra and path volumes are lazy caches. Refresh them with
`recompute=true`.

## Visualization Extension

Visualization is implemented as an optional package extension:

```toml
[weakdeps]
GraphMakie = ...
Makie = ...

[extensions]
BindingAndCatalysisVisualizationExt = ["GraphMakie", "Makie"]
```

The main module defines stubs in `src/visualize.jl`. If optional visualization
packages are unavailable, these stubs throw clear errors.

The extension file is:

```text
ext/BindingAndCatalysisVisualizationExt.jl
```

It includes the visualization sources into the parent module:

- `simo_plot.jl`
- `graphs.jl`
- `rop.jl`
- `poly_slices.jl`
- `regime_partition.jl`

Maintenance notes:

- Plotting-style kwargs may pass through `kwargs...` to Makie / GraphMakie.
- Package-specific mathematical semantics should use explicit keywords such as
  `chart`, `ranges`, `fixed`, and `asymptotic_only`.
- The extension can later be split into Makie-only and GraphMakie-specific
  extensions.

## Source Loading Order

`src/BindingAndCatalysis.jl` is the module entry point. Its load order is:

1. Dependencies and exact numeric types.
2. `volume_calc.jl` and shared core types.
3. `Bnc`, `CatalysisData`, `BindRegime`, and `Regimes` struct definitions.
4. `initialize.jl`.
5. Mathcore files:
   - `find_matrix_vertex.jl`
   - `d_stable.jl`
   - `perm_graph_core.jl`
   - `SparseSparse_modified.jl`
   - `matrix_inverse.jl`
   - `graph_propagate.jl`
6. Numerical and assignment layers:
   - `helperfunctions.jl`
   - `qK_x_mapping.jl`
   - `regime_assign.jl`
   - `volume_calc_impl.jl`
   - `numeric.jl`
7. Regime APIs:
   - `RegimeCore.jl`
   - `BindingRegime.jl`
   - `CatalysisRegime.jl`
   - `BncRegime.jl`
8. Linear BNC control API:
   - `BncControl.jl`
9. Graph APIs:
   - `BindingRegimeGraph.jl`
   - `CatalysisRegimeGraph.jl`
   - `BncRegimeGraph.jl`
10. High-level APIs:
   - `SIMO.jl`
   - `symbolics.jl`
   - `RegimeConstraints.jl`
   - `visualize.jl`
   - `old_api.jl`

New files should be added to the layer they belong to. Avoid introducing early
files that depend on methods loaded later.

## File Map

### Core Model and Regimes

- `src/BindingAndCatalysis.jl`: module entry, core type definitions, include
  order.
- `src/initialize.jl`: model construction, catalysis attachment, cache helpers.
- `src/RegimeCore.jl`: shared network accessors, regime getters, filters,
  affine maps, condition accessors.
- `src/BindingRegime.jl`: binding regime construction, summary, neighbors,
  volume access.
- `src/CatalysisRegime.jl`: catalysis regime construction and conditions.
- `src/BncRegime.jl`: Bnc regime matching, fixed-point conditions, stability.
- `src/BncControl.jl`: BNC linear control models, control metrics,
  steady-state invariance, input responsiveness, and H_bd provenance helpers.

### Mathcore

- `src/Mathcore/find_matrix_vertex.jl`: regime enumeration.
- `src/Mathcore/d_stable.jl`: d-stability classification.
- `src/Mathcore/perm_graph_core.jl`: permutation graph primitives.
- `src/Mathcore/matrix_inverse.jl`: exact / sparse affine inverse helpers.
- `src/Mathcore/graph_propagate.jl`: graph propagation of affine data.

### Numerical and Assignment

- `src/qK_x_mapping.jl`: qK/x maps, homotopy, catalysis simulations.
- `src/regime_assign.jl`: compiled classifiers and assignment.
- `src/numeric.jl`: Jacobian and reaction-order numerical helpers.
- `src/volume_calc.jl`: `Volume` type.
- `src/volume_calc_impl.jl`: Monte Carlo volume estimation and volume routing.
- `src/RegimeConstraints.jl`: analysis-time parameter constraints, restricted
  regime diagnostics, stable regime intersections, and constrained
  multistability/R-index sampling.

### Graphs, Symbolics, and Workflows

- `src/*RegimeGraph.jl`: binding, catalysis, and Bnc graph construction.
- `src/SIMO.jl` and `src/simo/*`: SIMO paths and path polyhedra.
- `src/symbolics.jl` and `src/symbolic/*`: symbolic API and rendering.
- `src/visualize.jl`, `src/visualization/*`,
  `ext/BindingAndCatalysisVisualizationExt.jl`: optional visualization.
- `src/old_api.jl`: compatibility aliases and deprecation wrappers.

## Legacy and Compatibility

`old_api.jl` keeps older terminology and aliases:

- vertex terminology -> regime terminology;
- legacy `get_mixed_*` aliases -> `get_bnc_*` APIs;
- SISO -> SIMO;
- qssKk -> wKk;
- `get_regime` / `get_regimes` binding convenience aliases.

Compatibility policy:

- Old aliases may call maintained APIs and issue deprecation warnings.
- New internal code should use maintained names, not legacy aliases.
- Avoid adding new APIs whose primary style depends on `return_idx`,
  `return_code`, `return_mask`, `input_logspace`, or `output_logspace`.

## Development Notes

- Formatter configuration lives in `.JuliaFormatter.toml`; the formatter
  environment is under `scripts/format`.
- Tests are split by subsystem under `test/`.
- Volume tests use smaller Monte Carlo batches for speed; heavy exploratory
  checks should be behind explicit environment switches.
- Visualization requires optional dependencies and is not loaded by plain
  `using BindingAndCatalysis`.
- `noback/` contains working notes and architecture feedback. Most of the folder
  is ignored unless files are explicitly tracked or force-added.
