# Developer Feedback from User-Guide Revision

This note records package-level API and workflow feedback found while revising
`docs/user_guide.md` and while using `BindingAndCatalysis.jl` for the competitive
adaptation analysis.  It is intentionally separate from the user guide: the user
guide should remain user-facing, while this file is for maintainers deciding what
to add or export in the package.

## Summary

The package already exposes most of the pieces needed for binding-regime,
catalysis-regime, and BNC-regime analysis.  The main problem is that some common
workflows still require downstream scripts to manually compose low-level pieces:

- BNC/catalysis index helpers exist but are not exported.
- Linearized BNC control quantities must be reconstructed by hand.
- Steady-state invariance and input-responsiveness standards are not package
  functions.
- Regime representative-point sampling and nonlinear step-response diagnostics
  are not first-class workflows.
- Singular-regime numerical derivative provenance is hard to inspect.

The items below are ordered by practical priority.

## Implementation Response, 2026-06-08

The first three package-level priorities have been implemented on branch `vibe`.
Each priority was committed separately.

### Completed

1. Symmetric public index/permutation API
   - Commit: `d0d9717 Export symmetric regime collection APIs`.
   - Exported catalysis/BNC index and permutation helpers.
   - Exported binding/BNC nullity collection helpers.
   - Added public docstrings and lightweight symmetry tests.

2. BNC linear control metrics
   - Commit: `17d3615 Add BNC linear control metrics`.
   - Added `src/BncControl.jl` and included it from the main module.
   - Public API now includes `linear_control_model`, `control_metrics`,
     `controllability_matrix`, `output_controllability_matrix`,
     `output_controllability_row`, `markov_coefficients`,
     `controllability_gramian`, and `steady_state_gain`.
   - Added `hbd_source` and `get_H_bd_info` so downstream analysis can inspect
     whether `H_bd` came from exact regime derivatives or numerical binding
     derivatives.
   - Used `timescale=:identity` by default. A positive timescale vector can be
     supplied to row-scale `A` and `B`; the package does not infer physical
     timescales from a regime alone.

3. Steady-state invariance and input responsiveness standards
   - Commit: `38a6f45 Add BNC responsiveness standards`.
   - Added `steady_state_invariance` and `is_steady_state_invariant`.
   - Added `input_responsiveness` and `input_responsive`.
   - Added `compare_input_responsiveness` for table-shaped regime comparisons.
   - Supported responsiveness standards:
     `:direct_flux`, `:output_controllability`, `:output_reachability`,
     `:gramian`, and `:steady_state_gain`.

### Verification

- Ran JuliaFormatter through the repository formatter environment on touched
  files, using `julia --project=scripts/format`.
- Ran the focused catalysis/BNC test file after the linear-control API change:
  `julia --project=. -e 'include("test/support/setup.jl"); include("test/bnc_regime/catalysis.jl")'`.
- Ran targeted smoke tests for the linear-control wrapper keywords and for the
  new responsiveness/invariance APIs.

### Deferred

- Regime representative-point and membership utilities remain deferred. These
  are useful, but they need careful chart semantics for `:qK`, `:qKk`, and
  `:wKk`.
- Nonlinear adaptation step-response diagnostics remain deferred. They are more
  adaptation-circuit-specific and should likely live in an extension-oriented
  file or submodule rather than the core regime API.
- `get_H_bd(rgm; return_info=true)` was not added. The implemented public path
  is `get_H_bd_info(rgm)`, which avoids changing the existing `get_H_bd(rgm)`
  call contract while still exposing provenance.

## 1. Export Symmetric Regime Index and Permutation Helpers

### Current State

`get_binding_indices` and `get_binding_perms` are exported and can be used after:

```julia
using BindingAndCatalysis
```

The catalysis/BNC counterparts are defined in `src/RegimeCore.jl`:

```julia
get_catalysis_indices(args...; kwargs...)
get_bnc_indices(args...; kwargs...)
get_catalysis_perms(args...; kwargs...)
get_bnc_perms(args...; kwargs...)
```

but they are not exported.  In the revised guide I had to use:

```julia
cat_ids = get_idx.(get_catalysis_regimes(model))
bnc_ids = get_idx.(get_bnc_regimes(model))
```

instead of the more natural:

```julia
cat_ids = get_catalysis_indices(model)
bnc_ids = get_bnc_indices(model)
```

### Why This Matters

The package presents three parallel regime layers:

- binding regimes,
- catalysis regimes,
- BNC regimes.

The public API should be symmetric across those layers.  Otherwise user
documentation must either recommend less explicit aliases like `get_idx.(...)`
or use fully qualified internal names.

### Suggested Change

Export the existing functions:

```julia
export get_catalysis_indices, get_bnc_indices
export get_catalysis_perms, get_bnc_perms
export get_binding_nullities, get_bnc_nullities
```

Potential location: `src/RegimeCore.jl`, alongside the existing regime-core
exports.

### Acceptance Tests

```julia
using BindingAndCatalysis

@test isdefined(Main, :get_catalysis_indices)
@test isdefined(Main, :get_bnc_indices)
@test isdefined(Main, :get_catalysis_perms)
@test isdefined(Main, :get_bnc_perms)

cat_rgms = get_catalysis_regimes(model)
bnc_rgms = get_bnc_regimes(model)

@test get_catalysis_indices(model) == get_idx.(cat_rgms)
@test get_bnc_indices(model) == get_idx.(bnc_rgms)
```

## 2. Add Public Linearized BNC Control Metrics

### Current State

For the adaptation report I needed the local linearized BNC dynamics:

```julia
δdot_qcat = A * δqcat + B * δinput
δoutput = C * δqcat + D * δinput
```

Currently this has to be reconstructed manually from package internals:

- `get_binding_network(rgm)`,
- `get_catalysis_network(rgm)`,
- `binding_order_matrix(rgm)`,
- `get_PΠ(cat_rgm)`,
- `get_H_bd(rgm)` or `get_H_bd_numerically(rgm)`,
- `locate_sym_*`,
- output rows extracted from the binding order matrix.

This is error-prone because it depends on vector ordering conventions and on
whether `H_bd` is exact or numerical.

### Why This Matters

This package is naturally used for regime-level control questions:

- Is a regime dynamically stable?
- Does an input affect an output locally?
- Is the output reachable through the local dynamics?
- Does the output have a nonzero long-time step gain?

Users should not have to rebuild the same `A, B, C, D` logic in every analysis.

### Suggested API

Add a public helper such as:

```julia
linear_control_model(
    rgm::BncRegime;
    input=:tI,
    output=:Astar,
    hbd=:auto,
)
```

Return a named tuple or small struct:

```julia
(;
    A,
    B,
    C,
    D,
    eigvals,
    stable,
    input,
    output,
    qcat_symbols,
    wKk_symbols,
    hbd_source,
)
```

Then add convenience metrics:

```julia
output_controllability_row(ctrl; order=size(ctrl.A, 1) - 1)
markov_coefficients(ctrl; order=size(ctrl.A, 1) - 1)
controllability_gramian(ctrl)
steady_state_gain(ctrl)
```

### Acceptance Tests

For a regular BNC regime:

```julia
ctrl = linear_control_model(rgm; input=:tI, output=:Astar)

@test size(ctrl.A, 1) == size(ctrl.A, 2)
@test length(ctrl.B) == size(ctrl.A, 1)
@test length(ctrl.C) == size(ctrl.A, 1)
@test ctrl.hbd_source == :exact_regime_derivative
```

For a singular BNC regime:

```julia
ctrl = linear_control_model(singular_rgm; input=:tI, output=:Astar)
@test ctrl.hbd_source in (:numerical_binding_derivative, :exact_regime_derivative)
```

## 3. Add Steady-State Invariance and Input-Responsiveness Standards

### Current State

The report had to implement several regime-classification standards outside the
package:

- steady-state invariance,
- direct-flux input responsiveness,
- output-controllability-row responsiveness,
- Gramian / thresholded output-reachability responsiveness,
- comparison tables across standards.

These standards are natural package-level analyses, not one-off report logic.

### Why This Matters

The standards depend on package-specific affine maps and symbol ordering.  If
each downstream project reimplements them, small differences in thresholds,
singular-regime handling, and output rows will produce inconsistent results.

### Suggested API

Add functions with explicit thresholds and output names:

```julia
steady_state_gain(
    rgm::BncRegime;
    input=:tI,
    output=:Astar,
    hbd=:auto,
)

is_steady_state_invariant(
    rgm::BncRegime;
    input=:tI,
    output=:Astar,
    atol=1e-8,
)

input_responsive(
    rgm::BncRegime;
    input=:tI,
    output=:Astar,
    standard=:output_reachability,
    threshold=0.1,
)

compare_input_responsiveness(
    rgms;
    input=:tI,
    outputs=(:Astar, :tAstar),
    standards=(:direct_flux, :output_controllability, :output_reachability),
    threshold=0.1,
)
```

Suggested supported `standard` values:

- `:direct_flux`,
- `:output_controllability`,
- `:output_reachability`,
- `:gramian`.

### Acceptance Tests

Use a small BNC model with known stable regimes:

```julia
rows = compare_input_responsiveness(stable_rgms; input=:tI, outputs=(:Astar, :tAstar))

@test all(hasproperty(row, :output) for row in rows)
@test all(hasproperty(row, :standard) for row in rows)
@test all(hasproperty(row, :responsive) for row in rows)
```

Also test that threshold changes are visible:

```julia
low = input_responsive(rgm; standard=:output_controllability, threshold=1e-8)
high = input_responsive(rgm; standard=:output_controllability, threshold=1e-1)
@test low || !high
```

## 4. Add Regime Representative-Point and Membership Utilities

### Current State

For nonlinear simulations, I needed to find a concrete point inside a regime,
perturb `tI`, and check whether the perturbed point stayed in the same regime.
That logic currently lives in downstream scripts.

The package exposes condition matrices, but users still have to build samplers
and membership checks manually.

### Why This Matters

Many workflows require a concrete representative point:

- plotting a nonlinear trajectory for one regime,
- validating a local linear conclusion,
- testing whether a parameter step stays in the same cone,
- debugging why a regime is feasible but hard to simulate.

### Suggested API

```julia
representative_point(
    rgm::Union{BindRegime,BncRegime};
    chart=:qK,
    margin=1.0,
    rng=Random.default_rng(),
)

point_in_regime(
    rgm::Union{BindRegime,BncRegime},
    point;
    chart=:qK,
    atol=1e-8,
)

step_stays_in_regime(
    rgm::BncRegime,
    point;
    chart=:wKk,
    input=:tI,
    fold=2.0,
    atol=1e-8,
)
```

For BNC regimes, `chart` should support at least `:qKk` and `:wKk`.

### Acceptance Tests

```julia
p = representative_point(rgm; chart=:wKk)
@test point_in_regime(rgm, p; chart=:wKk)

p2 = copy(p)
p2[locate_sym_wKk(model, :tI)] += log10(2.0)
@test step_stays_in_regime(rgm, p; chart=:wKk, input=:tI, fold=2.0) ==
      point_in_regime(rgm, p2; chart=:wKk)
```

## 5. Extend Nonlinear Adaptation Simulation Diagnostics

### Current State

`simulate_adaptation` is useful and should stay.  For report-grade analysis,
however, I also needed:

- pre-relaxation to a nearby steady state,
- a two-fold input step,
- full binding solves along the trajectory,
- output peak/final/baseline metrics,
- adaptation error,
- residual checks,
- explicit reporting of failed binding solves.

These diagnostics had to be added outside the package.

### Why This Matters

Users often want to validate a regime-level prediction with the full nonlinear
binding-update loop:

1. solve `x` from binding,
2. update qcat through catalysis,
3. solve `x` from binding again,
4. repeat along the trajectory.

The package already has the core machinery.  It needs a higher-level wrapper
that returns diagnostics, not only raw trajectories.

### Suggested API

Either extend `simulate_adaptation` or add:

```julia
simulate_step_response(
    model::Bnc;
    p,
    input=:tI,
    fold=2.0,
    observe=:Astar,
    logqcat0=nothing,
    pre_relax=(0.0, 200.0),
    tspan=(0.0, 500.0),
    saveat=range(0.0, 500.0, length=500),
    method=nothing,
    tol=1e-6,
)
```

Return:

```julia
(;
    t,
    loginput,
    logobserve,
    logqcat,
    baseline,
    peak,
    final,
    peak_shift,
    final_shift,
    adaptation_error,
    binding_residual_max,
    failed_observation_indices,
)
```

### Acceptance Tests

```julia
out = simulate_step_response(model; p, input=:tI, fold=2.0, observe=:Astar)

@test length(out.t) == length(out.logobserve)
@test hasproperty(out, :baseline)
@test hasproperty(out, :peak_shift)
@test hasproperty(out, :failed_observation_indices)
```

## 6. Expose `H_bd` Provenance for Singular Regimes

### Current State

The report needed to distinguish whether `H_bd` came from:

- exact regime derivatives for regular binding regimes, or
- numerical binding derivatives for singular binding regimes.

This distinction is scientifically important because conclusions based on
singular-regime numerical derivatives should be reported differently.

### Why This Matters

For regimes like the singular Astar-responsive cases in the adaptation report,
the conclusion depends on numerical `H_bd`.  A downstream script can infer this
from `is_singular(get_binding_regime(rgm))`, but the package should make the
source explicit.

### Suggested API

```julia
hbd_source(rgm::BncRegime)
```

returning one of:

```julia
:exact_regime_derivative
:numerical_binding_derivative
```

or:

```julia
get_H_bd(rgm; return_info=true)
```

returning:

```julia
(H=get_H_bd(rgm), source=:exact_regime_derivative)
```

### Acceptance Tests

```julia
info = get_H_bd(rgm; return_info=true)
@test hasproperty(info, :H)
@test hasproperty(info, :source)
@test info.source in (:exact_regime_derivative, :numerical_binding_derivative)
```

## 7. Add a Small Public "Analysis Table" Utility

### Current State

Several downstream scripts had to build CSV rows by hand.  The package already
contains the underlying regime objects and metrics, so it can provide a stable
table-shaped output.

### Suggested API

```julia
regime_analysis_table(
    rgms;
    input=:tI,
    outputs=(:Astar, :tAstar),
    include=(:stability, :invariance, :responsiveness, :hbd_source),
)
```

Return a vector of named tuples or a Tables.jl-compatible table.

### Acceptance Tests

```julia
tbl = regime_analysis_table(stable_rgms; input=:tI, outputs=(:Astar, :tAstar))
@test !isempty(tbl)
@test hasproperty(first(tbl), :regime)
@test hasproperty(first(tbl), :stable)
```

## Documentation Notes

`docs/user_guide.md` should not include this developer feedback directly.  It
should document stable user-facing workflows.  Once the API changes above are
implemented, the guide should be updated to use the new public names, especially:

- replace `get_idx.(get_bnc_regimes(model))` with `get_bnc_indices(model)`,
- replace manual control-metric reconstruction examples with
  `linear_control_model`,
- document `simulate_step_response` if it is added,
- document `hbd_source` whenever singular-regime derivative provenance matters.
