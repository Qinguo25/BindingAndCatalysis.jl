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
     `controllability_gramian`, `output_energy`,
     `dynamic_steady_state_gain`, `affine_steady_state_gain`, and
     `steady_state_gain`.
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
   - Added `input_drive` for explicit signed positive/negative input-drive
     terms.
   - Supported invariance standards: `:affine` and `:dynamic_dc_gain`.
   - Supported responsiveness standards:
     `:direct_feedthrough`, `:direct_flux`, `:output_controllability`,
     `:output_reachability`, `:gramian`, and `:steady_state_gain`.

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

## Validation on Competitive Adaptation Example, 2026-06-08

I tested the implemented APIs on the current competitive adaptation example in:

```text
/home/joker/Realizibility_index/NFBLB_saver/code/common.jl
```

using:

```julia
model = build_competitive_adaptation_model()
stable = filter(is_stable, get_bnc_regimes(model; singular=false))
```

This gives 116 stable BNC regimes under the same regular-regime filter used in
the report scripts.

### What Works

- `get_catalysis_indices`, `get_catalysis_perms`, `get_bnc_indices`, and
  `get_bnc_perms` are exported and callable.
- `compare_input_responsiveness` runs on the 116-regime stable set with
  two outputs and five standards, producing 1160 rows with zero API errors.
- `input_responsiveness(...; standard=:output_controllability, threshold=0.1)`
  reproduces the report's output-control threshold counts on the exact
  steady-state-invariant subset:

```text
Astar:  46 regimes
tAstar: 45 regimes
```

These match the report's `1e-1` output-control-row standard.

### Issue 1: `steady_state_invariance` Does Not Match the Xiao/Affine Steady-State Standard

The package currently implements `steady_state_invariance` as a local dynamic
DC-gain check:

```julia
D - C * inv(A) * B
```

Formula-level mismatch:

Let

```text
z = log(q_cat),    u = log(tI),    y = selected output
```

and let the package linear model be

```text
dz/dt = A z + B u
y     = C z + D u.
```

The current package criterion is:

```text
G_dyn = D - C A^{-1} B
steady-state invariant iff ||G_dyn||_∞ <= atol.
```

The Xiao/report criterion is instead the exact affine steady-state coefficient
from the regime's steady-state map.  If the output is an `x` species,

```text
log(x*) = H_x(wKk) * log(wKk) + h0
g_aff(output, input) = e_output' H_x(wKk) e_input.
```

If the output is a qcat coordinate,

```text
log(q_cat*) = F * log(wKk) + F0
g_aff(output, input) = e_output' F e_input.
```

The affine/Xiao standard is:

```text
steady-state invariant iff g_aff(output, input) = 0
```

or numerically:

```text
abs(Float64(g_aff(output, input))) <= atol.
```

Therefore the disagreement is not just a tolerance issue.  The package is
testing the DC gain of the local dynamic realization, while the report is
testing exact invariance of the regime's affine steady-state map.

That is not the same standard used in the report or in Xiao-style
steady-state invariance.  The report standard uses the exact affine
steady-state map:

- for `Astar`: the `Astar` row of `get_H(rgm)` with respect to `:tI`,
- for `tAstar`: the `tAstar` row of `get_qcat_F_F0(rgm)` with respect to `:tI`.

On the competitive adaptation example:

```text
output   package steady_state_invariance   exact affine/Xiao standard
Astar    27                               68
tAstar   26                               74
```

Concrete false-negative example:

```text
regime 10, Astar:
  exact affine gain = 0//1
  package DC residual = 9.983224234578937e-5
  package invariant = false

regime 10, tAstar:
  exact affine gain = 0//1
  package DC residual = 9.985548675947353e-5
  package invariant = false
```

Suggested fix:

- Keep the current dynamic DC-gain check, but rename it to something explicit
  like `dynamic_steady_state_gain` or `dc_gain_invariance`.
- Add a separate exact affine steady-state standard, for example:

```julia
steady_state_invariance(
    rgm::BncRegime;
    input=:tI,
    output=:Astar,
    standard=:affine,
    atol=1e-8,
)
```

where `standard=:affine` is the Xiao/report standard and
`standard=:dynamic_dc_gain` is the current `D - C*A\\B` calculation.

The user guide should explicitly distinguish these two meanings of
"steady-state invariance"; otherwise users will assume the current function
checks the Xiao condition.

### Issue 2: `:direct_flux` Is Implemented as Direct Feedthrough `D`, While the Report Needs the Input Drive into the Dynamics

The package currently scores `standard=:direct_flux` as:

```julia
maximum(abs, ctrl.D)
```

Formula-level mismatch:

Using the same local model,

```text
dz/dt = A z + B u
y     = C z + D u,
```

the package currently computes:

```text
s_feedthrough = ||D||_∞
responsive iff s_feedthrough > threshold.
```

This is a direct feedthrough test for the output equation.  It asks whether the
input appears immediately in

```text
y = C z + D u.
```

The report's `direct_flux` standard instead needs the input drive into the
dynamic equation, i.e. the `B`-side effect in

```text
dz/dt = A z + B u.
```

So the user's interpretation is essentially correct: the report needs a
`B`-like flux-drive quantity, while the package's current `:direct_flux`
returns a `D`-like direct-feedthrough score.

For this circuit the report's direct-flux drive was:

```text
Δ_f = ∂log(C1)/∂log(tI) - ∂log(C2)/∂log(tI).
```

Using the binding order matrix `H_bind = ∂log(x)/∂log(qK)`, this is:

```text
Δ_f = e_C1' H_bind e_tI - e_C2' H_bind e_tI.
```

The report's direct-flux responsiveness rule was:

```text
tAstar responsive iff
    Δ_f > θ.

Astar responsive iff
    Δ_f > θ
    and
    ∂log(Astar)/∂log(tAstar) > θ.
```

Equivalently, the Astar gate is:

```text
e_Astar' H_bind e_tAstar > θ.
```

Thus the package's current `:direct_flux` formula tests `D`, while the report's
formula tests a signed flux-drive difference in the binding/catalysis
structure that corresponds to an input drive into the catalytic state equation.

On the exact steady-state-invariant subset of the competitive adaptation
example, this gives:

```text
Astar:  0 regimes
tAstar: 0 regimes
```

The report's direct-flux rule gives:

```text
Astar:  14 regimes
tAstar: 20 regimes
```

The report rule was:

```julia
flux_drive = ∂log(C1)/∂log(tI) - ∂log(C2)/∂log(tI)

tAstar direct-flux responsive:
    flux_drive > threshold

Astar direct-flux responsive:
    flux_drive > threshold &&
    ∂log(Astar)/∂log(tAstar) > threshold
```

Suggested fix:

- Rename the current `:direct_flux` standard to `:direct_feedthrough` if it is
  meant to represent `D`.
- Implement a separate B/input-drive based standard for the report's
  `direct_flux` meaning.  For a generic package API this likely needs explicit
  positive and negative flux terms, for example `positive_flux=:C1` and
  `negative_flux=:C2`, instead of inferring them silently.
- Document the distinction:

```text
direct_feedthrough: output-equation score based on D
direct_flux:        dynamics/input-drive score based on B or flux-drive terms
```

### Issue 3: `:output_reachability` Needs an Explicit Direction Option

The package currently defines `:output_reachability` as the norm of the
output-controllability matrix:

```julia
norm([D C*B C*A*B ...])
```

Formula-level mismatch:

The package currently forms the output-control row/matrix:

```text
R_y = [D, C B, C A B, C A^2 B, ...]
```

and scores:

```text
s_norm = ||R_y||
responsive iff s_norm > θ.
```

This sign-agnostic default is a reasonable package-level interpretation of
"reachable": a large negative response and a large positive response are both
evidence that the input can move the output.  The mismatch is that the report
had an additional direction requirement.  That direction requirement should be
an explicit option, not silently assumed by the generic default.

For the directional report standard, define the Markov/output coefficients:

```text
m_0 = D,
m_1 = C B,
m_2 = C A B,
...
```

Let

```text
m_first = first m_j with abs(m_j) > atol.
```

The signed coefficient part of the report standard was:

```text
m_first > θ.
```

not:

```text
abs(m_first) > θ
```

and not:

```text
norm([m_0, m_1, m_2, ...]) > θ.
```

The report also allowed an energy-based route:

```text
E_y > θ^2,
```

where `E_y` is the output Gramian energy defined below in Issue 4.  Therefore
the report standard was:

```text
responsive iff (m_first > θ) or (E_y > θ^2).
```

The package's current norm test implements direction-free reachability.  The
report used positive-direction reachability.  These are both reasonable
standards, but they should have different names or an explicit keyword.

For a single input and single output, this is sign-agnostic and gives the same
responsive regimes as `:output_controllability` in the competitive adaptation
example:

```text
Astar:  output_controllability = 46, output_reachability = 46
tAstar: output_controllability = 45, output_reachability = 45
```

This is not the thresholded output-reachability standard used in the report.
The report standard was stricter and gave:

```text
Astar:  20 regimes
tAstar: 20 regimes
```

The report standard used:

```julia
first positive signed nonzero coefficient > threshold
```

or a Gramian energy condition:

```julia
output_energy > threshold^2
```

where the `Astar` output energy included the direct term:

```julia
D^2 + C * W * C'
```

Suggested fix:

- Keep direction-free reachability as a valid default, but document it as such.
- Add an explicit direction parameter, for example:

```julia
input_responsiveness(
    rgm;
    standard=:output_reachability,
    direction=:any,       # current norm / sign-agnostic behavior
)

input_responsiveness(
    rgm;
    standard=:output_reachability,
    direction=:positive,  # report behavior: first signed coefficient > θ
)
```

- If `direction=:positive`, the signed-first-coefficient part must preserve
  sign; using a norm is not enough.

### Issue 4: `:gramian` Threshold Semantics Need to State Whether the Threshold Is Energy-Like or Amplitude-Like

The package currently scores `:gramian` as:

```julia
norm(C * W * C')
```

and compares it directly to `threshold`.

Formula-level mismatch:

For the local state equation

```text
dz/dt = A z + B u,
```

with stable `A`, the controllability Gramian satisfies:

```text
A W + W A' + B B' = 0.
```

The package currently uses:

```text
s_pkg = ||C W C'||
responsive iff s_pkg > θ.
```

The report used an output energy and compared it to `θ^2`.  This does not mean
the package formula is mathematically invalid; it means the threshold has a
different unit/meaning.  If `θ` is intended as an energy threshold, comparing
`C W C'` to `θ` is coherent.  If `θ` is intended as an output-amplitude
threshold, then the corresponding energy cutoff is `θ^2`.

For a qcat output, where `D = 0`,

```text
E_y = C W C'.
```

For an `x` output with direct dependence on the input,

```text
E_y = D D' + C W C'.
```

For scalar output/input this is:

```text
E_y = D^2 + C W C'.
```

The report's energy criterion was:

```text
responsive iff E_y > θ^2.
```

The semantic mismatch is therefore:

1. the package appears to treat `threshold` as an energy threshold;
2. the report treated `threshold` as an amplitude threshold and therefore used
   `threshold^2` for energy;
3. for `x` outputs, the report included the direct output/input energy term
   `D D'`.

On the exact steady-state-invariant subset with `threshold=0.1`, it gives:

```text
Astar:  15 regimes
tAstar: 8 regimes
```

The report's Gramian/energy part was compared to `threshold^2`, and for `x`
outputs included the direct term `D^2`.  Under the report's thresholded
reachability rule, the accepted counts were:

```text
Astar:  20 regimes
tAstar: 20 regimes
```

Suggested fix:

- Decide and document whether `threshold` for `:gramian` is an energy threshold
  or an amplitude threshold.
- If the package keeps the current behavior, document it as
  `energy_threshold=threshold`.
- If users pass the same `threshold` used for output amplitudes, provide an
  option such as:

```julia
input_responsiveness(
    rgm;
    standard=:gramian,
    threshold=0.1,
    threshold_scale=:amplitude,  # compare energy to threshold^2
)
```

- For `output_space=:x`, include the direct term when reporting output energy:

```julia
D * D' + C * W * C'
```

- Document this clearly in `user_guide.md`, because users will otherwise expect
  `threshold=0.1` to mean the same threshold used by the other standards.

### Issue 5: `hbd_source` Needs a Documentation Note About BNC Regularity vs Binding Singularity

In the competitive adaptation example, some regimes selected by:

```julia
get_bnc_regimes(model; singular=false)
```

still report:

```julia
hbd_source(rgm) == :numerical_binding_derivative
```

For example regimes 10, 146, and 226 are stable under the BNC regular-regime
filter but have `hbd_source=:numerical_binding_derivative`.

This is not necessarily a bug: `singular=false` filters BNC regime nullity, not
whether the underlying binding regime is singular.  However, the user guide
should state this explicitly.  Otherwise users may assume `singular=false`
implies exact binding derivatives.

## Maintainer Response to Adaptation-Control Feedback, 2026-06-08

The feedback identifies real API ambiguity. The package should provide the
general state-space and regime-affine primitives, while downstream adaptation
projects should define biological labels, flux signs, and compound decision
rules.

### Implemented in the Generic Package Layer

- `linear_control_model` continues to return `A`, `B`, `C`, and `D`, with
  symbol-order metadata. These are the standard state-space matrices:
  state/system matrix, input/control matrix, output/observation matrix, and
  direct-feedthrough matrix.
- Added explicit steady-state gain names:
  `dynamic_steady_state_gain` for `D - C*(A \ B)` and
  `affine_steady_state_gain` for the exact affine regime map.
- Changed `steady_state_invariance` to accept `standard=:affine` and
  `standard=:dynamic_dc_gain`; the affine standard is the Xiao-style
  steady-state invariant condition.
- Added `output_energy(ctrl; include_direct=...)` and made
  `standard=:gramian` distinguish `threshold_scale=:energy` from
  `threshold_scale=:amplitude`.
- Renamed the `D`-based responsiveness meaning to
  `standard=:direct_feedthrough`.
- Kept `standard=:direct_flux`, but made it require explicit
  `positive_flux` and optional `negative_flux` selectors. This prevents the
  package from silently treating a direct-feedthrough matrix as a biological
  flux-drive criterion.
- Added signed reachability through
  `standard=:output_reachability, direction=:any/:positive/:negative`.
- Updated the user guide and architecture document to explain BNC regularity
  versus binding-derivative provenance.

### Left to User Projects

- Choosing that `C1` is the positive flux term and `C2` is the negative flux
  term.
- Deciding that an `Astar` direct-flux response requires both a positive
  `C1-C2` drive and a positive `Astar` sensitivity to `tAstar`.
- Choosing thresholds and deciding whether they are amplitude-like or
  energy-like.
- Choosing the analysis subset, such as stable regimes, exact affine-invariant
  regimes, or regimes with exact binding derivatives only.

This keeps the package general: it exposes the reusable mathematical
quantities, but it does not hard-code the biological interpretation of one
adaptation circuit.

## Retest After Maintainer Response, 2026-06-08

I retested the implemented API on the competitive adaptation example:

```text
/home/joker/Realizibility_index/NFBLB_saver/code/common.jl
```

using the same report subset:

```julia
model = build_competitive_adaptation_model()
stable = filter(is_stable, get_bnc_regimes(model; singular=false))
```

This gives 116 stable regular BNC regimes.

### Retest Result

The new API resolves the previous formula-level mismatches.  The package now
provides the mathematical primitives needed to reproduce the report standards,
while leaving circuit-specific biological composition rules in the downstream
project.

### Steady-State Invariance

Package call:

```julia
is_steady_state_invariant(
    rgm;
    input=:tI,
    output=output,
    standard=:affine,
    atol=1e-6,
)
```

matches the report's exact affine/Xiao steady-state invariant standard:

```text
output   package affine standard   report/local exact affine standard
Astar    68                        68
tAstar   74                        74
```

The previous dynamic DC-gain behavior is still available explicitly:

```julia
standard=:dynamic_dc_gain
```

For regime 10, the new distinction is clear:

```text
regime 10, Astar:
  affine residual     = 0.0
  affine invariant    = true
  dynamic DC residual = 9.983224234578937e-5
  dynamic invariant   = false

regime 10, tAstar:
  affine residual     = 0.0
  affine invariant    = true
  dynamic DC residual = 9.985548675947353e-5
  dynamic invariant   = false
```

This is the intended separation between exact affine steady-state invariance
and dynamic local DC gain.

### Direct Flux

Package call:

```julia
input_responsive(
    rgm;
    input=:tI,
    output=tA_ACTIVE,
    standard=:direct_flux,
    positive_flux=:C1,
    negative_flux=:C2,
    direction=:positive,
    threshold=0.1,
)
```

now represents the signed input-drive quantity

```text
Δ_f = ∂log(C1)/∂log(tI) - ∂log(C2)/∂log(tI)
```

instead of the output direct-feedthrough matrix `D`.

Using this package primitive plus the report's project-level Astar gate,

```text
∂log(Astar)/∂log(tAstar) > 0.1,
```

reproduces the report direct-flux counts:

```text
Astar:  14
tAstar: 20
```

This is acceptable: the package supplies the reusable signed input-drive
primitive, and the downstream project composes the Astar-specific gate.

### Output Reachability and Gramian Energy

Package calls:

```julia
input_responsive(
    rgm;
    input=:tI,
    output=output,
    standard=:output_reachability,
    direction=:positive,
    threshold=0.1,
)
```

and

```julia
input_responsive(
    rgm;
    input=:tI,
    output=output,
    standard=:gramian,
    threshold=0.1,
    threshold_scale=:amplitude,
)
```

now expose the two pieces needed for the report's thresholded reachability
standard:

```text
responsive iff (first signed Markov coefficient > 0.1)
              or (output energy > 0.1^2).
```

The recomposed report criterion matches the previous report counts:

```text
output   output_controllability   positive reachability   gramian amplitude   reachability OR gramian
Astar    46                       20                      15                  20
tAstar   45                       20                      10                  20
```

Note: the old report column named "gramian" actually used the compound
thresholded-reachability rule, i.e. signed first coefficient OR energy.  The
package's `standard=:gramian` is now correctly just the energy piece.

### Batch API

The batch helper also runs with the new explicit keywords:

```julia
rows = compare_input_responsiveness(
    stable;
    input=:tI,
    outputs=(A_ACTIVE, tA_ACTIVE),
    standards=(
        :direct_feedthrough,
        :direct_flux,
        :output_controllability,
        :output_reachability,
        :gramian,
    ),
    threshold=0.1,
    positive_flux=:C1,
    negative_flux=:C2,
    direction=:positive,
    threshold_scale=:amplitude,
)
```

Result:

```text
rows = 1160
errors = 0
missing responsiveness entries = 0
```

No further package-level API issue is identified from this retest.

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

`simulate_adaptation` is useful and should stay. The package now also exposes
`simulate_catalysis_trajectory` as the general reduced qcat trajectory wrapper:
callers can provide fixed or time-dependent combined `wKk` values, or split
`w`, `K`, and reduced `k` blocks. It returns matrix-shaped `logqcat` and `qcat`
trajectories for plotting and diagnostics.

For report-grade adaptation analysis, however, I also needed:

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

The package now has the model-independent trajectory wrapper. Higher-level
adaptation-specific diagnostics should be layered on top of that wrapper rather
than embedded into the core qK/x mapping code.

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
