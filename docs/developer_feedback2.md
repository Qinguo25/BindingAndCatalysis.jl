# Developer Feedback from Multistability Case Study

This note records feedback from using `BindingAndCatalysis.jl` for a
toggle-switch multistability R-index analysis under several parameter
constraints.  It complements `docs/developer_feedback.md`, which focused more on
adaptation and local BNC control workflows.

## Read this first

This file is intended to be self-contained.  The concrete analysis behind the
feedback was a toggle-switch multistability R-index study originally implemented
outside this repository at:

```text
~/Realizibility_index/CASE_STUDY/Toggle_switch_constraints/
```

The old reference table is:

```text
~/Realizibility_index/CASE_STUDY/Toggle_switch_constraints/results/summary.csv
```

The case study compared monomer-repressor and dimer-repressor toggle switches.
The monomer topology has `P1 + D2 <-> C1` and `P2 + D1 <-> C2`.  The dimer
topology adds TF self-dimerization first: `2P1 <-> Cp1`, `2P2 <-> Cp2`, then
`Cp1 + D2 <-> C1` and `Cp2 + D1 <-> C2`.  In the dimer model, `Kp1,Kp2` are
self-dimerization dissociation constants and `K1,K2` are dimer-DNA dissociation
constants, so stronger dimer-DNA binding is `K1<Kp1` and `K2<Kp2`.

For each constrained parameter family, the old analysis did this:

1. enumerate BNC regimes;
2. keep feasible, stable, nonsingular, full-dimensional BNC regimes in `wKk`
   space after constraints;
3. intersect every pair of stable regimes;
4. count a pair as bistable if the pair intersection is full-dimensional;
5. estimate the asymptotic solid-angle volume of the union of those pair
   intersections.  This was reported as the multistability R-index.

After the new constraint APIs were added, I reran this workflow using
`parameter_constraints`, `restrict_regimes`, `stable_regime_intersections`, and
`multistability_profile`.  The new APIs reproduce the deterministic polyhedral
counts exactly: reduced dimension, number of full-dimensional regimes, number of
stable full-dimensional regimes, and number of stable pair intersections all
match the old CSV.

The remaining issue is R-index semantics.  The current `multistability_profile`
does not always reproduce the old asymptotic R-index, especially after equality
constraints.  There are two concrete reasons:

- equality constraints currently use an SVD orthonormal nullspace basis, while
  the old analysis used named biological identified-parameter coordinates, for
  example one shared degradation coordinate copied into several original rate
  positions;
- `get(profile.R_atleast_stable_count, 2, 0.0)` is the at-least-two-regime
  estimate; its meaning is selected explicitly by `profile.mode` and recorded
  by `profile.denominator`.

The main development request is therefore not more regime enumeration.  That
part works.  The remaining request is an explicit constrained asymptotic
multistability estimator that:

- lets users choose the equality-measure convention, e.g. `basis=:orthonormal`
  versus `basis=:identified_parameters`;
- preserves biological reduced-coordinate labels when parameters are grouped;
- exposes `mode=:asymptotic_R` separately from finite-region sampling;
- returns the report-style regime counts together with
  `R_exact_stable_count`, `R_atleast_stable_count`, and
  `stderr_atleast_stable_count`.

## Implementation Response, 2026-06-08

The first implementation pass keeps these constraints purely analytical. It does
not store a constraint family inside `Bnc` and does not mutate the binding/BNC
regime caches. This keeps comparisons across multiple biological parameter
families cheap and explicit.

Implemented APIs:

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

Design choices:

- Matrix constraints are the core API, matching the package condition
  convention: first `nullity` rows are equalities and later rows are
  inequalities.
- Parameter charts now explicitly encode `old = F*new + F0`.
- Biological parameter identification can be expressed with `map` or `groups`,
  while advanced users can pass `F,F0` directly.
- Symbolic equalities and inequalities are convenience syntax compiled into the
  same matrix representation.
- Equality constraints are handled by affine reduction:

```text
z = offset + basis * y
```

  so sampling happens in the reduced constraint chart rather than by rejection
  against measure-zero equality rows.
- `chart=:auto` resolves to `:qK` for binding-only analysis and `:wKk` for BNC
  analysis.
- Strict inequalities such as `K1 < Kp1` are represented as closed halfspaces.
  This is appropriate for full-dimensional volume estimates because the
  boundary has zero measure.
- `multistability_profile(...; mode=:finite_region)` samples the constraint
  region first, then counts how many stable restricted BNC regimes contain each
  accepted sample. Therefore the denominator is explicitly
  `:constraint_region`.
- `multistability_profile(...; mode=:asymptotic_R)` strips offsets and samples
  recession-cone membership, with `denominator=:constraint_cone`.
- `multistability_R_index` is the report-oriented wrapper for conditional
  asymptotic multistability R-index summaries.
- Pairwise stable-regime intersections are retained as strict metadata.
  Higher-order multistability is currently exposed through sample hit
  combinations rather than exhaustive triple/k-tuple intersection enumeration.

Implemented return shape for constrained multistability:

```julia
profile = multistability_profile(model; constraints, samples=...)

profile.stable_count_histogram
profile.R_exact_stable_count
profile.R_atleast_stable_count
profile.max_stable_count
profile.combination_counts
profile.pair_table
profile.denominator
```

Deferred:

- exact or analytic union-volume computation for stable pair intersections;
- exhaustive triple/k-tuple intersection enumeration;
- persistent named constraint sets stored on `Bnc`;
- a table writer / Tables.jl-compatible wrapper around the profile object.

## Reproduction check after the implementation pass, 2026-06-08

I reran the toggle-switch case study using the new public APIs listed above,
rather than the original hand-written affine-reduction and polyhedron-intersection
code.  The test script rebuilt the monomer and dimer toggle models, expressed the
same parameter families with `parameter_constraints`, used `restrict_regimes` to
filter full-dimensional stable BNC regimes, used `stable_regime_intersections` to
find bistable stable-regime overlaps, and used `multistability_profile` for the
sampling estimate.

The deterministic parts reproduced correctly.  For every monomer and dimer
scenario checked, the reduced dimension, number of full-dimensional regimes,
number of stable full-dimensional regimes, and number of full-dimensional stable
pair intersections matched the earlier case-study CSV exactly.  This is a strong
sign that the new constraint/restriction/intersection APIs capture the core
polyhedral workflow.

The R-index values did not always reproduce the earlier case-study values,
especially for equality-heavy constrained families.  The table below used 50,000
samples per scenario, so small differences are Monte Carlo noise; large
differences are semantic.

| scenario | previous R-index | finite-region $R_{\ge 2}$ | comment |
| --- | ---: | ---: | --- |
| monomer unconstrained | 0.02762 | 0.02714 | matches within sampling error |
| monomer paired loss | 0.03886 | 0.03818 | close |
| monomer shared loss/beta | 0.06188 | 0.04828 | not reproduced |
| monomer shared loss/beta/K | 0.06270 | 0.04836 | not reproduced |
| monomer fully symmetric | 0.37451 | 0.25036 | not reproduced |
| dimer unconstrained | 0.01929 | 0.01904 | matches within sampling error |
| dimer unconstrained + `K_i<Kp_i` | 0.02784 | 0.02728 | close |
| dimer paired bound loss | 0.02350 | 0.02686 | not reproduced |
| dimer paired bound loss + `K_i<Kp_i` | 0.03496 | 0.03750 | not reproduced |
| dimer paired all loss | 0.03859 | 0.03536 | not reproduced |
| dimer paired all loss + `K_i<Kp_i` | 0.06143 | 0.05798 | not reproduced |
| dimer shared loss/beta/K | 0.07015 | 0.04862 | not reproduced |
| dimer shared loss/beta/K + `K_i<Kp_i` | 0.11228 | 0.07480 | not reproduced |
| dimer fully symmetric | 0.35180 | 0.26932 | not reproduced |
| dimer fully symmetric + `K_i<Kp_i` | 0.38737 | 0.37336 | close but still lower |

Two issues explain the mismatch.

### Equality constraints need an explicit measure convention

The original case-study script imposed equalities by building a biological
reduced coordinate chart directly.  For example, if several degradation rates
were constrained to share one value, the old script sampled one shared
degradation coordinate and copied it into all corresponding original positions.
In matrix form this was a non-orthonormal map `z = A*y`, where repeated columns
encode parameter identification.

The new `parameter_constraints` implementation absorbs equalities using an SVD
nullspace basis.  This produces an orthonormal coordinate system for the affine
subspace.  That is mathematically clean, but it changes the solid-angle measure
relative to the older biological-parameter chart.  For equality-heavy scenarios,
the R-index can change substantially even though the feasible regimes and pair
intersections are identical.

This means the API needs to make the measure convention explicit.  I would
suggest adding one of the following:

```julia
parameter_constraints(model; equalities, basis=:orthonormal)      # current behavior
parameter_constraints(model; groups, basis=:identified_parameters)
parameter_constraints(model; basis=A, offset=A0, reduced_symbols=...)
```

The `groups` form could express the older workflow directly, for example:

```julia
constraints = parameter_constraints(
    model;
    chart=:wKk,
    groups = Dict(
        :loss => [:γ1, :γ2, :η1, :η2, :δ1, :δ2],
        :beta => [:β1, :β2],
        :Kp => [:Kp1, :Kp2],
        :K => [:K1, :K2],
    ),
    inequalities = [(:K1, :<, :Kp1), (:K2, :<, :Kp2)],
)
```

The important point is that a user should be able to choose whether R-index is
measured in an orthonormal subspace chart or in named biological parameter
coordinates after parameter identification.  The guide should document the
default, because both conventions are defensible but they answer different
questions.

### `multistability_profile` should separate finite-region sampling from asymptotic R-index

The earlier report used asymptotic solid-angle R-index: after finding the
stable-pair intersection polyhedra, it used only the recession-cone halfspace
directions and ignored constant offsets.  The current `multistability_profile`
checks membership using the full restricted regime inequalities, including
`C0`.  In many scenarios these agree, but not always.  For example, in the
50,000-sample reproduction test, monomer fully symmetric gave about `0.36376`
when I sampled the stable-pair recession cones but only `0.25036` from
`get(profile.R_atleast_stable_count, 2, 0.0)`.

This should be made explicit in the API.  Possible fixes:

```julia
multistability_profile(model; constraints, asymptotic=true)
multistability_profile(model; constraints, sampler=:gaussian_cone)
multistability_profile(model; constraints, mode=:finite_region)
multistability_profile(model; constraints, mode=:asymptotic_R)
```

The user guide distinguishes this finite-region value from the asymptotic
R-index through `mode` and `denominator`; downstream reports should retain both
fields.

### Usability feedback from the reproduction

The new APIs are a major improvement over the old script.  The most comfortable
parts were:

- `parameter_constraints` avoids manual `C*A` and `C0 + C*A0` bookkeeping;
- `restrict_regimes` directly returns useful feasibility/dimension diagnostics;
- `stable_regime_intersections` replaces most hand-written pair-loop code;
- strict inequality notes are helpful and correctly remind users that strict
  constraints become closed halfspaces for volume calculations.

The remaining uncomfortable parts are:

- reduced coordinates are named `theta_i`, so report tables lose biological
  labels unless the user separately tracks the equality groups;
- the default equality basis silently changes the measure compared with the
  intuitive "merge these parameters into one biological parameter" workflow;
- `profile.mode` and `profile.denominator` must be carried into reports so the
  finite-region probability and asymptotic cone R-index remain distinct;
- there is still no direct one-call replacement for the earlier report's exact
  summary table: `full_dim_regimes`, `stable_full_dim_regimes`,
  `pair_intersections`, and asymptotic conditional R-index.

Highest-priority follow-up: add an explicit constrained asymptotic
multistability estimator that reuses `stable_regime_intersections` but preserves
the user's chosen reduced-parameter basis and strips offsets when
`asymptotic=true`.

## Maintainer Response and Second Implementation Pass, 2026-06-08

The feedback clarified that constrained R-index analysis needs two explicit
layers:

1. a parameter chart that defines how the original chart variables are expressed
   by reduced biological parameters;
2. constraints applied inside that chosen reduced chart.

The second implementation pass added this split.

### Implemented

- Added `ParameterChart` and `parameter_chart`.
- `ParameterChart` stores:
  - `original_symbols`,
  - `reduced_symbols`,
  - `F`,
  - `F0`,
  - `basis_kind`.
- `chart.basis` and `chart.offset` are aliases for `chart.F` and `chart.F0`,
  so the new API remains compatible with the existing restriction internals.
- `parameter_chart(...; map=...)` supports `old_symbol => new_symbol`
  parameter identification.
- `parameter_chart(...; groups=...)` supports
  `new_symbol => old_symbols` biological grouping.
- `parameter_chart(...; F, F0, reduced_symbols)` supports advanced affine
  reparameterizations directly.
- `parameter_constraints(chart; ...)` now applies constraints in reduced
  symbols by default.
- `parameter_constraints(chart; symbols=:original, ...)` applies constraints in
  original symbols and pulls them back through `old = F*new + F0`.
- The one-step convenience form is supported:

```julia
constraints = parameter_constraints(
    model;
    chart=:wKk,
    map=Dict(:K1 => :K, :K2 => :K),
    inequalities=[(:K1, :<, :Kp1), (:K2, :<, :Kp2)],
)
```

  In this one-step form constraints default to original symbols, preserving the
  old calling style.
- `multistability_profile` now accepts:

```julia
mode=:finite_region
mode=:asymptotic_R
```

- `mode=:finite_region` keeps offsets and returns
  `denominator=:constraint_region`.
- `mode=:asymptotic_R` strips offsets and returns
  `denominator=:constraint_cone`.
- Added `multistability_R_index`, a report-oriented wrapper that defaults to
  `mode=:asymptotic_R` and returns:
  - `full_dim_regimes`,
  - `stable_full_dim_regimes`,
  - `pair_intersections`,
  - `R_exact_stable_count`,
  - `R_atleast_stable_count`,
  - `stderr_atleast_stable_count`,
  - `samples`,
  - `basis_kind`,
  - `denominator`,
  - `pair_table`.

### Developer Position

The package should not decide which biological parameters should be grouped or
how a toggle-switch result should be interpreted. That remains project-level
analysis.

The package should be responsible for the reusable bookkeeping:

- preserving the chosen affine measure convention;
- pulling original-symbol constraints back into reduced coordinates;
- distinguishing finite-region sampling from asymptotic cone R-index;
- reporting the denominator and basis convention in every R-index summary.

This implementation follows that boundary. It makes the older report-style
R-index reproducible in the intended biological reduced chart without storing a
constraint family inside `Bnc` or mutating regime caches.

## Test Results After Second Implementation Pass, 2026-06-08

I retested after the package added `ParameterChart`, identified-parameter
`map`/`groups`, `mode=:asymptotic_R`, and the report-oriented
`multistability_R_index` wrapper.

First, the focused package test passed:

```text
BNC Analysis Constraints                  37 passed
Binding Analysis Constraints Default to qK 4 passed
```

Second, I reran the toggle-switch reproduction against the old reference CSV
using the new public API.  The reproduction used the old case-study parameter
identifications as `map=Dict(old_symbol => reduced_symbol, ...)`, original-symbol
inequalities such as `(:K1, :<, :Kp1)`, and `multistability_R_index`, which
defaults to `mode=:asymptotic_R`.

The main R-index issue is fixed.  The new R-index values reproduce the old
reference table within 50,000-sample Monte Carlo error.

Representative retest output:

| scenario | old R | new R, 50k samples | status |
| --- | ---: | ---: | --- |
| monomer unconstrained | 0.02762 | 0.02714 ± 0.00073 | OK |
| monomer shared loss/beta | 0.06188 | 0.06122 ± 0.00107 | OK |
| monomer fully symmetric | 0.37451 | 0.37540 ± 0.00217 | OK |
| dimer unconstrained | 0.01929 | 0.01904 ± 0.00061 | OK |
| dimer unconstrained + `K_i<Kp_i` | 0.02784 | 0.02728 ± 0.00073 | OK |
| dimer paired all loss + `K_i<Kp_i` | 0.06143 | 0.06182 ± 0.00108 | OK |
| dimer shared loss/beta/K + `K_i<Kp_i` | 0.11228 | 0.11238 ± 0.00141 | OK |
| dimer fully symmetric + `K_i<Kp_i` | 0.38737 | 0.38830 ± 0.00218 | OK |

The bistability-relevant deterministic counts also reproduce: reduced dimension,
stable full-dimensional regime count, and stable pair-intersection count match
the old CSV for all scenarios tested.

Two small wrapper semantics issues were found after the second pass.

First, the old CSV's `full_dim_regimes` column counted all full-dimensional
restricted BNC regimes, including singular regimes.  The second-pass
`multistability_R_index` default applied `singular=false` to the
`full_dim_restricted` count, so `full_dim_regimes` was lower in cases where some
full-dimensional regimes were singular.  Examples from the retest:

| scenario | old `full_dim_regimes` | current `full_dim_regimes` | stable count | pair count |
| --- | ---: | ---: | ---: | ---: |
| monomer fully symmetric | 14 | 11 | matches | matches |
| dimer fully symmetric | 38 | 35 | matches | matches |
| dimer fully symmetric + `K_i<Kp_i` | 35 | 32 | matches | matches |

This does not affect the values in `R_atleast_stable_count`, because the stable
nonsingular regimes and pair intersections still match.  But for a
report-oriented wrapper intended to reproduce the earlier table, the filters
used for summary columns should be separated:

```julia
full_dim_regimes = count all feasible full-dimensional restricted regimes
stable_full_dim_regimes = count stable, nonsingular, full-dimensional regimes
pair_intersections = stable nonsingular full-dimensional pair intersections
```

This has been patched: `full_dim_regimes` now counts all feasible
full-dimensional restricted BNC regimes, while `stable_full_dim_regimes`,
`pair_intersections`, and the stable-count R-index fields continue to use the
candidate filter controlled by `singular=false` by default.

Second, passing `map=Dict{Symbol,Symbol}()` marked the chart as
`basis_kind=:identified_parameters`, even though the map was empty and the chart
was effectively identity.  This was harmless numerically, but misleading in
report output.  Empty `map`/`groups` are now normalized to the identity chart.

This second issue was triggered by my reproduction script, where I passed an
empty map just to keep all scenarios on one code path.  A user can avoid it by
omitting `map` for unconstrained cases, but the package now handles empty maps
cleanly for report output.

## Context needed to read this file alone

This feedback comes from an actual downstream case study, not from a general API
review.  The case study asked how the asymptotic R-index for toggle-switch
multistability changes when biologically motivated parameter constraints are
imposed.

Two toggle-switch topologies were analyzed.

The monomer-repressor topology used two promoter-binding reactions:

```text
P1 + D2 <-> C1
P2 + D1 <-> C2
```

The dimer-repressor topology added TF self-dimerization before DNA binding:

```text
2P1 <-> Cp1
2P2 <-> Cp2
Cp1 + D2 <-> C1
Cp2 + D1 <-> C2
```

Here `Kp1,Kp2` are self-dimerization dissociation constants and `K1,K2` are
dimer-DNA dissociation constants.  Therefore stronger dimer-DNA binding than
self-dimerization is encoded as `K1 < Kp1` and `K2 < Kp2`.

For each topology, catalysis was attached so that the BNC regimes represent
local steady-state candidates of the reduced protein-total dynamics.  In the
analysis, a regime was treated as a candidate steady state only when it was:

- feasible;
- stable according to the package's BNC stability test;
- nonsingular;
- full-dimensional in the reduced logarithmic parameter chart after constraints
  were applied.

Multistability was detected by pairwise intersections of stable BNC-regime
polyhedra in `wKk` space.  A pair counted as bistable only if the two stable
regime polyhedra had a full-dimensional intersection.  The reported
multistability R-index was the asymptotic solid-angle volume of the union of all
such stable-pair intersections.

Several parameter families were compared.  Examples include affine equality
constraints such as `gamma1=delta1`, `gamma2=delta2`, `beta1=beta2`, `K1=K2`,
and `tD1=tD2`, plus inequality constraints such as `K1<Kp1` and `K2<Kp2`.  The
R-index was interpreted conditionally inside each constrained parameter family.
This conditional interpretation matters: after constraints are imposed, the
denominator is the feasible asymptotic region in the reduced chart, not the
original unconstrained high-dimensional space.

The package exposed the low-level ingredients needed to complete the analysis,
but the final script had to manually implement affine coordinate reduction,
halfspace constraints, constrained regime filtering, pairwise stable-regime
intersections, and conditional union-volume estimation.  The suggestions below
are based on those concrete implementation gaps.

## Overall impression of `docs/user_guide.md`

The guide is broadly aligned with how I want to call the package.  The most
useful parts are:

- the explicit split between binding, catalysis, and BNC regimes;
- the recommendation to use `input=:log` and `output=:log`;
- the emphasis on symbol helpers instead of hard-coded coordinate positions;
- the listing of condition getters such as `get_C_C0_nullity_qK` and
  `get_C_C0_nullity_wKk`;
- the BNC control section, which now gives a public route for many adaptation
  analyses.

For exploratory scripts, the guide gives enough to build models, enumerate
regimes, inspect conditions, and compute ordinary regime volumes.  For the
multistability analysis, however, I still had to write a fair amount of
downstream glue code around polyhedra, affine parameter reductions, regime-pair
intersections, and conditional R-index estimates.

## Small documentation/API consistency issues

The guide uses both `_sym` and `_symbol` helpers.  Both are exported, but the
distinction is easy to miss:

- `x_sym`, `qK_sym`, `wKk_sym` return symbolic `Num` objects;
- `x_symbol`, `qK_symbol`, `wKk_symbol` return plain `Symbol`s.

This distinction should be stated explicitly near the symbol-order section.
Otherwise it is not obvious why some examples use `wKk_sym(model)` while control
examples use `wKk_symbol(model)`.  The current guide is internally usable, but a
one-paragraph convention would avoid confusion.

The BNC section says `get_bnc_regimes(model; nullity=0)`.  That is convenient if
supported, but in practice I often reasoned from `is_singular(rgm)` and
`get_C_C0_nullity_wKk(rgm)`.  It would help to document exactly whether `nullity`
means the BNC polyhedron nullity in `wKk` space, the inner binding nullity, or
the package's regime-level `nlt`.

## What was missing for the multistability R-index workflow

The toggle-switch analysis needed the following steps:

1. enumerate BNC regimes;
2. retain feasible, stable, nonsingular, full-dimensional regimes;
3. impose affine parameter equalities such as `gamma1=delta1` or
   `beta1=beta2`;
4. impose inequality constraints such as `K1 < Kp1` and `K2 < Kp2`;
5. intersect every stable-regime pair;
6. keep full-dimensional pair intersections;
7. estimate the asymptotic solid-angle volume of the union of those pair
   intersections, optionally conditioned on the added inequality constraints;
8. return summary rows and enough regime-pair metadata for a report.

Most low-level pieces exist, but the workflow is not first-class.  I had to
write custom code for:

- pulling a regime polyhedron back through an affine coordinate map;
- detecting when a constraint becomes incompatible after parameter
  identifications;
- intersecting all stable regime pairs and checking full dimensionality;
- estimating union volume of pair-intersection cones;
- sampling conditionally inside added halfspaces;
- keeping output tables synchronized with the exact constraints used.

These are not project-specific operations.  They are natural package-level
operations for any R-index or multistability case study.

## Original suggested package additions, partly implemented

This section is retained as historical context for why the new constraint APIs
were added.  Some items below are already implemented in the first pass listed
at the top of this file.  The newer reproduction check above is the better guide
to the remaining gaps.

### 1. Affine constraint and inequality API for regime polyhedra

Add a small public representation for parameter constraints in `qK`, `qKk`, or
`wKk` coordinates.  For example:

```julia
constraints = parameter_constraints(
    model;
    chart = :wKk,
    equalities = [
        :γ1 => :δ1,
        :γ2 => :δ2,
        :β1 => :β2,
    ],
    inequalities = [
        (:K1, :<, :Kp1),
        (:K2, :<, :Kp2),
    ],
)
```

The object should expose:

- the reduced coordinate labels;
- the affine map from reduced coordinates to original coordinates;
- added halfspace rows in the reduced coordinates;
- a clear incompatible flag when an inequality contradicts an equality, for
  example `K1=Kp1` together with `K1<Kp1`.

This would replace hand-written `Dict{String,String}` maps and manual row
construction.

### 2. Pullback / restrict polyhedron helper

Add a helper like:

```julia
restrict_polyhedron(poly, constraints; canonicalize=true)
restrict_regime(rgm, constraints; chart=:wKk)
restrict_regimes(rgms, constraints; stable=true, singular=false, full_dim=true)
```

The function should:

- apply the affine map;
- apply halfspace constraints;
- canonicalize;
- return `nothing` or a diagnostic object if infeasible;
- report `dim`, `fulldim`, and nullity consistently.

This is the main missing primitive behind constrained R-index studies.

### 3. Stable-regime intersection utility

Add a public function for pairwise stable-regime intersections:

```julia
pairs = stable_regime_intersections(
    get_bnc_regimes(model);
    constraints,
    full_dim = true,
    singular = false,
)
```

The return value should include:

- regime indices/permutations for each pair;
- the intersection polyhedron;
- dimension/full-dimensionality;
- stability codes;
- optional volume estimates.

This would make the multistability detection criterion much less error-prone.

### 4. Union R-index / conditional solid-angle volume

`calc_volume` is useful for individual regimes, but multistability needs the
volume of a union of pair-intersection regions.  A public helper such as:

```julia
R = calc_union_R(
    pair_polys;
    asymptotic = true,
    conditioned_on = constraints,
    samples = 500_000,
    seed = 1,
)
```

would be very helpful.  It should make the denominator explicit:

- unconditional volume in the ambient reduced chart;
- conditional volume inside the constraint cone;
- whether the denominator was estimated by rejection sampling, analytic
  symmetry, or an exact/polyhedral method.

This matters because a halfspace does not always simply divide a feasible region
by one half after other constraints are imposed.  In the case study I had to be
careful not to assume that simplification.

### 5. One-call multistability summary

Once the above primitives exist, a higher-level helper would be natural:

```julia
summary = multistability_R_index(
    model;
    constraints,
    stable = true,
    singular = false,
    full_dim = true,
    samples = 500_000,
)
```

Expected fields:

- `reduced_symbols`;
- `total_bnc_regimes`;
- `full_dim_regimes`;
- `stable_full_dim_regimes`;
- `pair_intersections`;
- `stable_count_histogram`;
- `R_exact_stable_count`;
- `R_atleast_stable_count`;
- `stderr_atleast_stable_count`;
- `max_stable_count`;
- `samples`;
- `notes`;
- `pair_table`.

This is exactly the table shape needed for reports.

## Things that felt counterintuitive

### `get_polyhedron(rgm)` chart depends on regime type

For a binding regime, `get_polyhedron(rgm)` gives the `qK` condition.  For a BNC
regime, it gives the `wKk` condition.  That is sensible internally, but
downstream scripts become easier to audit if the chart can be requested
explicitly:

```julia
get_polyhedron(rgm; chart=:qK)
get_polyhedron(rgm; chart=:wKk)
```

Even if only one chart is valid for a regime type, an explicit keyword would
make report code clearer.

### Strict inequalities are represented as closed halfspaces

For asymptotic/full-dimensional R-index calculations this is usually fine,
because the boundary has zero volume.  But the API should say this explicitly
when users write constraints like `K1 < Kp1`.  Otherwise it looks like the
strictness is being ignored.

### Dimension checks require too much Polyhedra knowledge

The downstream script had to call `dim(poly) == ambient_dim`, then
`detecthlinearity!`, `removehredundancy!`, and `removevredundancy!` in several
places.  A package helper such as:

```julia
is_full_dimensional(poly; ambient_dim=nothing, canonicalize=true)
```

would make user code more reliable and easier to read.

### Constraint labels should remain symbolic

In the case study it was tempting to convert symbols to strings to build maps
like `"γ1" => "γδ"`.  This is fragile around Unicode, subscripts, and plain
ASCII fallbacks.  A package-level constraint API should accept both `Symbol` and
`Num` labels and normalize them using the same machinery as `locate_sym_*`.

## Original suggested priority, superseded by the reproduction check above

Highest priority for future case studies:

1. public parameter-constraint object for affine equalities and halfspace
   inequalities;
2. `restrict_regimes` / `restrict_polyhedron` in `wKk` and `qK` charts;
3. pairwise stable-regime intersection helper;
4. union/conditional R-index estimator with explicit denominator semantics;
5. documentation section: "Constrained multistability R-index workflow".

These additions would let a user express the biological model and constraints
directly, while leaving the package responsible for coordinate bookkeeping,
polyhedral feasibility, and volume semantics.

## Report-generation usability notes, 2026-06-08

These notes came from generating the polished toggle-switch report in
`/home/joker/Realizibility_index/CASE_STUDY/Toggle_switch_constraints`.  The
report enumerates monomer and dimer toggle-switch models under equality,
symmetry, and dimer-DNA-affinity constraints, then validates representative
bistable pairs by nonlinear phase portraits.

### `qcat_traj_cat` trajectory shape could be documented or wrapped

For plotting, `qcat_traj_cat` returned a vector of state vectors.  That is a
reasonable DifferentialEquations-style representation, but report code needed a
small conversion helper before it could plot trajectories as a `2 x T` matrix.
A documented helper such as `trajectory_matrix(us)` or a short note in the
simulation section would make downstream plotting scripts more direct.

### Dimer-toggle simulations can hit `maxiters` warnings

Several representative dimer-toggle trajectories emitted SciML `maxiters`
warnings during phase-portrait generation.  The trajectories were still usable
for the report, and every tested parameter family showed both attracting basins,
but this is easy for a user to misread as a package failure.  It would help if
`qcat_traj_cat` documented recommended stiff solver / `maxiters` settings for
toggle-like dimer regimes, or exposed a compact diagnostic summary that clearly
separates integration warnings from binding-network feasibility problems.

### Singular-regime propagation warnings need context or a quiet mode

Calling `match_regimes!` for the monomer and dimer toggle models produced
warnings such as `Inconsistent singular BncRegime H_inner directions found
during graph propagation`.  The report later filters to feasible,
full-dimensional, non-singular, stable regimes, so these warnings did not affect
the final R-index table or phase-portrait examples.  Still, the warning appears
before the user has filtered regimes, and it is not obvious whether it indicates
a real problem for the current analysis.  A short diagnostic note, a warning
category, or a keyword for suppressing warnings from regimes that will be marked
singular would make scripted report generation easier to audit.

### Follow-up observation from the expanded toggle report, 2026-06-09

After the report was expanded to 48 parameter families, I initially tried to run
finite `qcat_traj_cat` simulations for every selected stable-pair representative.
For several dimer examples, the reduced ODE path emitted repeated internal logs
from `src/qK_x_mapping.jl:686`:

```text
Error: M is still singular after maximum perturbation attempts.
```

The printed sparse matrices contained `Inf` entries.  The run was not used for
the final report because the R-index is asymptotic and a finite representative
point can be too close to a boundary or too extreme for robust trajectory
integration.  The final report instead uses `plot_qcat_slice_with_flux(...;
show_nullclines=true)` for representative nullcline diagnostics.

Minimal reproduction context from the case-study script:

```julia
include("/home/joker/Realizibility_index/CASE_STUDY/Toggle_switch_constraints/code/generate_bistability_report_assets.jl")
model = dimer_toggle_model()
sc = only(filter(s -> s.family_id == "D02", all_scenarios()))
constraints = constraints_for(model, sc)
pairs = stable_regime_intersections(get_bnc_regimes(model); constraints, full_dim=true)
best = choose_pair(model, constraints, pairs)
qcat_traj_cat(model, best.q1, best.logwKk, (0.0, 500.0); fail_on_binding_error=false)
```

This is not blocking the R-index workflow, but it would help if the trajectory
API exposed a quieter diagnostic mode for expected binding-solve failures, or if
the internal `M is still singular` path could be downgraded when
`fail_on_binding_error=false` and the caller is explicitly treating failures as
nonfatal.

### Maintainer response and implementation status

These report-generation notes are package-level usability issues, not
toggle-switch-specific code. The package response is:

- keep `qcat_traj_cat` as the low-level DifferentialEquations-style API, but
  add `trajectory_matrix(states)` and recommend `simulate_catalysis_trajectory`
  for matrix-shaped report output;
- add `simulate_catalysis_trajectory(...).diagnostics`, including solver
  `retcode`, success status, final-time reachability, saved-point count, and
  `maxiters`;
- add `bnc_regime_diagnostics(model)` and
  `match_regimes!(model; warn_singular_propagation=false)` so report scripts
  can suppress expected singular-propagation warnings after deciding to filter
  `singular=false`, while still retaining an auditable diagnostic record.

## Multi-fate multistability scaling feedback, 2026-06-09

Context:

- Case-study directory:
  `/home/joker/Realizibility_index/CASE_STUDY/Multi-fate`
- Script:
  `code/generate_multifate_rindex.jl`
- Model:
  multi-fate-3 with monomers `Ai`, homodimers `Hii`, heterodimers `Hij`,
  DNA states `Di/Ci`, protein totals
  `Ti = Ai + 2Hii + sum_j Hij`, and DNA totals `tDi = Di + Ci`.
- The binding network has 496 binding regimes and the catalysis network has
  496 catalysis regimes, so matching creates roughly 246k BNC regime
  combinations.

### Public CDD-based restriction is too heavy for exploratory multi-fate-3 scans

Using the public report path:

```julia
profile = multistability_profile(
    model;
    constraints,
    regimes,
    samples=200,
    stable=true,
    singular=false,
    feasible=true,
    full_dim=true,
    mode=:asymptotic_R,
    pair_intersections=false,
)
```

was still too slow for the unconstrained multi-fate-3 case.  The run was not
limited by the number of Monte Carlo samples; it was spending time in
`restrict_regimes` constructing and normalizing CDD polyhedra for a large number
of stable BNC regimes.  The observed stack was inside CDDLib
`removehredundancy!` called through:

```text
multistability_profile
  restrict_regimes
    restrict_regime
      get_polyhedron
        _build_polyhedron_from_C_C0
          _poly_normalize!
            CDDLib.removehredundancy!
```

This is a performance/API limitation rather than a correctness bug.

### Workaround used in the case study

For multi-fate-3 only, I used a fast ROP sampler in the case-study script:

1. iterate over stable, feasible, nonsingular BNC regimes;
2. call the internal `BindingAndCatalysis._regime_C_C0_nullity(rgm, :wKk)`;
3. call the internal `BindingAndCatalysis._pullback_constraints(C, C0, nlt, constraints)`;
4. skip regimes with residual equality rows, as the fast analogue of
   `full_dim=true` under continuous cone sampling;
5. sample the constrained asymptotic cone and count direct H-representation
   membership, without constructing a CDD polyhedron for every regime.

This made the multi-fate-3 exploratory scan practical and gave useful R-index
estimates, but it depends on internal functions and is less auditable than a
public API.

### Requested package feature

A public fast path would make large multistability scans much easier:

```julia
multistability_profile(
    model;
    constraints,
    mode=:asymptotic_R,
    full_dim=true,
    pair_intersections=false,
    restriction_backend=:hrep_membership,
)
```

or equivalently:

```julia
restricted = restrict_regimes(
    regimes,
    constraints;
    stable=true,
    singular=false,
    feasible=true,
    full_dim=:nullity_only,
    build_polyhedron=false,
)
```

The key need is to expose a documented, package-supported way to pull BNC
regime inequalities into a constrained parameter basis and use those
H-representations directly for Monte Carlo cone membership, while reserving CDD
polyhedron construction for selected regimes or validation examples.
