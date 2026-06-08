# Developer Feedback from Multistability Case Study

This note records feedback from using `BindingAndCatalysis.jl` for a
toggle-switch multistability R-index analysis under several parameter
constraints.  It complements `docs/developer_feedback.md`, which focused more on
adaptation and local BNC control workflows.

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

## Suggested package additions

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

- `ambient_dim`;
- `reduced_parameters`;
- `total_bnc_regimes`;
- `full_dim_regimes`;
- `stable_full_dim_regimes`;
- `multistable_pair_intersections`;
- `R_multistability`;
- `stderr`;
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

## Suggested priority

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
