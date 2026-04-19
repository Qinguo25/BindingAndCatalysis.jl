# Singular `x` Range Report

## Goal

For a singular binding regime, `show_expression_x` is not enough because some `x_i`
are not single-valued functions of `qK`. The practical target is:

- if `x_i` is fixed on the singular fiber, give an explicit expression
- otherwise, give a symbolic interval
  - `max(lower_1, lower_2, ...) < x_i < min(upper_1, upper_2, ...)`
- also keep any pure-`qK` consistency conditions that the singular regime imposes

## Recommended Construction

Work in log coordinates:

- `z = log10(x)`
- `y = log10(qK)`

For a binding regime we already have:

- `y = M z + M0`
- `C_x z + C0 > 0`

Equivalently, on variables `(z, y)`:

```math
\begin{aligned}
Mz - y + M_0 &= 0, \\
C_x z + C_0 &> 0.
\end{aligned}
```

For a chosen coordinate `x_i`, keep only `(z_i, y)` and project out the other
`z_j`. The projected polyhedron has the form

```math
\widetilde C_i
\begin{bmatrix}
z_i \\
y
\end{bmatrix}
+
\widetilde C_{0,i}
\ge 0,
```

with some leading equality rows.

Each projected row is then interpreted as follows:

- coefficient of `z_i` is zero:
  - this is a pure consistency condition on `qK`
- equality row with nonzero `z_i` coefficient:
  - this fixes `x_i` explicitly
- inequality row with positive `z_i` coefficient:
  - this is a lower bound on `x_i`
- inequality row with negative `z_i` coefficient:
  - this is an upper bound on `x_i`

So the symbolic range comes directly from polyhedral elimination. This is the
method now implemented in `src/output/singular_x_range.jl`.

## Why This Is Better Than The Regime-Graph Heuristic

The graph-based idea is useful as an intuition for nullity-1 regimes:

- move along the singular fiber
- look for the next regular regime reached when `x_i` decreases or increases
- use the neighboring regular regime expression as a bound

But this misses an important case:

- some interval endpoints are not regime-to-regime interfaces
- they are feasibility boundaries of the singular fiber itself

So a graph walk can be strictly incomplete even for nullity-1.

### Minimal counterexample

For `N = [2 1 -1]`, the singular regime gives:

- consistency: `q1 = 2 q2`
- fixed dominant expression: `x3 = q2 = q1 / 2`
- non-dominant range for `x1`:
  - lower bound: `x1^2 > K1`
  - upper bound: `x1 < q1`

The old graph-style construction only found the upper side for `x1`, and missed
the feasibility lower bound `x1 > sqrt(K1)`.

That is the key reason the projection method should be the primary algorithm.

## Nullity > 1

For nullity greater than 1, the graph heuristic stops being a natural primary
method because there is no longer a single 1D singular fiber to walk along.

The projection method still works:

- start from the same regime constraints in `(z, y)`
- project to `(z_i, y)`
- read off equalities / lower bounds / upper bounds / consistency conditions

So the answer is:

- yes, a symbolic range can still be produced for `nullity > 1`
- but it should be obtained by projection, not by graph traversal

The output is still of the same useful form:

- fixed expression, or
- `max(lower...) < x_i < min(upper...)`

## Larger-Model Check

I used a fixed sample generated from:

```julia
N_generator(4, 7; min_binder = 2, max_binder = 5)
```

with

```text
N =
[1 2 1 -1 0 0 0;
 1 1 1  0 -1 0 0;
 0 0 1  1  0 -1 0;
 2 1 0  0  1 0 -1]
```

Then I built:

- the original model with `L = L_from_N(N)`
- a modified model where one conservation row is replaced by the sum of two rows

```julia
Lnew[1, :] = L[1, :] + L[2, :]
```

### Regime counts

Original `L`:

- regular: 30
- nullity-1: 20
- nullity-2: 4

Modified `Lnew`:

- regular: 30
- nullity-1: 27
- nullity-2: 4

### Projection vs graph on nullity-1

I compared the projection method against the old graph-style heuristic on every
nullity-1 regime and every `x_i`, evaluating both at an interior `qK` point of
that regime.

Original `L`:

- total comparisons: 140
- mismatches: 17

Modified `Lnew`:

- total comparisons: 189
- mismatches: 42

The mismatches were not small numerical noise. Typical pattern:

- graph method gave one-sided or completely unbounded interval
- projection method gave a finite interval or even a fixed value

Examples from the modified model:

- graph: `(-Inf, Inf)`, projection: fixed value
- graph: `(-Inf, upper)`, projection: fixed value
- graph: `(lower, Inf)`, projection: fixed value

So the two methods are **not** generally consistent as constructions.

What *is* consistent is the following weaker statement:

- when a bound really comes from crossing into a neighboring regular regime,
  the graph expression matches one of the projected boundary expressions

But the graph method does not see all valid boundaries.

## About Replacing One `L` Row By A Sum

Replacing one conservation row by the sum of two rows preserves the span of the
left null space, so it is still a valid conservation basis.

It does change the coordinate system on `q`:

- e.g. if `q'_1 = q_1 + q_2` and `q'_2 = q_2`, then positivity implies
  `q'_1 > q'_2`

This is not an artificial extra constraint added by the range algorithm.
It is a consequence of choosing a different positive basis for the conserved
quantities.

So this experiment is best interpreted as a coordinate-change robustness check,
not as an invariance claim that the symbolic formulas should literally look the
same.

## Final Decision

Use the projection method as the package implementation.

Reason:

- complete for nullity-1
- still valid for nullity > 1
- naturally returns consistency conditions together with bounds
- directly matches the polyhedral semantics already used elsewhere in the package
- does not rely on guessing which side of a singular chain eventually reaches a
  regular regime

The graph heuristic can still be used as a debugging aid for nullity-1, but it
should not be the primary API.

## Code Changes

The implementation now lives in:

- `src/output/singular_x_range.jl`

and exposes:

- `get_singular_x_range`
- `show_singular_x_range`
- `show_expression_x_range`

with tests added in:

- `test/binding/basic.jl`
