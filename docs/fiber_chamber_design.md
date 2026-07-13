# Fiber/Chamber Design

This document defines the geometric object behind SIMO paths and records the
goals of the general variation-subspace and two-dimensional-fiber stages.
[`Architecture.md`](../Architecture.md) remains the canonical description of
code that is currently implemented.

## Geometric object

Let \(Q \cong \mathbb R^m\) be the logarithmic parameter space and let
\(\mathcal R=\{P_r\}\) be its labelled polyhedral regime complex. Let
\(U\subseteq Q\) be the allowed variation subspace, with \(\dim U=k\), and let

\[
\pi:Q\to B=Q/U
\]

be the quotient map. A base point \(b\in B\) selects the affine fiber
\(F_b=\pi^{-1}(b)\). The regime decomposition induced on that fiber is

\[
\mathcal S_b=\{P_r\cap F_b\ne\varnothing\}.
\]

The authoritative slice object is the labelled polyhedral cell complex
\(\mathcal S_b\), including its face-incidence data. As \(b\) varies, its
combinatorial type is locally constant away from a discriminant set. The
connected full-dimensional strata of \(B\) are chambers. Each chamber has one
exact slice type; chambers whose closures share a codimension-one facet are
adjacent. Lower-dimensional strata record degenerate transition types.

For \(k=1\), a generic fiber is a line and \(\mathcal S_b\) is an ordered
sequence of regimes and interfaces: a SIMO path. A node in the exact
path-chamber graph is a chamber labelled by a path, rather than merely the path
label itself. Two disconnected chambers may carry the same label unless
connectedness or convexity has been proved.

For \(k=2\), a generic fiber is a plane and \(\mathcal S_b\) is a labelled
planar polyhedral complex. Its two-dimensional cells are regime intersections,
its one-dimensional cells are regime interfaces, and its vertices are
higher-codimension intersections. A dual regime graph is a useful view, but is
not the complete object: it can lose face incidence, cyclic order, embeddings,
and multiple interfaces.

## Existence conditions are not yet a chamber decomposition

For a candidate slice type \(\tau\), define its existence set

\[
E_\tau=\{b\in B:\tau\text{ occurs on }F_b\}.
\]

A current SIMO path polyhedron represents a closed feasibility/existence
condition of this kind. It is feasible exactly when it is nonempty. A
full-dimensional condition describes generic behavior; a nonempty
lower-dimensional condition describes a boundary-only degeneracy and may have
zero volume.

The family \(\{E_\tau\}\) must not automatically be treated as the chamber
complex. Closed conditions may overlap on discriminant strata, and one
condition may require further refinement. An exact chamber complex is obtained
by refining all relevant boundaries/discriminants, taking connected strata,
and labelling each stratum by the exact slice type at a witness point.
Consequently, direct pairwise intersection of path polyhedra is not sufficient
to establish chamber adjacency.

## Current one-dimensional baseline

The supported production case is a coordinate-aligned one-dimensional
variation. `SIMOPaths` enumerates candidate ordered regime paths. Its default
condition backend is the pair-memoized DAG solver: reusable subpath conditions
are cached by regime pairs and solved in dependency order. The former vibe
suffix-DAG implementation remains available as `condition_method=:suffix_dag`
for comparison. Backend-specific source/sink, path, pair-cache, and DAG types
remain internal to the one-dimensional implementation.

Caller-supplied candidate paths must traverse real, correctly oriented edges
of the selected SIMO regime graph. An arbitrary sequence of regime labels is
not a path and is rejected before condition solving.

`path_feasible` is tri-state: `nothing` means the condition has not been
computed, while `true` or `false` records the nonempty/empty result.
`is_feasible(paths, path)` resolves it lazily.

Public keyword conventions follow the maintained SIMO API (`recompute`,
`reltol`, and `abstol`). Removed spellings such as `recalculate`, `rel_tol`,
and `abs_tol` are not silently translated; callers receive an error that names
the maintained replacement.

The same package-wide migration policy applies to stability: use
`recompute=true` and call `stability_code(...)` when a numeric code is needed;
the former `recalculate` and `return_code` keywords raise guided errors.

## Stage 2 — General variation-subspace layer

The goal is to separate fiber geometry from the one-dimensional path
algorithm.

- Represent \(U\) by a full-rank basis, with coordinate-index constructors as
  conveniences.
- Represent the base by a quotient map whose kernel is \(U\), plus a section
  when explicit fiber coordinates are needed.
- Introduce a fiber/slice problem abstraction that records the model,
  parameter chart, variation subspace, and base chart.
- Transform regime constraints into fiber/base coordinates and centralize
  projection, elimination, dimension, emptiness, witness, and canonicalization
  operations.
- Define backend-independent conditional-slice and chamber records; keep
  `SIMOPaths` as the \(k=1\) compatibility view backed by the pair-memoized DAG
  solver.
- Do not expose one-dimensional assumptions such as a global source, sink,
  order, or DAG in interfaces intended for \(k>1\).

Stage 2 is complete when coordinate and general linear subspaces share one
geometry interface, the \(k=1\) results match the established SIMO fixtures,
signatures are deterministic, and results are invariant under equivalent
choices of basis and quotient coordinates.

## Stage 3 — Two-dimensional fibers

The goal is to enumerate and condition the planar slice types for \(k=2\).

- Construct the labelled planar cell complex \(\mathcal S_b\), including cells,
  interfaces, vertices, incidence, and unbounded faces.
- Derive a dual regime graph as a view, while using the complete cell complex
  for equality and canonical signatures.
- Construct the base-space discriminant from projected critical faces and
  non-transverse events.
- Explore chambers from witness points and cross their facets to discover
  neighboring slice types, instead of enumerating arbitrary graphs in advance.
- Compute the exact condition and relative dimension for every discovered
  type, deduplicate by canonical signature, and retain lower-dimensional
  transition strata.
- Build a slice-type graph whose nodes are chambers with conditions and whose
  edges represent verified codimension-one adjacency.

Stage 3 is complete when small reference models have verified base-space
coverage, witness reconstruction agrees with stored signatures, adjacency is
facet-certified, results are invariant under equivalent fiber coordinates,
and all \(k=1\) regressions remain unchanged.
