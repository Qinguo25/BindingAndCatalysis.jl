# Relation Pruning Design Note

## Motivation

For large complete dimerization networks such as CDN5, the current calculation explores all feasible qK-space regimes and all compatible paths through the regime graph. This includes cases that are mathematically possible in the unconstrained model but impossible for a concrete experimental system.

For example, if the binding affinity order is known, then both

```text
K_d1,2 > K_d2,3
```

and the reverse order should not be explored. In a real system, the affinity values are fixed, so only one ordering is relevant. Removing regimes and interfaces that contradict known qK relations should simplify the regime graph before downstream calculations begin.

## Goal

Add support for user-supplied qK-space relations that prune the regime graph.

This feature should sit at the same level as the SISO graph/path-condition improvements: it is a general graph-simplification step that SISO can use first, but it should not be designed as a SISO-only test utility.

The first target is not a relabeling or symmetry feature. If users treat CDN monomers as placeholders and later map real monomers onto them, that interpretation should remain outside the core package. The package should simply accept predefined relations and calculate within the corresponding qK-space region.

## Core Idea

Given a relation polyhedron `R` in qK space:

1. Keep a regime vertex `v` only if

   ```text
   condition(v) ∩ R
   ```

   is nonempty.

2. Keep an interface or directed graph edge `v -> w` only if

   ```text
   condition(v) ∩ condition(w) ∩ R
   ```

   is nonempty.

3. Build downstream calculations, including SISO, from this pruned graph.

The edge check is important because two regimes may each be feasible under `R`, while their shared interface is not.

## Expected Benefits

- Fewer feasible regime vertices.
- Fewer feasible graph interfaces.
- Fewer source-to-sink paths for SISO.
- Fewer pair-condition subproblems in the memoized/DAG SISO solver.
- Earlier removal of impossible branches for CDN5 and other large systems.

This should be more effective than only intersecting predefined relations during final path-condition construction, because the graph topology itself becomes smaller before expensive path calculations start.

## API Direction

A possible public API for SISO is:

```julia
SISOPaths(model, change_qK; qK_constraints = relation_polyhedron)
```

or a more domain-specific name:

```julia
SISOPaths(model, change_qK; predefined_relations = relation_polyhedron)
```

At the graph level, we may also want a lower-level helper that returns a pruned qK graph before SISO is constructed.

The exact representation of `relation_polyhedron` still needs discussion. Candidate inputs include:

- a `Polyhedron` in full qK space,
- `(C, C0)` inequalities compatible with `get_polyhedron(C, C0)`,
- a higher-level relation specification for qK indices or symbolic qK names.

The low-level polyhedron input is probably the safest first implementation because it avoids prematurely designing a relation DSL.

## Implementation Sketch

1. Add an optional qK relation constraint to the regime graph or SISO construction path.
2. Compute which regimes survive intersection with the relation polyhedron.
3. Build a pruned directed qK graph containing only feasible vertices and feasible interfaces.
4. Recompute sources and sinks from the pruned graph when constructing SISO.
5. Ensure cached vertex/interface prisms and returned path condition polyhedra still include the predefined relation, so displayed conditions and volume calculations remain consistent with the user's assumptions.

## Open Questions

- Should relation constraints live in full qK space only, or should reduced-space constraints after eliminating `change_qK` also be accepted?
- Should empty-pruned graphs return an empty graph/object, or throw a clear error?
- What is the cleanest user-facing way to specify relations such as `K_d1,2 > K_d2,3` using existing qK symbols?
- Should the pruned graph expose diagnostics, for example how many regimes and edges were removed?

