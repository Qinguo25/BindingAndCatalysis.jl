# Relation Pruning Design Note

## Motivation

For large complete dimerization networks such as CDN5, the current calculation
explores all feasible qK-space regimes and all compatible paths through the
regime graph. This includes cases that are mathematically possible in the
unconstrained model but impossible for a concrete experimental system.

For example, if the binding affinity order is known, then both

```text
K_d1,2 > K_d2,3
```

and the reverse order should not be explored. In a real system, the affinity
values are fixed, so only one ordering is relevant. Removing regimes and
interfaces that contradict known qK relations should simplify the regime graph
before downstream calculations begin.

## Goal

Add support for user-supplied qK-space relations that prune the regime graph and
are carried through downstream path-condition calculations.

This feature should sit at the same level as the SISO graph/path-condition
improvements: it is a general qK-space graph-simplification step that SISO can
use first, but it should not be designed as a SISO-only test utility.

The first target is not a relabeling or symmetry feature. If users treat CDN
monomers as placeholders and later map real monomers onto them, that
interpretation should remain outside the core package. The package should simply
accept predefined relations and calculate within the corresponding qK-space
region.

## Core Semantics

Let `R` be a relation polyhedron in full qK space.

1. Keep a regime vertex `v` only if

   ```text
   condition(v) ∩ R
   ```

   is nonempty.

2. Keep an interface, and therefore any directed graph edge `v -> w` built from
   that interface, only if

   ```text
   condition(v) ∩ condition(w) ∩ R
   ```

   is nonempty.

3. Build downstream calculations, including SISO source/sink detection and path
   enumeration, from this pruned graph.

The edge check is required. Two regimes may each be feasible under `R`, while
their shared interface is not.

The relation must also remain part of all cached vertex/interface path
conditions. In particular, SISO eliminates the chosen `change_qK` coordinate
when constructing prisms, so the correct operation is

```text
project(condition ∩ R, eliminate = change_qK)
```

not

```text
project(condition, eliminate = change_qK) ∩ R
```

This distinction matters whenever `R` involves the eliminated coordinate.

## Initial API

The first implementation should support a low-level full qK-space polyhedron:

```julia
SISOPaths(model, change_qK; qK_constraints = relation_polyhedron)
```

where `relation_polyhedron` is a `Polyhedron` with the same ambient dimension as
the model's full qK space.

For convenience, a second accepted form can be added if it is cheap and
unambiguous:

```julia
SISOPaths(model, change_qK; qK_constraints = (C, C0))
SISOPaths(model, change_qK; qK_constraints = (C, C0, nullity))
```

These tuple forms should be interpreted exactly as inputs to `get_polyhedron`.

A higher-level relation DSL, for example specifying `K_d1,2 > K_d2,3` using qK
symbols, should be deferred. It is useful, but not needed to validate the core
graph and SISO behavior.

## Graph-Level API

SISO should not be the only entry point. Add a lower-level helper that returns a
pruned qK graph:

```julia
get_pruned_SISO_graph(model, change_qK; qK_constraints)
```

or, if we want a more general name:

```julia
get_relation_pruned_graph(model, change_qK; qK_constraints)
```

The helper should return enough information for diagnostics and testing. A
concrete shape could be:

```julia
graph, feasible_vertices, diagnostics
```

where:

- `graph` is a `SimpleDiGraph` with the original regime indices preserved,
- `feasible_vertices` is a `BitVector`,
- `diagnostics` records how many vertices and edges were removed.

Preserving original regime indices is important because existing APIs assume
graph vertex ids match regime ids. The pruned graph should therefore have
`n_regimes(model)` vertices and only the surviving directed edges, instead of
being compacted/reindexed.

## SISO State

Relation constraints should be stored on the SISO problem, not only used during
construction.

Current conceptual state:

```julia
struct SISOProblem
    bn
    change_qK_idx
    dag
end
```

Proposed conceptual state:

```julia
struct SISOProblem
    bn
    change_qK_idx
    dag
    qK_constraints
    feasible_vertices
    pruning_diagnostics
end
```

The exact type parameters can follow existing code style, but the key design
point is that helpers computing vertex prisms, interface prisms, path
polyhedra, and volumes can still see the relation constraint.

## Source and Sink Semantics

After pruning, sources and sinks should be recomputed from the pruned directed
graph.

Feasible vertices with no surviving incoming or outgoing SISO edges need special
treatment. For the initial SISO implementation, isolated vertices should not be
treated as valid length-1 paths unless we explicitly introduce a use case for
that behavior. SISO is about paths induced by changing one qK coordinate, so the
default should be to drop isolated feasible vertices from source/sink path
enumeration.

The graph-level helper can still expose them through `feasible_vertices` and
diagnostics.

## Empty Result Semantics

An empty pruned graph is a valid scientific result. If the relation polyhedron is
well-formed but excludes all regimes or all traversable interfaces, constructors
should return a valid object with no paths rather than throwing.

Throw clear errors for invalid inputs, such as:

- relation polyhedron dimension does not match full qK dimension,
- tuple constraint dimensions are inconsistent,
- unsupported `qK_constraints` type.

## Implementation Sketch

1. Normalize `qK_constraints` into either `nothing` or a full qK-space
   `Polyhedron`.
2. Validate that the constraint dimension matches the model's qK dimension.
3. Build or retrieve the full regime graph with qK interfaces.
4. Compute a `feasible_vertices` mask by testing `condition(v) ∩ R`.
5. Build the SISO-oriented directed edge list as usual, but keep only edges whose
   full-space interface satisfies `condition(v) ∩ condition(w) ∩ R`.
6. Construct a `SimpleDiGraph(n_regimes(model))` with original indices and only
   surviving directed edges.
7. Recompute SISO sources and sinks from the pruned graph, excluding isolated
   feasible vertices from path enumeration.
8. Store the normalized relation constraint on `SISOProblem`.
9. Update cached vertex/interface prism construction so constraints are applied
   before eliminating `change_qK`.
10. Ensure returned path condition polyhedra and volume calculations use these
    constrained prisms.

## Expected Benefits

- Fewer feasible regime vertices.
- Fewer feasible graph interfaces.
- Fewer source-to-sink paths for SISO.
- Fewer pair-condition subproblems in the memoized/DAG SISO solver.
- Earlier removal of impossible branches for CDN5 and other large systems.
- Returned conditions and volumes reflect the user's assumed qK region, not the
  unconstrained qK space.

This should be more effective than only intersecting predefined relations during
final path-condition construction, because the graph topology itself becomes
smaller before expensive path calculations start.

## Deferred Features

- A symbolic relation DSL for qK names.
- Reduced-space constraints after eliminating `change_qK`.
- Compact/reindexed induced graphs.
- Relation or symmetry handling for CDN monomer relabeling.
- Specialized diagnostics or visualization for removed vertices and interfaces.

## Open Questions

- What should the public graph-level helper be named?
- Should diagnostics be returned as a named tuple or a small concrete struct?
- Should there be an explicit keyword to allow isolated feasible vertices as
  length-1 SISO paths, or should that stay unsupported until a real use case
  appears?
