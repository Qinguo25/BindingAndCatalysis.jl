# Relation Pruning Technical Note

This note documents the current implementation details for qK preconstraint
pruning. It is intentionally technical and can later be rewritten into
user-facing documentation.

## Terminology

`qK_preconstraints` are user-supplied constraints in the full qK log-space that
are applied before SISO graph/path construction. They represent assumptions that
are known before the calculation starts, such as an ordering between binding
constants.

Internally, these constraints are stored as a full qK-space `Polyhedron`.

The older keyword spelling `qK_constraints` is currently accepted as a
compatibility alias. New code should use `qK_preconstraints`.

## Public Entry Points

The main SISO entry point accepts preconstraints:

```julia
SISOPaths(model, change_qK; qK_preconstraints = R)
```

The graph-level helper is:

```julia
graph, feasible_vertices, diagnostics =
    get_pruned_SISO_graph(model, change_qK; qK_preconstraints = R)
```

where:

- `graph` is a `SimpleDiGraph` with original regime indices preserved,
- `feasible_vertices` is a `BitVector`,
- `diagnostics` is a `RelationPruningDiagnostics`.

`RelationPruningDiagnostics` stores:

```julia
original_vertices::Int
feasible_vertices::Int
original_edges::Int
feasible_edges::Int
removed_vertices::Int
removed_edges::Int
```

Accessors:

```julia
get_qK_preconstraints(siso_or_helper_or_problem)
get_pruning_diagnostics(siso_or_helper_or_problem)
```

Compatibility accessor:

```julia
get_qK_constraints(siso_or_helper_or_problem)
```

## Accepted Constraint Inputs

`qK_preconstraints` can be:

```julia
nothing
Polyhedron
(C, C0)
(C, C0, nullity)
```

Tuple forms are interpreted through the existing `get_polyhedron` convention:

```julia
get_polyhedron(C, C0)
get_polyhedron(C, C0, nullity)
```

The resulting polyhedron must have the same ambient dimension as the model's
full qK dimension, currently `model.n`.

Invalid dimensions, unsupported types, and simultaneous use of both
`qK_preconstraints` and `qK_constraints` throw `ArgumentError`.

## Mathematical Semantics

Let `R` be the normalized preconstraint polyhedron in full qK space.

A regime vertex `v` survives only if:

```text
condition(v) ∩ R != empty
```

A directed SISO edge `v -> w` survives only if:

```text
condition(v) ∩ condition(w) ∩ R != empty
```

The interface check is separate from vertex feasibility. Two vertices can both
survive while the interface between them does not.

## Projection Semantics

SISO path conditions eliminate the selected `change_qK` coordinate. The
preconstraint is applied before elimination:

```text
project(condition ∩ R, eliminate = change_qK)
```

This is implemented by passing the stored preconstraint into cached vertex and
interface prism construction:

```julia
_get_polyhedron_prism(model, vertex, change_qK_idx, qK_preconstraints)
_get_interface_prism(model, from, to, change_qK_idx, qK_preconstraints)
```

This is important when `R` involves the eliminated coordinate. Applying `R` only
after projection would be mathematically wrong in that case.

## Graph Construction Details

The implementation starts from the normal SISO edge orientation:

```julia
_collect_oriented_edge_pairs(vertex_graph, change_qK_idx)
```

It then filters those edge pairs by:

1. source vertex feasible under `R`,
2. target vertex feasible under `R`,
3. full-space interface feasible under `R`.

The pruned graph is built as:

```julia
SimpleDiGraph(n_regimes(model))
```

with only surviving edges. Regime indices are not compacted or reindexed.

## Source And Sink Handling

For constrained SISO construction, sources and sinks are recomputed from the
pruned graph.

Isolated feasible vertices are not treated as length-1 SISO paths. They remain
visible in `feasible_vertices` and diagnostics, but they are removed from SISO
source/sink path enumeration.

Empty pruned graphs are valid. `SISOPaths` returns an object with empty
`rgm_paths`, empty sources, and empty sinks rather than throwing.

## Manual `rgm_paths`

When users provide `rgm_paths` manually together with `qK_preconstraints`, the
paths are validated strictly.

An error is thrown if any supplied path contains:

- a vertex infeasible under the preconstraint,
- an interface infeasible under the preconstraint.

This is intentional. Manual paths are an advanced override, and silently
dropping user-specified paths would hide mistakes.

## Symbolic Helper API

The symbolic helpers build ordinary full qK-space polyhedra.

Single relation:

```julia
R = qK_preconstraint(model, lhs, op, rhs)
R = qK_preconstraint(model, lhs, op, rhs, margin)
```

Multiple relations:

```julia
R = qK_preconstraints(model,
    (:K12, :>, :K23),
    (:K13, :<, :K14),
    (:tA, :>=, :tB, log10(2)),
)
```

or:

```julia
specs = [
    (:K12, :>, :K23),
    (:K13, :<, :K14),
]
R = qK_preconstraints(model, specs)
```

Supported operators:

```julia
:>, :>=, :gt, :ge
:<, :<=, :lt, :le
:(==), :(=), :eq
```

The relation is interpreted in log qK space. For example:

```julia
qK_preconstraint(model, :K12, :>, :K23, margin)
```

represents:

```text
logK12 - logK23 >= margin
```

Strict operators are represented as closed halfspaces. To enforce a strict
separation numerically, use a positive margin.

Equality relations are placed first in the generated H-representation and become
linear rows through the `nullity` argument to `get_polyhedron`.

## Numeric Right-Hand Sides

The helper also supports a floating-point numeric right-hand side:

```julia
qK_preconstraint(model, :K12, :>, 1.0)
```

This represents a direct log-space lower bound:

```text
logK12 >= 1.0
```

Integer right-hand sides are currently interpreted as qK indices by the existing
symbol lookup convention. Use floating-point literals for numeric bounds.

## Current Tests

The relation-pruning tests cover:

- unconstrained regression against normal `SISOPaths`,
- graph helper output and diagnostics for `qK_preconstraints = nothing`,
- impossible constraints producing empty feasible vertices and empty graphs,
- empty `SISOPaths` objects for empty pruned graphs,
- invalid input errors,
- conflict error when both `qK_preconstraints` and `qK_constraints` are supplied,
- point/equality constraints that keep isolated feasible vertices but no SISO
  paths,
- symbolic helper equivalence to manual `(C, C0)` construction,
- equality relation nullity handling,
- applying preconstraints before eliminating `change_qK` in vertex prism
  construction.

The tests are in:

```text
test/Relation_Prunning/relation_pruning_tests.jl
```

and are included from:

```text
test/runtests.jl
```

## Known Deferred Work

- User-facing documentation and examples.
- Large CDN benchmark on the remote machine.
- Domain-specific helpers such as affinity-order or dissociation-constant-order
  constructors.
- Possible future removal of the compatibility alias `qK_constraints`.
