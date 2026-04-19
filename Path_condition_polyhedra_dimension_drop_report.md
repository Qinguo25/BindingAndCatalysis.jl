# Path Condition Polyhedra Dimension Drop

## Question

For a `SIMOPaths` path

`a -> b -> c -> ... -> z`

the package builds its condition polyhedron by intersecting the projected edge polyhedra along the path. On somewhat larger random models, some path condition polyhedra lose dimension, or even become empty.

The goals here were:

1. find random examples on larger models
2. identify the smallest intersection of projected edge polyhedra that already causes dimension loss
3. look for a structural pattern

## Current construction in source

The relevant implementation is in [src/simo/polyhedra.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/simo/polyhedra.jl:33).

- Each edge polyhedron is built as
  `E_(u,v) = projection_{remove changing qK}(P_u ∩ P_v)`
  via `backend_intersect_eliminate` in `_ensure_edge_polyhedra!`.
- Each path polyhedron is then
  `P_path = ⋂_{e in path} E_e`
  via `backend_intersect_many` in `_build_path_polyhedron`.

So any dimension drop of `P_path` must come from incompatibility or pinching among the projected edge polyhedra, not from any extra path-specific constraint.

## Sampling setup

I used random `5 x 8` models from

```julia
N_generator(5, 8; min_binder=2, max_binder=5)
```

with

```julia
find_all_regimes!(model; mode=:float)
```

and then built `SIMOPaths(model, change_qK_idx)` for each changing `q/K` coordinate.

Two practical definitions are useful:

- `first-drop culprit`: the smallest subset `S` of projected edges with
  `dim(⋂_{e∈S} E_e) < ambient_dim`
- `final-dim culprit`: the smallest subset `S` with
  `dim(⋂_{e∈S} E_e) <= dim(P_path)`

These are the same for empty and codim-1 paths, but they can differ once the final path has codimension at least 2.

## What I actually sampled

I did not exhaust the full `5 x 8` search space. Instead I used random models and looked for dropped paths.

The concrete sample behind the conclusions below was:

- one random `5 x 8` model where I collected the first `8` dropped paths I encountered
  all `8` had ambient dimension `7`
  all `8` had individually full-dimensional projected edges
  all `8` had `first-drop culprit` size `2`
  these `8` were all empty-path examples
- a second random `5 x 8` model where I searched specifically for nonempty dropped paths
  the first `3` nonempty examples had path dimensions `6`, `5`, and `6`
  all `3` again had individually full-dimensional projected edges
  all `3` had `first-drop culprit` size `2`
  the codim-2 example needed a `4`-edge subset to reach its final dimension

So the report is based on a moderate random sample, not a proof. But the same mechanism repeated very consistently across the examples that did drop.

## Representative examples

### 1. Empty path from a pairwise contradiction

Random model:

```julia
N = [
    1 0 1 -1 0 0 0 0
    0 2 0 1 -1 0 0 0
    1 1 0 0 0 -1 0 0
    0 0 0 0 2 0 -1 0
    0 0 1 0 0 1 0 -1
]
```

For `change_qK_idx = 1`, one dropped path is

```julia
[1, 54, 56, 50, 45, 46, 11]
```

with ambient dimension `7`, but `dim(P_path) = -1`.

All projected edge polyhedra on this path are individually full-dimensional:

```julia
[7, 7, 7, 7, 7, 7]
```

One minimal culprit is the pair

```julia
[(1, 54), (45, 46)]
```

because:

- edge `(1,54)` contains `K₃ > q₂`
- edge `(45,46)` contains `q₂ > K₁` and `K₁ > K₃`

Together they force

```text
K₃ > q₂ > K₁ > K₃
```

which is impossible. So the path emptiness is already visible from a 2-edge projected intersection.

This example is useful because the two culprit edges are not adjacent on the path. The conflict appears only after projection to the reduced `qK` coordinates.

### 2. Codimension-1 path from a pairwise pinch

Random model:

```julia
N = [
    1 2 1 -1 0 0 0 0
    0 3 1 0 -1 0 0 0
    0 2 0 0 0 -1 0 0
    0 1 2 0 0 1 -1 0
    0 1 0 0 0 1 1 -1
]
```

For `change_qK_idx = 2`, one dropped path is

```julia
[1, 11, 17, 39, 40, 46, 34, 33]
```

with ambient dimension `7`, but `dim(P_path) = 6`.

Again, every projected edge polyhedron on the path is individually full-dimensional:

```julia
[7, 7, 7, 7, 7, 7, 7]
```

The minimal culprit is the pair

```julia
[(39, 40), (33, 34)]
```

The two projected edges contain opposite bounds on the same monomial:

- edge `(39,40)` contains
  `10 * K₁^4.1918 > K₃^2.0959 K₄^2.0959 K₅^2.0959`
- edge `(33,34)` contains
  `0.1 * K₃^2.0959 K₄^2.0959 K₅^2.0959 > K₁^4.1918`

Combining them gives equality:

```text
0.1 * K₃^2.0959 K₄^2.0959 K₅^2.0959  ~  K₁^4.1918
```

and that is exactly what shows up in the path polyhedron. So this is not a contradiction; it is a pairwise pinch onto a codimension-1 interface.

### 3. Codimension-2 path from two independent pairwise pinches

Using the same `N` as Example 2 and `change_qK_idx = 2`, another dropped path is

```julia
[1, 25, 17, 39, 43, 44, 46, 10, 34, 33]
```

Here the ambient dimension is `7`, but `dim(P_path) = 5`.

Again, all projected edge polyhedra on the path are full-dimensional:

```julia
[7, 7, 7, 7, 7, 7, 7, 7, 7]
```

Now the two notions above differ:

- the `first-drop culprit` is still size `2`
- the `final-dim culprit` is size `4`

Two pairwise pinchings already appear:

```julia
[(1, 25), (10, 34)]
[(43, 44), (33, 34)]
```

The first pair gives opposite bounds on `K₄ K₅` versus `K₃^4 q₃`:

- edge `(1,25)` contains
  `32 * K₄ K₅ > K₃^4 q₃`
- edge `(10,34)` contains
  `0.03125 * K₃^4 q₃ > K₄ K₅`

so they pinch to

```text
0.03125 * K₃^4 q₃  ~  K₄ K₅
```

The second pair gives the same codim-1 equality as Example 2:

```text
0.1 * K₃^2.0959 K₄^2.0959 K₅^2.0959  ~  K₁^4.1918
```

The smallest subset that already reaches the final path dimension `5` is

```julia
[(1, 25), (43, 44), (10, 34), (33, 34)]
```

So in this example the final codimension-2 drop is built from two independent pairwise pinches. A pair is enough to start the drop, but not enough to reproduce the final path polyhedron.

## Empirical pattern from the sampled random examples

Across the random examples I inspected:

- I did not find any case where a single projected edge polyhedron was already lower-dimensional.
- In all sampled dropped paths, the smallest subset causing the first loss of dimension was a pair of projected edges.
- The culprit pair does not need to be adjacent on the path.
- Empty paths arise when two projected edges impose incompatible strict inequalities after projection.
- Codim-1 paths arise when two projected edges impose opposite bounds on the same projected monomial, pinching it to equality.
- Higher codimension paths can be understood as several such pairwise pinches accumulating.

So the recurring mechanism is not "many edges gradually erode dimension in a diffuse way". It is much more localized:

1. a pairwise projected collision creates the first drop
2. additional independent pairwise collisions may reduce dimension further

## Mathematical interpretation

Write the projected edge polyhedra as

```text
E₁, E₂, ..., E_m ⊂ R^(d+r-1)
```

and the path polyhedron as

```text
P_path = E₁ ∩ E₂ ∩ ... ∩ E_m
```

In the observed examples, each `E_i` is full-dimensional. Therefore the drop in `dim(P_path)` is not inherited from any one edge; it is created by how their supporting halfspaces align after projection.

The two dominant mechanisms are:

1. contradiction
   two projected edges force `f > 0` and `f < 0` for the same projected affine or monomial expression, so the intersection is empty
2. pinch
   two projected edges force opposite inequalities on the same projected expression, and the only way to satisfy both is to land on the shared interface `f = 0`

For codimension larger than `1`, the examples suggest that the final path dimension is often the sum of several independent pinch events. In other words, codimension seems to count how many independent projected interfaces have been forced to equality.

This is still an empirical statement, not a theorem, but it matches every random example I checked.

## Working conjecture

For generic full-dimensional projected edge polyhedra on a path:

- the first dimension drop is typically triggered by a minimal 2-edge subset
- codim-`k` nonempty path polyhedra are typically produced by `k` independent pairwise pinches
- empty path polyhedra are the degenerate version where one of those pairwise pinches is inconsistent instead of exact

Equivalently: the important object is not the whole path at once, but the graph of pairwise collisions among projected edge polyhedra in the reduced coordinate space.

## Practical consequence for debugging

When a path polyhedron drops dimension, a good diagnosis strategy is:

1. check all projected edges on the path are individually full-dimensional
2. search for minimal 2-edge intersections with `dim < ambient`
3. separate the result into
   - contradictory pairs, which explain emptiness
   - pinching pairs, which explain equalities
4. if the final codimension is greater than `1`, look for multiple independent pinching pairs

That decomposition matched all random examples I inspected and seems to be the right abstraction for understanding path-level dimension loss.
