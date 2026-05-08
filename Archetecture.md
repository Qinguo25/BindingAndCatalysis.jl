# BindingAndCatalysis.jl Architecture

This package analyzes biochemical systems with a fast binding equilibrium layer
and an optional slower catalysis layer.  The main object is `Bnc`, which stores
the model definition plus lazily built caches for regimes, graphs, classifiers,
and numerical integration helpers.

## Binding Layer

A binding model is defined by

```math
q = Lx,\qquad N\log x = \log K.
```

`x` is the species vector, `q` contains conserved totals, and `K` contains
binding constants.  The exact local map from `(q,K)` to `x` is implicit, and its
log-derivative is

```math
H(x)=
\begin{bmatrix}
\Lambda_q^{-1}L\Lambda_x\\
N
\end{bmatrix}^{-1}.
```

Binding regimes approximate this manifold by dominance choices.  In each row of
`L`, one species dominates the corresponding total.  This gives

```math
\log q = P\log x + P_0,\qquad
\log(q,K)=
\begin{bmatrix}P\\N\end{bmatrix}\log x+
\begin{bmatrix}P_0\\0\end{bmatrix}.
```

The core binding-regime data are:

- `perm`: dominant species per total row
- `P, P0`: affine `x -> q` approximation
- `H, H0`: affine `qK -> x` map for regular regimes, or directional data for
  nullity-one singular regimes
- `C_x, C0_x`: dominance conditions in `x`
- `C_qK, C0_qK`: transported conditions in `qK`
- `nullity`: `0` regular, `1` singular facet, `>1` higher-codimension regime

## Catalysis Layer

After `update_catalysis!`, the model gains catalysis data

```math
\log v = \Pi\log x + \log k,\qquad \dot q = \Gamma v.
```

Rows of `q` are reordered into catalytic variables `q_cat` and conserved
slow variables `w`.  The reduced dynamics are

```math
\dot q_{cat}=Sv,\qquad \dot w=0.
```

Catalysis regimes encode flux dominance in `v` space and the equivalent
conditions in `(x,k)` space.  Their main data are:

- `P, P0`: fixed-point balance expression between positive and negative fluxes
- `C, C0`: flux dominance conditions in `v`
- `PΠ, CΠ`: the same information pulled back through `log v = Π log x + log k`

## Bnc Regimes

A `BncRegime` pairs one binding regime and one catalysis regime.  At a fixed
point, flux balance removes the free `q_cat` coordinates:

```math
\begin{bmatrix}
\log w\\
\log K\\
-P\log k
\end{bmatrix}
=
\begin{bmatrix}
P_w\\
N\\
P\Pi
\end{bmatrix}\log x+
\begin{bmatrix}
P_{0,w}\\
0\\
P_0
\end{bmatrix}.
```

For regular mixed regimes, this gives an affine map from `(w,K,k)` to `x` and
then to `q_cat`.  For singular mixed regimes, the package keeps consistency
conditions and directional data when available.

Stability is judged from

```math
H^{bd}=P\Pi H_{q_{cat}},
```

using the `d_stable` classifier.  If the inner binding regime is singular,
`get_H_bd_numerically` is used to obtain the binding response numerically.

## Regime Graphs

`RegimeGraph` stores one node per regime and one or more edge charts.  The edge
chart names are held in `space_idx`, so callers use symbols rather than fixed
integer slots:

- binding graph charts: `:x`, `:qK`
- catalysis graph charts: `:v`, `:xk`
- Bnc graph charts: `:xk`, `:qKk`, `:wKk`

`RegimeEdge` stores a vector of `(hyperplane_index, sign)` pairs.  Each index
points into the matching hyperplane pool for the chosen chart.

Use `draw_graph(grh; chart=:qK)` or `draw_graph(grh; chart=:wKk)` to choose the
displayed edge chart.  The older `edge_space` keyword is still accepted for
compatibility.

## Numerical qK -> x

`qK2x` supports several methods:

- `:free_energy`: default robust point solver; positive by construction
- `:newton_nullspace`: Newton solve on `x = x0 + N' m`, with free-energy fallback
- `:homotopy`: path following in log `qK` coordinates
- `:nlsolve`: package-independent full-space Newton
- `:regime`: asymptotic affine predictor, not an exact solver

`qK2x_residual` reports the actual equation residual in log coordinates.

## Visualization

The visualization layer contains:

- `draw_graph`: regime graph drawing, using `chart=...`
- `SIMO_plot`: one-dimensional parameter sweeps with regime shading
- `plot_binding_regime_partition`: 2D/3D binding partition plots
- `plot_bnc_regime_partition`: 2D/3D fixed-point mixed-regime partition plots
- `plot_qcat_slice_with_flux`: fixed `(w,K,k)` qcat slice with flux arrows and
  fixed-point markers

Invalid or infeasible partition grid points are rendered transparent.

## File Map

- `src/initialize.jl`: model construction and catalysis attachment
- `src/BindingRegime.jl`, `src/CatalysisRegime.jl`, `src/BncRegime.jl`: regime
  objects and affine/condition initialization
- `src/RegimeCore.jl`: shared getters for networks, regimes, affine maps, and
  conditions
- `src/*RegimeGraph.jl`, `src/Mathcore/perm_graph_core.jl`: graph construction
  and edge hyperplane storage
- `src/qK_x_mapping.jl`: numerical `qK <-> x` mapping and catalysis simulation
- `src/regime_assign.jl`: compiled regime classifiers
- `src/visualization/`: graph, SIMO, partition, and polyhedron plotting
