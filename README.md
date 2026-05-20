# BindingAndCatalysis.jl

This branch is the supplementary-information code snapshot for the paper
associated with the `ifac2026` branch.  It contains the Julia package and
notebooks used to reproduce the regime calculations, R-index estimates, and
adaptation checks reported in the paper.

`BindingAndCatalysis.jl` analyzes biochemical systems with a fast binding
equilibrium layer and an optional slower catalysis layer.  The package works in
dominance-regime coordinates: it enumerates asymptotic regimes, transports
dominance conditions between coordinate charts, builds regime graphs, and
computes volume-based realizability indices.

## SI Contents

- [`Examples/ifac2026_MM_and_HIll_part.ipynb`](Examples/ifac2026_MM_and_HIll_part.ipynb):
  Michaelis-Menten and Hill-function regime analysis.  This notebook enumerates
  binding dominance regimes, identifies the regimes matching the limiting
  Michaelis-Menten and Hill expressions, plots regime partitions, and computes
  R-index quantities for sequential Hill functions.
- [`Examples/ifac2026_Adaptation_circuit_part.ipynb`](Examples/ifac2026_Adaptation_circuit_part.ipynb):
  competitive negative-feedback adaptation.  This notebook builds the coupled
  binding-catalysis model, filters full-dimensional stable BNC regimes by
  invariance and responsiveness criteria, estimates the accepted-regime R-index,
  inspects the representative regime highlighted in the paper, and runs the
  numerical adaptation check.
- [`Archetecture.md`](Archetecture.md): implementation notes for the package
  internals, including binding regimes, catalysis regimes, BNC regimes, graph
  charts, numerical solvers, and visualization utilities.

All parameter coordinates used in the SI notebooks are `log10` coordinates.

## Reproducing The Notebooks

Start from this branch:

```bash
git clone git@github.com:Qinguo25/BindingAndCatalysis.jl.git
cd BindingAndCatalysis.jl
git checkout ifac2026
```

Instantiate the package environment:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Instantiate the notebook environment:

```julia
using Pkg
Pkg.activate("Examples")
Pkg.instantiate()
```

Then open the notebooks in `Examples/` with a Julia kernel, for example through
VS Code or Jupyter.  The notebook environment uses the package source from the
repository root:

```toml
[sources]
BindingAndCatalysis = {path = ".."}
```

The R-index cells use numerical volume estimates.  Re-running them may give
small differences within the reported tolerances.

## Package Capabilities

The package provides tools for:

- constructing binding models from `N` or `L`;
- mapping between species concentrations `x` and totals/constants `(q,K)`;
- enumerating binding, catalysis, and mixed BNC regimes;
- transporting regime inequalities between `x`, `qK`, `xk`, `qKk`, and `wKk`
  coordinate charts;
- checking fixed-point consistency and structural stability;
- computing regime-cone volumes and R-index sums;
- plotting regime graphs, SIMO sweeps, and 2D/3D regime partitions.

## Minimal Usage

```julia
using BindingAndCatalysis

model = Bnc(
    N = [1 1 -1],
    x_sym = [:S, :E, :C],
    q_sym = [:tS, :tE],
    K_sym = [:K],
)

show_conservation(model)
show_equilibrium(model)

rgms = get_regimes(model)
grh = get_regimes_graph!(model; full=true)
draw_graph(grh; chart=:qK)
```

For coupled binding-catalysis systems, attach the catalysis layer with the
model-specific flux-exponent matrix `Π` and stoichiometry matrix `Γ`, then
enumerate mixed regimes:

```julia
update_catalysis!(
    bnc_model;
    Π = Π,
    Γ = Γ,
    x_picked = x_picked,
    q_picked = q_picked,
    w_sym = w_symbols,
)

bnc_rgms = get_bnc_regimes(bnc_model)
bnc_grh = get_bnc_regimes_graph!(bnc_model) # not fully tested 
draw_graph(bnc_grh; chart=:wKk) # not fully tested
```

See the adaptation notebook for a complete coupled binding-catalysis model.

Supported graph charts are:

- binding graphs: `:x`, `:qK`;
- catalysis graphs: `:v`, `:xk`;
- BNC graphs: `:xk`, `:qKk`, `:wKk`.

## Development Checks

Run the package test suite from the repository root:

```julia
using Pkg
Pkg.activate(".")
Pkg.test()
```

The polyhedral computations use `Polyhedra.jl` with `CDDLib.jl`; plotting
examples use Makie/CairoMakie through the notebook environment.

## License

See [`LICENSE`](LICENSE).
