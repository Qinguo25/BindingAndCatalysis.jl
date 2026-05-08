# BindingAndCatalysis.jl

BindingAndCatalysis.jl analyzes equilibrium binding networks and catalysis-driven
slow dynamics in dominance-regime coordinates.

It provides tools for:

- building binding models from `N` or `L`
- mapping between species concentrations `x` and totals/constants `(q,K)`
- enumerating binding, catalysis, and mixed Bnc regimes
- constructing regime graphs in several charts
- checking fixed-point consistency and structural stability
- plotting regime graphs, SIMO sweeps, and 2D/3D regime partitions

## Installation

For local development:

```julia
using Pkg
Pkg.develop(path="/path/to/BindingAndCatalysis.jl")
Pkg.instantiate()
```

For notebooks and plotting examples:

```julia
using Pkg
Pkg.activate("Examples")
Pkg.instantiate()
```

The polyhedral backend uses `Polyhedra.jl` and `CDDLib.jl`.

## Quick Start: Binding Model

```julia
using BindingAndCatalysis

binding = Bnc(
    N = [1 1 -1],
    x_sym = [:S, :E, :C],
    q_sym = [:tS, :tE],
    K_sym = [:K],
)

show_conservation(binding)
show_equilibrium(binding)
```

Map between `qK` and `x` in log10 coordinates:

```julia
logqK = [0.0, 0.0, -1.0]
logx = qK2x(binding, logqK; input_logspace=true, output_logspace=true)
qK2x_residual(binding, logx, logqK; input_logspace=true)
```

Available `qK2x` methods:

```julia
qK2x(binding, logqK; method=:free_energy, input_logspace=true, output_logspace=true)
qK2x(binding, logqK; method=:newton_nullspace, input_logspace=true, output_logspace=true)
qK2x(binding, logqK; method=:homotopy, input_logspace=true, output_logspace=true)
qK2x(binding, logqK; method=:nlsolve, input_logspace=true, output_logspace=true)
qK2x(binding, logqK; method=:regime, input_logspace=true, output_logspace=true) # predictor
```

`method=:free_energy` is the robust pointwise default.  `method=:homotopy` is
for path-following when the path itself matters.

## Regimes and Graphs

```julia
rgms = get_regimes(binding)
grh = get_regimes_graph!(binding; full=true)

draw_graph(grh; chart=:x)
draw_graph(grh; chart=:qK)
```

`draw_graph` uses `chart` to choose which edge hyperplanes to display.  Supported
charts are:

- binding graphs: `:x`, `:qK`
- catalysis graphs: `:v`, `:xk`
- Bnc graphs: `:xk`, `:qKk`, `:wKk`

## Adding Catalysis

```julia
model = Bnc(
    N = [1 0 1 -1 0;
         0 1 1  0 -1],
    x_sym = [:S, :P, :E, :C1, :C2],
    q_sym = [:tS, :tP, :tE],
    K_sym = [:K1, :K2],
)

Π = [1 0 0 0 0;
     0 1 0 0 0]

Γ = [1 -1;
    -1  1]

update_catalysis!(
    model;
    Π = Π,
    Γ = Γ,
    x_picked = [:C1, :C2],
    q_picked = [:tP, :tS],
    w_sym = [:TS],
)

bnc_rgms = get_bnc_regimes(model)
bnc_grh = get_bnc_regimes_graph!(model)
draw_graph(bnc_grh; chart=:wKk)
```

## Visualization

Binding regime partition:

```julia
plot_binding_regime_partition(
    model;
    axes = [:TS, :tP],
    fixed = Dict(:tE => 0, :K1 => -1, :K2 => 1),
    ranges = (-6, 6),
    n = 300,
    chart = :x,
)
```

Invalid or infeasible grid points are transparent.

Bnc fixed-point partition:

```julia
plot_bnc_regime_partition(
    model;
    axes = [:K1, :k1],
    fixed = Dict(:TS => 0, :tE => 0, :K2 => 0, :k2 => 0),
    chart = :wKk,
)
```

SIMO sweep:

```julia
SIMO_plot(
    binding,
    [0.0, -1.0],
    :tS;
    observe_x = [:S, :C],
    show_regime_label = true,
)
```

## Symbol Helpers

Symbolic helpers return `Symbolics.Num` values:

```julia
x_sym(model), q_sym(model), K_sym(model), qK_sym(model)
k_sym(model), v_sym(model), wKk_sym(model)
```

Plain-symbol helpers return `Vector{Symbol}`:

```julia
x_symbol(model), q_symbol(model), K_symbol(model), qK_symbol(model)
k_symbol(model), v_symbol(model), wKk_symbol(model)
```

## Documentation

- [Archetecture.md](Archetecture.md): current internal architecture
- [Examples/Minimal_example.ipynb](Examples/Minimal_example.ipynb): step-by-step
  binding-regime workflow
- [noback/Visualization_demo.ipynb](noback/Visualization_demo.ipynb): generated
  visualization examples

## License

See [LICENSE](LICENSE).
