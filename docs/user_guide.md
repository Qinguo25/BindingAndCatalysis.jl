# BindingAndCatalysis.jl User Guide

This guide is for users who want to build binding/catalysis models, solve
concentrations, enumerate regimes, inspect regime formulas, and run numerical
workflows.  It emphasizes the public API that should be used in scripts,
notebooks, and downstream projects.

## 1. Quick Start

For numerical work:

```julia
using BindingAndCatalysis
```

For plotting, load a Makie backend and `GraphMakie` before loading the package:

```julia
using CairoMakie, GraphMakie
using BindingAndCatalysis
```

Most examples in this guide use log10 coordinates.  In new code, prefer the
explicit coordinate keywords:

```julia
input = :log      # input vector is already log10-scaled
output = :log     # return log10-scaled output
```

The older `input_logspace=true` and `output_logspace=true` keywords still work,
but new examples should use `input=:log` and `output=:log`.

## 2. Minimal Binding Workflow

Build a binding model from a reaction matrix `N`:

```julia
model = Bnc(
    N = [1 1 -1],
    x_sym = [:S, :E, :C],
    q_sym = [:tS, :tE],
    K_sym = [:K],
)
```

Inspect the model:

```julia
summary(model)
show_conservation(model)
show_equilibrium(model)
```

Solve `qK -> x`, assign the point to a regime, and inspect the local formula:

```julia
logqK = [0.0, 0.0, -1.0]

logx = qK2x(model, logqK; input=:log, output=:log)
idx = assign_regime_qK_index(model, logqK; input=:log, asymptotic_only=false)
rgm = get_binding_regime(model, idx)

rgm
show_condition_qK(rgm)
show_expression_x(rgm)
```

This is the basic workflow to remember:

1. build `Bnc`,
2. solve `qK2x`,
3. assign a binding regime,
4. fetch the regime object,
5. inspect its condition and affine map.

## 3. Coordinate and Symbol Order

The package uses several coordinate vectors.  Their order matters.

| Vector | Meaning |
| --- | --- |
| `x` | species concentrations |
| `q` | conservation totals |
| `K` | binding constants |
| `qK` | `[q; K]` |
| `k` | catalysis rate constants |
| `qKk` | `[q; K; k]` |
| `wKk` | reduced BNC coordinates `[w; K; k]` |

Use symbol helpers instead of hard-coding positions:

```julia
x_sym(model)
qK_sym(model)

iC = locate_sym_x(model, :C)
iK = locate_sym_qK(model, :K)
```

After adding catalysis, these are also useful:

```julia
q_cat_sym(model)
wKk_sym(model)

itI = locate_sym_wKk(model, :tI)
```

There are two symbol-helper styles:

- `_sym` helpers, such as `x_sym` and `wKk_sym`, return Symbolics `Num`
  objects for display and symbolic expressions.
- `_symbol` helpers, such as `x_symbol` and `wKk_symbol`, return plain
  `Symbol`s for programmatic selectors and table outputs.

Common model fields:

```julia
model.n       # number of species
model.d       # number of conservation totals
model.r       # number of binding constants
model.x_sym
model.q_sym
model.K_sym
```

## 4. Converting Between `x` and `qK`

Map species concentrations to conservation totals and binding constants:

```julia
logx = [-1.0, -1.0, -2.0]
logqK = x2qK(model, logx; input=:log, output=:log)
```

Solve concentrations from totals/constants:

```julia
logqK = [0.0, 0.0, -1.0]
logx = qK2x(model, logqK; input=:log, output=:log)
```

Check the residual:

```julia
qK2x_residual(model, logx, logqK; input=:log)
```

Available `qK2x` methods:

```julia
qK2x(model, logqK; input=:log, output=:log, method=:homotopy)
qK2x(model, logqK; input=:log, output=:log, method=:free_energy)
qK2x(model, logqK; input=:log, output=:log, method=:newton_nullspace)
qK2x(model, logqK; input=:log, output=:log, method=:nlsolve)
qK2x(model, logqK; input=:log, output=:log, method=:regime)
```

Practical guidance:

- If unsure, omit `method`; the package chooses a default.
- Use `:homotopy` when robustness matters more than speed.
- Use `:regime` as a local affine predictor near a known regular regime.
- `:free_energy` is only valid for models satisfying its structural condition;
  otherwise the package redirects to `:homotopy`.

## 5. Binding Regimes

Enumerate binding regimes:

```julia
find_all_regimes!(model)
```

Most getters also build the cache automatically:

```julia
rgms = get_binding_regimes(model)
idxs = get_binding_indices(model)
perms = get_binding_perms(model)
```

Fetch one regime by index or permutation:

```julia
rgm1 = get_binding_regime(model, 1)
rgm2 = get_binding_regime(model, [1, 2])
```

Common regime helpers:

```julia
get_binding_index(rgm1)
get_binding_perm(rgm1)
get_nullity(rgm1)
is_singular(rgm1)
is_asymptotic(rgm1)
```

Filter regimes:

```julia
regular = get_binding_regimes(model; singular=false)
singular = get_binding_regimes(model; singular=true)
asymptotic = get_binding_regimes(model; asymptotic=true)

regular_ids = get_binding_indices(model; singular=false)
low_nullity = get_binding_regimes(model; max_nullity=1)
```

Filter a candidate list:

```julia
candidates = get_binding_indices(model)

selected = filter_regimes(model, candidates; singular=false)
mask = filter_regimes_mask(model, candidates; singular=false)
selected2, mask2 = filter_regimes_with_mask(model, candidates; singular=false)
```

Recommended convention:

- use `get_binding_regimes` when you need objects,
- use `get_binding_indices` when you need IDs,
- use `get_binding_perms` when you need permutations,
- avoid `return_idx=true` in new code when a dedicated getter exists.

## 6. Assigning Points to Regimes

Assign a `qK` point:

```julia
perm = assign_regime_qK(
    model,
    logqK;
    input=:log,
    asymptotic_only=false,
)

idx = assign_regime_qK_index(
    model,
    logqK;
    input=:log,
    asymptotic_only=false,
)
```

Assign an `x` point:

```julia
logx = qK2x(model, logqK; input=:log, output=:log)

perm_from_x = assign_regime_x(model, logx; input=:log)
idx_from_x = assign_regime_x_index(model, logx; input=:log)
```

Meaning of `asymptotic_only`:

- `true`: use asymptotic dominance only; usually faster.
- `false`: use coefficient-aware conditions; better for concrete numerical
  points.

## 7. Regime Conditions and Affine Maps

A binding regime condition is represented by equality and inequality rows:

```julia
C, C0, nullity = get_C_C0_nullity_qK(rgm)
```

Interpretation:

- first `nullity` rows: `C * logqK + C0 == 0`,
- remaining rows: `C * logqK + C0 >= 0`.

Common binding condition getters:

```julia
get_C_C0_qK(rgm)
get_C_qK(rgm)
get_C0_qK(rgm)

get_C_C0_x(rgm)
get_C_x(rgm)
get_C0_x(rgm)
```

Common binding affine maps:

```julia
get_affine_x2q(rgm)       # log x -> log q
get_affine_x2qK(rgm)      # log x -> log qK
get_affine_qK2x(rgm)      # log qK -> log x

get_P_P0(rgm)             # compatibility alias for log x -> log q
get_M_M0(rgm)             # compatibility alias for log x -> log qK
get_H_H0(rgm)             # compatibility alias for log qK -> log x
```

Display symbolic conditions and formulas:

```julia
show_condition_qK(rgm)
show_condition_x(rgm)
show_expression_x(rgm)
```

For singular regimes, `log qK -> log x` may not have a unique affine
representation.  If expression rendering fails, inspect:

```julia
get_nullity(rgm)
is_singular(rgm)
get_C_C0_nullity_qK(rgm)
```

## 8. Regime Graphs and Visualization

Build the binding regime graph:

```julia
grh = get_regimes_graph!(model)
```

Query neighbors and edges:

```julia
get_neighbors(model, 1)
get_neighbors(model, 1; return_idx=true)

edge = get_edge(model, 1, 2)
```

Build chart-specific graph views:

```julia
get_neighbor_graph_x(model)
get_neighbor_graph_qK(model)
```

Draw the graph:

```julia
using CairoMakie, GraphMakie
using BindingAndCatalysis

grh = get_regimes_graph!(model)
draw_graph(grh; chart=:qK)
```

Plot a 2D binding partition in `qK` space:

```julia
plot_binding_regime_partition(
    model;
    axes = [:tS, :K],
    fixed = Dict(:tE => 0.0),
    ranges = (-4.0, 4.0),
    n = 200,
    chart = :qK,
)
```

`plot_binding_regime_partition` returns:

```julia
fig, ax, data = plot_binding_regime_partition(...)
data.values
```

If `draw_graph` or partition plotting says optional visualization packages are
required, verify that the active Julia environment can load:

```julia
using CairoMakie, GraphMakie
```

## 9. Adding Catalysis

Start with a binding model:

```julia
model = Bnc(
    N = [1 0 1 -1 0;
         0 1 1  0 -1],
    x_sym = [:S, :P, :E, :C1, :C2],
    q_sym = [:tS, :tP, :tE],
    K_sym = [:K1, :K2],
)
```

Attach a catalysis layer:

```julia
Pi = [1 0;
      0 1]

Gamma = [1 -1;
        -1  1]

update_catalysis!(
    model;
    Γ = Gamma,
    Π = Pi,
    x_picked = [:C1, :C2],
    q_picked = [:tP, :tS],
    k_sym = [:k1, :k2],
    w_sym = [:TS],
)
```

Parameter meanings:

- `Π`: exponent matrix of flux monomials with respect to the picked species.
- `Γ`: catalysis change matrix in the reduced conservation coordinates.
- `x_picked`: species corresponding to columns of `Π`.
- `q_picked`: conservation coordinates changed by catalysis.
- `k_sym`: catalysis rate-constant names.
- `w_sym`: catalysis-induced conserved quantities.

If `Π` already has one column per species in `x`, `x_picked` can be omitted.

## 10. Catalysis and BNC Regimes

Enumerate catalysis regimes:

```julia
cat_rgms = get_catalysis_regimes(model)
cat_ids = get_catalysis_indices(model)
cat_perms = get_catalysis_perms(model)
```

Build Binding-and-Catalysis regimes:

```julia
bnc_rgms = get_bnc_regimes(model)
bnc_ids = get_bnc_indices(model)
bnc_perms = get_bnc_perms(model)
```

Fetch one BNC regime:

```julia
bnc_rgm = get_bnc_regime(model, 1)
bnc_rgm = get_bnc_regime(model, bind_idx, cat_idx)
```

Access component regimes:

```julia
bind_rgm = get_binding_regime(bnc_rgm)
cat_rgm = get_catalysis_regime(bnc_rgm)
```

Filter BNC regimes:

```julia
feasible = get_bnc_regimes(model; feasible=true)
all_bnc = get_bnc_regimes(model; feasible=nothing)
regular = get_bnc_regimes(model; singular=false)
stable = get_bnc_regimes(model; stable=true)
```

For BNC regimes, `singular=false` refers to the BNC regime nullity `rgm.nlt`.
It does not guarantee that the underlying binding regime has an exact binding
derivative.  Use `hbd_source(bnc_rgm)` when derivative provenance matters.

Check stability:

```julia
is_stable(bnc_rgm)
stability_code(bnc_rgm)
```

BNC condition getters:

```julia
get_C_C0_nullity_qKk(bnc_rgm)
get_C_C0_nullity_wKk(bnc_rgm)
get_C_C0_xk(bnc_rgm)
```

BNC affine maps:

```julia
get_affine_wKk2x(bnc_rgm)
get_affine_wKk2xk(bnc_rgm)
get_affine_wKk2v(bnc_rgm)
get_affine_wKk2qcat(bnc_rgm)
```

The local binding derivative used in BNC dynamics is:

```julia
get_H_bd(bnc_rgm)
```

To inspect whether this derivative came from an exact regime derivative or a
numerical binding derivative:

```julia
info = get_H_bd_info(bnc_rgm)
info.H
info.source
hbd_source(bnc_rgm)
```

Assign a `wKk` point to a BNC regime:

```julia
idx = assign_bnc_regime_wKk(model, logwKk; tol=1e-8, max_nullity=0)
```

Plot a BNC partition:

```julia
plot_bnc_regime_partition(
    model;
    axes = [:K1, :k1],
    fixed = Dict(:TS => 0.0, :K2 => 0.0, :k2 => 0.0),
    ranges = (-4.0, 4.0),
    n = 100,
    chart = :wKk,
)
```

## 11. Linear BNC Control and Responsiveness

For a BNC regime, build the local linearized control model in `log(q_cat)`
coordinates:

```julia
ctrl = linear_control_model(
    bnc_rgm;
    input = :all,      # symbols from wKk_symbol(model), or one symbol
    output = :x,       # :x, :qcat, or one output symbol
)
```

This is the standard state-space form:

```text
d state / dt = A * state + B * input
output       = C * state + D * input
```

In control terminology, `A` is the state/system matrix, `B` is the input/control
matrix, `C` is the output/observation matrix, and `D` is the direct-feedthrough
matrix. The model also stores symbol order metadata so the rows and columns can
be interpreted safely:

```julia
ctrl.A
ctrl.B
ctrl.C
ctrl.D
ctrl.eigvals
ctrl.stable
ctrl.input
ctrl.output
ctrl.hbd_source
```

Use `control_metrics` when you want a named tuple:

```julia
metrics = control_metrics(bnc_rgm; input=:all, output=:qcat)
metrics.A
metrics.eigvals
```

Common regime-level control summaries:

```julia
controllability_matrix(ctrl)
output_controllability_matrix(ctrl)
output_controllability_row(ctrl)
markov_coefficients(ctrl)
steady_state_gain(ctrl)
```

`steady_state_gain(ctrl)` is the dynamic DC gain `D - C * (A \ B)`. If you need
the exact affine steady-state coefficient from the regime map, use:

```julia
affine_steady_state_gain(ctrl)
dynamic_steady_state_gain(ctrl)
```

For stable `A`, compute the infinite-horizon controllability Gramian:

```julia
W = controllability_gramian(ctrl)
E = output_energy(ctrl; include_direct=true)
```

Steady-state invariance and responsiveness helpers return diagnostic named
tuples:

```julia
inv = steady_state_invariance(ctrl; standard=:affine, atol=1e-8)
inv.invariant
inv.residual

dc_inv = steady_state_invariance(ctrl; standard=:dynamic_dc_gain, atol=1e-8)

resp = input_responsiveness(
    bnc_rgm;
    input = first(wKk_symbol(model)),
    output = first(x_symbol(model)),
    standard = :output_controllability,
    threshold = 1e-8,
)
resp.responsive
resp.score
```

Supported responsiveness standards are:

- `:direct_feedthrough`, the direct `D` matrix term;
- `:direct_flux`, a signed input-drive score that requires explicit
  `positive_flux` and optional `negative_flux` terms;
- `:output_controllability`,
- `:output_reachability`, with `direction=:any`, `:positive`, or `:negative`;
- `:gramian`, with `threshold_scale=:energy` or `:amplitude`;
- `:steady_state_gain`, the dynamic DC-gain magnitude.

For a report-specific flux-drive standard, pass the positive and negative terms
explicitly. The package does not infer circuit-specific biology from names such
as `C1` and `C2`:

```julia
drive = input_drive(
    bnc_rgm;
    input = :tI,
    positive = :C1,
    negative = :C2,
    target_space = :x,
    direction = :positive,
)

flux_resp = input_responsiveness(
    bnc_rgm;
    input = :tI,
    output = :tAstar,
    standard = :direct_flux,
    positive_flux = :C1,
    negative_flux = :C2,
    direction = :positive,
    threshold = 0.1,
)
```

For signed reachability, use:

```julia
input_responsiveness(
    bnc_rgm;
    input = :tI,
    output = :Astar,
    standard = :output_reachability,
    direction = :positive,
    threshold = 0.1,
)
```

For Gramian-based output energy, `threshold_scale=:amplitude` compares output
energy against `threshold^2`, while `threshold_scale=:energy` compares directly:

```julia
input_responsiveness(
    bnc_rgm;
    input = :tI,
    output = :Astar,
    standard = :gramian,
    threshold = 0.1,
    threshold_scale = :amplitude,
    include_direct = true,
)
```

For batch comparisons across regimes and outputs:

```julia
rows = compare_input_responsiveness(
    get_bnc_regimes(model; stable=true);
    input = first(wKk_symbol(model)),
    outputs = (first(x_symbol(model)), first(q_cat_symbol(model))),
    standards = (:direct_feedthrough, :output_controllability),
    threshold = 1e-8,
)
```

## 12. Constrained Multistability and R-Index

Analysis-time parameter constraints let you study regime feasibility, stable
overlap, and R-index inside a constrained parameter family without modifying the
model's regime cache.

The default chart is context dependent:

- binding-only models use `chart=:qK`;
- BNC/catalysis models use `chart=:wKk`.

For constrained R-index work, it is useful to separate two concepts:

1. a parameter chart, which says how the original coordinates are represented
   by new reduced coordinates;
2. constraints inside that reduced chart.

For example, several original parameters can be identified with one biological
parameter:

```julia
chart = parameter_chart(
    model;
    chart = :wKk,
    map = Dict(
        :γ1 => :loss,
        :γ2 => :loss,
        :δ1 => :loss,
        :δ2 => :loss,
        :β1 => :beta,
        :β2 => :beta,
    ),
)
```

This means:

```text
old_wKk = chart.F * new_wKk + chart.F0
```

The same identification can be written by groups:

```julia
chart = parameter_chart(
    model;
    chart = :wKk,
    groups = Dict(
        :loss => [:γ1, :γ2, :δ1, :δ2],
        :beta => [:β1, :β2],
    ),
)
```

Advanced users can pass the affine map directly:

```julia
chart = parameter_chart(
    model;
    chart = :wKk,
    F = F,
    F0 = F0,
    reduced_symbols = [:loss, :beta, :K],
)
```

The fields `chart.basis` and `chart.offset` are aliases for `chart.F` and
`chart.F0`, matching the lower-level restriction internals.

Matrix constraints use the package condition convention:

```julia
C * z + C0 == 0   # first nullity rows
C * z + C0 >= 0   # remaining rows
```

For a BNC model:

```julia
constraints = parameter_constraints(
    model;
    chart = :auto,
    C = C,
    C0 = C0,
    nullity = nlt,
)
```

Convenience symbolic constraints are also supported:

```julia
constraints = parameter_constraints(
    model;
    equalities = [:k1 => :k2],
    inequalities = [(:K1, :<, :Kp1), (:K2, :<, :Kp2)],
)
```

When using a `ParameterChart`, constraints are written in the reduced symbols by
default:

```julia
constraints = parameter_constraints(
    chart;
    inequalities = [(:K, :<, :Kp)],
)
```

If you prefer to write inequalities in the original symbols and pull them back
through the chart, use `symbols=:original`:

```julia
constraints = parameter_constraints(
    chart;
    inequalities = [(:K1, :<, :Kp1), (:K2, :<, :Kp2)],
    symbols = :original,
)
```

The one-step form is also supported. In that form, symbolic constraints are
interpreted in the original chart unless `constraint_symbols=:reduced` is
provided:

```julia
constraints = parameter_constraints(
    model;
    chart = :wKk,
    map = Dict(:K1 => :K, :K2 => :K),
    inequalities = [(:K1, :<, :Kp1), (:K2, :<, :Kp2)],
)
```

Strict inequalities are represented as closed halfspaces for volume/R-index
calculations.  This does not affect full-dimensional asymptotic volume because
the boundary has zero measure.

Restrict regimes under the constraints:

```julia
restricted = restrict_regimes(
    get_bnc_regimes(model),
    constraints;
    stable = true,
    singular = false,
    feasible = true,
    full_dim = true,
)
```

Each entry records feasibility and dimension diagnostics:

```julia
rr = first(restricted)
rr.poly
rr.feasible
rr.dim
rr.ambient_dim
rr.full_dim
rr.reason
```

Find full-dimensional stable pair intersections:

```julia
pairs = stable_regime_intersections(
    get_bnc_regimes(model);
    constraints,
    full_dim = true,
)
```

For a constrained sampling estimate of multistability:

```julia
profile = multistability_profile(
    model;
    constraints,
    samples = 500_000,
    mode = :finite_region,
)
```

The denominator is the constraint region: sampling first accepts points that
satisfy `constraints`, then counts how many stable restricted BNC regimes contain
each accepted point.

Use `mode=:asymptotic_R` for the asymptotic solid-angle R-index convention. This
strips offsets and samples recession-cone membership:

```julia
profile = multistability_profile(
    model;
    constraints,
    samples = 500_000,
    mode = :asymptotic_R,
)
```

For report-style summaries, use:

```julia
summary = multistability_R_index(
    model;
    constraints,
    samples = 500_000,
)
```

`multistability_R_index` defaults to `mode=:asymptotic_R` and returns
`full_dim_regimes`, `stable_full_dim_regimes`, `pair_intersections`,
`R_multistability`, `stderr`, `basis_kind`, and `denominator`.

`full_dim_regimes` is a broad diagnostic count of all feasible full-dimensional
restricted BNC regimes. The stable candidate counts, pair intersections, and
`R_multistability` use the candidate filter controlled by `singular`, which
defaults to nonsingular regimes.

Empty `map` or `groups` inputs are normalized to the identity chart. This is
useful when a script shares one code path across unconstrained and grouped
parameter families.

Useful fields:

```julia
profile.mode
profile.denominator
profile.basis_kind
profile.R_atleast_1    # any stable regime
profile.R_atleast_2    # bistability R-index
profile.R_atleast_3    # multistability hint
profile.max_hit_count
profile.combination_counts
profile.pair_table
```

`combination_counts` is useful for multistability exploration.  If many sampled
points hit the same triple of stable regimes, that combination is a good
candidate for a later explicit intersection check.

## 13. Catalysis and Adaptation Simulations

For full catalysis dynamics from an initial `x`:

```julia
logx0 = fill(-1.0, model.n)
t, logx_traj = x_traj_cat(
    model,
    logx0,
    (0.0, 100.0);
    input=:log,
    output=:log,
)
```

For reduced qcat dynamics with a fixed or time-dependent `wKk`:

```julia
logqcat0 = fill(-2.0, length(q_cat_sym(model)))
logwKk = zeros(length(wKk_sym(model)))

t, logqcat_traj = qcat_traj_cat(
    model,
    logqcat0,
    logwKk,
    (0.0, 100.0);
    input=:log,
    output=:log,
)
```

For a step-response/adaptation workflow, use `simulate_adaptation`:

```julia
base = zeros(length(wKk_sym(model)))
itI = locate_sym_wKk(model, :tI)

result = simulate_adaptation(
    model;
    p = base,
    logtI = t -> t < 10.0 ? base[itI] : base[itI] + log10(2.0),
    tspan = (0.0, 200.0),
    observe = :Astar,
)

result.t
result.logtI
result.logobserve
result.logqcat
```

`simulate_adaptation` solves the catalysis dynamics and observes either a qcat
coordinate or an `x` species.  When observing an `x` species, it repeatedly
solves the binding problem along the trajectory.  Always inspect `NaN` values
in `result.logobserve`; they indicate that the binding solve failed at that
time point.

## 14. Symbolic Output

The symbolic display helpers return equations or inequalities that are useful
for notebooks and reports:

```julia
show_condition_qK(rgm; log_space=false)
show_condition_x(rgm; log_space=false)
show_expression_x(rgm; log_space=false)

show_condition_qKk(bnc_rgm; kind=:binding)
show_condition_qKk(bnc_rgm; kind=:catalysis)
show_expression_qcat(bnc_rgm)
show_catalysis_dynamics(bnc_rgm)
```

Use `log_space=false` when you want expressions in ordinary symbolic variables
rather than log variables.

## 15. SIMO Workflows

For one-input sweeps and path summaries:

```julia
paths = SIMOPaths(model, :tS)
draw_graph(paths)

SIMO_plot(
    model,
    [0.0, -1.0],
    :tS;
    observe_x = [:S, :C],
    show_regime_label = true,
)
```

The SIMO tools are useful when one coordinate changes and the other coordinates
are fixed.

## 16. Recommended Names and Legacy API

Use explicit names in new code:

```julia
get_binding_regime(model, idx)
get_binding_regimes(model)
get_binding_indices(model)
get_binding_perms(model)

get_catalysis_regime(model, idx)
get_catalysis_regimes(model)

get_bnc_regime(model, idx)
get_bnc_regimes(model)
```

Legacy and convenience names still work:

```julia
get_regime(model, idx)
get_regimes(model)
get_indices(model)
get_perms(model)
```

Recommended rules:

- `get_regime` is a binding-regime alias, not a generic regime getter.
- Use `get_bnc_regime` / `get_bnc_regimes` for BNC regimes.
- Use `get_catalysis_regime` / `get_catalysis_regimes` for catalysis regimes.
- Prefer explicit `binding`, `catalysis`, and `bnc` names in user-facing code.

## 17. Common Issues

### `qK2x` warns that `L * N' is nonzero`

This means the model does not satisfy the structural condition required by the
free-energy solver.  The package defaults to `:homotopy`, or redirects
`method=:free_energy` to `:homotopy`.  This is not an error.

### `draw_graph` says optional visualization packages are required

Confirm that the current environment can load:

```julia
using CairoMakie, GraphMakie
```

In notebooks, use this order:

```julia
using CairoMakie, GraphMakie
using BindingAndCatalysis
```

If the error persists, restart the Julia kernel and make sure the active
environment contains `GraphMakie` and a Makie backend.

### `show_expression_x` fails on a singular regime

Check:

```julia
get_nullity(rgm)
is_singular(rgm)
```

For a singular regime, inspect the condition matrix instead:

```julia
get_C_C0_nullity_qK(rgm)
```

### Regime assignment is not unique

The point may lie on a boundary, or the tolerance may be too strict.  Try:

```julia
assign_regime_qK_index(model, logqK; input=:log, eps=1e-8)
```

or use coefficient-aware numerical conditions:

```julia
assign_regime_qK_index(model, logqK; input=:log, asymptotic_only=false)
```

### BNC regimes silently exclude infeasible points

By default, `get_bnc_regimes(model)` returns feasible regimes only.  To inspect
everything:

```julia
get_bnc_regimes(model; feasible=nothing)
```

## 18. Complete Small Example

```julia
using CairoMakie, GraphMakie
using BindingAndCatalysis

model = Bnc(
    N = [1 1 -1],
    x_sym = [:S, :E, :C],
    q_sym = [:tS, :tE],
    K_sym = [:K],
)

logqK = [0.0, 0.0, -1.0]
logx = qK2x(model, logqK; input=:log, output=:log)

idx = assign_regime_qK_index(model, logqK; input=:log, asymptotic_only=false)
rgm = get_binding_regime(model, idx)

rgm
show_condition_qK(rgm)
show_expression_x(rgm)

grh = get_regimes_graph!(model)
draw_graph(grh; chart=:qK)
```
