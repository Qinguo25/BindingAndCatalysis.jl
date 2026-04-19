---
name: bnc-julia-maintainer
description: Use when working inside BindingAndCatalysis.jl on Julia code, especially for regime enumeration, exact/float polyhedral backends, vendored local cdd builds, SIMO workflows, symbolic rendering, tests, and Makie graph visualization. This skill is for repo-specific debugging and maintenance, not general Julia help.
---

# BindingAndCatalysis.jl Maintainer Skill

Use this skill when modifying this repository. It captures repo-specific workflows and failure modes that are easy to miss if you approach it like a generic Julia package.

Source priority for this repo is:

1. current Julia source under `src/`
2. tests under `test/`
3. docs such as `Archetecture.md`

`Archetecture.md` is still useful for orientation, but parts of it may lag behind backend refactors.

## What To Read First

- Read `src/BindingAndCatalysis.jl` for module layout and exported surface.
- Read `Archetecture.md` for the high-level structure, then verify backend details against source.
- If the task touches exact/log arithmetic, read:
  - `src/ExactTypes.jl`
- If the task touches polyhedra or projection, read:
  - `src/PolyBackend.jl`
  - `src/CddBridge.jl`
  - `src/NativePolyhedra/NativePolyhedra.jl`
  - `src/NativePolyhedra/polyhedra_core.jl`
  - `src/NativePolyhedra/vrep_core.jl`
- If the task touches mixed regimes, read:
  - `src/mixed_regime/bnc_core.jl`
  - `src/mixed_regime/bnc_conditions.jl`
- If the task touches path conditions or reaction-order workflows, read:
  - `src/SIMO.jl`
  - `src/simo/core.jl`
  - `src/simo/polyhedra.jl`
  - `src/simo/reaction_order.jl`
- If the task touches symbolic output, read:
  - `src/output/symbolic_api.jl`
  - `src/output/symbolic_paths.jl`
  - `src/output/symbolic_renderers.jl`
  - `src/output/singular_x_range.jl`
- If the task touches graph drawing, read:
  - `src/visualization/graphs.jl`
  - `src/visualization/simo_plot.jl`
  - `src/visualization/poly_slices.jl`
  - `src/visualize.jl`

## Mental Model

There are four layers that matter most in maintenance work:

1. Domain logic
   - binding regimes, mixed regimes, qK/x mapping, SIMO paths, symbolic output
2. In-memory polyhedron representation
   - `NativePolyhedra.Polyhedron` and related helpers
   - still used for data structures and some geometry helpers such as `vrep` and `interior_point`
3. Polyhedral backend facade and bridge
   - `src/PolyBackend.jl`
   - `src/CddBridge.jl`
   - runtime elimination/intersection/projection go through local `cdd/cddlog`
4. Build and artifact plumbing
   - `deps/build.jl`
   - `scripts/build_local_cdd.sh`
   - `Artifacts.toml`

Important current invariants:

- Project-level H-representation uses `C * x + C0 >= 0`
- `NativePolyhedra` stores halfspaces as `a * x <= beta`
- Conversion is `a = -C`, `beta = C0`

Do not mix these conventions.

## Backend Rules

- The repository no longer uses `CDDLib.jl` or `Polyhedra.jl`.
- The runtime polyhedral backend is local vendored `cdd/cddlog` only.
- There is no runtime fallback from `PolyBackend` to `NativePolyhedra`.
- `NativePolyhedra` remains in the repo as:
  - the in-memory polyhedron type
  - standalone geometry algorithms
  - an independently runnable test suite
- Float mode may use the local `cdd` fastpath.
- Exact mode does not use the float-style fastpath; exact elimination/projection go through local `cddlog`.
- `cddlog` currently assumes exact polyhedra with rational coefficients and `ExactLogExpr` right-hand sides.
- Business logic should call `PolyBackend` helpers, not backend-specific command runners.
- Backend selection belongs in `src/PolyBackend.jl`.
- H-representation serialization, command execution, and parsing belong in `src/CddBridge.jl`.

Current relevant files:

- `src/PolyBackend.jl`
- `src/CddBridge.jl`
- `src/NativePolyhedra/NativePolyhedra.jl`
- `deps/build.jl`
- `scripts/build_local_cdd.sh`
- `Artifacts.toml`

## Build And Tooling Notes

- `Pkg.build()` runs `deps/build.jl`.
- By default, `deps/build.jl` installs the source artifact pinned in `Artifacts.toml` and builds local tools from it.
- You can override the source tree with `BNC_CDDLOG_SOURCE_DIR=/path/to/cddlib-logarithmic`.
- Local vendored binaries live under `.build/cddlog/src`.
- The build now fails hard if required tools are missing. There is no silent downgrade.
- Required tools include at least:
  - `projection`
  - `redcheck`
  - `scdd`
  - `projection_log`
  - `redcheck_log`
  - `scdd_log`
- Force a rebuild with:

```bash
BNC_FORCE_REBUILD_CDD=1 julia --project=. --startup-file=no deps/build.jl
```

- The build script expects:
  - a C compiler: `gcc`, `cc`, or `clang`
  - GMP development headers and libraries for the logarithmic build
- On Debian/Ubuntu/WSL the usual install is:
  - `build-essential`
  - `libgmp-dev`
  - `pkg-config`
- If `pkg-config` cannot find GMP, the build script also probes common Julia/conda/artifact locations under `~/.julia`.

Useful environment toggles:

- `BNC_FORCE_REBUILD_CDD=1`
  - force rebuild even if `.build/cddlog/src` already has the required tools
- `BNC_CDDLOG_SOURCE_DIR`
  - override the source tree used by `deps/build.jl`
- `BNC_CDD_BINDIR`, `BNC_CDDLOG_BINDIR`
  - point runtime lookup at a custom binary directory
- `BNC_CDD_BUILD_DIR`, `BNC_CDDLOG_BUILD_DIR`
  - point runtime lookup at a build root whose binaries live in `src/`
- `BNC_DISABLE_LOCAL_CDD=1`
  - disable local backend discovery for tests
- `BNC_DISABLE_CDDLOG=1`
  - disable only the logarithmic backend for tests

Do not reintroduce eager warnings at `using BindingAndCatalysis` time.

## Shell And Julia Invocation Gotcha

This repo frequently uses Julia functions ending with `!`, for example `find_all_regimes!`.

When using `zsh`, `!` can trigger history expansion and corrupt command lines.

Safe patterns:

- use a here-doc
- or call the function via `getfield(Module, Symbol("find_all_regimes!"))`

Avoid raw inline shell commands containing unescaped Julia identifiers with `!`.

## Test Structure

Tests are intentionally split by domain:

- `test/binding/`
- `test/concurrency/`
- `test/mixed_regime/`
- `test/output/`
- `test/simo/`
- `test/backends/`
- `test/legacy/`
- `test/NativePolyhedra/`

Important current split:

- `test/runtests.jl` is the default package suite
- `test/NativePolyhedra/runtests.jl` is standalone and intentionally excluded from the default suite

Useful targeted runs:

```bash
julia --project=. --startup-file=no -e 'include("test/support/setup.jl"); include("test/backends/cdd_bridge.jl")'
```

```bash
julia --project=. --startup-file=no -e 'include("test/support/setup.jl"); include("test/mixed_regime/catalysis.jl")'
```

```bash
julia --project=. --startup-file=no -e 'include("test/support/setup.jl"); include("test/simo/workflows.jl")'
```

```bash
julia --project=. --startup-file=no test/NativePolyhedra/runtests.jl
```

```bash
julia --project=. --startup-file=no test/runtests.jl
```

Useful environment sanity checks:

```bash
julia --project=Examples --startup-file=no -e 'using BindingAndCatalysis'
```

Prefer focused validation first, then full `test/runtests.jl`.

## Common Debugging Patterns

### 1. Symbolic output prints decimals instead of exact expressions

Check:

- `src/output/symbolic_renderers.jl`
- `src/output/symbolic_paths.jl`
- `_exp10_factor`
- `ExactLogExpr` handling

If exact data reaches the renderer but output still shows `0.25`, the bug is usually in factor reconstruction or exact-expression rendering, not regime computation.

### 2. Graph drawing behaves oddly

Check:

- whether labels are rendered by `graphplot!` or post-rendered by helper text calls
- whether displayed node indices are true regime indices or temporary plotting indices
- whether the layout is built on the qK neighbor graph, x-space graph, or a SIMO path graph

For 3D Makie graphs, explicit post-render labels are usually safer than relying on `ilabels`.

### 3. Polyhedral backend errors or missing-tool failures

Check:

- `src/PolyBackend.jl`
- `src/CddBridge.jl`
- `.build/cddlog/src`
- `deps/build.jl`
- `scripts/build_local_cdd.sh`

Remember:

- runtime eliminate/intersect/project paths now require local tools
- exact-log cases require `cddlog`
- `cddlog` support is narrower than float `cdd` support
- missing tools should error, not warn and continue

If the failure is exact-only, inspect `_require_cddlog_support!` and the coefficient types reaching the bridge.

### 4. SIMO path polyhedra crash or hang

Check:

- `src/simo/polyhedra.jl`
- `backend_prefers_fastpath`
- `backend_intersect_eliminate`
- `backend_intersect_many`
- `backend_eliminate`
- `backend_from_fastpath`

Exact mode does not take the float bulk fastpath, so do not assume the exact path shares the same backend behavior as float SIMO.

### 5. Build appears fine but old `.log` / `.trs` files look wrong

These are often stale autotools artifacts from an older source layout. Check whether the actual runtime binaries in `.build/cddlog/src` are current before treating old logs as authoritative.

### 6. Notebook or Examples environment drifts from the package

Check:

- `Examples/Project.toml`
- `Examples/Manifest.toml`
- whether `Pkg.build()` was run in the active environment

The Examples environment should track the package without reintroducing old `Polyhedra` / `CDDLib` dependencies.

## Editing Guidance

- Keep backend selection in `PolyBackend`, not in callers.
- Keep H-representation conversion logic in `CddBridge`, not in domain modules.
- Keep project H-representation semantics explicit when converting to/from `NativePolyhedra`.
- Do not mutate a polyhedron as a side effect of serialization unless the change is explicitly intended.
- When changing `draw_graph`, test:
  - `Bnc` graph cases
  - `SIMOPaths` graph cases
  - 2D and 3D if touched
- When changing symbolic output, validate both:
  - exact binding-only cases
  - exact mixed/SIMO cases
- When changing build logic, validate real tool behavior, not just file existence.
  - In particular, exact-log paths should prove that `projection_log` or `scdd_log` can read `logarithmic` input.

## Good Minimal Validation Cases

Small exact binding case:

```julia
model = Bnc(N = [1 1 -1])
find_all_regimes!(model; mode = :exact)
```

Exact SIMO rational-printing case:

```julia
model = Bnc(N = [2 1 -1])
find_all_regimes!(model; mode = :exact)
simo = SIMOPaths(model, 1)
show_condition(simo, 1; log_space = false)
```

Small float SIMO case:

```julia
N = [1 1 0 -1 0 0;
     0 1 1 0 -1 0;
     1 0 1 0 0 -1]
model = Bnc(N = N)
simo = SIMOPaths(model, 1)
get_polyhedra(simo)
```

Backend-focused regression run:

```bash
julia --project=. --startup-file=no -e 'include("test/support/setup.jl"); include("test/backends/cdd_bridge.jl")'
```

## What Not To Do

- Do not reintroduce direct `CDDLib.jl` / `Polyhedra.jl` dependencies.
- Do not reintroduce runtime fallback from `PolyBackend` to `NativePolyhedra`.
- Do not bypass `PolyBackend` from business logic for eliminate/intersect/project operations.
- Do not assume `Archetecture.md` is fresher than source.
- Do not assume old autotools logs describe the current vendored source layout.
- Do not trust shell one-liners with Julia `!` names under `zsh`.
- Do not validate graph text placement only in 2D if you changed 3D behavior.

## Expected Deliverable Style

When finishing repo work, report:

- what changed
- which focused checks were run
- whether full `test/runtests.jl` was run
- whether `test/NativePolyhedra/runtests.jl` was run when relevant
- any remaining environment dependency, especially compiler/GMP requirements or local `cdd/cddlog` availability
