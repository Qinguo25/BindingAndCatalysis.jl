---
name: bnc-julia-maintainer
description: Use when working inside BindingAndCatalysis.jl on Julia code, especially for regime enumeration, exact/float polyhedral backends, vendored local cdd builds, SISO workflows, symbolic rendering, tests, and Makie graph visualization. This skill is for repo-specific debugging and maintenance, not general Julia help.
---

# BindingAndCatalysis.jl Maintainer Skill

Use this skill when modifying this repository. It captures repo-specific workflows and failure modes that are easy to miss if you approach it like a generic Julia package.

## What To Read First

- Read `src/BindingAndCatalysis.jl` for module layout and exported surface.
- Read `Archetecture.md` for the current high-level structure.
- If the task touches polyhedra or projection, read:
  - `src/PolyBackend.jl`
  - `src/CddBridge.jl`
  - `src/NativePolyhedra/NativePolyhedra.jl`
- If the task touches graph drawing, read:
  - `src/visualization/graphs.jl`
  - `src/visualize.jl`
- If the task touches symbolic output, read:
  - `src/output/symbolic_api.jl`
  - `src/output/symbolic_renderers.jl`
- If the task touches SISO path conditions, read:
  - `src/siso/core.jl`
  - `src/siso/polyhedra.jl`

## Mental Model

There are three layers that matter most in maintenance work:

1. Domain logic
   - binding regimes, mixed regimes, SISO paths, symbolic output
2. Polyhedral facade
   - `src/PolyBackend.jl`
   - business code should call facade functions, not backend details
3. Polyhedral implementations
   - local vendored cdd via `src/CddBridge.jl`
   - native fallback via `src/NativePolyhedra`

Important current invariant:

- Project-level H-representation uses `C * x + C0 >= 0`
- `NativePolyhedra` stores halfspaces as `a * x <= β`
- Conversion is `a = -C`, `β = C0`

Do not mix these conventions.

## Backend Rules

- Float fast path should prefer local vendored `cdd` if available.
- Exact path should prefer local vendored `cddlog` if available.
- If local tools fail or are unavailable, the code should fall back to `NativePolyhedra`.
- Business logic should not directly import or call `CDDLib.jl` / `Polyhedra.jl`. This repo removed those dependencies.

Current relevant files:

- `src/CddBridge.jl`
- `src/PolyBackend.jl`
- `deps/build.jl`
- `scripts/build_local_cdd.sh`

## Build And Tooling Notes

- Local vendored cdd binaries live under `.build/cddlog/src`.
- `Pkg.build()` runs `deps/build.jl`.
- Build script expects:
  - a C compiler: `gcc`, `cc`, or `clang`
  - GMP development files for the logarithmic build
- On Debian/Ubuntu/WSL the expected install is:
  - `build-essential`
  - `libgmp-dev`
  - `pkg-config`

Runtime warnings already exist:

- missing local `cdd` triggers a once-only warning and falls back
- missing local `cddlog` triggers a once-only warning and falls back

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
- `test/mixed/`
- `test/siso/`
- `test/output/`
- `test/backends/`
- `test/NativePolyhedra/`

Useful targeted runs:

```bash
julia --project=. --startup-file=no -e 'using BindingAndCatalysis, Test; include("test/support/setup.jl"); include("test/backends/cdd_bridge.jl")'
```

```bash
julia --project=. --startup-file=no -e 'using BindingAndCatalysis, Test; include("test/support/setup.jl"); include("test/siso/workflows.jl")'
```

```bash
julia --project=. --startup-file=no test/runtests.jl
```

Prefer focused validation first, then full `test/runtests.jl`.

## Common Debugging Patterns

### 1. Symbolic output prints decimals instead of rationals

Check:

- `src/output/symbolic_renderers.jl`
- `_exp10_factor`
- `ExactLogExpr` handling

If exact data reaches the renderer but output still shows `0.25`, the bug is usually in factor reconstruction, not in regime computation.

### 2. Graph drawing behaves oddly

Check:

- whether labels are rendered by `graphplot!` or post-rendered by `add_nodes_text!`
- whether node indices shown are current display indices or true regime indices
- whether you are laying out on qK neighbor graph or x-space neighbor graph

For 3D Makie graphs, post-render labels are safer than relying on `ilabels`.

### 3. SISO path polyhedra crash or hang

Check:

- `src/siso/polyhedra.jl`
- `backend_prefers_fastpath`
- `backend_intersect_eliminate`
- `backend_eliminate`

If local cdd tools are unstable on a case, the fallback path must preserve correctness.

### 4. Build appears fine but old `.log` / `.trs` files look wrong

These are often stale autotools artifacts from an older source layout. Check whether runtime binaries in `.build/cddlog/src` are current before treating old logs as authoritative.

## Editing Guidance

- Keep backend selection in `PolyBackend`, not in callers.
- Keep project H-representation semantics explicit when converting to/from `NativePolyhedra`.
- When changing `draw_graph`, test:
  - `Bnc` graph
  - `SISOPaths` graph
  - 2D and 3D if touched
- When changing symbolic output, validate both:
  - exact binding-only case
  - exact mixed/SISO case

## Good Minimal Validation Cases

Small exact binding case:

```julia
model = Bnc(N = [1 1 -1])
find_all_regimes!(model; mode = :exact)
```

Exact SISO rational-printing case:

```julia
model = Bnc(N = [2 1 -1])
find_all_regimes!(model; mode = :exact)
siso = SISOPaths(model, 1)
show_condition(siso, 1; log_space = false)
```

Small float SISO case:

```julia
N = [1 1 0 -1 0 0;
     0 1 1 0 -1 0;
     1 0 1 0 0 -1]
model = Bnc(N = N)
siso = SISOPaths(model, 1)
get_polyhedra(siso)
```

## What Not To Do

- Do not reintroduce direct `CDDLib.jl` / `Polyhedra.jl` dependencies.
- Do not bypass `PolyBackend` from business logic.
- Do not assume old autotools logs describe the current vendored source layout.
- Do not trust shell one-liners with Julia `!` names under `zsh`.
- Do not validate graph text placement only in 2D if you changed 3D behavior.

## Expected Deliverable Style

When finishing repo work, report:

- what changed
- which focused checks were run
- whether full `test/runtests.jl` was run
- any remaining fallback or environment dependency, especially compiler/GMP requirements
