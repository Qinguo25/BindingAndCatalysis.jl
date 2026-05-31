# Suggestions Of Args

Date: 2026-05-31

This note reviews public and semi-public argument styles from the perspective of Julia package maintenance and user ergonomics.

## Overall Judgment

The current API is convenient for interactive exploration, but many public wrappers accept `args...; kwargs...` without documenting which keywords are consumed locally and which are forwarded. This is flexible, but it makes typo detection poor and makes future refactors risky.

Recommended rule:

- Keep `kwargs...` only at true forwarding boundaries, such as Makie plotting functions, ODE/nonlinear solver wrappers, and compatibility/deprecation wrappers.
- Prefer explicit keyword signatures for user-facing model/regime APIs.
- If forwarding is necessary, document the exact target, for example "forwarded to `ODE.solve`" or "forwarded to `get_bind_regimes` filter keywords".

## High Priority Changes

1. `get_regime`, `get_idx`, `get_perm`, `get_nullity`, `is_singular`, `is_asymptotic`

These are core lookup APIs and currently rely on broad forwarding in `RegimeCore.jl`.

Suggestion:

- Keep the ergonomic overloads, but avoid forwarding arbitrary kwargs through more than one layer.
- Public signatures should expose only `check::Bool=false` and regime-initialization switches that are actually supported.
- For model lookup, prefer:
  - `get_regime(model, idx; check=false, inv_info=true)`
  - `get_regime(model, perm; check=false, inv_info=true)`
  - `get_bnc_regime(model, bind, cat; check=false)`

2. `return_idx::Bool`

Used in `get_regimes`, `get_neighbors`, `assign_regime_*`, path APIs, and filtering APIs. It is common, but it changes the return type from regime/permutation to integer index.

Suggestion:

- Keep `return_idx` for now because it is already widespread.
- Add explicit aliases for common user workflows:
  - `get_regime_indices(...)`
  - `get_neighbor_indices(...)`
  - `assign_regime_idx(...)`
- Long term, prefer separate functions over return-type-changing booleans for exported APIs.

3. `summary_regime`

Now changed to `summary_regime(...; compute_volume=false)`.

Suggestion:

- Keep `compute_volume` explicit and default false.
- Only accept volume-related kwargs when `compute_volume=true`.
- Consider adding a future structured summary API:
  - `regime_summary(rgm; compute_volume=false) -> NamedTuple`
  - `summary_regime(...)` only prints that result.

4. Volume APIs

`calc_volume`, `get_volume`, `get_volumes`, and internal volume helpers use broad `kwargs...`.

Suggestion:

- Public signatures should list sampling controls explicitly: `method`, `n_samples`, `seed`, `recalculate`, `asymptotic`, `rebase_K`, `rebase_mat`.
- Internal helpers can keep `kwargs...` if they only forward to the selected backend.
- Avoid passing unrelated filter keywords into numerical volume routines.

5. Constructor `Bnc(; ..., kwargs...)`

The constructor forwards remaining keywords to `update_catalysis!`. This is convenient but surprising because binding-network construction and catalysis attachment are separate concepts.

Suggestion:

- Keep current behavior for compatibility.
- Document that remaining kwargs are catalysis kwargs.
- Consider adding `catalysis_kwargs=(;)` or a named constructor:
  - `Bnc(; N, L, ...)`
  - `with_catalysis(bnc; Γ, Π, ...)`

## Medium Priority Changes

1. Log-space flags

Keywords like `input_logspace`, `output_logspace`, `log_space`, and `output_logspace` are intuitive individually, but there are many combinations across mapping, assignment, trajectory, and symbolic rendering functions.

Suggestion:

- Keep `input_logspace` and `output_logspace` for numeric transformations.
- Keep `log_space` only for display/rendering.
- Avoid adding more variants such as `is_log`, `log_input`, or `use_log`.

2. Filter keywords

`singular`, `asymptotic`, `feasible`, and `asymptotic_only` are semantically close but not identical.

Suggestion:

- Use `singular` and `asymptotic` for filtering collections.
- Use `asymptotic_only` only for assignment/search algorithms.
- Document accepted values:
  - `singular`: `true`, `false`, integer threshold, or `nothing`.
  - `asymptotic`: `true`, `false`, or `nothing`.
  - `feasible`: `true`, `false`, or `nothing`.

3. `check::Bool`

`check=false` is idiomatic for fast lookup, but it should mean only bounds/existence validation.

Suggestion:

- Do not let `check=true` trigger expensive regime construction unless the function name makes that clear.
- Prefer `ensure_*` functions for explicit construction.

4. Plotting APIs

Makie-style `kwargs...` is idiomatic and should remain, because users expect to pass arbitrary recipe attributes.

Suggestion:

- Keep `kwargs...` in visualization functions.
- Put domain-specific keywords before `kwargs...`; pass the rest only to Makie calls.
- Now that visualization is an extension, error messages should name the needed optional packages.

## Low Priority Cleanup

- Rename local variables `vtx`, `vtxs`, and `vertices` to `rgm`, `rgms`, and `regimes` when touching files. Do this gradually to avoid noisy diffs.
- Keep deprecated `vertex` APIs in `old_api.jl` for now, but route new docs and examples through `regime` names only.
- Avoid adding exported functions with typo-compatible names. Keep typo wrappers deprecated and unexport them in the next breaking release.

## Suggested Migration Order

1. Stabilize core lookup signatures and document exact kwargs.
2. Add explicit index-return aliases before removing any `return_idx` usage.
3. Narrow volume API kwargs and keep solver/plot kwargs flexible.
4. Split constructor catalysis kwargs from core binding-network kwargs.
5. Remove old `vertex` aliases only in a planned breaking release.
