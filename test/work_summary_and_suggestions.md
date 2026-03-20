# Work Summary And Suggestions

## What this package is doing

This package has three closely related layers:

1. `BindRegime`
   A dominance regime for the binding map `(L, N)`, with
   - dominant monomial map `log q = P log x + P0`
   - regime inequalities in `x` and transported inequalities in `(q, K)`
   - inverse-like map `H` / `H0` when the regime is regular.

2. `CatalysisRegime`
   A dominance regime for the reduced catalysis selector `S_pos_neg`, with
   - steady-state selector `Pθ`
   - dominance selector `Cθ`
   - lifted matrices `PΠ`, `CΠ`
   - conditions naturally written in `(x, k)`.

3. `BncRegime`
   A mixed regime built from one binding regime and one catalysis regime, with
   - `H_bd = Pθ Π H[:, 1:r_v]` for stability screening
   - `(q, K, k)` consistency conditions
   - `(q_ss, K, k)` steady-state consistency conditions
   - the reduced map from `(q_ss, K, k)` to `x`
   - when regular, an explicit affine expression for `q_cat`.

## What was added in this round

- Completed access APIs for `CatalysisRegime` and `BncRegime`.
- Added pair-based mixed-regime retrieval from binding perm + catalysis perm.
- Added symbol helpers for `k`, `q_cat`, `w`, `q_para`, `q_ss`.
- Added symbolic rendering for
  - full catalysis dynamics
  - reduced catalysis dynamics
  - catalysis conditions in `(x, k)`
  - mixed conditions in `(x, k)`, `(q, K, k)`, `(q_ss, K, k)`
  - regular `q_cat` expressions in `(q_ss, K, k)`.
- Added `is_stable` / `judge_stability!` for `BncRegime`.
- Moved mixed-regime stability judgment to on-demand evaluation instead of computing it during construction.
- Fixed cache invalidation for `BncRegimes`.
- Fixed the existing `qK_x_mapping.jl` regression caused by the removed `_LN_sparse` field.
- Added tests for catalytic and mixed-regime APIs.

## Main design choices

- Kept the existing binding-regime API style.
- Avoided a big data-structure rewrite; the mixed regime table is still a matrix of `Union{BncRegime,Nothing}`.
- Used explicit names when the base variables differ:
  - `show_condition_xk`
  - `show_condition_qKk`
  - `show_condition_qssKk`
  - `show_expression_qcat`
- Kept backward-flavored aliases in `src/old_api.jl`.

## Suggestions

1. Separate “dominance condition” from “steady-state equation” a little more explicitly in the public API.

Right now the catalysis layer still has both concepts, but some getters are dominance-only (`get_C_xk`) while some renderers show the full mixed set. This is mathematically fine, but clearer names like `get_dominance_*` and `get_steady_state_*` would make notebooks easier to read.

2. Consider giving `BncRegimes` its own small container type.

At the moment it is a matrix, which is simple and fast, but a wrapper with
`bind_perm_dict`, `cat_perm_dict`, and `data` would make introspection cleaner and would mirror `Regimes`.

3. Consider caching “catalysis-only in `(q, K, k)`” conditions.

Those are currently derived on demand. That keeps the stored object simple, but if you inspect many mixed regimes interactively, caching that block may be worthwhile.

4. Consider documenting the three bases side-by-side in the README.

- `(x, k)` for regime selection / catalytic dominance
- `(q, K, k)` for mixed consistency before eliminating `q_cat`
- `(q_ss, K, k)` for steady-state reduced consistency

That distinction is the most important conceptual bridge in this package.

5. `d_w` was easy to misread.

I fixed one bug where it was accidentally taken from the wrong dimension of `L_Γ`. That suggests this field is subtle enough that a short constructor comment would help future maintenance.
