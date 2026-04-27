# Session Handoff: Pair-Memo / SISO Investigation

## Current branch

- `xiaoyu_pair_memo`


## Main goal of this session

We focused on the single-in single-out path-condition calculation in `src/SISO.jl`, with three main questions:

- compare the current `pair_memo` backend against the other algorithm on branch `recontrol`
- understand whether a DAG-style rewrite helps
- identify what actually dominates runtime and pruning, so a future parallel version targets the right work


## Files changed in this session

- [src/SISO.jl](/Users/wuxiaoyu/Documents/GitHub/Bnc_julia/src/SISO.jl)
- [test/runtests.jl](/Users/wuxiaoyu/Documents/GitHub/Bnc_julia/test/runtests.jl)
- [test/cdn4_path_condition_benchmark.jl](/Users/wuxiaoyu/Documents/GitHub/Bnc_julia/test/cdn4_path_condition_benchmark.jl)
- [test/cdn4_pair_memo_reference.md](/Users/wuxiaoyu/Documents/GitHub/Bnc_julia/test/cdn4_pair_memo_reference.md)
- [test/pair_memo_branch_analysis.jl](/Users/wuxiaoyu/Documents/GitHub/Bnc_julia/test/pair_memo_branch_analysis.jl)


## Architecture and codebase reading

Read [Archetecture.md](/Users/wuxiaoyu/Documents/GitHub/Bnc_julia/Archetecture.md) first.

Important takeaways:

- `Bnc` is the main root object.
- `SISOPaths` is the high-level wrapper for path/polyhedron work.
- The current SISO path-condition backend lives in `src/SISO.jl`.
- The hot recursive solver is `_find_pair_path_conditions!`.
- The branch logic inside `_find_pair_path_conditions!` is:
  - no-bridge early exit
  - middle-overlap branch:
    `n_solved_successors == 0 && n_solved_predecessors == 0`
  - suffix branch:
    `solved_successor_ratio > solved_predecessor_ratio`
  - otherwise prefix branch


## Branch comparison work

We compared the current branch against `recontrol` using the `CDN4` example.

### Benchmark script

- [test/cdn4_path_condition_benchmark.jl](/Users/wuxiaoyu/Documents/GitHub/Bnc_julia/test/cdn4_path_condition_benchmark.jl)

This script:

- builds the `CDN4` model
- runs `find_all_vertices!`
- builds `SISOPaths`
- runs `get_polyhedra`
- prints timing and backend statistics
- supports `condition_solver=:recursive` and `condition_solver=:dag`

### Saved recursive baseline

- [test/cdn4_pair_memo_reference.md](/Users/wuxiaoyu/Documents/GitHub/Bnc_julia/test/cdn4_pair_memo_reference.md)

Original saved single-thread recursive baseline on `xiaoyu_pair_memo`:

- `find_all_vertices_seconds = 4.232400958`
- `build_paths_seconds = 1.429451958`
- `get_polyhedra_seconds = 6.057668875`
- `cached_vertex_prisms = 51`
- `cached_interface_prisms = 495`
- `cached_pairs = 1172`
- `cached_path_condition_entries = 6836`

A later rerun gave slightly different timing but same overall picture:

- `find_all_vertices_seconds ≈ 4.53`
- `build_paths_seconds ≈ 1.93`
- `get_polyhedra_seconds ≈ 6.71`

### `recontrol` result

Earlier in the session, the `recontrol` branch was benchmarked on the same `CDN4` case while keeping `CDDLib`.

Observed result:

- `find_all_vertices_seconds ≈ 8.77`
- `build_paths_seconds ≈ 0.30`
- `get_polyhedra_seconds ≈ 218.08`
- `computed_node_polyhedra = 170`
- `computed_edge_polyhedra = 495`

Interpretation:

- with `CDDLib`, `recontrol` spends a lot of time materializing node/edge polyhedra up front
- on this workload, current `pair_memo` is much faster

Also noted:

- `recontrol` has a constructor bug around `SIMOPaths` using an undefined `paths` variable
- the benchmark script was written to fall back around that bug when possible


## `xiaoyu_suffix_DAG` reading

We also checked `xiaoyu_suffix_DAG`.

Conclusion:

- algorithmically useful as reference
- not a direct apples-to-apples comparison for the requested test, because it uses a different polyhedra backend rather than the same `CDDLib` path


## DAG experiment on current `pair_memo`

I added an experimental DAG-scheduled variant into [src/SISO.jl](/Users/wuxiaoyu/Documents/GitHub/Bnc_julia/src/SISO.jl).

### What was added

- `condition_solver` option to `SISOPaths`
- a new `:dag` backend path
- DAG helper logic intended to compute pair conditions in a scheduled order
- tests in [test/runtests.jl](/Users/wuxiaoyu/Documents/GitHub/Bnc_julia/test/runtests.jl) to compare recursive and DAG outputs on a smaller notebook model

### Important conclusion

The first DAG attempt was too eager.

Main issue:

- it expanded a much larger pair-dependency closure than the recursive memoized solver actually touches

Measured effect on `CDN4`:

- recursive cache footprint was about `1172` cached pairs in the original benchmark
- one DAG dependency-closure probe found `2752` required pairs from the source-sink roots

This explains why the DAG idea, implemented naively, did not help.

Current understanding:

- the correct pair-DAG idea should be:
  - top-down to discover only needed pairs
  - bottom-up to solve them
- but the top-down phase must be selective, not broad
- a good next step would be a branch-planned DAG:
  - decide which dependency branch each pair will use
  - only then schedule bottom-up evaluation


## Branch and pruning analysis

To understand what really matters for performance, I added:

- [test/pair_memo_branch_analysis.jl](/Users/wuxiaoyu/Documents/GitHub/Bnc_julia/test/pair_memo_branch_analysis.jl)

This script can:

- instrument the current recursive `pair_memo` solver on `CDN4`
- count branch usage
- count pruning events
- report inclusive time by branch family
- sweep `CDN2` to `CDN5` for longest SISO path length and total path count

### Useful commands

- `julia --project=. test/pair_memo_branch_analysis.jl branch`
- `julia --project=. test/pair_memo_branch_analysis.jl scaling`
- `julia --project=. test/cdn4_path_condition_benchmark.jl`


## Most important performance result

On `CDN4`, the expensive part of the recursive solver is the middle-overlap branch:

- `middle_overlap_pairs = 145`
- `suffix_pairs = 244`
- `prefix_pairs = 66`

By branch count, suffix happens more often.

By work and time, middle-overlap dominates:

- `middle_generated_paths = 4089`
- `suffix_generated_paths = 2305`
- `prefix_generated_paths = 101`
- `middle_overlap_seconds ≈ 6.16`
- `suffix_seconds ≈ 1.75`
- `prefix_seconds ≈ 0.037`

This means:

- the runtime bottleneck is the branch
  `n_solved_successors == 0 && n_solved_predecessors == 0`
- that is the middle bridge-through-`(successor, predecessor)` case


## Tree pruning result

This was the most useful finding for future parallelization.

Question investigated:

- does the solver prune mostly because candidate polyhedra fail geometric overlap tests?

Answer on `CDN4`:

- no

Measured pruning counters:

- `middle_empty_subproblem_skips = 430`
- `no_bridge_pairs = 340`
- `middle_intersection_empty_skips = 0`
- `suffix_intersection_empty_skips = 0`
- `prefix_intersection_empty_skips = 0`
- all interface-empty skip counters were also `0`
- suffix/prefix empty-subproblem skips were also `0`

Interpretation:

- the most important pruning is not the final geometric intersection test
- the most important pruning is:
  - the recursive middle subproblem is empty:
    `isempty(middle_conditions)`
- the next important prune is structural:
  - no available bridge successors/predecessors

Memoization also matters a lot:

- `cache_hits = 1348`

### Practical conclusion for parallelization

This is the key design insight from the session.

For `CDN4`, a good parallel attempt should:

- preserve selective top-down discovery
- preserve memoization, including caching of empty pair results
- target the expensive middle-overlap work
- avoid broad eager pair expansion
- avoid over-investing in earlier geometric overlap checks, because those were not the dominant pruning source here

In short:

- empty middle subproblems matter more than polyhedron non-overlap on this workload


## Path-length growth across CDN family

Using the complete-dimerization family generated from the `CDN4` pattern:

- `CDN2`: `n_regimes = 4`, `n_paths = 2`, longest path `2` edges / `3` vertices
- `CDN3`: `n_regimes = 25`, `n_paths = 36`, longest path `5` edges / `6` vertices
- `CDN4`: `n_regimes = 218`, `n_paths = 3936`, longest path `10` edges / `11` vertices
- `CDN5`: `n_regimes = 2451`, `n_paths = 6243216`, longest path `19` edges / `20` vertices

Important takeaway:

- longest path length grows fast
- total path count grows explosively


## Current repo state note

At the time this handoff was written, `git status` showed:

- modified:
  - `src/SISO.jl`
  - `test/runtests.jl`
- untracked:
  - `test/cdn4_pair_memo_reference.md`
  - `test/cdn4_path_condition_benchmark.jl`
  - `test/pair_memo_branch_analysis.jl`
- several deleted files under `test/` were also present in status and appear to be part of a separate cleanup

Be careful not to accidentally undo unrelated user cleanup work.


## Suggested next step

Best next investigation:

- break down `middle_empty_subproblem_skips = 430` by which `(successor, predecessor)` middle pairs are empty most often

Why:

- if a small set of empty middle-pair patterns accounts for much of the pruning, there may be a cheap pre-check or scheduling strategy that preserves recursive selectivity while making later parallelization cleaner

Alternative next step:

- redesign the DAG experiment so it first records a selective dependency plan for each pair, instead of expanding a broad dependency closure


## Follow-up investigation on empty middle pairs

I continued the suggested next step by extending [test/pair_memo_branch_analysis.jl](/Users/wuxiaoyu/Documents/GitHub/Bnc_julia/test/pair_memo_branch_analysis.jl) to record:

- how often each middle pair `(successor, predecessor)` is queried from the middle-overlap branch
- how often that middle pair returns empty vs nonempty
- which outer pairs `(from, to)` experience the most empty-middle skips

### Updated command

- `julia --project=. test/pair_memo_branch_analysis.jl branch`

### New CDN4 result

Relevant new summary fields from the rerun:

- `middle_empty_subproblem_skips = 430`
- `distinct_empty_middle_pairs = 248`
- `max_empty_hits_per_pair = 3`
- `pairs_with_single_empty_hit = 123`
- `pairs_with_multiple_empty_hits = 125`
- `top_10_empty_hit_share ≈ 0.0698`

Top empty middle pairs all had only `3` empty hits each, for example:

- `(2, 21)`
- `(5, 18)`
- `(7, 18)`
- `(7, 30)`
- `(10, 21)`

### Interpretation

This result is important for parallelization design:

- empty middle subproblems are real and valuable pruning
- but they are not concentrated in a tiny set of repeated middle-pair motifs
- the top 10 empty middle pairs explain only about `7%` of all empty-middle skips

So the likely conclusion is:

- a cheap lookup table for only a few “known empty” middle pairs probably will not buy much on `CDN4`
- preserving selective discovery still matters more than mining a tiny hot set of empty pairs

### Updated likely next step

The most promising next investigation now looks like:

- record the full middle-branch dependency graph for only the actually visited pairs
- then test whether those middle-branch child pairs can be scheduled in parallel without broadening discovery

In other words:

- not “precompute a small empty-pair blacklist”
- more likely “parallelize the expensive middle branch while keeping the recursive/top-down discovery policy”


## Good prompt to restart on another machine

When reopening Codex at home, a good starting prompt is:

`Read test/work_summary_and_suggestions.md and continue the pair_memo parallelization investigation.`
