# Session Handoff: Pair-Memo / SISO Investigation

## Current branch

- `xiaoyu_parallel_pair_memo`


## Main goal of this session

We focused on the single-in single-out path-condition calculation in `src/SISO.jl`, with three main questions:

- compare the current `pair_memo` backend against the other algorithm on branch `recontrol`
- understand whether a DAG-style rewrite helps
- identify what actually dominates runtime and pruning, so a future parallel version targets the right work


## Files changed in this session

- [src/SISO.jl](/Users/wuxiaoyu/Documents/GitHub/Bnc_julia/src/SISO.jl)
- [test/runtests.jl](/Users/wuxiaoyu/Documents/GitHub/Bnc_julia/test/runtests.jl)
- [test/cdn4_path_condition_benchmark.jl](/Users/wuxiaoyu/Documents/GitHub/Bnc_julia/test/cdn4_path_condition_benchmark.jl)
- [test/cdn_overnight_benchmark.jl](/Users/wuxiaoyu/Documents/GitHub/Bnc_julia/test/cdn_overnight_benchmark.jl)
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

At the time this handoff was originally written, `git status` showed:

- modified:
  - `src/SISO.jl`
  - `test/runtests.jl`
- untracked:
  - `test/cdn4_pair_memo_reference.md`
  - `test/cdn4_path_condition_benchmark.jl`
  - `test/pair_memo_branch_analysis.jl`
- several deleted files under `test/` were also present in status and appear to be part of a separate cleanup

Be careful not to accidentally undo unrelated user cleanup work.

Current later-session note:

- the active branch is now `xiaoyu_parallel_pair_memo`
- the current DAG backend has weighted progress diagnostics added
- benchmark artifacts under `test/` and remote run artifacts on XiaoLab should be treated as data, not code to casually clean up


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


## 2026-04-28 CDN5 remote benchmark update

Remote machine:

- host alias: `XiaoLab`
- CPU allowance observed: `0-191`
- memory observed: about `502 GiB`
- Julia: `1.12.6`

### Pair-memo DAG CDN5 run

Run directory:

- `/raid/users/xiaoyu/bnc_cdn5_20260427_114626`

This run completed successfully.

Important result fields:

- `cdn = 5`
- `condition_solver = "dag"`
- `julia_threads = 100`
- `n_regimes = 2451`
- `n_sources = 625`
- `n_sinks = 1`
- `n_paths = 6243216`
- `n_polyhedra = 6243216`
- `elapsed_seconds = 19095.315429210663`
- `get_polyhedra_seconds = 19086.222969167`
- `cached_pairs = 32435`
- `cached_path_condition_entries = 10496673`
- `cached_vertex_prisms = 452`
- `cached_interface_prisms = 7488`
- `dag_planned_pairs = 32435`
- `dag_pair_solve_calls = 32435`
- `dag_middle_join_pairs = 28868`
- `dag_middle_parallel_nodes = 962`
- `dag_middle_serial_nodes = 2187`
- `dag_middle_collect_seconds = 34.091755846`
- `dag_middle_compute_seconds = 39382.462410701`
- `dag_middle_merge_seconds = 6341.168057572`
- `dag_pair_solve_seconds = 19080.445789332`

CPU/core-use result from `artifacts/cdn5_julia_core_usage.tsv`:

- monitor interval: 60 seconds
- samples: `321`
- average process CPU: `251.5%`, about `2.5` cores
- peak process CPU: `299%`, about `3.0` cores
- all samples were below `500%`, so this run never used 5 effective cores in a sampled minute
- average active threads above 1% CPU: `52.5`
- maximum active threads above 1% CPU: `127`
- active threads above 10% CPU: about `1` throughout
- maximum RSS: about `38.3 GiB`

Interpretation:

- the current pair-memo DAG code is memory-light enough for CDN5
- it is not using the available parallel hardware well
- simply giving the same implementation 50 or 100 cores will not materially speed it up
- the observed long run behaves like a roughly 3-core job

### Recontrol CDN5 run

Run directory:

- `/raid/users/xiaoyu/bnc_cdn5_recontrol_20260427_192359`

This run did not complete.

Notes:

- `recontrol` does not accept `condition_solver = :dag`, so the branch default algorithm was used and recorded as `condition_solver = "recontrol_default"`
- a temporary remote-copy fix was needed in `src/simo/core.jl`: `paths` -> `rgm_paths`
- the benchmark script also needed to read `sources` and `sinks` fields directly because this branch does not provide `get_sources(::SIMOPaths)` / `get_sinks(::SIMOPaths)`

Last observed state:

- `stage = "solving_polyhedra"`
- `n_paths = 6243216`
- stderr reached suffix-DAG path-polyhedra construction at layer `10/20`, about `50%`
- no `cdn5_result.json` was written
- process disappeared after RSS climbed to about `479 GiB`, around `90.8%` of machine memory

Interpretation:

- recontrol exposed much more CPU parallelism during suffix-DAG construction
- but it appears to be memory-prohibitive for CDN5 on this machine
- likely failure mode is memory pressure / OOM, although kernel OOM logs were not accessible/visible

### Weighted progress implementation

The pair-memo DAG backend now uses a weighted progress bar rather than raw pair-count progress.

Key idea:

- raw pair count is misleading because pair work is heavy-tailed
- each scheduled pair gets an adaptive weight
- diagonal and no-bridge pairs have weight `1`
- other pairs are weighted by dependency count plus already-cached child condition entries
- the progress bar is updated once per solved pair, not inside inner loops

Diagnostics added to the DAG profile and benchmark JSON:

- `dag_weighted_work_done`
- `dag_weighted_work_total`
- `dag_weighted_progress`
- `dag_weighted_progress_units`
- `dag_largest_pair_seconds`
- `dag_largest_pair`
- `dag_current_pair`
- `dag_current_pair_branch`
- `dag_current_pair_weight`
- `dag_current_pair_elapsed_seconds`
- `dag_current_pair_running`
- `dag_current_pair_output_entries`

This should make future long runs easier to monitor, but it does not by itself improve throughput.


## Critical rethink: should we implement a new parallel scheduler?

Initial tempting proposal:

- build a dependency-aware task queue of pair plans
- run all ready pairs across worker tasks
- commit results into the memo cache as dependencies finish

This is directionally attractive because the CDN5 run used only about 3 effective cores. But it needs skepticism before implementation.

### Why a fully dynamic pair task queue may be risky

- The current selective DAG plan is already ordered by dependencies, but not necessarily designed for concurrent mutation of `helper`.
- `SISOHelper` caches are shared mutable structures:
  - `pair_conditions`
  - `vertex_prisms`
  - `interface_prisms`
- Concurrent writes into these caches would require careful locking or per-worker staging.
- Locks around polyhedron/cache operations could erase the expected speedup.
- Pair costs are heavy-tailed, so static level scheduling may leave many workers idle near the tail.
- Dynamic scheduling could improve load balance but complicates reproducibility and correctness.
- Some expensive pair work may call CDDLib/polyhedron routines whose thread behavior or allocation behavior may not scale cleanly.
- The existing CDN5 result says the current code is CPU-underutilized, but it does not yet prove which part is serial:
  - pair scheduling
  - cache lock/lookup shape
  - polyhedron operations
  - middle merge
  - allocation/GC
  - C library calls

### Better implementation sequence

Do not jump straight to a large concurrent pair-memo cache rewrite.

Recommended sequence:

1. **Profile first with sampled stacks on CDN4 and a short CDN5 window.**

   Goal:

   - identify where the 19086 seconds would go if sampled
   - separate Julia time, CDDLib time, GC, cache lookup, middle compute, and merge

2. **Measure dependency-layer width and weighted work per layer.**

   Goal:

   - answer whether a simple layer-parallel scheduler has enough ready work
   - estimate tail width before adding concurrency

3. **Prototype layer-parallel execution, not a full dynamic queue.**

   Safer first step:

   - keep the existing selective plan
   - solve only one dependency layer at a time
   - inside each layer, run independent pairs in parallel
   - stage each pair result locally
   - commit layer results serially or with minimal locking

   Why this is attractive:

   - correctness is easier because dependencies only point to earlier layers
   - cache writes can be batched
   - it preserves selective discovery
   - it gives a real scaling test without the complexity of a full scheduler

4. **Only then consider a dynamic ready queue.**

   Use it if:

   - layer width is high but pair costs are too skewed
   - layer-parallel execution shows good speedup early but stalls badly at the tail

5. **Split very heavy middle pairs only after profiling.**

   This may be necessary for 50-core scaling, but it is the most delicate change.

   The split should avoid shared writes inside inner loops:

   - each chunk computes local condition candidates
   - merge/deduplicate after chunk completion
   - update the cache once per pair

### Current recommendation

Implementing some parallel structure is probably worthwhile, because the successful CDN5 run is clearly underusing cores.

But the first implementation should be conservative:

- add diagnostics for planned-pair dependency layers and per-layer work estimates
- then implement layer-parallel pair solving with staged results
- defer a fully dynamic work queue until measurements show layer scheduling is insufficient

Expected upside:

- if layers are wide, this could convert part of the calculation from about 3 effective cores toward the 50-core budget

Main risk:

- if the long tail is dominated by a very small number of huge middle pairs, layer parallelism alone will not solve the tail
- in that case, chunking heavy middle joins becomes the next target


## 2026-04-28 parallel-structure experiments

Added:

- [test/dag_parallel_diagnostics.jl](/Users/wuxiaoyu/Documents/GitHub/Bnc_julia/test/dag_parallel_diagnostics.jl)

Purpose:

- inspect the selective pair-DAG dependency layers without changing solver behavior
- optionally solve planned pairs and record per-pair timings
- estimate the upper bound for layer-parallel scheduling before implementing it

Useful commands:

- `julia --project=. test/dag_parallel_diagnostics.jl`
- `BNC_DIAG_SOLVE=true julia --project=. test/dag_parallel_diagnostics.jl`
- `BNC_DIAG_CDN_N=5 julia --project=. test/dag_parallel_diagnostics.jl`

### CDN4 diagnostic results

Planning-only:

- `planned_pairs = 1172`
- `n_layers = 9`
- layer pair counts: `645, 189, 131, 93, 57, 37, 14, 5, 1`
- widest layer is layer 1, but it is mostly trivial diagonal/no-bridge work
- final layer has only one middle pair

Measured pair-solve diagnostic:

- measured sequential pair-solve time: about `6.20s`
- ideal layer-parallel time with infinite workers: about `2.16s`
- ideal layer-parallel time with 50 workers: about `2.16s`
- ideal layer-parallel time with 8 workers: about `2.22s`

Interpretation:

- for CDN4, layer parallelism has a hard upper bound of only about `2.8x` to `2.9x`
- adding more than 8 workers would not help this case much
- the final layers are narrow enough that barriers matter

Existing middle-join threading benchmark on CDN4:

- `JULIA_NUM_THREADS=1 BNC_CDN4_SOLVER=dag`: `get_polyhedra_seconds ≈ 6.15`
- `JULIA_NUM_THREADS=8 BNC_CDN4_SOLVER=dag`: `get_polyhedra_seconds ≈ 4.33`
- this is only about `1.4x`
- inner middle-join threading helps, but it is not enough

### CDN5 planning diagnostic results

Planning-only:

- `planned_pairs = 32435`
- `n_layers = 18`
- layer pair counts:
  - `11321, 3543, 3676, 3561, 2835, 2148, 1607, 1123, 867, 614, 407, 310, 213, 113, 57, 28, 10, 2`
- branch counts:
  - `no_bridge = 10869`
  - `diagonal = 452`
  - `middle = 3149`
  - `suffix = 10481`
  - `prefix = 7484`

Static dependency-weight proxy:

- `total_static_weight = 115470`
- ideal layer schedule with 50 workers: `2655`
- ideal layer schedule with 8 workers: `14502`
- ideal layer schedule with infinite workers: `1402`
- perfect global 50-worker lower bound would be roughly `115470 / 50 = 2309`

Interpretation:

- CDN5 has much more exploitable layer width than CDN4
- by static weights, 50-worker layer scheduling is only about `15%` worse than a perfect global 50-worker queue
- this weakens the case for implementing a complex dynamic ready queue first
- the final layers are narrow, but their static weights are not large enough to dominate the static proxy

### Updated recommendation after testing

The best next implementation to test is:

- **layer-parallel pair solving with staged results**

Why:

- it preserves the selective DAG plan
- it avoids the complexity of a dynamic ready queue
- CDN5 static layer diagnostics suggest it could expose most of the useful 50-core parallelism
- CDN4 measured timings show not to expect miracles on small cases, which is fine

What not to implement first:

- a fully dynamic ready queue

Reason:

- by the CDN5 static proxy, dynamic scheduling may only improve about `15%` over layer scheduling
- it would require more concurrency machinery and more ways to get cache correctness wrong

Critical implementation constraints:

- precompute or safely serialize vertex/interface prism cache writes before solving a layer
- let each pair own its output map locally
- commit pair-condition results to `helper.pair_conditions` after the layer finishes, preferably serially at first
- avoid shared `Dict` writes inside threaded loops
- keep the existing middle-join chunking inside each pair only if it does not oversubscribe threads badly

Likely staged implementation plan:

1. Add layer construction from the existing planned pair DAG.
2. Add a serial layer solver that should reproduce the current DAG result exactly.
3. Add threaded layer solving where each pair returns `(pair, conditions)` into a per-layer result vector.
4. Commit the result vector to `helper.pair_conditions` after the threaded region.
5. Benchmark CDN4 with 1, 8, and maybe 16 threads.
6. If CDN4 is correct, run CDN5 on XiaoLab with 50 threads and monitor CPU/RSS.


## 2026-04-28 layer-parallel prototype

Implemented a default-off prototype in [src/SISO.jl](/Users/wuxiaoyu/Documents/GitHub/Bnc_julia/src/SISO.jl).

Enable with:

- `BNC_SISO_DAG_LAYER_PARALLEL=true`

Current behavior:

- only activates for `condition_solver = :dag`
- only activates when `Threads.nthreads() > 1`
- preserves the existing selective pair-DAG plan
- groups planned pairs into dependency layers
- prewarms vertex/interface prism caches serially before each layer
- computes each pair into a local `SISOPathConditionMap`
- commits layer results into `helper.pair_conditions` after threaded computation
- disables inner middle-join threading while layer-parallel execution is active to avoid nested oversubscription

The normal DAG solver path remains the default when the environment flag is absent.

### Local CDN4 prototype benchmarks

Baseline existing DAG path:

- `JULIA_NUM_THREADS=1 BNC_CDN4_SOLVER=dag`: `get_polyhedra_seconds ≈ 6.15`
- `JULIA_NUM_THREADS=8 BNC_CDN4_SOLVER=dag`: `get_polyhedra_seconds ≈ 4.33`

Layer-parallel prototype:

- `JULIA_NUM_THREADS=8 BNC_SISO_DAG_LAYER_PARALLEL=true BNC_CDN4_SOLVER=dag`: `get_polyhedra_seconds ≈ 3.71`
- `JULIA_NUM_THREADS=16 BNC_SISO_DAG_LAYER_PARALLEL=true BNC_CDN4_SOLVER=dag`: `get_polyhedra_seconds ≈ 3.47`

Correctness smoke:

- CDN4 result still had `n_polyhedra = 3936`
- `cached_pairs = 1172`
- `cached_path_condition_entries = 6836`
- full local `test/runtests.jl` passed after the prototype

Interpretation:

- layer-parallel execution is a real improvement on CDN4, but CDN4 is too small/narrow to show 50-core-style scaling
- 16 threads only modestly improve over 8 threads, consistent with the earlier dependency-layer upper-bound analysis
- the result is good enough to justify a CDN5 remote test

### Remote CDN5 layer-parallel result

Run directory:

- `/raid/users/xiaoyu/bnc_cdn5_layer_parallel_20260428_184549`

Run setup:

- `JULIA_NUM_THREADS = 50`
- `BNC_SISO_DAG_LAYER_PARALLEL = true`
- `BNC_CDN_N = 5`
- `BNC_CDN_SOLVER = dag`

Startup note:

- the first launch failed before benchmark start because the copied remote environment was not fully instantiated
- error: `failed to find source of parent package: "RecursiveArrayTools"`
- fixed by running `Pkg.instantiate(); Pkg.precompile(); using BindingAndCatalysis`
- failed logs were moved under `artifacts/failed_uninstantiated_env_attempt/`

Completed result:

- `stage = "completed"`
- `finished_at = "2026-04-28T21:45:44.396"`
- `n_regimes = 2451`
- `n_sources = 625`
- `n_sinks = 1`
- `n_paths = 6243216`
- `n_polyhedra = 6243216`
- `elapsed_seconds = 9469.500964164734`
- `get_polyhedra_seconds = 9438.820804268`
- `cached_pairs = 32435`
- `cached_path_condition_entries = 10496673`
- `cached_vertex_prisms = 452`
- `cached_interface_prisms = 7488`
- `dag_planned_pairs = 32435`
- `dag_pair_solve_calls = 32435`
- `dag_middle_join_pairs = 29530`
- `dag_middle_serial_nodes = 3149`
- `dag_middle_parallel_nodes = 0`
- `dag_pair_solve_seconds = 75591.480857416`
- `dag_middle_compute_seconds = 52146.752355458`
- `dag_middle_collect_seconds = 4.202313223`
- `dag_middle_merge_seconds = 0.0`

Comparison to previous pair-memo DAG CDN5 run:

- previous `get_polyhedra_seconds = 19086.222969167`
- layer-parallel `get_polyhedra_seconds = 9438.820804268`
- speedup is about `2.02x`
- previous total elapsed was about `19095s`; layer-parallel total elapsed was about `9469s`

CPU/memory monitor summary:

- monitor file: `artifacts/cdn5_julia_core_usage.tsv`
- 60-second monitor samples: `158`
- average process CPU from monitor: `1330.7%`, about `13.3` effective cores
- peak process CPU from monitor: `2028%`, about `20.3` effective cores
- user observed via `top` that instantaneous/short-window peak CPU reached about `4900%`, about `49` effective cores
- average active threads above 1% CPU: `55.0`
- maximum active threads above 1% CPU: `100`
- average active threads above 10% CPU: `45.6`
- maximum active threads above 10% CPU: `51`
- maximum RSS: about `36.3 GiB`

Interpretation:

- the layer-parallel scheduler can expose near-50-core parallelism in bursts
- the sustained average is much lower because late layers have a small number of very heavy pairs
- memory remained healthy and similar to the previous pair-memo DAG run
- the next algorithmic bottleneck is the heavy middle-pair tail, not layer-level worker availability

Tail evidence:

- `dag_largest_pair = [1251, 155]`
- `dag_largest_pair_seconds = 3129.701002878`
- final current pair `[2447, 155]` took about `3109.117938709s`
- final heavy middle pairs produced hundreds of thousands of condition entries each

Decision:

- pause further algorithm improvement for now
- next work focus should be preparing this branch for merge into `main`
- before merging, decide whether the layer-parallel prototype should stay default-off behind `BNC_SISO_DAG_LAYER_PARALLEL=true` or be promoted to a documented option
