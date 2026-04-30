# CDN4 Pair-Memo Reference

Saved on `xiaoyu_pair_memo` before adding the single-thread DAG-scheduled variant.

Model:
- `CDN4`
- `218` regimes
- `64` sources
- `1` sink
- `3936` source-to-sink paths

Reference timing:
- `find_all_vertices_seconds = 4.232400958`
- `build_paths_seconds = 1.429451958`
- `get_polyhedra_seconds = 6.057668875`

Reference backend cache summary:
- `cached_vertex_prisms = 51`
- `cached_interface_prisms = 495`
- `cached_pairs = 1172`
- `cached_path_condition_entries = 6836`

Notes:
- These numbers came from `test/SISO_test/benchmarks/cdn4_path_condition_benchmark.jl` in single-thread mode with the recursive pair-memo solver.
- Use this file as the baseline when comparing `condition_solver=:dag`.


## CDN5 Remote Pair-Memo DAG Reference

Saved from XiaoLab run:

- `/raid/users/xiaoyu/bnc_cdn5_20260427_114626`

Model:

- `CDN5`
- `2451` regimes
- `625` sources
- `1` sink
- `6243216` source-to-sink paths

Run setup:

- branch family: pair-memo DAG
- `condition_solver = "dag"`
- `JULIA_NUM_THREADS = 100`
- Julia `1.12.6`
- machine CPU allowance: `0-191`
- machine memory: about `502 GiB`

Completed timing:

- `find_all_vertices_seconds = 5.156616476`
- `build_paths_seconds = 3.851884115`
- `get_polyhedra_seconds = 19086.222969167`
- `elapsed_seconds = 19095.315429210663`

Backend cache summary:

- `cached_vertex_prisms = 452`
- `cached_interface_prisms = 7488`
- `cached_pairs = 32435`
- `cached_path_condition_entries = 10496673`

DAG summary:

- `dag_planned_pairs = 32435`
- `dag_pair_solve_calls = 32435`
- `dag_middle_join_pairs = 28868`
- `dag_middle_parallel_nodes = 962`
- `dag_middle_serial_nodes = 2187`
- `dag_pair_solve_seconds = 19080.445789332`
- `dag_middle_collect_seconds = 34.091755846`
- `dag_middle_compute_seconds = 39382.462410701`
- `dag_middle_merge_seconds = 6341.168057572`

CPU/memory monitor summary:

- monitor file: `artifacts/cdn5_julia_core_usage.tsv`
- monitor interval: 60 seconds
- samples: `321`
- average process CPU: `251.5%`, about `2.5` effective cores
- peak process CPU: `299%`, about `3.0` effective cores
- all samples were below `500%`
- max RSS: about `38.3 GiB`

Interpretation:

- CDN5 pair-memo DAG completes within memory
- the run is strongly CPU-underutilized despite 100 Julia threads
- the current algorithm shape is the bottleneck, not core availability


## CDN5 Remote Recontrol Reference

Saved from XiaoLab run:

- `/raid/users/xiaoyu/bnc_cdn5_recontrol_20260427_192359`

Run setup:

- branch: `recontrol`
- commit: `96e47d1e21ce701872fb43bba2e8d62b66b44bbb`
- `JULIA_NUM_THREADS = 100`
- `condition_solver = "recontrol_default"` because this branch does not accept `condition_solver = :dag`

Outcome:

- did not complete
- no `cdn5_result.json` was written
- last status was `stage = "solving_polyhedra"`
- stderr reached suffix-DAG path-polyhedra construction at layer `10/20`, about `50%`
- RSS rose to about `479 GiB` before the process disappeared

Interpretation:

- recontrol appears to expose much more CPU parallelism
- but for CDN5 it is likely memory-prohibitive on the XiaoLab machine


## CDN5 Remote Layer-Parallel Pair-Memo DAG Reference

Saved from XiaoLab run:

- `/raid/users/xiaoyu/bnc_cdn5_layer_parallel_20260428_184549`

Run setup:

- `condition_solver = "dag"`
- `BNC_SISO_DAG_LAYER_PARALLEL = true`
- `JULIA_NUM_THREADS = 50`
- Julia `1.12.6`
- machine CPU allowance: `0-191`
- machine memory: about `502 GiB`

Completed timing:

- `find_all_vertices_seconds = 5.553000724`
- `build_paths_seconds = 4.381636538`
- `get_polyhedra_seconds = 9438.820804268`
- `elapsed_seconds = 9469.500964164734`
- `finished_at = "2026-04-28T21:45:44.396"`

Backend cache summary:

- `cached_vertex_prisms = 452`
- `cached_interface_prisms = 7488`
- `cached_pairs = 32435`
- `cached_path_condition_entries = 10496673`
- `n_polyhedra = 6243216`

DAG summary:

- `dag_planned_pairs = 32435`
- `dag_pair_solve_calls = 32435`
- `dag_middle_join_pairs = 29530`
- `dag_middle_serial_nodes = 3149`
- `dag_middle_parallel_nodes = 0`
- `dag_pair_solve_seconds = 75591.480857416`
- `dag_middle_compute_seconds = 52146.752355458`
- `dag_middle_collect_seconds = 4.202313223`
- `dag_middle_merge_seconds = 0.0`

CPU/memory monitor summary:

- monitor file: `artifacts/cdn5_julia_core_usage.tsv`
- monitor interval: 60 seconds
- samples: `158`
- average process CPU: `1330.7%`, about `13.3` effective cores
- peak sampled process CPU: `2028%`, about `20.3` effective cores
- user-observed `top` peak: about `4900%`, about `49` effective cores
- max RSS: about `36.3 GiB`

Comparison to previous pair-memo DAG CDN5 run:

- previous `get_polyhedra_seconds = 19086.222969167`
- layer-parallel `get_polyhedra_seconds = 9438.820804268`
- speedup: about `2.02x`

Tail note:

- `dag_largest_pair = [1251, 155]`
- `dag_largest_pair_seconds = 3129.701002878`
- final current pair `[2447, 155]` took about `3109.117938709s`

Interpretation:

- layer-parallel pair solving substantially improves CDN5 runtime while keeping memory under control
- it can reach near-50-core utilization in bursts
- sustained CPU remains much lower because the late heavy middle-pair tail is still narrow


## CDN5 Remote Pair+Chunk Queue Reference

Saved from XiaoLab run:

- `/raid/users/xiaoyu/bnc_cdn5_chunk_queue_20260429_193504`

Run setup:

- `condition_solver = "dag"`
- `BNC_SISO_DAG_SCHEDULER = "queue"` / queue scheduler enabled
- `BNC_SISO_DAG_CHUNK_QUEUE = true`
- `BNC_SISO_DAG_INNER_PARALLEL_MIN_WEIGHT = 50000`
- `BNC_SISO_DAG_INNER_PARALLEL_TARGET_ENTRIES = 5000`
- `JULIA_NUM_THREADS = 50`
- Julia `1.12.6`
- machine CPU allowance: `0-191`
- machine memory: about `502 GiB`

Completed timing:

- `find_all_vertices_seconds = 6.627733651`
- `build_paths_seconds = 4.710966198`
- `get_polyhedra_seconds = 2689.327333405`
- `elapsed_seconds = 2714.28760099411`
- `finished_at = "2026-04-29T20:22:27.291"`

Backend cache summary:

- `cached_vertex_prisms = 452`
- `cached_interface_prisms = 7488`
- `cached_pairs = 32435`
- `cached_path_condition_entries = 10496673`
- `n_polyhedra = 6243216`

DAG / queue summary:

- `dag_planned_pairs = 32435`
- `dag_pair_solve_calls = 32435`
- `dag_middle_join_pairs = 29530`
- `dag_middle_serial_nodes = 3121`
- `dag_middle_parallel_nodes = 28`
- `dag_pair_solve_seconds = 83235.935102122`
- `dag_middle_compute_seconds = 82135.632922283`
- `dag_middle_collect_seconds = 5.187464642`
- `dag_middle_merge_seconds = 0.0`

Tail note:

- `dag_largest_pair = [1009, 155]`
- `dag_largest_pair_seconds = 842.551622233`
- final current pair `[2447, 155]` took about `645.809801625s`
- final current pair output entries: `413334`

CPU/memory monitor summary:

- monitor file: `artifacts/cdn5_julia_core_usage.tsv`
- monitor interval: 60 seconds
- sampled process CPU near the tail: about `4091%` to `4470%`
- live `top` observation near heavy chunked work: about `4900%`
- monitor values are once-per-minute samples and under-report short peaks
- max sampled RSS near tail: about `37.7 GiB`

Comparison:

- original pair-memo DAG `get_polyhedra_seconds = 19086.222969167`
- layer-parallel `get_polyhedra_seconds = 9438.820804268`
- conservative layer-inner test `get_polyhedra_seconds = 10427.390524772`
- pair+chunk queue `get_polyhedra_seconds = 2689.327333405`
- speedup vs layer-parallel: about `3.5x`
- speedup vs original pair-memo DAG: about `7.1x`

Interpretation:

- global pair queue plus weighted middle-join chunk tasks is the best current CDN5 scheduler
- the bad tail is still visible, but the worst pair drops from about `3100s` to about `843s`
- CPU utilization during tail improves to roughly the 40-50 core range
- threaded `auto` scheduling should use the pair+chunk queue by default; serial scheduling remains useful for deterministic debugging and single-thread runs


## CDN5 Remote Pair+Chunk Queue Monitor Reference

Saved from XiaoLab run:

- `/raid/users/xiaoyu/bnc_cdn5_chunk_queue_monitor5s_20260429_205904`

Completed timing:

- `get_polyhedra_seconds = 2703.630988755`
- `elapsed_seconds = 2719.921362876892`
- `finished_at = "2026-04-29T21:48:22.593"`
- `n_polyhedra = 6243216`
- `cached_pairs = 32435`
- `cached_path_condition_entries = 10496673`

DAG / queue summary:

- `dag_middle_parallel_nodes = 28`
- `dag_largest_pair = [1596, 155]`
- `dag_largest_pair_seconds = 851.888723548`
- final current pair `[2447, 155]` took about `709.108619793s`
- final current pair output entries: `413334`

Enhanced CPU monitor summary:

- monitor file: `artifacts/cdn5_julia_cpu_5s.tsv`
- monitor interval: about 5 seconds
- interval summed thread CPU samples: `381`
- average interval summed thread CPU: about `4223%`
- maximum interval summed thread CPU: about `5052%`
- samples above `4000%`: `302`
- samples above `4500%`: `299`
- samples above `4800%`: `198`

Estimator note:

- Previous fixed chunk target: `BNC_SISO_DAG_INNER_PARALLEL_TARGET_ENTRIES = 5000`
- Observed task-timer throughput: about `10,609,440 / 84,678.71 = 125` estimated entries/sec
- The adaptive estimator therefore uses `BNC_SISO_DAG_TARGET_CHUNK_SECONDS = 40` by default, with a fallback rate of `125` entries/sec.
- Initial target entries are `125 * 40 = 5000`, so the estimator reproduces the successful CDN5 chunk size before it has runtime samples.
