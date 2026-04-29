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
- These numbers came from `test/SISO_test/cdn4_path_condition_benchmark.jl` in single-thread mode with the recursive pair-memo solver.
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
- further algorithm work is paused for now; next focus is branch merge preparation
