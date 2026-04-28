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
- These numbers came from `test/cdn4_path_condition_benchmark.jl` in single-thread mode with the recursive pair-memo solver.
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
