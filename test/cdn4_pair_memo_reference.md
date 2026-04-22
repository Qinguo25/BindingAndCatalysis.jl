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
