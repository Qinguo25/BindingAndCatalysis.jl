# SISO Test Assets

This folder holds SISO-specific scripts and notes that are too specialized or too slow for the package-level `test/runtests.jl` entry point.

- `benchmarks/`: runnable timing scripts for local scheduler and path-condition experiments.
- `benchmarks/results/`: generated benchmark outputs. This directory is ignored by git.
- `diagnostics/`: exploratory scripts that inspect planning, pair memoization, and DAG scheduling structure.
- `long_runs/`: launchers and benchmark harnesses intended for remote or overnight runs.
- `references/`: small curated reference notes and benchmark summaries worth keeping in version control.
- `docs/`: working notes and handoff summaries.

Use `test/runtests.jl` for correctness/CI. Use scripts here for focused SISO performance investigation.

## DAG Scheduler Notes

Threaded DAG path-condition solving uses the pair+chunk queue scheduler by default. Set `BNC_SISO_DAG_SCHEDULER=serial` to force the serial DAG scheduler, or `BNC_SISO_DAG_SCHEDULER=queue` to request the threaded queue scheduler explicitly.

The queue scheduler can split large middle-join pairs into weighted chunk tasks. The main tuning knob is `BNC_SISO_DAG_TARGET_CHUNK_SECONDS`, which defaults to `40` and is converted into a target entry count using the observed chunk throughput. The default fallback throughput is `125` entries/sec, so the initial target is about `5000` entries/chunk, matching the CDN5 benchmark that saturated roughly 50 cores.

Useful tuning knobs:

- `BNC_SISO_DAG_CHUNK_QUEUE=true|false`
- `BNC_SISO_DAG_TARGET_CHUNK_SECONDS`
- `BNC_SISO_DAG_CHUNK_SIZE_GATE=true|false`
- `BNC_SISO_DAG_CHUNK_WIDTH_GATE=true|false`
- `BNC_SISO_DAG_CHUNK_THREAD_GATE=true|false`
- `BNC_SISO_DAG_INNER_PARALLEL_MIN_WEIGHT`
- `BNC_SISO_DAG_INNER_PARALLEL_MAX_CHUNKS`
- `BNC_SISO_DAG_INNER_PARALLEL_TARGET_ENTRIES` as a compatibility override for a fixed entry count

Benchmark outputs include queue counters such as chunked pair count, chunk task count, chunk load estimates, max chunk runtime, finalize time, gate skip counts, and the estimator's current entries/sec and target entries. These fields are intended for threshold tuning.
