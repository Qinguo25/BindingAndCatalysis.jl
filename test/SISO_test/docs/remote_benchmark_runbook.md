# Remote CDN Benchmark Runbook

This note records the repeatable workflow for launching long CDN/SISO benchmarks
on a remote Linux machine. It intentionally does not store machine IP addresses,
usernames, passwords, or private paths.

Before launching a remote run, ask the user for:

- SSH alias or host name, for example `<SSH_ALIAS>`
- SSH username, if it is not already part of the alias
- SSH password, if key-based auth is not configured
- Remote base directory for benchmark runs, for example `<REMOTE_BASE>`
- Julia executable path on the remote, for example `<JULIA_BIN>`
- Desired CDN size, thread count, and model variant

Do not write the password into scripts, docs, shell commands, metadata files, or
git history. Let `ssh` / `scp` prompt for it interactively.

## Local Inputs

The long-run benchmark and monitor live at:

- `test/SISO_test/long_runs/cdn_overnight_benchmark.jl`
- `test/SISO_test/long_runs/monitor_julia_cpu.sh`

Useful benchmark environment variables:

- `JULIA_NUM_THREADS`: Julia worker threads, for example `50`
- `BNC_CDN_N`: CDN size, for example `5`
- `BNC_CDN_SOLVER`: usually `dag`
- `BNC_CDN_INCLUDE_HOMODIMERS`: `false` for heterodimers only, `true` for heterodimers plus homodimers
- `BNC_SISO_DAG_SCHEDULER`: usually `auto`
- `BNC_SISO_DAG_CHUNK_QUEUE`: usually `true`
- `BNC_SISO_DAG_TARGET_CHUNK_SECONDS`: default is `40`
- `BNC_HEARTBEAT_SECONDS`: status JSON update interval
- `BNC_STATUS_PATH`: remote status JSON path
- `BNC_RESULT_PATH`: remote result JSON path

For homodimers, the benchmark generator uses pairs `i <= j`; diagonal
homodimer rows add the monomer twice, so their stoichiometric coefficient is
`2`.

## Prepare A Remote Run Directory

From the local repository root:

```bash
RUN_NAME="bnc_cdn5_all_dimers_$(date +%Y%m%d_%H%M%S)"
REMOTE_RUN="<REMOTE_BASE>/${RUN_NAME}"

git archive --format=tar HEAD -o /tmp/bnc_remote_benchmark_head.tar
ssh <SSH_ALIAS> "mkdir -p ${REMOTE_RUN}/repo ${REMOTE_RUN}/artifacts"
scp /tmp/bnc_remote_benchmark_head.tar \
  test/SISO_test/long_runs/cdn_overnight_benchmark.jl \
  test/SISO_test/long_runs/monitor_julia_cpu.sh \
  <SSH_ALIAS>:"${REMOTE_RUN}/"
ssh <SSH_ALIAS> "
  tar -xf ${REMOTE_RUN}/bnc_remote_benchmark_head.tar -C ${REMOTE_RUN}/repo
  mkdir -p ${REMOTE_RUN}/repo/test/SISO_test/long_runs
  cp ${REMOTE_RUN}/cdn_overnight_benchmark.jl ${REMOTE_RUN}/repo/test/SISO_test/long_runs/cdn_overnight_benchmark.jl
  chmod +x ${REMOTE_RUN}/monitor_julia_cpu.sh
"
```

Use `git archive HEAD` for a clean committed snapshot. If an uncommitted script
change is intentional, copy that script over after extracting the archive, as
shown above.

## Instantiate The Project

Fresh archives may not have all Julia dependencies available in the remote depot.
Instantiate once before launching:

```bash
ssh <SSH_ALIAS> "
  cd ${REMOTE_RUN}/repo
  <JULIA_BIN> --project=. -e 'using Pkg; Pkg.instantiate()'
"
```

If this fails because the remote has no network access or the registry is stale,
ask the user whether to use an existing prepared run directory or remote depot
instead of silently changing the environment.

## Launch CDN5, Heterodimers Only

```bash
ssh <SSH_ALIAS> "
  run=${REMOTE_RUN}
  art=\$run/artifacts
  julia_bin=<JULIA_BIN>
  cd \$run/repo

  {
    echo run_dir=\$run
    echo started_at=\$(date -Is)
    echo hostname=\$(hostname)
    \$julia_bin --version
    echo julia_threads=50
    echo cdn=5
    echo include_homodimers=false
    echo condition_solver=dag
    echo scheduler=auto
    echo target_chunk_seconds=40
    echo monitor_interval_seconds=5
    echo nproc=\$(nproc)
    uname -a
    lscpu
    free -h
  } > \$art/run_metadata.txt

  env JULIA_NUM_THREADS=50 \
    BNC_CDN_N=5 \
    BNC_CDN_SOLVER=dag \
    BNC_CDN_INCLUDE_HOMODIMERS=false \
    BNC_SISO_DAG_SCHEDULER=auto \
    BNC_SISO_DAG_CHUNK_QUEUE=true \
    BNC_SISO_DAG_TARGET_CHUNK_SECONDS=40 \
    BNC_HEARTBEAT_SECONDS=30 \
    BNC_STATUS_PATH=\$art/cdn5_status.json \
    BNC_RESULT_PATH=\$art/cdn5_result.json \
    nohup \$julia_bin --project=. test/SISO_test/long_runs/cdn_overnight_benchmark.jl \
      > \$art/cdn5_stdout.log \
      2> \$art/cdn5_stderr.log &

  echo \$! > \$art/cdn5_julia.pid
  sleep 5

  nohup bash \$run/monitor_julia_cpu.sh \
    \$(cat \$art/cdn5_julia.pid) \
    \$art/cdn5_cpu_5s.tsv \
    5 \
    > \$art/monitor_stdout.log \
    2> \$art/monitor_stderr.log &

  echo \$! > \$art/cdn5_monitor.pid
  ps -p \$(cat \$art/cdn5_julia.pid) -o pid,etime,pcpu,pmem,nlwp,rss,stat,cmd
"
```

## Launch CDN5, Heterodimers Plus Homodimers

Use the same launch script, but set:

```bash
BNC_CDN_INCLUDE_HOMODIMERS=true
```

Use distinct artifact names to avoid mixing runs:

```bash
BNC_STATUS_PATH=$art/cdn5_all_dimers_status.json
BNC_RESULT_PATH=$art/cdn5_all_dimers_result.json
stdout: $art/cdn5_all_dimers_stdout.log
stderr: $art/cdn5_all_dimers_stderr.log
monitor: $art/cdn5_all_dimers_cpu_5s.tsv
pid: $art/cdn5_all_dimers_julia.pid
```

## Check A Running Job

```bash
ssh <SSH_ALIAS> "
  art=${REMOTE_RUN}/artifacts
  ps -p \$(cat \$art/cdn5_julia.pid) -o pid,etime,pcpu,pmem,nlwp,rss,stat,cmd
  ps -p \$(cat \$art/cdn5_monitor.pid) -o pid,etime,pcpu,pmem,rss,stat,cmd
  cat \$art/cdn5_status.json 2>/dev/null || true
  tail -n 8 \$art/cdn5_cpu_5s.tsv 2>/dev/null || true
  tail -n 40 \$art/cdn5_stderr.log 2>/dev/null || true
"
```

For all-dimer runs, replace the artifact names with the `cdn5_all_dimers_*`
names used at launch.

The CPU monitor records 5-second interval thread CPU. The most useful column is
`interval_thread_cpu_sum`; values near `5000` mean about 50 cores are active.

## Check Completion

```bash
ssh <SSH_ALIAS> "
  art=${REMOTE_RUN}/artifacts
  cat \$art/cdn5_result.json 2>/dev/null || true
  tail -n 20 \$art/cdn5_stderr.log 2>/dev/null || true
  awk -F '\t' 'NR>1 && NF>=22 && \$18+0>0 {
    n++;
    sum+=\$18;
    if (\$18>max) max=\$18;
  } END {
    if (n>0) printf \"samples=%d avg_interval_thread_cpu=%.1f max=%.1f\\n\", n, sum/n, max;
  }' \$art/cdn5_cpu_5s.tsv
"
```

Record conclusions in a small reference Markdown note. Do not commit raw status
JSON, stdout/stderr logs, monitor TSV files, launcher logs, or full remote
artifacts.

## Stop A Run

Only stop a job when the user explicitly asks or when it is clearly failed.

```bash
ssh <SSH_ALIAS> "
  art=${REMOTE_RUN}/artifacts
  kill \$(cat \$art/cdn5_julia.pid) 2>/dev/null || true
  kill \$(cat \$art/cdn5_monitor.pid) 2>/dev/null || true
"
```

For all-dimer runs, use `cdn5_all_dimers_julia.pid` and
`cdn5_all_dimers_monitor.pid`.

## Common Failure Modes

- Missing package error: run `Pkg.instantiate()` in the remote repo.
- Immediate process exit: inspect stderr before relaunching.
- Empty `ps` after launch: the process likely exited during startup; check
  `*_stderr.log`.
- Monitor exists but Julia is gone: inspect result/status/stderr; the monitor
  exits after the PID disappears.
- Huge memory growth: report RSS and stage to the user before taking action.
- Password prompt in non-interactive SSH: rerun with a TTY and enter the
  password interactively; never place it in the command string.
