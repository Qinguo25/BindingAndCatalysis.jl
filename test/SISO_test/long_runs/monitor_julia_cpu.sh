#!/usr/bin/env bash
set -u

PID="$1"
OUT="$2"
INTERVAL="${3:-5}"
CLK_TCK="$(getconf CLK_TCK 2>/dev/null || echo 100)"

declare -A PREV_THREAD_TICKS=()
INTERVAL_THREAD_CPU_SUM="0.0"
INTERVAL_ACTIVE_THREADS_GT1PCT=0
INTERVAL_ACTIVE_THREADS_GT10PCT=0
INTERVAL_MAX_THREAD_PCPU="0.0"

read_proc_ticks() {
  local stat_line rest
  stat_line="$(cat "/proc/$PID/stat" 2>/dev/null)" || return 1
  rest="${stat_line##*) }"
  awk '{print $12 + $13}' <<< "$rest"
}

update_thread_interval_stats() {
  local dt="$1"
  local sum_cpu="0"
  local max_cpu="0"
  local active1=0
  local active10=0
  local tid stat_line rest ticks prev delta cpu task

  for task in "/proc/$PID/task/"*; do
    [[ -e "$task/stat" ]] || continue
    tid="${task##*/}"
    stat_line="$(cat "$task/stat" 2>/dev/null)" || continue
    rest="${stat_line##*) }"
    ticks="$(awk '{print $12 + $13}' <<< "$rest")"
    prev="${PREV_THREAD_TICKS[$tid]:-}"
    PREV_THREAD_TICKS[$tid]="$ticks"
    [[ -n "$prev" ]] || continue
    delta=$((ticks - prev))
    cpu="$(awk -v d="$delta" -v hz="$CLK_TCK" -v dt="$dt" 'BEGIN { if (dt > 0) printf "%.1f", 100 * d / hz / dt; else printf "0.0" }')"
    sum_cpu="$(awk -v a="$sum_cpu" -v b="$cpu" 'BEGIN { printf "%.1f", a + b }')"
    max_cpu="$(awk -v a="$max_cpu" -v b="$cpu" 'BEGIN { printf "%.1f", (b > a ? b : a) }')"
    awk -v c="$cpu" 'BEGIN { exit !(c > 1.0) }' && active1=$((active1 + 1))
    awk -v c="$cpu" 'BEGIN { exit !(c > 10.0) }' && active10=$((active10 + 1))
  done

  INTERVAL_THREAD_CPU_SUM="$sum_cpu"
  INTERVAL_ACTIVE_THREADS_GT1PCT="$active1"
  INTERVAL_ACTIVE_THREADS_GT10PCT="$active10"
  INTERVAL_MAX_THREAD_PCPU="$max_cpu"
}

echo -e "timestamp\tpid\tetimes\tps_pcpu\tpmem\tnlwp\trss_kb\tvsz_kb\tstat\tlast_cpu\tthread_cpu_sum\tactive_threads_gt1pct\tactive_threads_gt10pct\tunique_last_cpus\ttop_pcpu_snapshot\tinterval_seconds\tinterval_proc_pcpu\tinterval_thread_cpu_sum\tinterval_active_threads_gt1pct\tinterval_active_threads_gt10pct\tinterval_max_thread_pcpu\tcpus_allowed" >> "$OUT"

prev_wall="$(date +%s.%N)"
prev_proc_ticks="$(read_proc_ticks || echo 0)"
while kill -0 "$PID" 2>/dev/null; do
  sleep "$INTERVAL"
  ts=$(date -Is)
  proc=$(ps -p "$PID" -o pid=,etimes=,pcpu=,pmem=,nlwp=,rss=,vsz=,stat=,psr= | awk '{print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9}')
  thread_stats=$(ps -L -p "$PID" -o psr=,pcpu= 2>/dev/null | awk '{sum+=$2; if ($2>1) n1++; if ($2>10) n10++; cores[$1]=1} END {u=0; for (c in cores) u++; printf "%.1f\t%d\t%d\t%d", sum, n1+0, n10+0, u}')
  top_cpu=$(top -b -n 1 -p "$PID" 2>/dev/null | awk -v pid="$PID" '$1 == pid {print $9; found=1} END {if (!found) print "NA"}')
  now_wall="$(date +%s.%N)"
  now_proc_ticks="$(read_proc_ticks || echo "$prev_proc_ticks")"
  interval_seconds="$(awk -v a="$prev_wall" -v b="$now_wall" 'BEGIN { printf "%.3f", b - a }')"
  proc_delta=$((now_proc_ticks - prev_proc_ticks))
  interval_proc_pcpu="$(awk -v d="$proc_delta" -v hz="$CLK_TCK" -v dt="$interval_seconds" 'BEGIN { if (dt > 0) printf "%.1f", 100 * d / hz / dt; else printf "0.0" }')"
  update_thread_interval_stats "$interval_seconds"
  allowed=$(grep Cpus_allowed_list "/proc/$PID/status" 2>/dev/null | awk '{print $2}')
  echo -e "$ts\t$proc\t$thread_stats\t$top_cpu\t$interval_seconds\t$interval_proc_pcpu\t$INTERVAL_THREAD_CPU_SUM\t$INTERVAL_ACTIVE_THREADS_GT1PCT\t$INTERVAL_ACTIVE_THREADS_GT10PCT\t$INTERVAL_MAX_THREAD_PCPU\t$allowed" >> "$OUT"
  prev_wall="$now_wall"
  prev_proc_ticks="$now_proc_ticks"
done
