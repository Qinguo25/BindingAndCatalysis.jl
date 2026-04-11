#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC_ROOT="$ROOT/src/cddlib-logarithmic-complete"
OUT_ROOT="$ROOT/.build/cddlog"
OUT_SRC="$OUT_ROOT/src"

if [[ -n "${CC:-}" ]]; then
  CC=${CC}
elif command -v gcc >/dev/null 2>&1; then
  CC=gcc
elif command -v cc >/dev/null 2>&1; then
  CC=cc
elif command -v clang >/dev/null 2>&1; then
  CC=clang
else
  echo "[FAIL] no C compiler found. Set CC, or install gcc/cc/clang." >&2
  exit 1
fi
CFLAGS=${CFLAGS:--std=c11 -O2}

COMMON_SRCS=(
  lib-src/cddcore.c
  lib-src/cddio.c
  lib-src/cddlib.c
  lib-src/cddlogarithmic.c
  lib-src/cddlp.c
  lib-src/cddmp.c
  lib-src/cddproj.c
  lib-src/setoper.c
)

PROGS=(
  adjacency
  allfaces
  cddexec
  fourier
  lcdd
  projection
  redcheck
  redexter
  redundancies
  redundancies_clarkson
  scdd
)

build_variant() {
  local mode=$1
  local outdir=$2
  local cppflags=$3
  local libs=$4
  local libname=$5

  mkdir -p "$outdir"
  rm -f "$outdir"/*.o "$outdir/$libname" "$outdir"/adjacency* "$outdir"/allfaces* "$outdir"/cddexec* \
    "$outdir"/fourier* "$outdir"/lcdd* "$outdir"/projection* "$outdir"/redcheck* "$outdir"/redexter* \
    "$outdir"/redundancies* "$outdir"/scdd*

  for src in "${COMMON_SRCS[@]}"; do
    obj="$outdir/$(basename "${src%.c}").o"
    "$CC" $CFLAGS $cppflags -I"$SRC_ROOT/lib-src" -c "$SRC_ROOT/$src" -o "$obj"
  done
  ar rcs "$outdir/$libname" "$outdir"/*.o

  for prog in "${PROGS[@]}"; do
    local exe_name=$prog
    if [[ "$mode" == "log" ]]; then
      exe_name="${prog}_log"
    fi
    "$CC" $CFLAGS $cppflags -I"$SRC_ROOT/lib-src" "$SRC_ROOT/src/$prog.c" "$outdir/$libname" $libs -o "$outdir/$exe_name"
  done
}

mkdir -p "$OUT_ROOT/default" "$OUT_ROOT/log" "$OUT_SRC"

build_variant default "$OUT_ROOT/default" "" "-lm" libcdd.a
build_variant log "$OUT_ROOT/log" "-DCDDLOGARITHMIC" "-lgmp -lm" libcddlog.a

rm -f "$OUT_SRC"/*
cp "$OUT_ROOT/default"/adjacency "$OUT_SRC"/
cp "$OUT_ROOT/default"/allfaces "$OUT_SRC"/
cp "$OUT_ROOT/default"/cddexec "$OUT_SRC"/
cp "$OUT_ROOT/default"/fourier "$OUT_SRC"/
cp "$OUT_ROOT/default"/lcdd "$OUT_SRC"/
cp "$OUT_ROOT/default"/projection "$OUT_SRC"/
cp "$OUT_ROOT/default"/redcheck "$OUT_SRC"/
cp "$OUT_ROOT/default"/redexter "$OUT_SRC"/
cp "$OUT_ROOT/default"/redundancies "$OUT_SRC"/
cp "$OUT_ROOT/default"/redundancies_clarkson "$OUT_SRC"/
cp "$OUT_ROOT/default"/scdd "$OUT_SRC"/

cp "$OUT_ROOT/log"/adjacency_log "$OUT_SRC"/
cp "$OUT_ROOT/log"/allfaces_log "$OUT_SRC"/
cp "$OUT_ROOT/log"/cddexec_log "$OUT_SRC"/
cp "$OUT_ROOT/log"/fourier_log "$OUT_SRC"/
cp "$OUT_ROOT/log"/lcdd_log "$OUT_SRC"/
cp "$OUT_ROOT/log"/projection_log "$OUT_SRC"/
cp "$OUT_ROOT/log"/redcheck_log "$OUT_SRC"/
cp "$OUT_ROOT/log"/redexter_log "$OUT_SRC"/
cp "$OUT_ROOT/log"/redundancies_log "$OUT_SRC"/
cp "$OUT_ROOT/log"/redundancies_clarkson_log "$OUT_SRC"/
cp "$OUT_ROOT/log"/scdd_log "$OUT_SRC"/

cat > "$OUT_ROOT/BUILD_INFO.txt" <<EOF
CC=$CC
CFLAGS=$CFLAGS
built_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

echo "[OK] local cdd backend built at $OUT_SRC"
