#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CC=${CC:-gcc}
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
  testcdd1
  testshoot
  testcdd2
  testlp1
  testlp2
  testlp3
)

build_variant() {
  local mode=$1
  local outdir=$2
  local cppflags=$3
  local libs=$4
  local libname=$5

  mkdir -p "$ROOT/$outdir"
  rm -f "$ROOT/$outdir"/*.o "$ROOT/$outdir/$libname" "$ROOT/$outdir"/*_log "$ROOT/$outdir"/testlogarithmic

  for src in "${COMMON_SRCS[@]}"; do
    obj="$ROOT/$outdir/$(basename "${src%.c}").o"
    $CC $CFLAGS $cppflags -I"$ROOT/lib-src" -c "$ROOT/$src" -o "$obj"
  done
  ar rcs "$ROOT/$outdir/$libname" "$ROOT/$outdir"/*.o

  for prog in "${PROGS[@]}"; do
    local exe_name=$prog
    if [[ "$mode" == log ]]; then
      if [[ "$prog" == scdd || "$prog" == cddexec ]]; then
        exe_name="${prog}_log"
      fi
    fi
    $CC $CFLAGS $cppflags -I"$ROOT/lib-src" "$ROOT/src/$prog.c" "$ROOT/$outdir/$libname" $libs -o "$ROOT/$outdir/$exe_name"
  done

  if [[ "$mode" == log ]]; then
    $CC $CFLAGS -DCDDLOGARITHMIC -I"$ROOT/lib-src" "$ROOT/src/testlogarithmic.c" "$ROOT/$outdir/$libname" -lgmp -lm -o "$ROOT/$outdir/testlogarithmic"
  fi
}

build_variant default build-default "" "-lm" libcdd.a
build_variant log build-log "-DCDDLOGARITHMIC" "-lgmp -lm" libcddlog.a

echo "[OK] manual build completed: build-default and build-log"
