#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BASE=${BASE:-"${TMPDIR:-/tmp}/cddlib-autotools.$$"}
SRC="$BASE/source"
BUILD="$BASE/build"
STAGE="$BASE/stage"
rm -rf "$BASE"
mkdir -p "$SRC" "$BUILD" "$STAGE"

echo "[1/5] copying source tree"
(
  cd "$ROOT"
  tar \
    --exclude='./.autotools-tmp' \
    --exclude='./.autotools-build' \
    --exclude='./.autotools-stage' \
    --exclude='./autom4te.cache' \
    --exclude='./.deps' \
    --exclude='./.libs' \
    --exclude='./src/.deps' \
    --exclude='./src/.libs' \
    --exclude='./lib-src/.deps' \
    --exclude='./lib-src/.libs' \
    --exclude='./doc/.deps' \
    --exclude='./doc/.libs' \
    -cf - .
) | (
  cd "$SRC"
  tar -xf -
)

echo "[2/5] cleaning copied source state"
rm -f "$SRC/config.status" "$SRC/config.log" "$SRC/libtool"
find "$SRC" -name Makefile -type f -delete
find "$SRC" -name '*.o' -type f -delete
find "$SRC" -name '*.lo' -type f -delete
find "$SRC" -name '*.la' -type f -delete
find "$SRC" -name '.deps' -type d -prune -exec rm -rf {} +
find "$SRC" -name '.libs' -type d -prune -exec rm -rf {} +

echo "[3/5] configuring out-of-tree build"
(
  cd "$BUILD"
  "$SRC/configure" --disable-shared --disable-gmp-backend --prefix=/usr
)

echo "[4/5] building"
(
  cd "$BUILD"
  make -j"${MAKE_JOBS:-1}"
)

echo "[5/5] installing into stage"
(
  cd "$BUILD"
  make DESTDIR="$STAGE" install
)

require_file() {
  local path=$1
  [ -e "$path" ] || { echo "[FAIL] missing file: $path" >&2; exit 1; }
}

require_grep() {
  local pattern=$1
  local file=$2
  grep -Fq -- "$pattern" "$file" || { echo "[FAIL] missing pattern: $pattern in $file" >&2; exit 1; }
}

require_file "$STAGE/usr/include/cddlib/cddlogarithmic.h"
require_file "$STAGE/usr/lib/pkgconfig/cddlog.pc"
require_file "$STAGE/usr/bin/scdd_log"
require_file "$STAGE/usr/bin/cddexec_log"
require_file "$STAGE/usr/bin/fourier_log"
require_file "$STAGE/usr/bin/projection_log"
require_file "$STAGE/usr/bin/redcheck_log"
require_file "$STAGE/usr/share/doc/cddlib/examples/samplelog_box3d.ine"

echo "[verify] pkg-config"
PKG_CONFIG_PATH="$STAGE/usr/lib/pkgconfig" pkg-config --libs cddlog > "$BUILD/pkglibs.out"
require_grep "-lcddlog" "$BUILD/pkglibs.out"
require_grep "-lgmp" "$BUILD/pkglibs.out"

echo "[verify] installed scdd_log"
cp "$SRC/examples/samplelog_box3d.ine" "$BUILD/samplelog_box3d.ine"
"$STAGE/usr/bin/scdd_log" "$BUILD/samplelog_box3d.ine" >/dev/null 2>&1
require_grep "V-representation" "$BUILD/samplelog_box3d.ext"
require_grep "8 4 logarithmic" "$BUILD/samplelog_box3d.ext"
require_grep "log(5/1)" "$BUILD/samplelog_box3d.ext"

echo "[OK] autotools configure/build/install flow passed"
