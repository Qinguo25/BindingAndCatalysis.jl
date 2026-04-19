#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
BUILD=${BUILD:-"${TMPDIR:-/tmp}/cdd-build-check.$$"}
STAGE=${STAGE:-"${TMPDIR:-/tmp}/cdd-stage-check.$$"}
rm -rf "$BUILD" "$STAGE"
mkdir -p "$BUILD" "$STAGE"

(
  cd "$BUILD"
  "$ROOT/configure" --disable-shared --disable-gmp-backend >/dev/null
  make -j"${MAKE_JOBS:-1}" >/dev/null
  make check >/dev/null
  make DESTDIR="$STAGE" install >/dev/null
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
require_file "$STAGE/usr/bin/fourier_log"
require_file "$STAGE/usr/bin/projection_log"
require_file "$STAGE/usr/bin/redcheck_log"
require_file "$STAGE/usr/share/doc/cddlib/examples/samplelog_box4d.ine"

PKG_CONFIG_PATH="$STAGE/usr/lib/pkgconfig" pkg-config --libs cddlog > "$BUILD/pkglibs.out"
require_grep "-lcddlog" "$BUILD/pkglibs.out"
require_grep "-lgmp" "$BUILD/pkglibs.out"

cp "$ROOT/examples/samplelog_box3d.ine" "$BUILD/samplelog_box3d.ine"
"$STAGE/usr/bin/scdd_log" "$BUILD/samplelog_box3d.ine" >/dev/null 2>&1
require_grep "V-representation" "$BUILD/samplelog_box3d.ext"
require_grep "8 4 logarithmic" "$BUILD/samplelog_box3d.ext"
require_grep "log(5/1)" "$BUILD/samplelog_box3d.ext"

echo "[OK] configure/build/check/install smoke passed"
