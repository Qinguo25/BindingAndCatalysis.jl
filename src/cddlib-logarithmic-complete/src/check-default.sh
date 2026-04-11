#!/bin/sh
set -eu
TMPDIR=${TMPDIR:-"$(pwd)/.tmp-default"}
mkdir -p "$TMPDIR"
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT INT TERM

require_grep() {
  pattern=$1
  file=$2
  if ! grep -Fq "$pattern" "$file"; then
    echo "[FAIL] missing pattern: $pattern" >&2
    echo "[FAIL] file: $file" >&2
    exit 1
  fi
}

"$top_builddir/src/testcdd2" > "$TMPDIR/testcdd2.out" 2>&1
require_grep "V-representation" "$TMPDIR/testcdd2.out"

printf '%s\n' "$top_srcdir/examples/samplelp1.ine" | "$top_builddir/src/testlp3" > "$TMPDIR/testlp3.out" 2>&1
require_grep "optimal_value :   3" "$TMPDIR/testlp3.out"

cp "$top_srcdir/examples/samplev1.ext" "$TMPDIR/samplev1.ext"
"$top_builddir/src/scdd" "$TMPDIR/samplev1.ext" > "$TMPDIR/scdd.out" 2>&1
require_grep "H-representation" "$TMPDIR/samplev1.ine"
require_grep "real" "$TMPDIR/samplev1.ine"
