#!/bin/sh
set -eu
TMPDIR=${TMPDIR:-"$(pwd)/.tmp-log"}
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

printf '%s\n' "$top_srcdir/examples/samplelog1.ine" | "$top_builddir/src/testcdd1_log" > "$TMPDIR/testcdd1.out" 2>&1
require_grep "V-representation" "$TMPDIR/testcdd1.out"
require_grep "2 2 logarithmic" "$TMPDIR/testcdd1.out"
require_grep "1 log(2/1)" "$TMPDIR/testcdd1.out"

printf '%s\n' "$top_srcdir/examples/sampleloglp1.ine" | "$top_builddir/src/testlp3_log" > "$TMPDIR/testlp3.out" 2>&1
require_grep "optimal_value :  log(2/1)" "$TMPDIR/testlp3.out"
require_grep "An interior point found: ( 1/2*log(2/1))" "$TMPDIR/testlp3.out"

cp "$top_srcdir/examples/samplelog_box.ine" "$TMPDIR/samplelog_box.ine"
"$top_builddir/src/scdd_log" "$TMPDIR/samplelog_box.ine" > "$TMPDIR/scdd_box.out" 2>&1
require_grep "V-representation" "$TMPDIR/samplelog_box.ext"
require_grep "4 3 logarithmic" "$TMPDIR/samplelog_box.ext"
require_grep "1 log(2/1) 0" "$TMPDIR/samplelog_box.ext"
require_grep "1 0 log(3/1)" "$TMPDIR/samplelog_box.ext"

cp "$TMPDIR/samplelog_box.ext" "$TMPDIR/samplelog_box_roundtrip.ext"
"$top_builddir/src/scdd_log" "$TMPDIR/samplelog_box_roundtrip.ext" > "$TMPDIR/scdd_box_roundtrip.out" 2>&1
require_grep "H-representation" "$TMPDIR/samplelog_box_roundtrip.ine"
require_grep "4 3 logarithmic" "$TMPDIR/samplelog_box_roundtrip.ine"
require_grep "log(2/1) -1 0" "$TMPDIR/samplelog_box_roundtrip.ine"
require_grep "log(3/1) 0 -1" "$TMPDIR/samplelog_box_roundtrip.ine"

cp "$top_srcdir/examples/samplelog_box3d.ine" "$TMPDIR/samplelog_box3d.ine"
"$top_builddir/src/scdd_log" "$TMPDIR/samplelog_box3d.ine" > "$TMPDIR/scdd_box3d.out" 2>&1
require_grep "V-representation" "$TMPDIR/samplelog_box3d.ext"
require_grep "8 4 logarithmic" "$TMPDIR/samplelog_box3d.ext"
require_grep "1 log(2/1) log(3/1) log(5/1)" "$TMPDIR/samplelog_box3d.ext"

printf '1\n1\n' | "$top_builddir/src/fourier_log" "$top_srcdir/examples/samplelog_box.ine" > "$TMPDIR/fourier.out" 2>&1
require_grep "Nonredundant representation:" "$TMPDIR/fourier.out"
require_grep "2 2 logarithmic" "$TMPDIR/fourier.out"
require_grep "log(2/1) -1" "$TMPDIR/fourier.out"

printf '1\n2\n' | "$top_builddir/src/projection_log" "$top_srcdir/examples/samplelog_box.ine" > "$TMPDIR/projection.out" 2>&1
require_grep "H-representation" "$TMPDIR/projection.out"
require_grep "2 2 logarithmic" "$TMPDIR/projection.out"
require_grep "log(2/1) -1" "$TMPDIR/projection.out"

"$top_builddir/src/redcheck_log" "$top_srcdir/examples/samplelog_linearity.ine" > "$TMPDIR/redcheck.out" 2>&1
require_grep "Canonicalize the matrix." "$TMPDIR/redcheck.out"
require_grep "Redundant rows are: 5" "$TMPDIR/redcheck.out"
require_grep "linearity 1  1" "$TMPDIR/redcheck.out"
require_grep "3 3 logarithmic" "$TMPDIR/redcheck.out"
require_grep "log(3/1) 0 -1" "$TMPDIR/redcheck.out"

"$top_builddir/src/redundancies_log" "$top_srcdir/examples/samplelog_linearity.ine" > "$TMPDIR/redundancies.out" 2>&1
require_grep "redundant rows: 5" "$TMPDIR/redundancies.out"
require_grep "linearity 2  1 2" "$TMPDIR/redundancies.out"
require_grep "4 3 logarithmic" "$TMPDIR/redundancies.out"
require_grep "log(3/1) 0 -1" "$TMPDIR/redundancies.out"
