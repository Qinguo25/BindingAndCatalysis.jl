#!/bin/sh
set -eu
TMPDIR=${TMPDIR:-"$(pwd)/.tmp-log-large"}
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

cp "$top_srcdir/examples/samplelog_box3d.ine" "$TMPDIR/samplelog_box3d.ine"
"$top_builddir/src/scdd_log" "$TMPDIR/samplelog_box3d.ine" > "$TMPDIR/box3d.out" 2>&1
require_grep "V-representation" "$TMPDIR/samplelog_box3d.ext"
require_grep "8 4 logarithmic" "$TMPDIR/samplelog_box3d.ext"
require_grep "log(5/1)" "$TMPDIR/samplelog_box3d.ext"

"$top_builddir/src/scdd_log" "$TMPDIR/samplelog_box3d.ext" > "$TMPDIR/box3d-roundtrip.out" 2>&1
require_grep "H-representation" "$TMPDIR/samplelog_box3d.ine"
require_grep "6 4 logarithmic" "$TMPDIR/samplelog_box3d.ine"
require_grep "log(3/1)" "$TMPDIR/samplelog_box3d.ine"

cp "$top_srcdir/examples/samplelog_box4d.ine" "$TMPDIR/samplelog_box4d.ine"
"$top_builddir/src/scdd_log" "$TMPDIR/samplelog_box4d.ine" > "$TMPDIR/box4d.out" 2>&1
require_grep "V-representation" "$TMPDIR/samplelog_box4d.ext"
require_grep "16 5 logarithmic" "$TMPDIR/samplelog_box4d.ext"
require_grep "log(7/1)" "$TMPDIR/samplelog_box4d.ext"

printf '1\n2\n3\n' | "$top_builddir/src/fourier_log" "$top_srcdir/examples/samplelog_box4d.ine" > "$TMPDIR/fourier4d.out" 2>&1
require_grep "Nonredundant representation:" "$TMPDIR/fourier4d.out"
require_grep "log(2/1) -1" "$TMPDIR/fourier4d.out"

printf '1\n2\n3\n4\n' | "$top_builddir/src/projection_log" "$top_srcdir/examples/samplelog_box4d.ine" > "$TMPDIR/projection4d.out" 2>&1
require_grep "H-representation" "$TMPDIR/projection4d.out"
require_grep "log(2/1) -1" "$TMPDIR/projection4d.out"

cat > "$TMPDIR/samplelog_zero_proj.ine" <<'EOF'
H-representation
begin
 6 7 logarithmic
0 -1 1 0 0 0 0
0 -1 0 1 0 0 0
0 -1 0 0 1 0 0
0 -2 0 1 1 -1 1
0 -1 0 0 0 0 1
0 -2 1 0 1 -1 1
end
EOF
printf '1\n1\n' | "$top_builddir/src/projection_log" "$TMPDIR/samplelog_zero_proj.ine" > "$TMPDIR/projection_zero.out" 2>&1
require_grep "H-representation" "$TMPDIR/projection_zero.out"
require_grep "0 6 logarithmic" "$TMPDIR/projection_zero.out"
