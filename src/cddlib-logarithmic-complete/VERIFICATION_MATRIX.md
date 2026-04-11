# Verification matrix

This document records the commands that were actually used to validate the delivered workflows.

## 1. In-tree regression suite

From the working tree used to produce this delivery:

```bash
make -j1 check
```

Observed result:

- `check-default.sh` PASS
- `testlogarithmic` PASS
- `check-logarithmic.sh` PASS
- `check-logarithmic-large.sh` PASS

## 2. H -> V and V -> H round-trip

### 2D box

```bash
./src/scdd_log examples/samplelog_box.ine
./src/scdd_log /tmp/samplelog_box_clean.ext
```

Observed features:

- H -> V output contains `4 3 logarithmic`
- vertices include `1 log(2/1) 0`, `1 0 log(3/1)`, `1 log(2/1) log(3/1)`
- V -> H round-trip prints back clean inequalities containing:
  - `log(2/1) -1 0`
  - `log(3/1) 0 -1`

### 3D / 4D boxes

```bash
./src/scdd_log examples/samplelog_box3d.ine
./src/scdd_log examples/samplelog_box4d.ine
./src/scdd_log <generated .ext file>
```

Observed features:

- 3D output contains `8 4 logarithmic`
- 4D output contains `16 5 logarithmic`
- round-tripped H-representations print back as clean logarithmic box inequalities

## 3. Fourier-Motzkin elimination

```bash
printf '1\n1\n' | ./src/fourier_log examples/samplelog_box.ine
printf '1\n1\n' | ./src/fourier_log examples/samplelog_box3d.ine
printf '1\n2\n3\n' | ./src/fourier_log examples/samplelog_box4d.ine
```

Observed features:

- no aborts
- outputs remain `H-representation`
- reduced systems preserve clean logarithmic rows such as `log(2/1) -1`

## 4. Block elimination / projection

```bash
printf '1\n2\n' | ./src/projection_log examples/samplelog_box.ine
printf '1\n2\n' | ./src/projection_log examples/samplelog_box3d.ine
printf '1\n2\n3\n4\n' | ./src/projection_log examples/samplelog_box4d.ine
```

Observed features:

- no aborts
- outputs remain `H-representation`
- reduced systems preserve clean logarithmic rows such as `log(2/1) -1`

## 5. detectlinearity / canonicalization

```bash
./src/redcheck_log examples/samplelog_linearity.ine
```

Observed features:

- no abort
- redundant row 5 detected
- output remains `H-representation`
- output contains `linearity 1  1`
- remaining logarithmic constraint prints as `log(3/1) 0 -1`

## 6. redundancy removal

```bash
./src/redundancies_log examples/samplelog_linearity.ine
```

Observed features:

- no abort
- `redundant rows: 5`
- output contains `linearity 2  1 2`
- remaining logarithmic constraint prints as `log(3/1) 0 -1`

## 7. LP

```bash
printf '%s\n' examples/sampleloglp1.ine | ./src/testlp3_log
```

Observed features:

- `optimal_value :  log(2/1)`
- interior point contains `1/2*log(2/1)`

## 8. Manual autotools install smoke

A clean copied source tree was validated step-by-step with:

```bash
mkdir -p /tmp/cdd-install-debug/source /tmp/cdd-install-debug/build /tmp/cdd-install-debug/stage
# copy source tree into /tmp/cdd-install-debug/source
cd /tmp/cdd-install-debug/build
../source/configure --disable-shared --disable-gmp-backend --prefix=/usr
make -j1
make check
make DESTDIR=/tmp/cdd-install-debug/stage install
PKG_CONFIG_PATH=/tmp/cdd-install-debug/stage/usr/lib/pkgconfig pkg-config --libs cddlog
cp ../source/examples/samplelog_box3d.ine .
/tmp/cdd-install-debug/stage/usr/bin/scdd_log samplelog_box3d.ine
```

Observed features:

- configure completed
- build completed
- `make check` completed
- install completed
- installed stage contained `scdd_log`, `fourier_log`, `projection_log`, `redcheck_log`, `cddlog.pc`, and `cddlogarithmic.h`
- installed `scdd_log` produced `V-representation` with `8 4 logarithmic`

