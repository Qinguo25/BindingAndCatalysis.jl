# cddlib logarithmic backend

This tree adds a `logarithmic` number type to cddlib for polyhedra of the form

`A x >= b`

with the intended user-facing contract:

- `A` stays rational.
- `b` may contain exact logarithmic constants of the form `m*log(x/y)`, where `m`, `x`, and `y` are integers and `x/y > 0`.
- the core cddlib workflows continue to read and write files marked `logarithmic`.

The implementation goal in this delivery is not to build a general transcendental CAS. The goal is narrower and engineering-focused: make the base cddlib workflows work on logarithmic right-hand sides while preserving clean `logarithmic` input/output on the supported regression set.

## Verified workflows

The following workflows are covered by the regression suite and were re-verified in this session on the provided logarithmic examples:

- H-representation -> V-representation conversion
- V-representation -> H-representation round-trip
- Fourier-Motzkin elimination
- block elimination / projection
- detectlinearity / canonicalization (`redcheck_log`)
- redundancy removal (`redundancies_log`)
- LP solve and interior-point path on a logarithmic interval
- autotools build + `make check`
- autotools install smoke with installed `pkg-config` and installed `scdd_log`

## Regression examples

The main logarithmic examples are:

- `examples/samplelog1.ine`
- `examples/sampleloglp1.ine`
- `examples/samplelog_box.ine`
- `examples/samplelog_box3d.ine`
- `examples/samplelog_box4d.ine`
- `examples/samplelog_linearity.ine`

These exercise:

- 1D interval conversion and LP
- 2D box H/V round-trip
- 3D and 4D box conversion
- FM elimination and projection on box examples
- linearity detection and redundancy removal on a logarithmic equality/redundancy case

## What changed technically

### 1. Exact symbolic backend improvements

The logarithmic backend in `lib-src/cddlogarithmic.c` now keeps symbolic expressions and adds an exact zero-detection path based on conversion into a rational function over formal log atoms.

That addresses the practical failure mode that previously broke the backend:

- expressions like `x + (-x)`
- expressions like `log(a)/log(a) - 1`
- rows that are symbolically zero but only numerically close to zero

Before this work, those cases often reached `ddl_sgn()` and aborted. Now they are reduced through exact symbolic zero detection before the sign decision falls back to numeric magnitude.

### 2. Output normalization for clean logarithmic files

The internal elimination and DD steps can temporarily build more complicated symbolic expressions than the input language. To keep file I/O clean, the backend now performs targeted print-time normalization:

- `ddl_fprint()` reconstructs log-linear expressions when possible.
- `ddl_try_fprint_hrep_row()` re-normalizes H-representation rows so that round-tripped inequalities print back in clean `logarithmic` form instead of leaking intermediate expressions such as `1/log(...)`.
- `lib-src/cddio.c` uses the row-aware printer for H-representation output under `CDDLOGARITHMIC`.

### 3. Regression coverage

`src/check-logarithmic.sh` and `src/check-logarithmic-large.sh` now cover the user-requested base workflows directly.

## Build and test

### Recommended local validation

From a clean checkout of this tree:

```bash
mkdir /tmp/cdd-build
cd /tmp/cdd-build
/path/to/cddlib-logarithmic-complete/configure --disable-shared --disable-gmp-backend --prefix=/usr
make -j1
make check
make DESTDIR=/tmp/cdd-stage install
PKG_CONFIG_PATH=/tmp/cdd-stage/usr/lib/pkgconfig pkg-config --libs cddlog
cp /path/to/cddlib-logarithmic-complete/examples/samplelog_box3d.ine .
/tmp/cdd-stage/usr/bin/scdd_log samplelog_box3d.ine
```

### Convenience wrappers

- `run-tests.sh` is the repository-level smoke entrypoint.
- `scripts/test-autotools-install.sh` exercises the clean-copy out-of-tree install flow.

For restricted execution environments, it is safer to run the configure / make / make check / install steps separately, exactly as shown above. Those step-by-step commands are the authoritative validation path used for this delivery.

## Key documentation

- `ARCHITECTURE_OVERVIEW.md`
- `IMPLEMENTATION_DETAILS.md`
- `VERIFICATION_MATRIX.md`
- `DELIVERY_NOTES.md`
- `scripts/test-autotools-install.sh`

