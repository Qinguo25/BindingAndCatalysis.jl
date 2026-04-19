# Architecture overview

## 1. High-level structure

Relevant directories:

- `lib-src/`
  - cddlib core library sources
  - logarithmic backend lives in `cddlogarithmic.c` / `cddlogarithmic.h`
  - output integration lives in `cddio.c`
- `src/`
  - command-line drivers (`scdd`, `fourier`, `projection`, `redcheck`, etc.)
  - logarithmic regression scripts and `testlogarithmic.c`
- `examples/`
  - shipped logarithmic and default examples
- `scripts/`
  - install/build helpers
- `doc/`
  - generated manual assets

## 2. Numeric backend layering

cddlib routes its arithmetic through numeric macros that map to backend-specific functions.

Under `CDDLOGARITHMIC`:

- `mytype` is defined in `cddlogarithmic.h` as a reference-counted symbolic expression handle.
- arithmetic operations dispatch into `ddl_*` functions:
  - `ddl_add`
  - `ddl_sub`
  - `ddl_mul`
  - `ddl_div`
  - `ddl_neg`
  - `ddl_inv`
  - `ddl_cmp`
  - `ddl_sgn`
  - parsing / printing helpers

This means the rest of cddlib still runs its normal DD, pivot, elimination, LP, and redundancy code; only the scalar backend changes.

## 3. Internal data model

`lib-src/cddlogarithmic.c` stores each scalar as a reference-counted expression node:

- `DDL_CONST`
- `DDL_LOGATOM`
- `DDL_ADD`
- `DDL_SUB`
- `DDL_MUL`
- `DDL_DIV`
- `DDL_NEG`

A `DDL_LOGATOM` represents an exact constant:

`q * log(num / den)`

where:

- `q` is an exact GMP rational (`mpq_t`)
- `num`, `den` are exact GMP integers (`mpz_t`)
- the ratio is canonicalized and required to be positive

So the user-facing atomic form `m*log(x/y)` is represented exactly.

## 4. Data flow

### Input path

1. text matrix file is read by cddlib input code in `cddio.c`
2. logarithmic numeric tokens are parsed by `dd_sread_rational_value()`
3. that delegates into `ddl_parse_expression()` for `logarithmic` matrices
4. symbolic `ddl_expr` nodes are created
5. cddlib core algorithms consume those scalars through the backend arithmetic API

### Computation path

Core cddlib algorithms are unchanged in structure:

- H/V conversion
- FM elimination
- block elimination
- linearity / redundancy checks
- LP solve

These operations generate intermediate scalar expressions by repeated backend arithmetic.

### Output path

1. cddlib asks the backend to print numbers
2. `ddl_fprint()` tries to reconstruct clean log-linear text first
3. for H-representation matrices, `dd_WriteMatrix()` calls `ddl_try_fprint_hrep_row()` to print an entire inequality row in clean `logarithmic` form
4. when row-wise reconstruction is not possible, output falls back to the generic scalar printer

## 5. Why the row printer matters

The key engineering problem was not just computing with symbolic expressions; it was preventing round-tripped H-representations from printing internal scaling artifacts.

Without row-aware output normalization, a valid inequality could print as something like:

- `1/log(2/1)`
- `log(2/1)/log(2/1)`

That is mathematically equivalent but violates the user-facing requirement that output stay in the same logarithmic style.

`ddl_try_fprint_hrep_row()` solves this at the matrix-row level by identifying a rational rescaling that turns the row back into:

- rational coefficients on variable columns
- logarithmic constants in the first column

## 6. Test architecture

### Unit / small regression

- `src/testlogarithmic.c`
  - builds a tiny logarithmic interval in memory
  - checks H->V conversion
  - checks round-trip through file I/O
  - checks LP optimum

### Workflow regression

- `src/check-logarithmic.sh`
  - 1D / 2D features
  - LP
  - H/V round-trip
  - FM elimination
  - projection
  - redcheck
  - redundancies
- `src/check-logarithmic-large.sh`
  - 3D / 4D conversion
  - 4D elimination / projection coverage

## 7. Installation layout

Installed outputs include:

- library: `libcddlog`
- header: `include/cddlib/cddlogarithmic.h`
- pkg-config: `cddlog.pc`
- CLI tools:
  - `scdd_log`
  - `lcdd_log`
  - `fourier_log`
  - `projection_log`
  - `redcheck_log`
  - `adjacency_log`
  - `allfaces_log`
  - `cddexec_log`

