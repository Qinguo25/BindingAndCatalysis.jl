# Implementation details

## 1. Core representation

The logarithmic backend is implemented in `lib-src/cddlogarithmic.c`.

Each scalar is a `ddl_expr` DAG node. Exact constants are stored with GMP rationals and integers; logarithmic atoms store exact rational multipliers and exact integer ratios.

This gives exact representation for:

- rational constants
- `m * log(x / y)` atoms
- symbolic combinations produced internally by cddlib arithmetic

## 2. Exact zero detection

### Problem

The original symbolic-log prototype failed when the core algorithms generated expressions that were algebraically zero but not syntactically identical. Typical examples:

- `x + (-x)`
- `log(a)/log(a) - 1`
- duplicate rows differing only by symbolic cancellation

Those expressions reached `ddl_sgn()` and caused aborts.

### Solution

The backend now converts expressions into a rational function over formal log atoms:

- numerator polynomial over monomials in log atoms
- denominator polynomial over monomials in log atoms

Key functions:

- `ddl_expr_to_rfunc`
- `ddl_expr_is_zero_exact`
- `ddl_sgn`
- `ddl_cmp`

`ddl_sgn()` now behaves as follows:

1. obtain a fast long-double estimate
2. if it is clearly nonzero, use it
3. if it is numerically near zero, invoke exact symbolic zero detection
4. return zero when the symbolic numerator vanishes exactly
5. only abort if the expression is neither numerically clear nor symbolically reducible to zero

For the shipped regression cases, this is enough to stabilize the required workflows.

## 3. Print normalization

### Scalar normalization

`ddl_fprint()` first tries to print through `ddl_fprint_linear_from_rfunc()`.

This reconstructs expressions of the form:

- rational constant
- plus/minus a sum of rational multiples of `log(num/den)` atoms

so human-readable output stays close to the original input language.

### H-row normalization

`ddl_try_fprint_hrep_row()` analyzes a full inequality row.

It detects whether the row can be rescaled back into the clean format expected by cddlib H-representation files:

- first column: logarithmic / rational constant term
- remaining columns: rational coefficients

This is why V->H round-trips of the box examples print back as clean inequalities instead of raw symbolic scaling artifacts.

## 4. Parser behavior

Accepted atomic logarithmic terms are of the form:

- `log(x/y)`
- `m*log(x/y)`
- rational combinations composed by existing arithmetic syntax used internally by the backend parser

Validation rules:

- `x/y` must be positive
- ratios are normalized by gcd
- `log(1)` is folded to exact zero
- zero multipliers are folded to exact zero

## 5. Boundary cases

Handled explicitly:

- `log(1/1)` -> `0`
- `0 * log(x/y)` -> `0`
- duplicate / opposite expressions -> canonical zero via exact zero detection
- rows that are all rational still print as rational rows
- rows with zero variable coefficients are treated as rational zeros by the row printer

Known practical scope boundary:

- this backend is designed to support the required cddlib workflows and regression cases
- it is not positioned as a full symbolic algebra system for arbitrary transcendental identities

## 6. Error management

The backend uses fail-fast behavior for invalid numeric states that should never survive parsing or core arithmetic:

- invalid log arguments (non-positive ratio)
- division by exact zero
- out-of-memory allocation failures

When these occur, the backend aborts with a descriptive `cddlogarithmic:` message rather than silently producing a wrong result.

This matches cddlib's existing style for unrecoverable scalar-backend failures.

## 7. Performance assessment

### Cost drivers

The dominant costs are:

- expression DAG growth under elimination / pivoting
- repeated conversion to rational-function form for exact zero checks and clean printing
- GMP rational/integer operations on normalization paths

### Practical behavior on shipped regression set

Observed behavior in this session:

- 1D / 2D / 3D / 4D shipped logarithmic examples complete quickly
- 4D box conversion and elimination remain practical
- no pathological blow-up was observed on the bundled regression corpus

### Tradeoff

This implementation intentionally favors correctness of the required workflows over aggressive symbolic simplification.

There is still room for future optimization:

- memoization of rational-function conversions
- better common-subexpression reuse
- stronger canonicalization of symbolic products / quotients
- specialized fast paths for pure log-linear rows

## 8. Files changed conceptually

The main logical changes are concentrated in:

- `lib-src/cddlogarithmic.c`
- `lib-src/cddlogarithmic.h`
- `lib-src/cddio.c`
- `src/check-logarithmic.sh`
- `src/check-logarithmic-large.sh`
- `src/testlogarithmic.c`
- `examples/samplelog*.ine`
- autotools files that install `libcddlog`, `cddlog.pc`, and the `*_log` tools

