#include "setoper.h"
#include "cdd.h"
#include "cddlogarithmic.h"

#if defined(CDDLOGARITHMIC)

#include <assert.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

enum ddl_kind {
  DDL_CONST = 1,
  DDL_LOGATOM,
  DDL_ADD,
  DDL_SUB,
  DDL_MUL,
  DDL_DIV,
  DDL_NEG
};

struct ddl_expr {
  int refcount;
  enum ddl_kind kind;
  int has_q;
  int has_numden;
  mpq_t q;
  mpz_t num;
  mpz_t den;
  struct ddl_expr *a;
  struct ddl_expr *b;
};

static void ddl_expr_incref(ddl_expr *e)
{
  if (e != NULL) e->refcount++;
}

static void ddl_expr_decref(ddl_expr *e)
{
  if (e == NULL) return;
  e->refcount--;
  if (e->refcount > 0) return;
  ddl_expr_decref(e->a);
  ddl_expr_decref(e->b);
  if (e->has_q) mpq_clear(e->q);
  if (e->has_numden) {
    mpz_clear(e->num);
    mpz_clear(e->den);
  }
  free(e);
}

static ddl_expr *ddl_expr_alloc(enum ddl_kind kind)
{
  ddl_expr *e = (ddl_expr *) calloc(1, sizeof(ddl_expr));
  if (e == NULL) {
    fprintf(stderr, "cddlogarithmic: out of memory\n");
    abort();
  }
  e->refcount = 1;
  e->kind = kind;
  e->has_q = 0;
  e->has_numden = 0;
  e->a = NULL;
  e->b = NULL;
  return e;
}

static ddl_expr *ddl_expr_const_q(const mpq_t q)
{
  ddl_expr *e = ddl_expr_alloc(DDL_CONST);
  e->has_q = 1;
  mpq_init(e->q);
  mpq_set(e->q, q);
  mpq_canonicalize(e->q);
  return e;
}

static ddl_expr *ddl_expr_const_si(signed long v)
{
  mpq_t q;
  ddl_expr *e;
  mpq_init(q);
  mpq_set_si(q, v, 1UL);
  e = ddl_expr_const_q(q);
  mpq_clear(q);
  return e;
}

static int ddl_expr_is_const(const ddl_expr *e)
{
  return e != NULL && e->kind == DDL_CONST;
}

static int ddl_expr_is_zero(const ddl_expr *e)
{
  return ddl_expr_is_const(e) && mpq_sgn(e->q) == 0;
}

static int ddl_expr_is_one(const ddl_expr *e)
{
  return ddl_expr_is_const(e) && mpq_cmp_si(e->q, 1L, 1UL) == 0;
}

static int ddl_expr_is_minus_one(const ddl_expr *e)
{
  return ddl_expr_is_const(e) && mpq_cmp_si(e->q, -1L, 1UL) == 0;
}

static int ddl_expr_same(const ddl_expr *x, const ddl_expr *y)
{
  if (x == y) return 1;
  if (x == NULL || y == NULL) return 0;
  if (x->kind != y->kind) return 0;
  switch (x->kind) {
    case DDL_CONST:
      return mpq_cmp(x->q, y->q) == 0;
    case DDL_LOGATOM:
      return mpq_cmp(x->q, y->q) == 0 && mpz_cmp(x->num, y->num) == 0 && mpz_cmp(x->den, y->den) == 0;
    case DDL_NEG:
      return ddl_expr_same(x->a, y->a);
    case DDL_ADD:
    case DDL_SUB:
    case DDL_MUL:
    case DDL_DIV:
      return ddl_expr_same(x->a, y->a) && ddl_expr_same(x->b, y->b);
    default:
      return 0;
  }
}

static ddl_expr *ddl_expr_alias(ddl_expr *e)
{
  ddl_expr_incref(e);
  return e;
}

static int ddl_expr_is_neg_of(const ddl_expr *x, const ddl_expr *y)
{
  if (x == NULL || y == NULL) return 0;
  if (x->kind == DDL_NEG) return ddl_expr_same(x->a, y);
  if (y->kind == DDL_NEG) return ddl_expr_same(y->a, x);
  return 0;
}

static ddl_expr *ddl_expr_logatom_q(const mpq_t coeff, const mpz_t num_in, const mpz_t den_in)
{
  ddl_expr *e;
  mpz_t num, den, g;
  int sign;

  if (mpq_sgn(coeff) == 0) return ddl_expr_const_si(0);

  mpz_init(num);
  mpz_init(den);
  mpz_init(g);
  mpz_set(num, num_in);
  mpz_set(den, den_in);
  sign = mpz_sgn(num) * mpz_sgn(den);
  if (sign <= 0) {
    fprintf(stderr, "cddlogarithmic: log(x/y) requires positive rational x/y\n");
    abort();
  }
  mpz_abs(num, num);
  mpz_abs(den, den);
  mpz_gcd(g, num, den);
  if (mpz_cmp_ui(g, 1UL) > 0) {
    mpz_divexact(num, num, g);
    mpz_divexact(den, den, g);
  }
  if (mpz_cmp(num, den) == 0) {
    mpz_clear(num);
    mpz_clear(den);
    mpz_clear(g);
    return ddl_expr_const_si(0);
  }

  e = ddl_expr_alloc(DDL_LOGATOM);
  e->has_q = 1;
  mpq_init(e->q);
  mpq_set(e->q, coeff);
  mpq_canonicalize(e->q);
  e->has_numden = 1;
  mpz_init(e->num);
  mpz_init(e->den);
  mpz_set(e->num, num);
  mpz_set(e->den, den);

  mpz_clear(num);
  mpz_clear(den);
  mpz_clear(g);
  return e;
}

static ddl_expr *ddl_expr_logatom_si(signed long coeff, const mpz_t num, const mpz_t den)
{
  mpq_t q;
  ddl_expr *e;
  mpq_init(q);
  mpq_set_si(q, coeff, 1UL);
  e = ddl_expr_logatom_q(q, num, den);
  mpq_clear(q);
  return e;
}

static ddl_expr *ddl_expr_neg(ddl_expr *x);
static ddl_expr *ddl_expr_add(ddl_expr *x, ddl_expr *y);
static ddl_expr *ddl_expr_sub(ddl_expr *x, ddl_expr *y);
static ddl_expr *ddl_expr_mul(ddl_expr *x, ddl_expr *y);
static ddl_expr *ddl_expr_div(ddl_expr *x, ddl_expr *y);

static ddl_expr *ddl_expr_neg(ddl_expr *x)
{
  ddl_expr *e;
  if (x == NULL) return NULL;
  if (ddl_expr_is_zero(x)) return ddl_expr_alias(x);
  if (x->kind == DDL_CONST) {
    mpq_t q;
    mpq_init(q);
    mpq_neg(q, x->q);
    e = ddl_expr_const_q(q);
    mpq_clear(q);
    return e;
  }
  if (x->kind == DDL_LOGATOM) {
    mpq_t q;
    mpq_init(q);
    mpq_neg(q, x->q);
    e = ddl_expr_logatom_q(q, x->num, x->den);
    mpq_clear(q);
    return e;
  }
  if (x->kind == DDL_NEG) return ddl_expr_alias(x->a);
  if (x->kind == DDL_DIV && ddl_expr_is_const(x->a)) {
    ddl_expr *na = ddl_expr_neg(x->a);
    e = ddl_expr_div(na, x->b);
    ddl_expr_decref(na);
    return e;
  }
  if (x->kind == DDL_MUL && ddl_expr_is_const(x->a)) {
    ddl_expr *na = ddl_expr_neg(x->a);
    e = ddl_expr_mul(na, x->b);
    ddl_expr_decref(na);
    return e;
  }
  e = ddl_expr_alloc(DDL_NEG);
  e->a = ddl_expr_alias(x);
  return e;
}

static ddl_expr *ddl_expr_add(ddl_expr *x, ddl_expr *y)
{
  ddl_expr *e;
  if (ddl_expr_is_zero(x)) return ddl_expr_alias(y);
  if (ddl_expr_is_zero(y)) return ddl_expr_alias(x);
  if (ddl_expr_is_neg_of(x, y)) return ddl_expr_const_si(0);
  if (ddl_expr_is_const(x) && ddl_expr_is_const(y)) {
    mpq_t q;
    mpq_init(q);
    mpq_add(q, x->q, y->q);
    e = ddl_expr_const_q(q);
    mpq_clear(q);
    return e;
  }
  if (x->kind == DDL_LOGATOM && y->kind == DDL_LOGATOM &&
      mpz_cmp(x->num, y->num) == 0 && mpz_cmp(x->den, y->den) == 0) {
    mpq_t q;
    mpq_init(q);
    mpq_add(q, x->q, y->q);
    e = ddl_expr_logatom_q(q, x->num, x->den);
    mpq_clear(q);
    return e;
  }
  e = ddl_expr_alloc(DDL_ADD);
  e->a = ddl_expr_alias(x);
  e->b = ddl_expr_alias(y);
  return e;
}

static ddl_expr *ddl_expr_sub(ddl_expr *x, ddl_expr *y)
{
  ddl_expr *ny, *e;
  if (ddl_expr_same(x, y)) return ddl_expr_const_si(0);
  ny = ddl_expr_neg(y);
  e = ddl_expr_add(x, ny);
  ddl_expr_decref(ny);
  return e;
}

static ddl_expr *ddl_scale_logatom_const(const ddl_expr *atom, const ddl_expr *constant)
{
  ddl_expr *e;
  mpq_t q;
  mpq_init(q);
  mpq_mul(q, atom->q, constant->q);
  e = ddl_expr_logatom_q(q, atom->num, atom->den);
  mpq_clear(q);
  return e;
}

static ddl_expr *ddl_expr_mul(ddl_expr *x, ddl_expr *y)
{
  ddl_expr *e;
  if (ddl_expr_is_zero(x) || ddl_expr_is_zero(y)) return ddl_expr_const_si(0);
  if (ddl_expr_is_one(x)) return ddl_expr_alias(y);
  if (ddl_expr_is_one(y)) return ddl_expr_alias(x);
  if (ddl_expr_is_minus_one(x)) return ddl_expr_neg(y);
  if (ddl_expr_is_minus_one(y)) return ddl_expr_neg(x);
  if (ddl_expr_is_const(x) && ddl_expr_is_const(y)) {
    mpq_t q;
    mpq_init(q);
    mpq_mul(q, x->q, y->q);
    e = ddl_expr_const_q(q);
    mpq_clear(q);
    return e;
  }
  if (x->kind == DDL_LOGATOM && ddl_expr_is_const(y)) return ddl_scale_logatom_const(x, y);
  if (y->kind == DDL_LOGATOM && ddl_expr_is_const(x)) return ddl_scale_logatom_const(y, x);
  if (y->kind == DDL_DIV && ddl_expr_is_const(y->a) && ddl_expr_same(x, y->b)) {
    return ddl_expr_alias(y->a);
  }
  if (x->kind == DDL_DIV && ddl_expr_is_const(x->a) && ddl_expr_same(y, x->b)) {
    return ddl_expr_alias(x->a);
  }

  e = ddl_expr_alloc(DDL_MUL);
  e->a = ddl_expr_alias(x);
  e->b = ddl_expr_alias(y);
  return e;
}

static ddl_expr *ddl_expr_div(ddl_expr *x, ddl_expr *y)
{
  ddl_expr *e;
  if (ddl_expr_is_zero(y)) {
    fprintf(stderr, "cddlogarithmic: division by zero expression\n");
    abort();
  }
  if (ddl_expr_is_zero(x)) return ddl_expr_const_si(0);
  if (ddl_expr_is_one(y)) return ddl_expr_alias(x);
  if (ddl_expr_is_minus_one(y)) return ddl_expr_neg(x);
  if (ddl_expr_is_const(x) && ddl_expr_is_const(y)) {
    mpq_t q;
    mpq_init(q);
    mpq_div(q, x->q, y->q);
    e = ddl_expr_const_q(q);
    mpq_clear(q);
    return e;
  }
  if (ddl_expr_is_const(x) && y->kind == DDL_DIV && ddl_expr_is_const(y->a)) {
    ddl_expr *num = ddl_expr_mul(x, y->b);
    e = ddl_expr_div(num, y->a);
    ddl_expr_decref(num);
    return e;
  }
  if (x->kind == DDL_LOGATOM && ddl_expr_is_const(y)) {
    mpq_t q;
    mpq_init(q);
    mpq_div(q, x->q, y->q);
    e = ddl_expr_logatom_q(q, x->num, x->den);
    mpq_clear(q);
    return e;
  }
  e = ddl_expr_alloc(DDL_DIV);
  e->a = ddl_expr_alias(x);
  e->b = ddl_expr_alias(y);
  return e;
}

static void ddl_assign_expr(mytype dst, ddl_expr *src)
{
  ddl_expr_decref(dst[0].ptr);
  dst[0].ptr = src;
}

void ddl_init(mytype a)
{
  a[0].ptr = ddl_expr_const_si(0);
}

void ddl_clear(mytype a)
{
  ddl_expr_decref(a[0].ptr);
  a[0].ptr = NULL;
}

void ddl_set(mytype a, mytype b)
{
  ddl_assign_expr(a, ddl_expr_alias(b[0].ptr));
}

void ddl_set_d(mytype a, double b)
{
  mpq_t q;
  mpq_init(q);
  mpq_set_d(q, b);
  ddl_assign_expr(a, ddl_expr_const_q(q));
  mpq_clear(q);
}

void ddl_set_si(mytype a, signed long b)
{
  ddl_assign_expr(a, ddl_expr_const_si(b));
}

void ddl_set_si2(mytype a, signed long b, unsigned long c)
{
  mpq_t q;
  mpq_init(q);
  mpq_set_si(q, b, c);
  ddl_assign_expr(a, ddl_expr_const_q(q));
  mpq_clear(q);
}

void ddl_add(mytype a, mytype b, mytype c)
{
  ddl_assign_expr(a, ddl_expr_add(b[0].ptr, c[0].ptr));
}

void ddl_sub(mytype a, mytype b, mytype c)
{
  ddl_assign_expr(a, ddl_expr_sub(b[0].ptr, c[0].ptr));
}

void ddl_mul(mytype a, mytype b, mytype c)
{
  ddl_assign_expr(a, ddl_expr_mul(b[0].ptr, c[0].ptr));
}

void ddl_div(mytype a, mytype b, mytype c)
{
  ddl_assign_expr(a, ddl_expr_div(b[0].ptr, c[0].ptr));
}

void ddl_neg(mytype a, mytype b)
{
  ddl_assign_expr(a, ddl_expr_neg(b[0].ptr));
}

void ddl_inv(mytype a, mytype b)
{
  mytype one;
  ddl_init(one);
  ddl_set_si(one, 1L);
  ddl_assign_expr(a, ddl_expr_div(one[0].ptr, b[0].ptr));
  ddl_clear(one);
}

static int ddl_eval_ld_expr(const ddl_expr *e, long double *out)
{
  long double av = 0.0L, bv = 0.0L;
  switch (e->kind) {
    case DDL_CONST:
      *out = (long double) mpq_get_d(e->q);
      return 1;
    case DDL_LOGATOM:
      *out = (long double) mpq_get_d(e->q) *
             (logl((long double) mpz_get_d(e->num)) - logl((long double) mpz_get_d(e->den)));
      return isfinite((double) *out);
    case DDL_NEG:
      if (!ddl_eval_ld_expr(e->a, &av)) return 0;
      *out = -av;
      return 1;
    case DDL_ADD:
      if (!ddl_eval_ld_expr(e->a, &av) || !ddl_eval_ld_expr(e->b, &bv)) return 0;
      *out = av + bv;
      return 1;
    case DDL_SUB:
      if (!ddl_eval_ld_expr(e->a, &av) || !ddl_eval_ld_expr(e->b, &bv)) return 0;
      *out = av - bv;
      return 1;
    case DDL_MUL:
      if (!ddl_eval_ld_expr(e->a, &av) || !ddl_eval_ld_expr(e->b, &bv)) return 0;
      *out = av * bv;
      return 1;
    case DDL_DIV:
      if (!ddl_eval_ld_expr(e->a, &av) || !ddl_eval_ld_expr(e->b, &bv)) return 0;
      if (fabsl(bv) < LDBL_MIN) return 0;
      *out = av / bv;
      return 1;
    default:
      return 0;
  }
}


typedef struct ddl_atom_id ddl_atom_id;
typedef struct ddl_poly_term ddl_poly_term;

typedef struct {
  size_t n;
  int *ids;
  unsigned long *exp;
} ddl_monom;

struct ddl_poly_term {
  mpq_t coeff;
  ddl_monom mono;
  ddl_poly_term *next;
};

typedef struct {
  ddl_poly_term *head;
} ddl_poly;

typedef struct {
  ddl_poly num;
  ddl_poly den;
} ddl_rfunc;

struct ddl_atom_id {
  int id;
  mpz_t num;
  mpz_t den;
  ddl_atom_id *next;
};

typedef struct {
  int next_id;
  ddl_atom_id *head;
} ddl_atom_table;

static void ddl_monom_init_identity(ddl_monom *m)
{
  m->n = 0;
  m->ids = NULL;
  m->exp = NULL;
}

static void ddl_monom_clear(ddl_monom *m)
{
  free(m->ids);
  free(m->exp);
  m->ids = NULL;
  m->exp = NULL;
  m->n = 0;
}

static void ddl_monom_copy(ddl_monom *dst, const ddl_monom *src)
{
  size_t i;
  dst->n = src->n;
  if (src->n == 0) {
    dst->ids = NULL;
    dst->exp = NULL;
    return;
  }
  dst->ids = (int *) calloc(src->n, sizeof(int));
  dst->exp = (unsigned long *) calloc(src->n, sizeof(unsigned long));
  if (dst->ids == NULL || dst->exp == NULL) {
    fprintf(stderr, "cddlogarithmic: out of memory\n");
    abort();
  }
  for (i = 0; i < src->n; ++i) {
    dst->ids[i] = src->ids[i];
    dst->exp[i] = src->exp[i];
  }
}

static int ddl_monom_equal(const ddl_monom *a, const ddl_monom *b)
{
  size_t i;
  if (a->n != b->n) return 0;
  for (i = 0; i < a->n; ++i) {
    if (a->ids[i] != b->ids[i] || a->exp[i] != b->exp[i]) return 0;
  }
  return 1;
}

static void ddl_monom_init_atom(ddl_monom *m, int id)
{
  m->n = 1;
  m->ids = (int *) calloc(1, sizeof(int));
  m->exp = (unsigned long *) calloc(1, sizeof(unsigned long));
  if (m->ids == NULL || m->exp == NULL) {
    fprintf(stderr, "cddlogarithmic: out of memory\n");
    abort();
  }
  m->ids[0] = id;
  m->exp[0] = 1UL;
}

static void ddl_monom_mul(ddl_monom *dst, const ddl_monom *a, const ddl_monom *b)
{
  size_t i = 0, j = 0, k = 0;
  dst->ids = (int *) calloc(a->n + b->n, sizeof(int));
  dst->exp = (unsigned long *) calloc(a->n + b->n, sizeof(unsigned long));
  if (dst->ids == NULL || dst->exp == NULL) {
    fprintf(stderr, "cddlogarithmic: out of memory\n");
    abort();
  }
  while (i < a->n || j < b->n) {
    if (j >= b->n || (i < a->n && a->ids[i] < b->ids[j])) {
      dst->ids[k] = a->ids[i];
      dst->exp[k] = a->exp[i];
      ++i;
    } else if (i >= a->n || b->ids[j] < a->ids[i]) {
      dst->ids[k] = b->ids[j];
      dst->exp[k] = b->exp[j];
      ++j;
    } else {
      dst->ids[k] = a->ids[i];
      dst->exp[k] = a->exp[i] + b->exp[j];
      ++i;
      ++j;
    }
    ++k;
  }
  dst->n = k;
}

static void ddl_poly_init(ddl_poly *p)
{
  p->head = NULL;
}

static void ddl_poly_clear(ddl_poly *p)
{
  ddl_poly_term *t = p->head;
  while (t != NULL) {
    ddl_poly_term *next = t->next;
    mpq_clear(t->coeff);
    ddl_monom_clear(&t->mono);
    free(t);
    t = next;
  }
  p->head = NULL;
}

static void ddl_poly_add_term(ddl_poly *p, const mpq_t coeff, const ddl_monom *mono)
{
  ddl_poly_term *t, *prev = NULL;
  if (mpq_sgn(coeff) == 0) return;
  for (t = p->head; t != NULL; prev = t, t = t->next) {
    if (ddl_monom_equal(&t->mono, mono)) {
      mpq_add(t->coeff, t->coeff, coeff);
      if (mpq_sgn(t->coeff) == 0) {
        if (prev != NULL) prev->next = t->next;
        else p->head = t->next;
        mpq_clear(t->coeff);
        ddl_monom_clear(&t->mono);
        free(t);
      }
      return;
    }
  }
  t = (ddl_poly_term *) calloc(1, sizeof(ddl_poly_term));
  if (t == NULL) {
    fprintf(stderr, "cddlogarithmic: out of memory\n");
    abort();
  }
  mpq_init(t->coeff);
  mpq_set(t->coeff, coeff);
  mpq_canonicalize(t->coeff);
  ddl_monom_copy(&t->mono, mono);
  t->next = p->head;
  p->head = t;
}

static void ddl_poly_set_const_q(ddl_poly *p, const mpq_t q)
{
  ddl_monom m;
  ddl_poly_init(p);
  ddl_monom_init_identity(&m);
  ddl_poly_add_term(p, q, &m);
  ddl_monom_clear(&m);
}

static void ddl_poly_copy(ddl_poly *dst, const ddl_poly *src)
{
  ddl_poly_term *t;
  ddl_poly_init(dst);
  for (t = src->head; t != NULL; t = t->next) ddl_poly_add_term(dst, t->coeff, &t->mono);
}

static void ddl_poly_add_poly(ddl_poly *dst, const ddl_poly *a, const ddl_poly *b, int subtract_b)
{
  ddl_poly_term *t;
  ddl_poly_init(dst);
  for (t = a->head; t != NULL; t = t->next) ddl_poly_add_term(dst, t->coeff, &t->mono);
  for (t = b->head; t != NULL; t = t->next) {
    mpq_t q;
    mpq_init(q);
    mpq_set(q, t->coeff);
    if (subtract_b) mpq_neg(q, q);
    ddl_poly_add_term(dst, q, &t->mono);
    mpq_clear(q);
  }
}

static void ddl_poly_mul_poly(ddl_poly *dst, const ddl_poly *a, const ddl_poly *b)
{
  ddl_poly_term *ta, *tb;
  ddl_poly_init(dst);
  for (ta = a->head; ta != NULL; ta = ta->next) {
    for (tb = b->head; tb != NULL; tb = tb->next) {
      ddl_monom m;
      mpq_t q;
      mpq_init(q);
      mpq_mul(q, ta->coeff, tb->coeff);
      ddl_monom_mul(&m, &ta->mono, &tb->mono);
      ddl_poly_add_term(dst, q, &m);
      ddl_monom_clear(&m);
      mpq_clear(q);
    }
  }
}

static void ddl_poly_negate(ddl_poly *dst, const ddl_poly *src)
{
  ddl_poly_term *t;
  ddl_poly_init(dst);
  for (t = src->head; t != NULL; t = t->next) {
    mpq_t q;
    mpq_init(q);
    mpq_neg(q, t->coeff);
    ddl_poly_add_term(dst, q, &t->mono);
    mpq_clear(q);
  }
}

static void ddl_atom_table_init(ddl_atom_table *tab)
{
  tab->next_id = 1;
  tab->head = NULL;
}

static void ddl_atom_table_clear(ddl_atom_table *tab)
{
  ddl_atom_id *a = tab->head;
  while (a != NULL) {
    ddl_atom_id *next = a->next;
    mpz_clear(a->num);
    mpz_clear(a->den);
    free(a);
    a = next;
  }
  tab->head = NULL;
}

static int ddl_atom_table_get_id(ddl_atom_table *tab, const mpz_t num, const mpz_t den)
{
  ddl_atom_id *a;
  for (a = tab->head; a != NULL; a = a->next) {
    if (mpz_cmp(a->num, num) == 0 && mpz_cmp(a->den, den) == 0) return a->id;
  }
  a = (ddl_atom_id *) calloc(1, sizeof(ddl_atom_id));
  if (a == NULL) {
    fprintf(stderr, "cddlogarithmic: out of memory\n");
    abort();
  }
  a->id = tab->next_id++;
  mpz_init(a->num);
  mpz_init(a->den);
  mpz_set(a->num, num);
  mpz_set(a->den, den);
  a->next = tab->head;
  tab->head = a;
  return a->id;
}

static void ddl_rfunc_init(ddl_rfunc *rf)
{
  ddl_poly_init(&rf->num);
  ddl_poly_init(&rf->den);
}

static void ddl_rfunc_clear(ddl_rfunc *rf)
{
  ddl_poly_clear(&rf->num);
  ddl_poly_clear(&rf->den);
}

static void ddl_rfunc_set_const_q(ddl_rfunc *rf, const mpq_t q)
{
  mpq_t one;
  ddl_rfunc_init(rf);
  mpq_init(one);
  mpq_set_si(one, 1L, 1UL);
  ddl_poly_set_const_q(&rf->num, q);
  ddl_poly_set_const_q(&rf->den, one);
  mpq_clear(one);
}

static void ddl_rfunc_set_atom(ddl_rfunc *rf, const mpq_t coeff, int atom_id)
{
  ddl_monom m;
  mpq_t one;
  ddl_rfunc_init(rf);
  ddl_monom_init_atom(&m, atom_id);
  ddl_poly_init(&rf->num);
  ddl_poly_add_term(&rf->num, coeff, &m);
  ddl_monom_clear(&m);
  mpq_init(one);
  mpq_set_si(one, 1L, 1UL);
  ddl_poly_set_const_q(&rf->den, one);
  mpq_clear(one);
}

static void ddl_rfunc_addsub(ddl_rfunc *dst, const ddl_rfunc *a, const ddl_rfunc *b, int subtract_b)
{
  ddl_poly ad, bc, num, den;
  ddl_poly_mul_poly(&ad, &a->num, &b->den);
  ddl_poly_mul_poly(&bc, &b->num, &a->den);
  ddl_poly_add_poly(&num, &ad, &bc, subtract_b);
  ddl_poly_mul_poly(&den, &a->den, &b->den);
  ddl_rfunc_init(dst);
  dst->num = num;
  dst->den = den;
  ddl_poly_clear(&ad);
  ddl_poly_clear(&bc);
}

static void ddl_rfunc_mul(ddl_rfunc *dst, const ddl_rfunc *a, const ddl_rfunc *b)
{
  ddl_poly num, den;
  ddl_poly_mul_poly(&num, &a->num, &b->num);
  ddl_poly_mul_poly(&den, &a->den, &b->den);
  ddl_rfunc_init(dst);
  dst->num = num;
  dst->den = den;
}

static void ddl_rfunc_div(ddl_rfunc *dst, const ddl_rfunc *a, const ddl_rfunc *b)
{
  ddl_poly num, den;
  ddl_poly_mul_poly(&num, &a->num, &b->den);
  ddl_poly_mul_poly(&den, &a->den, &b->num);
  ddl_rfunc_init(dst);
  dst->num = num;
  dst->den = den;
}

static int ddl_expr_to_rfunc(const ddl_expr *e, ddl_atom_table *tab, ddl_rfunc *rf)
{
  switch (e->kind) {
    case DDL_CONST:
      ddl_rfunc_set_const_q(rf, e->q);
      return 1;
    case DDL_LOGATOM: {
      int id = ddl_atom_table_get_id(tab, e->num, e->den);
      ddl_rfunc_set_atom(rf, e->q, id);
      return 1;
    }
    case DDL_NEG: {
      ddl_rfunc child;
      ddl_poly negnum;
      if (!ddl_expr_to_rfunc(e->a, tab, &child)) return 0;
      ddl_poly_negate(&negnum, &child.num);
      ddl_rfunc_init(rf);
      rf->num = negnum;
      rf->den = child.den;
      child.den.head = NULL;
      ddl_rfunc_clear(&child);
      return 1;
    }
    case DDL_ADD:
    case DDL_SUB:
    case DDL_MUL:
    case DDL_DIV: {
      ddl_rfunc a, b, out;
      if (!ddl_expr_to_rfunc(e->a, tab, &a)) return 0;
      if (!ddl_expr_to_rfunc(e->b, tab, &b)) { ddl_rfunc_clear(&a); return 0; }
      if (e->kind == DDL_ADD) ddl_rfunc_addsub(&out, &a, &b, 0);
      else if (e->kind == DDL_SUB) ddl_rfunc_addsub(&out, &a, &b, 1);
      else if (e->kind == DDL_MUL) ddl_rfunc_mul(&out, &a, &b);
      else ddl_rfunc_div(&out, &a, &b);
      ddl_rfunc_clear(&a);
      ddl_rfunc_clear(&b);
      *rf = out;
      return 1;
    }
    default:
      return 0;
  }
}

static int ddl_expr_is_zero_exact(const ddl_expr *e)
{
  ddl_atom_table tab;
  ddl_rfunc rf;
  int result;
  ddl_atom_table_init(&tab);
  if (!ddl_expr_to_rfunc(e, &tab, &rf)) {
    ddl_atom_table_clear(&tab);
    return 0;
  }
  result = (rf.num.head == NULL);
  ddl_rfunc_clear(&rf);
  ddl_atom_table_clear(&tab);
  return result;
}


static ddl_atom_id *ddl_atom_table_find(const ddl_atom_table *tab, int id)
{
  ddl_atom_id *a;
  for (a = tab->head; a != NULL; a = a->next) if (a->id == id) return a;
  return NULL;
}

static void ddl_monom_intersect_min(ddl_monom *dst, const ddl_monom *a, const ddl_monom *b)
{
  size_t i = 0, j = 0, k = 0;
  dst->ids = (int *) calloc((a->n < b->n ? a->n : b->n), sizeof(int));
  dst->exp = (unsigned long *) calloc((a->n < b->n ? a->n : b->n), sizeof(unsigned long));
  if ((a->n > 0 || b->n > 0) && (dst->ids == NULL || dst->exp == NULL)) {
    fprintf(stderr, "cddlogarithmic: out of memory\n");
    abort();
  }
  while (i < a->n && j < b->n) {
    if (a->ids[i] < b->ids[j]) {
      ++i;
    } else if (b->ids[j] < a->ids[i]) {
      ++j;
    } else {
      dst->ids[k] = a->ids[i];
      dst->exp[k] = (a->exp[i] < b->exp[j] ? a->exp[i] : b->exp[j]);
      ++i; ++j; ++k;
    }
  }
  dst->n = k;
}

static void ddl_monom_divide_by(ddl_monom *dst, const ddl_monom *src, const ddl_monom *factor)
{
  size_t i = 0, j = 0, k = 0;
  dst->ids = (int *) calloc(src->n, sizeof(int));
  dst->exp = (unsigned long *) calloc(src->n, sizeof(unsigned long));
  if (src->n > 0 && (dst->ids == NULL || dst->exp == NULL)) {
    fprintf(stderr, "cddlogarithmic: out of memory\n");
    abort();
  }
  while (i < src->n) {
    unsigned long sub = 0;
    if (j < factor->n && factor->ids[j] == src->ids[i]) {
      sub = factor->exp[j];
      ++j;
    }
    if (src->exp[i] > sub) {
      dst->ids[k] = src->ids[i];
      dst->exp[k] = src->exp[i] - sub;
      ++k;
    }
    ++i;
  }
  dst->n = k;
}

static void ddl_poly_common_monom(ddl_monom *out, const ddl_poly *p)
{
  ddl_poly_term *t;
  ddl_monom tmp;
  ddl_monom_init_identity(out);
  if (p->head == NULL) return;
  ddl_monom_copy(out, &p->head->mono);
  for (t = p->head->next; t != NULL; t = t->next) {
    ddl_monom_intersect_min(&tmp, out, &t->mono);
    ddl_monom_clear(out);
    *out = tmp;
  }
}

static void ddl_poly_divide_by_monom(ddl_poly *dst, const ddl_poly *src, const ddl_monom *factor)
{
  ddl_poly_term *t;
  ddl_poly_init(dst);
  for (t = src->head; t != NULL; t = t->next) {
    ddl_monom m;
    ddl_monom_divide_by(&m, &t->mono, factor);
    ddl_poly_add_term(dst, t->coeff, &m);
    ddl_monom_clear(&m);
  }
}

static void ddl_rfunc_cancel_common_monom(ddl_rfunc *rf)
{
  ddl_monom nfac, dfac, common;
  ddl_poly newnum, newden;
  if (rf->num.head == NULL || rf->den.head == NULL) return;
  ddl_poly_common_monom(&nfac, &rf->num);
  ddl_poly_common_monom(&dfac, &rf->den);
  ddl_monom_intersect_min(&common, &nfac, &dfac);
  ddl_monom_clear(&nfac);
  ddl_monom_clear(&dfac);
  if (common.n == 0) {
    ddl_monom_clear(&common);
    return;
  }
  ddl_poly_divide_by_monom(&newnum, &rf->num, &common);
  ddl_poly_divide_by_monom(&newden, &rf->den, &common);
  ddl_poly_clear(&rf->num);
  ddl_poly_clear(&rf->den);
  rf->num = newnum;
  rf->den = newden;
  ddl_monom_clear(&common);
}

static int ddl_poly_constant_term_q(const ddl_poly *p, mpq_t q)
{
  if (p->head == NULL || p->head->next != NULL || p->head->mono.n != 0) return 0;
  mpq_set(q, p->head->coeff);
  return 1;
}

static int ddl_poly_is_linear_form(const ddl_poly *p)
{
  ddl_poly_term *t;
  for (t = p->head; t != NULL; t = t->next) {
    if (t->mono.n == 0) continue;
    if (t->mono.n == 1 && t->mono.exp[0] == 1UL) continue;
    return 0;
  }
  return 1;
}

static int ddl_term_sort_key(const ddl_poly_term *t)
{
  if (t->mono.n == 0) return 0;
  return t->mono.ids[0] + 1;
}

static void ddl_fprint_logatom_term(FILE *f, const mpq_t coeff, const ddl_atom_id *atom, int first)
{
  int sgn = mpq_sgn(coeff);
  mpq_t absq;
  mpq_init(absq);
  mpq_abs(absq, coeff);
  if (!first) fprintf(f, sgn >= 0 ? "+" : "-");
  else if (sgn < 0) fprintf(f, "-");
  if (mpq_cmp_si(absq, 1L, 1UL) != 0) {
    mpq_out_str(f, 10, absq);
    fprintf(f, "*");
  }
  fprintf(f, "log(");
  mpz_out_str(f, 10, atom->num);
  fprintf(f, "/");
  mpz_out_str(f, 10, atom->den);
  fprintf(f, ")");
  mpq_clear(absq);
}

static void ddl_fprint_const_term(FILE *f, const mpq_t coeff, int first)
{
  int sgn = mpq_sgn(coeff);
  mpq_t absq;
  mpq_init(absq);
  mpq_abs(absq, coeff);
  if (!first) fprintf(f, sgn >= 0 ? "+" : "-");
  else if (sgn < 0) fprintf(f, "-");
  mpq_out_str(f, 10, absq);
  mpq_clear(absq);
}

static int ddl_fprint_linear_from_rf(FILE *f, const ddl_rfunc *rf, const ddl_atom_table *tab);

static int ddl_fprint_linear_from_rfunc(FILE *f, const ddl_expr *e)
{
  ddl_atom_table tab;
  ddl_rfunc rf;
  int ok;
  ddl_atom_table_init(&tab);
  if (!ddl_expr_to_rfunc(e, &tab, &rf)) {
    ddl_atom_table_clear(&tab);
    return 0;
  }
  ddl_rfunc_cancel_common_monom(&rf);
  ok = ddl_fprint_linear_from_rf(f, &rf, &tab);
  ddl_rfunc_clear(&rf);
  ddl_atom_table_clear(&tab);
  return ok;
}


static int ddl_poly_equal(const ddl_poly *a, const ddl_poly *b)
{
  ddl_poly_term *ta;
  int count_a = 0, count_b = 0;
  for (ta = a->head; ta != NULL; ta = ta->next) {
    ddl_poly_term *tb;
    int found = 0;
    ++count_a;
    for (tb = b->head; tb != NULL; tb = tb->next) {
      if (ddl_monom_equal(&ta->mono, &tb->mono) && mpq_cmp(ta->coeff, tb->coeff) == 0) {
        found = 1;
        break;
      }
    }
    if (!found) return 0;
  }
  for (ta = b->head; ta != NULL; ta = ta->next) ++count_b;
  return count_a == count_b;
}

static int ddl_rfunc_is_zero(const ddl_rfunc *rf)
{
  return rf->num.head == NULL;
}

static int ddl_rfunc_is_rational_const(const ddl_rfunc *rf, mpq_t q)
{
  mpq_t nq, dq;
  if (ddl_rfunc_is_zero(rf)) {
    mpq_set_si(q, 0L, 1UL);
    return 1;
  }
  mpq_init(nq); mpq_init(dq);
  if (!ddl_poly_constant_term_q(&rf->num, nq) || !ddl_poly_constant_term_q(&rf->den, dq) || mpq_sgn(dq) == 0) {
    mpq_clear(nq); mpq_clear(dq);
    return 0;
  }
  mpq_div(q, nq, dq);
  mpq_clear(nq); mpq_clear(dq);
  return 1;
}

static void ddl_rfunc_mul_by_poly(ddl_rfunc *dst, const ddl_rfunc *src, const ddl_poly *poly)
{
  ddl_poly num, den;
  ddl_poly_mul_poly(&num, &src->num, poly);
  ddl_poly_copy(&den, &src->den);
  ddl_rfunc_init(dst);
  dst->num = num;
  dst->den = den;
}

static int ddl_fprint_linear_from_rf(FILE *f, const ddl_rfunc *rf, const ddl_atom_table *tab)
{
  ddl_poly_term *t;
  mpq_t denq, termq;
  int printed = 0;
  mpq_init(denq);
  if (!ddl_poly_constant_term_q(&rf->den, denq) || mpq_sgn(denq) == 0 || !ddl_poly_is_linear_form(&rf->num)) {
    mpq_clear(denq);
    return 0;
  }
  mpq_init(termq);
  for (t = rf->num.head; t != NULL; t = t->next) {
    if (t->mono.n == 0) {
      mpq_div(termq, t->coeff, denq);
      ddl_fprint_const_term(f, termq, !printed);
      printed = 1;
    }
  }
  for (t = rf->num.head; t != NULL; t = t->next) {
    if (t->mono.n == 1 && t->mono.exp[0] == 1UL) {
      ddl_atom_id *atom = ddl_atom_table_find(tab, t->mono.ids[0]);
      if (atom == NULL) continue;
      mpq_div(termq, t->coeff, denq);
      ddl_fprint_logatom_term(f, termq, atom, !printed);
      printed = 1;
    }
  }
  if (!printed) fprintf(f, "0");
  mpq_clear(termq);
  mpq_clear(denq);
  return 1;
}

int ddl_try_fprint_hrep_row(FILE *f, mytype *row, long d)
{
  ddl_atom_table tab;
  ddl_rfunc *rf = NULL;
  ddl_poly *scale = NULL;
  ddl_rfunc tmp;
  mpq_t q;
  long j;
  int ok = 1;
  ddl_atom_table_init(&tab);
  rf = (ddl_rfunc *) calloc((size_t)d, sizeof(ddl_rfunc));
  if (rf == NULL) {
    ddl_atom_table_clear(&tab);
    return 0;
  }
  for (j = 0; j < d; ++j) {
    if (!ddl_expr_to_rfunc(row[j][0].ptr, &tab, &rf[j])) { ok = 0; goto done; }
    ddl_rfunc_cancel_common_monom(&rf[j]);
  }
  scale = NULL;
  mpq_init(q);
  for (j = 1; j < d; ++j) {
    if (ddl_rfunc_is_rational_const(&rf[j], q)) continue;
    if (!ddl_poly_constant_term_q(&rf[j].num, q)) { ok = 0; goto done; }
    if (scale == NULL) {
      scale = (ddl_poly *) calloc(1, sizeof(ddl_poly));
      if (scale == NULL) { ok = 0; goto done; }
      ddl_poly_copy(scale, &rf[j].den);
    } else if (!ddl_poly_equal(scale, &rf[j].den)) {
      ok = 0;
      goto done;
    }
  }
  if (scale != NULL) {
    for (j = 0; j < d; ++j) {
      ddl_rfunc_mul_by_poly(&tmp, &rf[j], scale);
      ddl_rfunc_cancel_common_monom(&tmp);
      ddl_rfunc_clear(&rf[j]);
      rf[j] = tmp;
    }
  }
  if (!ddl_fprint_linear_from_rf(f, &rf[0], &tab)) { ok = 0; goto done; }
  fprintf(f, " ");
  for (j = 1; j < d; ++j) {
    if (!ddl_rfunc_is_rational_const(&rf[j], q)) { ok = 0; goto done; }
    mpq_out_str(f, 10, q);
    fprintf(f, " ");
  }
  fprintf(f, "\n");
  ok = 1;
done:
  mpq_clear(q);
  if (scale != NULL) { ddl_poly_clear(scale); free(scale); }
  if (rf != NULL) {
    for (j = 0; j < d; ++j) ddl_rfunc_clear(&rf[j]);
    free(rf);
  }
  ddl_atom_table_clear(&tab);
  return ok;
}

static void ddl_print_expr_prec(FILE *f, const ddl_expr *e, int parent_prec)
{
  int my_prec = 100;
  switch (e->kind) {
    case DDL_CONST:
      mpq_out_str(f, 10, e->q);
      return;
    case DDL_LOGATOM:
      if (mpq_cmp_si(e->q, -1L, 1UL) == 0) {
        fprintf(f, "-");
      } else if (mpq_cmp_si(e->q, 1L, 1UL) != 0) {
        mpq_out_str(f, 10, e->q);
        fprintf(f, "*");
      }
      fprintf(f, "log(");
      mpz_out_str(f, 10, e->num);
      fprintf(f, "/");
      mpz_out_str(f, 10, e->den);
      fprintf(f, ")");
      return;
    case DDL_NEG:
      my_prec = 30;
      if (my_prec < parent_prec) fprintf(f, "(");
      fprintf(f, "-");
      ddl_print_expr_prec(f, e->a, my_prec);
      if (my_prec < parent_prec) fprintf(f, ")");
      return;
    case DDL_ADD:
    case DDL_SUB:
      my_prec = 10;
      break;
    case DDL_MUL:
    case DDL_DIV:
      my_prec = 20;
      break;
    default:
      break;
  }
  if (my_prec < parent_prec) fprintf(f, "(");
  ddl_print_expr_prec(f, e->a, my_prec);
  switch (e->kind) {
    case DDL_ADD: fprintf(f, "+"); break;
    case DDL_SUB: fprintf(f, "-"); break;
    case DDL_MUL: fprintf(f, "*"); break;
    case DDL_DIV: fprintf(f, "/"); break;
    default: break;
  }
  ddl_print_expr_prec(f, e->b, my_prec + (e->kind == DDL_SUB || e->kind == DDL_DIV));
  if (my_prec < parent_prec) fprintf(f, ")");
}

void ddl_fprint(FILE *f, mytype x)
{
  if (ddl_fprint_linear_from_rfunc(f, x[0].ptr)) return;
  ddl_print_expr_prec(f, x[0].ptr, 0);
}

static void ddl_abort_unresolved(const char *what, const ddl_expr *e)
{
  fprintf(stderr, "cddlogarithmic: unable to resolve %s for expression: ", what);
  ddl_print_expr_prec(stderr, e, 0);
  fprintf(stderr, "\n");
  abort();
}

int ddl_sgn(mytype a)
{
  long double v, tol;
  if (ddl_expr_is_const(a[0].ptr)) return mpq_sgn(a[0].ptr->q);
  if (!ddl_eval_ld_expr(a[0].ptr, &v)) ddl_abort_unresolved("sign", a[0].ptr);
  tol = 1024.0L * LDBL_EPSILON * fmaxl(1.0L, fabsl(v));
  if (v > tol) return 1;
  if (v < -tol) return -1;
  if (ddl_expr_is_zero(a[0].ptr) || ddl_expr_is_zero_exact(a[0].ptr)) return 0;
  ddl_abort_unresolved("sign", a[0].ptr);
  return 0;
}

int ddl_cmp(mytype a, mytype b)
{
  mytype t;
  int s;
  ddl_init(t);
  ddl_sub(t, a, b);
  s = ddl_sgn(t);
  ddl_clear(t);
  return s;
}

double ddl_get_d(mytype a)
{
  long double v;
  if (!ddl_eval_ld_expr(a[0].ptr, &v)) ddl_abort_unresolved("double conversion", a[0].ptr);
  return (double) v;
}

int ddl_is_const(mytype a)
{
  return a[0].ptr != NULL && a[0].ptr->kind == DDL_CONST;
}

struct ddl_parser {
  const char *s;
  size_t pos;
  int err;
};

static void ddl_parser_skip_ws(struct ddl_parser *p)
{
  while (isspace((unsigned char) p->s[p->pos])) p->pos++;
}

static int ddl_parser_peek(struct ddl_parser *p)
{
  ddl_parser_skip_ws(p);
  return (unsigned char) p->s[p->pos];
}

static int ddl_parser_match(struct ddl_parser *p, char c)
{
  ddl_parser_skip_ws(p);
  if (p->s[p->pos] == c) {
    p->pos++;
    return 1;
  }
  return 0;
}

static ddl_expr *ddl_parse_expr(struct ddl_parser *p);

static ddl_expr *ddl_parse_rational_literal(struct ddl_parser *p)
{
  size_t start, len;
  char buf[dd_wordlenmax];
  mpq_t q;
  ddl_expr *e;
  ddl_parser_skip_ws(p);
  start = p->pos;
  if (!isdigit((unsigned char) p->s[p->pos])) {
    p->err = 1;
    return NULL;
  }
  while (isdigit((unsigned char) p->s[p->pos])) p->pos++;
  if (p->s[p->pos] == '/') {
    p->pos++;
    if (!isdigit((unsigned char) p->s[p->pos])) {
      p->err = 1;
      return NULL;
    }
    while (isdigit((unsigned char) p->s[p->pos])) p->pos++;
  }
  len = p->pos - start;
  if (len >= sizeof(buf)) {
    p->err = 1;
    return NULL;
  }
  memcpy(buf, p->s + start, len);
  buf[len] = '\0';
  mpq_init(q);
  if (mpq_set_str(q, buf, 10) != 0) {
    mpq_clear(q);
    p->err = 1;
    return NULL;
  }
  mpq_canonicalize(q);
  e = ddl_expr_const_q(q);
  mpq_clear(q);
  return e;
}

static ddl_expr *ddl_parse_factor(struct ddl_parser *p)
{
  ddl_expr *e = NULL, *inner = NULL;
  int c = ddl_parser_peek(p);
  if (c == '+') {
    p->pos++;
    return ddl_parse_factor(p);
  }
  if (c == '-') {
    p->pos++;
    inner = ddl_parse_factor(p);
    if (inner == NULL) return NULL;
    e = ddl_expr_neg(inner);
    ddl_expr_decref(inner);
    return e;
  }
  if (c == '(') {
    p->pos++;
    e = ddl_parse_expr(p);
    if (e == NULL || !ddl_parser_match(p, ')')) {
      p->err = 1;
      ddl_expr_decref(e);
      return NULL;
    }
    return e;
  }
  if (strncmp(p->s + p->pos, "log(", 4) == 0) {
    mpz_t num, den, g;
    p->pos += 4;
    ddl_parser_skip_ws(p);
    /* Restrict the exact log atom syntax to positive rational literals. */
    if (!isdigit((unsigned char) p->s[p->pos])) {
      p->err = 1;
      return NULL;
    }
    mpz_init(num); mpz_init(den); mpz_init(g);
    {
      size_t start = p->pos, len;
      char buf[dd_wordlenmax];
      while (isdigit((unsigned char) p->s[p->pos])) p->pos++;
      len = p->pos - start;
      if (len >= sizeof(buf)) { p->err = 1; goto log_fail; }
      memcpy(buf, p->s + start, len); buf[len] = '\0';
      mpz_set_str(num, buf, 10);
    }
    if (!ddl_parser_match(p, '/')) { p->err = 1; goto log_fail; }
    ddl_parser_skip_ws(p);
    if (!isdigit((unsigned char) p->s[p->pos])) { p->err = 1; goto log_fail; }
    {
      size_t start = p->pos, len;
      char buf[dd_wordlenmax];
      while (isdigit((unsigned char) p->s[p->pos])) p->pos++;
      len = p->pos - start;
      if (len >= sizeof(buf)) { p->err = 1; goto log_fail; }
      memcpy(buf, p->s + start, len); buf[len] = '\0';
      mpz_set_str(den, buf, 10);
    }
    if (!ddl_parser_match(p, ')')) { p->err = 1; goto log_fail; }
    if (mpz_sgn(num) <= 0 || mpz_sgn(den) <= 0) { p->err = 1; goto log_fail; }
    mpz_gcd(g, num, den);
    if (mpz_cmp_ui(g, 1UL) > 0) {
      mpz_divexact(num, num, g);
      mpz_divexact(den, den, g);
    }
    e = ddl_expr_logatom_si(1L, num, den);
log_fail:
    mpz_clear(num); mpz_clear(den); mpz_clear(g);
    return e;
  }
  return ddl_parse_rational_literal(p);
}

static ddl_expr *ddl_parse_term(struct ddl_parser *p)
{
  ddl_expr *lhs = ddl_parse_factor(p);
  while (!p->err) {
    ddl_expr *rhs = NULL, *next = NULL;
    int c = ddl_parser_peek(p);
    if (c != '*' && c != '/') break;
    p->pos++;
    rhs = ddl_parse_factor(p);
    if (rhs == NULL) {
      ddl_expr_decref(lhs);
      return NULL;
    }
    if (c == '*') next = ddl_expr_mul(lhs, rhs);
    else next = ddl_expr_div(lhs, rhs);
    ddl_expr_decref(lhs);
    ddl_expr_decref(rhs);
    lhs = next;
  }
  return lhs;
}

static ddl_expr *ddl_parse_expr(struct ddl_parser *p)
{
  ddl_expr *lhs = ddl_parse_term(p);
  while (!p->err) {
    ddl_expr *rhs = NULL, *next = NULL;
    int c = ddl_parser_peek(p);
    if (c != '+' && c != '-') break;
    p->pos++;
    rhs = ddl_parse_term(p);
    if (rhs == NULL) {
      ddl_expr_decref(lhs);
      return NULL;
    }
    if (c == '+') next = ddl_expr_add(lhs, rhs);
    else next = ddl_expr_sub(lhs, rhs);
    ddl_expr_decref(lhs);
    ddl_expr_decref(rhs);
    lhs = next;
  }
  return lhs;
}

int ddl_parse_expression(mytype value, const char *s)
{
  struct ddl_parser p;
  ddl_expr *e;
  p.s = s;
  p.pos = 0;
  p.err = 0;
  e = ddl_parse_expr(&p);
  ddl_parser_skip_ws(&p);
  if (p.err || e == NULL || p.s[p.pos] != '\0') {
    ddl_expr_decref(e);
    return -1;
  }
  ddl_assign_expr(value, e);
  return 0;
}

#else
/* Keep an object file for the non-logarithmic builds. */
int cddlogarithmic_unused_translation_unit;
#endif
