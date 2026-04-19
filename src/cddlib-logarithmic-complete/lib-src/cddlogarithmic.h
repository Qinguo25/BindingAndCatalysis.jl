#ifndef __CDDLOGARITHMIC_H
#define __CDDLOGARITHMIC_H

#if defined(CDDLOGARITHMIC)

#include <stdio.h>
#include <gmp.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct ddl_expr ddl_expr;
typedef struct ddl_cell { ddl_expr *ptr; } mytype[1];

void ddl_init(mytype a);
void ddl_clear(mytype a);
void ddl_set(mytype a, mytype b);
void ddl_set_d(mytype a, double b);
void ddl_set_si(mytype a, signed long b);
void ddl_set_si2(mytype a, signed long b, unsigned long c);
void ddl_add(mytype a, mytype b, mytype c);
void ddl_sub(mytype a, mytype b, mytype c);
void ddl_mul(mytype a, mytype b, mytype c);
void ddl_div(mytype a, mytype b, mytype c);
void ddl_neg(mytype a, mytype b);
void ddl_inv(mytype a, mytype b);
int ddl_cmp(mytype a, mytype b);
int ddl_sgn(mytype a);
double ddl_get_d(mytype a);
int ddl_is_const(mytype a);

int ddl_parse_expression(mytype value, const char *s);
void ddl_fprint(FILE *f, mytype x);
int ddl_try_fprint_hrep_row(FILE *f, mytype *row, long d);

#if defined(__cplusplus)
}
#endif

#endif /* CDDLOGARITHMIC */

#endif /* __CDDLOGARITHMIC_H */
