#include "setoper.h"
#include "cdd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *msg)
{
  fprintf(stderr, "testlogarithmic: %s\n", msg);
  exit(1);
}

static void set_expr(mytype dst, const char *expr)
{
  dd_sread_rational_value(expr, dst);
}

static int equal_expr(mytype lhs, const char *expr)
{
  mytype rhs;
  int ok;
  dd_init(rhs);
  set_expr(rhs, expr);
  ok = dd_Equal(lhs, rhs);
  dd_clear(rhs);
  return ok;
}

int main(void)
{
  dd_ErrorType err = dd_NoError;
  dd_MatrixPtr M = NULL, G = NULL, Mround = NULL, Mlp = NULL;
  dd_PolyhedraPtr poly = NULL;
  dd_LPPtr lp = NULL;
  FILE *tmp = NULL;
  int found_zero = 0, found_log2 = 0;
  dd_rowrange i;

  dd_set_global_constants();

  M = dd_CreateMatrix(2, 2);
  M->representation = dd_Inequality;
  M->numbtype = dd_Logarithmic;

  set_expr(M->matrix[0][0], "0");
  set_expr(M->matrix[0][1], "1");
  set_expr(M->matrix[1][0], "log(2/1)");
  set_expr(M->matrix[1][1], "-1");

  poly = dd_DDMatrix2Poly(M, &err);
  if (err != dd_NoError || poly == NULL) fail("dd_DDMatrix2Poly failed");

  G = dd_CopyGenerators(poly);
  if (G == NULL || G->rowsize != 2 || G->colsize != 2) fail("unexpected generator matrix size");

  for (i = 0; i < G->rowsize; ++i) {
    if (!equal_expr(G->matrix[i][0], "1")) continue;
    if (equal_expr(G->matrix[i][1], "0")) found_zero = 1;
    if (equal_expr(G->matrix[i][1], "log(2/1)")) found_log2 = 1;
  }
  if (!found_zero || !found_log2) fail("generator vertices do not match {0, log(2)}");

  tmp = fopen("/tmp/cddlog_roundtrip.ext", "w+");
  if (tmp == NULL) fail("cannot open roundtrip file");
  dd_WriteMatrix(tmp, G);
  rewind(tmp);
  Mround = dd_PolyFile2Matrix(tmp, &err);
  fclose(tmp);
  tmp = NULL;
  if (err != dd_NoError || Mround == NULL) fail("roundtrip read failed");
  if (Mround->rowsize != G->rowsize || Mround->colsize != G->colsize) fail("roundtrip size mismatch");

  Mlp = dd_CreateMatrix(2, 2);
  Mlp->representation = dd_Inequality;
  Mlp->numbtype = dd_Logarithmic;
  Mlp->objective = dd_LPmax;
  set_expr(Mlp->matrix[0][0], "0");
  set_expr(Mlp->matrix[0][1], "1");
  set_expr(Mlp->matrix[1][0], "log(2/1)");
  set_expr(Mlp->matrix[1][1], "-1");
  set_expr(Mlp->rowvec[0], "0");
  set_expr(Mlp->rowvec[1], "1");

  lp = dd_Matrix2LP(Mlp, &err);
  if (err != dd_NoError || lp == NULL) fail("dd_Matrix2LP failed");
  dd_LPSolve(lp, dd_DualSimplex, &err);
  if (err != dd_NoError) fail("dd_LPSolve failed");
  if (!equal_expr(lp->optvalue, "log(2/1)")) fail("LP optimum is not log(2)");

  dd_FreeLPData(lp);
  dd_FreeMatrix(Mlp);
  dd_FreeMatrix(Mround);
  dd_FreeMatrix(G);
  dd_FreePolyhedra(poly);
  dd_FreeMatrix(M);
  dd_free_global_constants();

  printf("testlogarithmic: ok\n");
  return 0;
}
