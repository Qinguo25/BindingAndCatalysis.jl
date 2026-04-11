/* projection.c: Test program to call the cdd library cddlib
   written by Komei Fukuda, fukuda@ifor.math.ethz.ch
*/

/*  This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*/

#include "setoper.h"
#include "cdd.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

dd_boolean SetInputFile(FILE **f, dd_DataFileType fname)
{
  dd_boolean success=dd_FALSE;
  success=dd_FALSE;

  if ( ( *f = fopen(fname, "r") )!= NULL) {
    printf("input file %s is open\n", fname);
    success=dd_TRUE;
  }
  else{
    printf("The input file %s not found\n",fname);
  }
  return success;
}

dd_boolean SetWriteFile(FILE **f, dd_DataFileType fname)
{
  dd_boolean success=dd_FALSE;

  if ( (*f = fopen(fname, "w")) != NULL){
    printf("output file %s is open\n",fname);
    success=dd_TRUE;
  }
  else{
    printf("The output file %s cannot be opened\n",fname);
  }
  return success;
}

static dd_MatrixPtr dd_PermuteMatrixColumns(dd_MatrixPtr M, dd_colindex perm)
{
  dd_MatrixPtr Mperm=NULL;
  dd_rowrange i,m;
  dd_colrange j,d;

  if (M==NULL || perm==NULL) return NULL;

  m=M->rowsize;
  d=M->colsize;
  Mperm=dd_CreateMatrix(m, d);
  dd_CopyArow(Mperm->rowvec, M->rowvec, d);
  set_copy(Mperm->linset, M->linset);
  Mperm->numbtype=M->numbtype;
  Mperm->representation=M->representation;
  Mperm->objective=M->objective;

  for (i=1; i<=m; i++){
    for (j=1; j<=d; j++){
      dd_set(Mperm->matrix[i-1][j-1], M->matrix[i-1][perm[j]-1]);
    }
  }
  for (j=1; j<=d; j++){
    dd_set(Mperm->rowvec[j-1], M->rowvec[perm[j]-1]);
  }
  return Mperm;
}

static dd_MatrixPtr dd_ProjectByRepeatedFourier(dd_MatrixPtr M, dd_colset delset, dd_ErrorType *error)
{
  dd_MatrixPtr Mwork=NULL, Mnext=NULL;
  dd_rowset redset=NULL, impl_linset=NULL;
  dd_rowindex newpos=NULL;
  dd_colindex perm=NULL;
  dd_colrange j,d,k,delsize;

  *error=dd_NoError;
  if (M==NULL) return NULL;

  d=M->colsize;
  delsize=0;
  perm=(long*)calloc(d+1, sizeof(long));
  if (perm==NULL){
    *error=dd_ImproperInputFormat;
    return NULL;
  }

  perm[1]=1;  /* keep the constant column fixed */
  k=1;
  for (j=2; j<=d; j++){
    if (!set_member(j, delset)){
      k++;
      perm[k]=j;
    } else {
      delsize++;
    }
  }
  for (j=2; j<=d; j++){
    if (set_member(j, delset)){
      k++;
      perm[k]=j;
    }
  }

  Mwork=dd_PermuteMatrixColumns(M, perm);
  if (Mwork==NULL){
    *error=dd_ImproperInputFormat;
    goto _L99;
  }

  for (j=1; j<=delsize; j++){
    Mnext=dd_FourierElimination(Mwork, error);
    dd_FreeMatrix(Mwork);
    Mwork=NULL;
    if (*error!=dd_NoError || Mnext==NULL) goto _L99;

    if (Mnext->rowsize>0){
      dd_MatrixCanonicalize(&Mnext, &impl_linset, &redset, &newpos, error);
      if (*error!=dd_NoError) goto _L99;
      set_free(redset); redset=NULL;
      set_free(impl_linset); impl_linset=NULL;
      free(newpos); newpos=NULL;
    }

    Mwork=Mnext;
    Mnext=NULL;
  }

_L99:
  if (redset!=NULL) set_free(redset);
  if (impl_linset!=NULL) set_free(impl_linset);
  if (newpos!=NULL) free(newpos);
  if (perm!=NULL) free(perm);
  if (*error!=dd_NoError){
    if (Mnext!=NULL) dd_FreeMatrix(Mnext);
    if (Mwork!=NULL) dd_FreeMatrix(Mwork);
    return NULL;
  }
  return Mwork;
}


int main(int argc, char *argv[])
{
  dd_MatrixPtr M=NULL,M1=NULL;
  dd_colrange j,s,t,d;
  dd_ErrorType err=dd_NoError;
  dd_rowset redset,impl_linset;
  dd_colset delset;
  dd_rowindex newpos;
  mytype val;
  dd_DataFileType inputfile;
  FILE *reading=NULL;

  dd_set_global_constants();  /* First, this must be called. */

  dd_init(val);
  if (argc>1) strcpy(inputfile,argv[1]);
  if (argc<=1 || !SetInputFile(&reading,argv[1])){
    dd_WriteProgramDescription(stdout);
    fprintf(stdout,"\ncddlib test program to apply the Block Elimination to an H-polyhedron.\n");
    dd_SetInputFile(&reading,inputfile, &err);
  }
  if (err==dd_NoError) {
    M=dd_PolyFile2Matrix(reading, &err);
  }
  else {
    fprintf(stderr,"Input file not found\n");
    goto _L99;
  }

  if (err!=dd_NoError) goto _L99;

  d=M->colsize;
  set_initialize(&delset, d);

  printf("How many variables to eliminate? (max %ld): ",d-1);
  scanf("%ld",&s);

  for (j=1; j<=s; j++){
    printf("\n%ld th deletion variable): ",j);
    scanf("%ld",&t);
    set_addelem(delset, t+1);
  }

  if (set_card(M->linset)==0){
    M1=dd_ProjectByRepeatedFourier(M, delset, &err);
  } else {
    M1=dd_BlockElimination(M, delset, &err);
  }

  if (err!=dd_NoError || M1==NULL) goto _L99;

  dd_WriteMatrix(stdout, M1);

  if (M1->rowsize>0){
    dd_MatrixCanonicalize(&M1,&impl_linset,&redset,&newpos,&err);
    if (err!=dd_NoError) goto _L99;
  } else {
    set_initialize(&redset, 1);
    set_initialize(&impl_linset, 1);
    newpos=(long*)calloc(1,sizeof(long));
  }

  fprintf(stdout, "\nRedundant rows: ");
  set_fwrite(stdout, redset);
  fprintf(stdout, "\n");

  dd_WriteMatrix(stdout, M1);

  dd_FreeMatrix(M);
  dd_FreeMatrix(M1);
  set_free(delset);
  set_free(redset);
  set_free(impl_linset);
  free(newpos);

_L99:;
  /* if (err!=dd_NoError) dd_WriteErrorMessages(stderr,err); */
  dd_free_global_constants();  /* At the end, this should be called. */
  return 0;
}


/* end of projection.c */
