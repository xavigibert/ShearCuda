/*
     Copyright (C) 2012  GP-you Group (http://gp-you.org)
 
     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.
 
     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.
 
     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <string.h>

#ifdef UNIX
#include <stdint.h>
#endif

#include "mex.h"

#include "cublas.h"

#include "GPUcommon.hh"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  if (nrhs != 13)
    mexErrMsgTxt("Wrong number of arguments");

  //transa, transb, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc
  char transa;
  char transb;
  int m;
  int n;
  int k;

  float alpha;
  float beta;

  float *d_A;
  float *d_B;
  float *d_C;

  int lda;
  int ldb;
  int ldc;

  transa = (char) mxGetScalar(prhs[0]);
  transb = (char) mxGetScalar(prhs[1]);

  m = (int) mxGetScalar(prhs[2]);
  n = (int) mxGetScalar(prhs[3]);
  k = (int) mxGetScalar(prhs[4]);

  alpha = (float) mxGetScalar(prhs[5]);
  d_A = (float *) (UINTPTR mxGetScalar(prhs[6]));
  lda = (int) mxGetScalar(prhs[7]);

  d_B = (float *) (UINTPTR mxGetScalar(prhs[8]));
  ldb = (int) mxGetScalar(prhs[9]);
  beta = (float) mxGetScalar(prhs[10]);

  d_C = (float *) (UINTPTR mxGetScalar(prhs[11]));
  ldc = (int) mxGetScalar(prhs[12]);

  cublasSgemm(transa,transb, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);

}
