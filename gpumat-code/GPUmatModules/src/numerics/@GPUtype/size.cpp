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
#include <stdarg.h>

#ifdef UNIX
#include <stdint.h>
#endif

#include "mex.h"


// CUDA
#include "cuda.h"
#include "cuda_runtime.h"

#include "GPUmat.hh"
#include "numerics.hh"


// static paramaters

static int init = 0;
static GPUmat *gm;

/*    D = SIZE(X), for M-by-N matrix X, returns the two-element row vector
    D = [M,N] containing the number of rows and columns in the matrix.
    For N-D arrays, SIZE(X) returns a 1-by-N vector of dimension lengths.
    Trailing singleton dimensions are ignored.

    [M,N] = SIZE(X) for matrix X, returns the number of rows and columns in
    X as separate output variables.

    [M1,M2,M3,...,MN] = SIZE(X) for N>1 returns the sizes of the first N
    dimensions of the array X.  If the number of output arguments N does
    not equal NDIMS(X), then for:

    N > NDIMS(X), SIZE returns ones in the "extra" variables, i.e., outputs
                  NDIMS(X)+1 through N.
    N < NDIMS(X), MN contains the product of the sizes of dimensions N
                  through NDIMS(X).

    M = SIZE(X,DIM) returns the length of the dimension specified
    by the scalar DIM.  For example, SIZE(X,1) returns the number
    of rows. If DIM > NDIMS(X), M will be 1.
*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  CUresult cudastatus = CUDA_SUCCESS;

  // no more than 2 arguments expected
  if (nrhs > 2)
    mexErrMsgTxt("Wrong number of arguments");

  if (init == 0) {
    // Initialize function
    //mexLock();

    // load GPUmat
    gm = gmGetGPUmat();

    // check gm
    gmCheckGPUmat(gm);

    // load module
    // NO MODULE REQUIRED

    // load float GPU function
    // NO FUNCTION REQUIRED

    init = 1;
  }

  // log
  gm->debug.log("> SIZE\n",0);
  gm->debug.logPush();

  if (gm->comp.getCompileMode() == 1) {
    mexWarnMsgTxt(WARNING_NUMERICS_COMPNOTIMPLEMENTED);
  }


  // mex parameters are:
  // IN: GPUtype variable

  GPUtype IN = gm->gputype.getGPUtype(prhs[0]);
  const int *in_size = gm->gputype.getSize(IN);
  int in_ndims = gm->gputype.getNdims(IN);
  // 2 cases
  // 1. s = size(A)
  // 2. s = size(A,dim)
  if (nrhs == 1) {
    // 2 cases:
    // 1. s = size(A)
    // 2. [a,b,c,...] = size(A)
    if (nlhs<=1) {
      // 1. s = size(A)
      // create output plhs[0]
      plhs[0] = mxCreateDoubleMatrix(1, in_ndims, mxREAL);
      // fill in plhs[0] with IN dimensions
      double *plhs_size = mxGetPr(plhs[0]);
      for (int i = 0; i < in_ndims; i++)
        plhs_size[i] = (double) in_size[i];
    } else {
      // 2. [a,b,c,...] = size(A)
      for (int i=0;i<nlhs;i++) {
        // create output
        // create output plhs[i]
        int r = 1;
        // if i is greater than IN dims return 1
        if (i>(in_ndims-1)) {
          // r = 1
        } else {
          r = in_size[i];
        }
        plhs[i] = mxCreateDoubleScalar(r);
      }
    }
  } else {
    // retrieve dim
    int dim  =  (int) mxGetScalar(prhs[1]);
    int r = 1;
    // if dim is greater than IN dims return 1
    if (dim>in_ndims) {
      // r = 1
    } else {
      r = in_size[dim-1];
    }
    // create output plhs[0]
    plhs[0] = mxCreateDoubleScalar(r);
  }



}
