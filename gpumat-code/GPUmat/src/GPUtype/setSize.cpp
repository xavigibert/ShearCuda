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

#ifndef MATLAB
#define MATLAB
#endif

//#include "cutil.h"
#include "cublas.h"
#include "cuda_runtime.h"
#include "cuda.h"

#include "cudalib_common.h"
#include "cudalib_error.h"
#include "cudalib.h"

#include "GPUcommon.hh"
#include "GPUerror.hh"
#include "Queue.hh"
#include "GPUstream.hh"
#include "GPUmanager.hh"
#include "GPUtype.hh"
#include "GPUop.hh"

#include "GPUtypeMat.hh"
#include "MatlabTemplates.hh"
static int init = 0;
static GPUmanager *GPUman;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  // tmp
  mxArray *lhs[2];

  if (nrhs != 2)
    mexErrMsgTxt("Wrong number of arguments");

  // check input
  MATCHECKINPUT(prhs[0])

  if (init == 0) {
    // Initialize function
    mexLock();
    // load GPUmanager
    mexCallMATLAB(2, &lhs[0], 0, NULL, "GPUmanager");
    GPUman = (GPUmanager *) (UINTPTR mxGetScalar(lhs[0]));
    mxDestroyArray(lhs[0]);
    init = 1;
  }
  GPUtype *p = mxToGPUtype(prhs[0], GPUman);

  if (GPUman->getCompileMode()==1) {
      GPUman->compAbort(ERROR_GPUMANAGER_COMPNOTIMPLEMENTED);
  }


  if (mxIsComplex(prhs[1]) || (mxGetNumberOfDimensions(prhs[1]) != 2)
      || (mxGetM(prhs[1]) != 1))
    mexErrMsgTxt("Size vector must be a row vector with real elements.");

  int *mysize = NULL;
  int ndims;

  if (mxGetNumberOfElements(prhs[1]) == 1) {
    mysize = (int*) Mymalloc(2 * sizeof(int));
    mysize[0] = 1;
    mysize[1] = (int) mxGetScalar(prhs[1]);
    ndims = 2;
  } else {
    int n = mxGetNumberOfElements(prhs[1]);
    double *tmp = mxGetPr(prhs[1]);
    mysize = (int*) Mymalloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
      mysize[i] = (int) floor(tmp[i]);
    }
    ndims = n;

  }

  p->setSize(ndims, mysize);

  // clean up
  if (mysize != NULL)
    Myfree(mysize);
}
