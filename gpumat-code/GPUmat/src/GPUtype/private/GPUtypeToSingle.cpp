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
static int init = 0;
static GPUmanager *GPUman;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  GPUmatResult_t status = GPUmatSuccess;
  // tmp
  mxArray *lhs[2];

  if (nrhs != 1)
    mexErrMsgTxt("Wrong number of arguments");

  // the passed element should be a GPUsingle
  if (!(mxIsClass(prhs[0], "GPUsingle")))
    mexErrMsgTxt(ERROR_EXPECTED_GPUSINGLE);

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

  int numel = p->getNumel();
  int ndims = p->getNdims();
  int *size = p->getSize();
  int mysize = p->getMySize();
  gpuTYPE_t type = p->getType();

  // create dest array
  // dims re set to [1 numel]. A reshape is required outside
  // this function
  mwSize dims[2];

  // create destination
  if (type == gpuFLOAT) {
    dims[0] = 1;
    dims[1] = numel;

    plhs[0] = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
  } else if (type == gpuCFLOAT) {
    dims[0] = 1;
    dims[1] = 2 * numel;

    plhs[0] = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
  }

  try {
      status = GPUopCudaMemcpy(mxGetPr(plhs[0]), p->getGPUptr(),
          mysize * numel, cudaMemcpyDeviceToHost, p->getGPUmanager());

  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }

}
