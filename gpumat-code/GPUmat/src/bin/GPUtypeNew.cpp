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

#ifdef UNIX
#include <stdint.h>
#endif

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
#include "GPUnumeric.hh"

#include "MatlabTemplates.hh"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  // My garbage collector
  MyGC mgc = MyGC();    // Garbage collector for Malloc
  MyGCObj<GPUtype> mgcobj; // Garbage collector for GPUtype


  // tmp
  mxArray *lhs[2];

  if (nrhs<=2)
    mexErrMsgTxt("Wrong number of arguments");

  /* Arguments
  * type
  * GPUmanager
  * list of dimensions
  */
  gpuTYPE_t type      =  (gpuTYPE_t) ((int) mxGetScalar(prhs[0]));
  GPUmanager *GPUman    =  (GPUmanager *) (UINTPTR mxGetScalar(prhs[1]));


  // Make the following test
  // 1) complex elements
  // 2)  should be either a scalar or a vector with dimension 2
  // 3) If there are more arguments, each argument is checked to be scalar afterwards

  int nrhsstart = 2; // first 2 arguments are type and GPUmanager
  int nrhseff = nrhs-2;

  for (int i = nrhsstart; i < nrhs; i++) {
    if (mxIsComplex(prhs[i]) || (mxGetNumberOfDimensions(prhs[i]) != 2)
      || (mxGetM(prhs[i]) != 1))
      mexErrMsgTxt("Size vector must be a row vector with real elements.");
  }

  int *mysize = NULL;
  int ndims;

  if (nrhseff == 1) {
    if (mxGetNumberOfElements(prhs[nrhsstart]) == 1) {
      mysize = (int*) Mymalloc(2 * sizeof(int),&mgc);
      mysize[0] = (int) mxGetScalar(prhs[nrhsstart]);
      mysize[1] = mysize[0];
      ndims = 2;
    } else {
      int n = mxGetNumberOfElements(prhs[nrhsstart]);
      double *tmp = mxGetPr(prhs[nrhsstart]);
      mysize = (int*) Mymalloc(n * sizeof(int),&mgc);
      for (int i = 0; i < n; i++) {
        mysize[i] = (int) floor(tmp[i]);
      }
      ndims = n;

    }
  } else {
    int n = nrhseff;
    mysize = (int*) Mymalloc(n * sizeof(int),&mgc);
    for (int i = nrhsstart; i < nrhs; i++) {
      if (mxGetNumberOfElements(prhs[i]) == 1) {
        mysize[i-nrhsstart] = (int) mxGetScalar(prhs[i]);
      } else {
        mexErrMsgTxt("Input arguments must be scalar.");
      }
    }
    ndims = n;
  }

  if (mysize == NULL)
    mexErrMsgTxt("Unexpected error in GPUtypeNew.");

  // Check all dimensions different from 0
  for (int i = 0; i < ndims; i++) {
    if (mysize[i] == 0)
      mexErrMsgTxt("Dimension cannot be zero.");

  }

  // remove any one at the end
  int finalndims = ndims;
  for (int i = ndims - 1; i > 1; i--) {
    if (mysize[i] == 1)
      finalndims--;
    else
      break;
  }
  ndims = finalndims;

  GPUtype *r = new GPUtype(type, ndims , mysize, GPUman);
  mgcobj.setPtr(r); // should delete this pointer
  r->setSize(ndims, mysize);


  try {
    GPUopAllocVector(*r);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }


  // create output result

  mgcobj.remPtr(r);
  plhs[0] = toMx(r);


}

