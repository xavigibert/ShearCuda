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

  // arguments
  // 1. GPUsingle
  //END(A,K,N) is called for indexing expressions involving the object A
  //when END is part of the K-th index out of N indices.  For example,
  //the expression A(end-1,:) calls A's END method with END(A,1,2).

  // tmp
  mxArray *lhs[2];

  if (nrhs != 3)
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


    //GPUmanager *GPUman = p->getGPUmanager();
    try {
      if (GPUman->getCompileMode()==1) {
        GPUman->compAbort(ERROR_GPUMANAGER_COMPNOTIMPLEMENTED);
      }
    } catch (GPUexception ex) {
      mexErrMsgTxt(ex.getError());
    }



    int k         =  (int) mxGetScalar(prhs[1]);
    int n         =  (int) mxGetScalar(prhs[2]);


    int r = p->getEnd(k-1,n)+1; // index in Matlab starts from 1, function geEnd requires index from 0.


    plhs[0] = mxCreateDoubleScalar(r);


    //mxDestroyArray(lhs[1]);

}
