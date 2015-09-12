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

#include "GPUmat.hh"


// static paramaters
static int init = 0;
static GPUmat *gm;

/*
 * RAND(N, GPUsingle) is the N-by-N random matrix.
 * RAND(N, GPUdouble) is the N-by-N random matrix
 *
 * RAND(M,N, GPUsingle) or RAND([M,N], GPUsingle) is an M-by-N matrix with random numbers
 *
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  CUresult cudastatus = CUDA_SUCCESS;

  // At least 2 arguments expected
  // The last argument is always a GPUtype
  if (nrhs<2)
     mexErrMsgTxt("Wrong number of arguments");

  if (init == 0) {
    // Initialize function
    //mexLock();

    // load GPUmat
    gm = gmGetGPUmat();

    // check gm
    gmCheckGPUmat(gm);


    init = 1;
  }


  // log
  gm->debug.log("> RANDN\n",0);
  gm->debug.logPush();

  // This function is called such as the last argument is always of type
  // GPUtype
  // For example:
  // rand(3,4,5,GPUsingle)
  //
  // mex parameters are:
  // LAST parameter -> IN
  // 0:LAST -> dimensions

  GPUtype IN = gm->gputype.getGPUtype(prhs[nrhs-1]);

  if (gm->comp.getCompileMode() == 1) {
    GPUtype R = gm->gputype.create(gpuFLOAT, 0, NULL, NULL);
    gm->comp.pushGPUtype(&R);

    gm->comp.functionStart("GPUMAT_mxRandnDrv");
    gm->comp.functionSetParamGPUtype(&R);
    gm->comp.functionSetParamGPUtype(&IN);
    gm->comp.functionSetParamMxMx(nrhs-1, prhs);
    gm->comp.functionEnd();
    plhs[0] = gm->gputype.createMxArray(R);
  } else {
    // nrhs-1 because last argument is a GPUtype
    // r is the returned output
    GPUtype R = gm->rand.mxRandnDrv(IN, nrhs-1, prhs);
    plhs[0] = gm->gputype.createMxArray(R);
  }

  gm->debug.logPop();
}
