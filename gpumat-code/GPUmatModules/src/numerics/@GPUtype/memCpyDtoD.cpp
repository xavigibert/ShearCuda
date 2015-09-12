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

/* Copy from device to device memory */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  //CUresult cudastatus = CUDA_SUCCESS;

  // simple garbage collection
  MyGCObj<Range> mygc1;

  // more than 4 arguments expected
  if (nrhs != 4)
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


  // mex parameters are:
  // DST: Destination GPUtype variable
  // SRC: Source GPUtype variable
  // dst_index: position where we copy in DST
  // count: number of elements to copy

  GPUtype DST = gm->gputype.getGPUtype(prhs[0]);
  GPUtype SRC = gm->gputype.getGPUtype(prhs[1]);
  if (gm->comp.getCompileMode() == 1) {
    gm->comp.functionStart("GPUMAT_mxMemCpyDtoD");
    gm->comp.functionSetParamGPUtype(&DST);
    gm->comp.functionSetParamGPUtype(&SRC);
    gm->comp.functionSetParamMxMx(nrhs-2, &prhs[2]);
    gm->comp.functionEnd();
  } else {
    gm->gputype.mxMemCpyDtoD(DST, SRC, nrhs-2, &prhs[2]);
  }




}
