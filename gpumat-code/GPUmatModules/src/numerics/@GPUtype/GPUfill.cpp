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

/* Interface to GPUmat function fill */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  //CUresult cudastatus = CUDA_SUCCESS;

  // simple garbage collection
  MyGCObj<Range> mygc1;

  // more than 4 arguments expected
  if (nrhs != 7)
    mexErrMsgTxt(ERROR_GPUFILL_WRONGARGS);

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
  gm->debug.log("> GPUfill\n",0);
  gm->debug.logPush();

  // mex parameters are:
  // DST: Destination GPUtype variable
  // offset
  // incr
  // m
  // p
  // type

  GPUtype DST = gm->gputype.getGPUtype(prhs[0]);
  if (gm->comp.getCompileMode() == 1) {

    gm->comp.functionStart("GPUMAT_mxFill");
    gm->comp.functionSetParamGPUtype(&DST);
    gm->comp.functionSetParamMxMx(nrhs-1, &(prhs[1]));
    gm->comp.functionEnd();

  } else {
    gm->gputype.mxFill(DST, nrhs-1, &(prhs[1]));
  }
  gm->debug.logPop();


}
