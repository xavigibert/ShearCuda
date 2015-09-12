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

// static paramaters
static int init = 0;
static GPUmat *gm;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  CUresult cudastatus = CUDA_SUCCESS;

  if (nrhs != 1)
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
  gm->debug.log("> GPUrand\n",0);
  gm->debug.logPush();

  // mex parameters are:
  // 1. OUT
  //OUT is the output GPU array (result)
  GPUtype OUT = gm->gputype.getGPUtype(prhs[0]);

  if (gm->comp.getCompileMode() == 1) {
    gm->comp.functionStart("GPUMAT_Rand");
    gm->comp.functionSetParamGPUtype(&OUT);
    gm->comp.functionEnd();
  } else {
    gm->rand.rand(OUT);
  }

  gm->debug.logPop();

}
