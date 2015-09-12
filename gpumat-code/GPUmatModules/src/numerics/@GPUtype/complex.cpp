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
static CUfunction drvfuns[4];
static int init = 0;
static GPUmat *gm;
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  if ((nrhs > 2) || (nrhs == 0))
      mexErrMsgTxt("Wrong number of arguments");

  if (init == 0) {
    // Initialize function
    //mexLock();
    // load GPUmat
    gm = gmGetGPUmat();
    gmCheckGPUmat(gm);
    init = 1;
  }
  // log
  gm->debug.log("> COMPLEX\n",0);
  gm->debug.logPush();
  // mex parameters are:
  // RE
  // IM (optional)

  if (gm->comp.getCompileMode() == 1) {
    GPUtype R = gm->gputype.create(gpuFLOAT, 0, NULL, NULL);
    gm->comp.pushGPUtype(&R);

    if (nrhs==1) {
      GPUtype RE  = gm->gputype.getGPUtype(prhs[0]);
      gm->comp.functionStart("GPUMAT_ComplexDrv");
      gm->comp.functionSetParamGPUtype(&R);
      gm->comp.functionSetParamGPUtype(&RE);
      gm->comp.functionEnd();
    }

    if (nrhs==2) {
      GPUtype RE  = gm->gputype.getGPUtype(prhs[0]);
      GPUtype IM  = gm->gputype.getGPUtype(prhs[1]);
      gm->comp.functionStart("GPUMAT_ComplexDrv1");
      gm->comp.functionSetParamGPUtype(&R);
      gm->comp.functionSetParamGPUtype(&RE);
      gm->comp.functionSetParamGPUtype(&IM);
      gm->comp.functionEnd();
    }

    plhs[0] = gm->gputype.createMxArray(R);
  } else {
    if (nrhs==1) {
      GPUtype RE  = gm->gputype.getGPUtype(prhs[0]);
      GPUtype R = gm->gputype.realToComplex(RE);
      plhs[0] = gm->gputype.createMxArray(R);
    }
    if (nrhs==2) {
      GPUtype RE  = gm->gputype.getGPUtype(prhs[0]);
      GPUtype IM  = gm->gputype.getGPUtype(prhs[1]);
      GPUtype R = gm->gputype.realImagToComplex(RE, IM);
      plhs[0] = gm->gputype.createMxArray(R);
    }


  }
  gm->debug.logPop();
}
