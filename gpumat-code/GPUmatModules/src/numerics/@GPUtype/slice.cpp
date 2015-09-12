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



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  CUresult cudastatus = CUDA_SUCCESS;

  // simple garbage collection
  MyGCObj<Range> mygc1;

  // more than 2 arguments expected
  if (nrhs < 2)
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
  gm->debug.log("> SLICE\n",0);
  gm->debug.logPush();

  // mex parameters are:
  // RHS: GPUtype variable (right hand side)
  //      or Matlab array, automatically converted
  // ...: variable number of arguments after 'RHS' representing the range

  if (gm->comp.getCompileMode() == 1) {
    GPUtype RHS;

    if ((mxGetClassID(prhs[0]) == mxDOUBLE_CLASS)||(mxGetClassID(prhs[0]) == mxSINGLE_CLASS)) {
      // compile this option
      // create dummy RHS
      RHS = gm->gputype.create(gpuFLOAT, 0, NULL, NULL);
      gm->comp.pushGPUtype(&RHS);

      gm->comp.functionStart("GPUMAT_mxToGPUtype");
      gm->comp.functionSetParamGPUtype(&RHS);
      gm->comp.functionSetParamMx(prhs[0]);
      gm->comp.functionEnd();
      //RHS = gm->gputype.mxToGPUtype(prhs[0]);
    } else {
      RHS = gm->gputype.getGPUtype(prhs[0]);
    }
    // create dummy OUT
    GPUtype OUT = gm->gputype.create(gpuFLOAT, 0, NULL, NULL);
    gm->comp.pushGPUtype(&OUT);

    gm->comp.functionStart("GPUMAT_mxSliceDrv");
    gm->comp.functionSetParamGPUtype(&OUT);
    gm->comp.functionSetParamGPUtype(&RHS);
    gm->comp.functionSetParamMxMx(nrhs-1, &prhs[1]);
    gm->comp.functionEnd();
    plhs[0] = gm->gputype.createMxArray(OUT);
  } else {
    GPUtype RHS;
    // convert Matlab array to GPUtype
    if ((mxGetClassID(prhs[0]) == mxDOUBLE_CLASS)||(mxGetClassID(prhs[0]) == mxSINGLE_CLASS)) {
      RHS = gm->gputype.mxToGPUtype(prhs[0]);
    } else {
      RHS = gm->gputype.getGPUtype(prhs[0]);
    }
    GPUtype OUT = gm->aux.mxSliceDrv(RHS, nrhs-1, &prhs[1] );
    plhs[0] = gm->gputype.createMxArray(OUT);
  }
  gm->debug.logPop();

}
