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



  // more than 4 arguments expected
  if (nrhs < 4)
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
  gm->debug.log("> ASSIGN\n",0);
  gm->debug.logPush();

  // mex parameters are:
  // dir: direction. Range is applied to the left or the right
  // LHS: GPUtype variable (left hand side)
  // RHS: GPUtype variable (right hand side)
  //      or Matlab array (converted to GPUtype)
  // ...: variable number of arguments after 'dir' representing the range

  if (mxGetClassID(prhs[0]) != mxDOUBLE_CLASS) {
    mexErrMsgTxt(ERROR_ASSIGN_FIRSTARG);
  }
  int dir = (int) mxGetScalar(prhs[0]);
  GPUtype LHS = gm->gputype.getGPUtype(prhs[1]);

  if (gm->comp.getCompileMode() == 1) {
    GPUtype RHS;
    if ((mxGetClassID(prhs[2]) == mxDOUBLE_CLASS)||(mxGetClassID(prhs[2]) == mxSINGLE_CLASS)) {
      // compile this option
      // create dummy RHS
      RHS = gm->gputype.create(gpuFLOAT, 0, NULL, NULL);
      gm->comp.pushGPUtype(&RHS);

      gm->comp.functionStart("GPUMAT_mxToGPUtype");
      gm->comp.functionSetParamGPUtype(&RHS);
      gm->comp.functionSetParamMx(prhs[2]);
      gm->comp.functionEnd();
      //RHS = gm->gputype.mxToGPUtype(prhs[2]);
    } else {
      RHS = gm->gputype.getGPUtype(prhs[2]);
    }

    gm->comp.functionStart("GPUMAT_mxAssign");
    gm->comp.functionSetParamGPUtype(&LHS);
    gm->comp.functionSetParamGPUtype(&RHS);
    gm->comp.functionSetParamInt(dir);
    gm->comp.functionSetParamMxMx(nrhs-3, &prhs[3]);
    gm->comp.functionEnd();


  } else {
    GPUtype RHS;
    // convert Matlab array to GPUtype
    if ((mxGetClassID(prhs[2]) == mxDOUBLE_CLASS)||(mxGetClassID(prhs[2]) == mxSINGLE_CLASS)) {
      RHS = gm->gputype.mxToGPUtype(prhs[2]);
    } else {
      RHS = gm->gputype.getGPUtype(prhs[2]);
    }

    gm->aux.mxAssign(LHS, RHS, dir, nrhs-3, &prhs[3] );
  }

  gm->debug.logPop();


}
