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

static CUfunction drvfuns[4];
static int init = 0;
static GPUmat *gm;


/*
 * B = repmat(A,M,N) creates a large matrix B consisting of an M-by-N
 * tiling of copies of A. The size of B is [size(A,1)*M, size(A,2)*N].
 * The statement repmat(A,N) creates an N-by-N tiling.
 *
 * B = REPMAT(A,[M N]) accomplishes the same result as repmat(A,M,N).
 *
 * B = REPMAT(A,[M N P ...]) tiles the array A to produce a
 * multidimensional array B composed of copies of A. The size of B is
 * [size(A,1)*M, size(A,2)*N, size(A,3)*P, ...].
 *
 *
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  CUresult cudastatus = CUDA_SUCCESS;



  // At least 2 arguments expected
  // The last argument is always a GPUtype
  if (nrhs < 2)
    mexErrMsgTxt("Wrong number of arguments");

  if (init == 0) {
    // Initialize function

    // load GPUmat
    gm = gmGetGPUmat();

    // check gm
    gmCheckGPUmat(gm);

    // init done
    init = 1;
  }

  // log
  gm->debug.log("> REPMAT\n",0);
  gm->debug.logPush();

  // This function is called such as the first argument is always of type
  // GPUtype
  // For example:
  // repmat(A,1,2)
  //
  // mex parameters are:
  // Par 0 -> IN
  // Par 0:END -> dimensions

  GPUtype IN = gm->gputype.getGPUtype(prhs[0]);
  if (gm->comp.getCompileMode() == 1) {

    GPUtype R = gm->gputype.create(gpuFLOAT, 0, NULL, NULL);
    gm->comp.pushGPUtype(&R);

    gm->comp.functionStart("GPUMAT_mxRepmatDrv");
    gm->comp.functionSetParamGPUtype(&R);
    gm->comp.functionSetParamGPUtype(&IN);
    gm->comp.functionSetParamMxMx(nrhs-1, &(prhs[1]));
    gm->comp.functionEnd();
    plhs[0] = gm->gputype.createMxArray(R);

  } else {

    GPUtype R = gm->gputype.mxRepmatDrv(IN, nrhs-1, &(prhs[1]));
    plhs[0] = gm->gputype.createMxArray(R);
  }

  // log
  gm->debug.logPop();

}

