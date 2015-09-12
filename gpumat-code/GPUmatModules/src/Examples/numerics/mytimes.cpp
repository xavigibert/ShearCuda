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
static CUfunction drvfunf; // float
static CUfunction drvfunc; // complex
static CUfunction drvfund; // double
static CUfunction drvfuncd;//double complex

static int init = 0;

static GPUmat *gm;


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  CUresult cudastatus = CUDA_SUCCESS;

  if (nrhs != 3)
    mexErrMsgTxt("Wrong number of arguments");

  if (init == 0) {
    // Initialize function
    //mexLock();

    // load GPUmat
		gm = gmGetGPUmat();

		// load module
		CUmodule *drvmod = gmGetModule("examples_numerics");

    // load float GPU function
    CUresult status = cuModuleGetFunction(&drvfunf, *drvmod, "TIMESF");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    // load complex GPU function
    status = cuModuleGetFunction(&drvfunc, *drvmod, "TIMESC");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    // load double GPU function
    status = cuModuleGetFunction(&drvfund, *drvmod, "TIMESD");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    // load complex GPU function
    status = cuModuleGetFunction(&drvfuncd, *drvmod, "TIMESCD");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    init = 1;
  }

  // mex parameters are:

  // 1. IN1
  // 2. IN2
  // 3. OUT

  //IN1 is the input GPU array
    GPUtype IN1 = gm->gputype.getGPUtype(prhs[0]);

    //IN2 is the input GPU array
    GPUtype IN2 = gm->gputype.getGPUtype(prhs[1]);

    //OUT is the output GPU array (result)
    GPUtype OUT = gm->gputype.getGPUtype(prhs[2]);

    // number of elements
    int nin1 = gm->gputype.getNumel(IN1);
    int nin2 = gm->gputype.getNumel(IN2);
    int nout = gm->gputype.getNumel(OUT);

    gpuTYPE_t tin1 = gm->gputype.getType(IN1);
    gpuTYPE_t tin2 = gm->gputype.getType(IN2);
    gpuTYPE_t tout = gm->gputype.getType(OUT);

    // check input/out size and type
    if (nin1!=nin2)
      mexErrMsgTxt("Input arguments must have the same number of elements.");

    if (nin1!=nout)
      mexErrMsgTxt("Input and output arguments must have the same number of elements.");

    if (tin1!=tin2)
      mexErrMsgTxt("Input arguments must be of the same type.");

    if (tin1!=tout)
      mexErrMsgTxt("Input and output arguments must be of the same type.");

    // I need the pointers to GPU memory
    CUdeviceptr d_IN1  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IN1));
    CUdeviceptr d_IN2  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IN2));
    CUdeviceptr d_OUT = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(OUT));

  // The GPU kernel depends on the type of input/output
  CUfunction drvfun;
  if (tin1 == gpuFLOAT) {
    drvfun = drvfunf;
  } else if (tin1 == gpuCFLOAT) {
    drvfun = drvfunc;
  } else if (tin1 == gpuDOUBLE) {
    drvfun = drvfund;
  } else if (tin1 == gpuCDOUBLE) {
    drvfun = drvfuncd;
  }

  hostdrv_pars_t gpuprhs[3];
  int gpunrhs = 3;
  gpuprhs[0] = hostdrv_pars(&d_IN1,sizeof(d_IN1),__alignof(d_IN1));
  gpuprhs[1] = hostdrv_pars(&d_IN2,sizeof(d_IN2),__alignof(d_IN2));
  gpuprhs[2] = hostdrv_pars(&d_OUT,sizeof(d_OUT),__alignof(d_OUT));

  int N = nin1;

  hostGPUDRV(drvfun, N, gpunrhs, gpuprhs);

}
