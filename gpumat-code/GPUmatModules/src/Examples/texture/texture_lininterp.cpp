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

static CUtexref texf; // float  texture
static CUtexref texd; // double texture

static int init = 0;

static GPUmat *gm;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  CUresult cudastatus = CUDA_SUCCESS;

  if (nrhs != 2)
    mexErrMsgTxt("Wrong number of arguments");

  if (init == 0) {
    // Initialize function
    //mexLock();

    // load GPUmat
		gm = gmGetGPUmat();

		// load module
		CUmodule *drvmod = gmGetModule("examples_texture");


    // load float GPU function
    CUresult status = cuModuleGetFunction(&drvfunf, *drvmod, "LININTERF");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    // load double GPU function
    status = cuModuleGetFunction(&drvfund, *drvmod, "LININTERD");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    // load textures defined in module
    status = cuModuleGetTexRef(&texf, *drvmod, "texref_f1_a");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load texture.");
    }

    status = cuModuleGetTexRef(&texd, *drvmod, "texref_d1_a");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load texture.");
    }

    // no complex function support
    init = 1;
  }

  // mex parameters are:

  // 1. IN1. Input array
  // 2. IN2. Input indexes array

  //IN1 is the input GPU array
  GPUtype IN1 = gm->gputype.getGPUtype(prhs[0]);

  //IN2 is the input GPU array
  GPUtype IN2 = gm->gputype.getGPUtype(prhs[1]);

  //OUT is the output GPU array (result)
  // Create of the same size of IN1
  gpuTYPE_t in1_t = gm->gputype.getType(IN1);
  int in1_d = gm->gputype.getNdims(IN1);
  const int * in1_s = gm->gputype.getSize(IN1);
  int in1_n = gm->gputype.getNumel(IN1);
  int in1_b = gm->gputype.getDataSize(IN1);

  gpuTYPE_t in2_t = gm->gputype.getType(IN2);
  int in2_d = gm->gputype.getNdims(IN2);
  const int * in2_s = gm->gputype.getSize(IN2);
  int in2_n = gm->gputype.getNumel(IN2);

  if ((in1_t==gpuCFLOAT) || (in1_t==gpuCDOUBLE)) {
    mexErrMsgTxt("Complex TYPE not supported");
  }

  if (in1_t != in2_t) {
    mexErrMsgTxt("Input arguments must be of the same type");
  }

  if (in1_n != in2_n) {
    mexErrMsgTxt("Input arguments must have the same number of elements");
  }

  //OUT is the output GPU array (result)
  // Create of the same size of IN1
  GPUtype OUT = gm->gputype.create(in1_t, in1_d, in1_s, NULL);



  // I need the pointers to GPU memory
  CUdeviceptr d_IN1  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IN1));
  CUdeviceptr d_IN2  = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(IN2));
  CUdeviceptr d_OUT = (CUdeviceptr) (UINTPTR gm->gputype.getGPUptr(OUT));

  // The GPU kernel depends on the type of input/output
  CUfunction drvfun;
  CUtexref   drvtex;
  CUarray_format_enum drvtexformat;
  int drvtexnum;

  if (in1_t == gpuFLOAT) {
    drvfun = drvfunf;
    drvtex = texf;
    drvtexformat = CU_AD_FORMAT_FLOAT;
    drvtexnum = 1;
  } else if (in1_t == gpuDOUBLE) {
    drvfun = drvfund;
    drvtex = texd;
    drvtexformat = CU_AD_FORMAT_SIGNED_INT32;
    drvtexnum = 2;
  }

  if (CUDA_SUCCESS != cuTexRefSetFormat(drvtex, drvtexformat, drvtexnum)) {
    mexErrMsgTxt("Execution error (texture).");
  }
  if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, drvtex, UINTPTR d_IN1, in1_n*in1_b)) {
    mexErrMsgTxt("Execution error (texture).");
  }

  if (CUDA_SUCCESS != cuParamSetTexRef(drvfun, CU_PARAM_TR_DEFAULT, drvtex)) {
    mexErrMsgTxt("Execution error (texture1).");
  }


  hostdrv_pars_t gpuprhs[2];
	int gpunrhs = 2;
	gpuprhs[0] = hostdrv_pars(&d_IN2,sizeof(d_IN2),__alignof(d_IN2));
	gpuprhs[1] = hostdrv_pars(&d_OUT,sizeof(d_OUT),__alignof(d_OUT));

	int N = in1_n;

	hostGPUDRV(drvfun, N, gpunrhs, gpuprhs);


	// return result
	plhs[0] = gm->gputype.createMxArray(OUT);


}
