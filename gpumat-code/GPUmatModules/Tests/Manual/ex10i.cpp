#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

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



void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {

  CUresult cudastatus = CUDA_SUCCESS;

  if (nrhs != 1)
    mexErrMsgTxt("Wrong number of arguments");


  if (init == 0) {
    // Initialize function
    //mexLock();

    // load GPUmat
    gm = gmGetGPUmat();


    init = 1;
  }

  // mex parameters are:
  GPUtype B = gm->gputype.getGPUtype(prhs[0]);
  GPUtype A = gm->gputype.mxSlice(B, Range(1,1,10));
  plhs[0] = gm->gputype.createMxArray(A);



}
