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

#include "GPUkernel.hh"
#include "GPUmat.hh"


/*************************************************
 * Simple garbage collector
 **************************************************/
MyGC::MyGC() :
  ptr(NULL), mysize(10), idx(0) {

  ptr = (void **) malloc(mysize * sizeof(void *));

  for (int i = 0; i < mysize; i++)
    ptr[i] = NULL;
}

void MyGC::setPtr(void *p) {
  if (idx == mysize) {
    // increase size
    int newmysize = mysize + 10;
    void **tmpptr = (void **) malloc(newmysize * sizeof(void *));
    for (int i = 0; i < newmysize; i++)
      tmpptr[i] = NULL;

    memcpy(tmpptr, ptr, mysize * sizeof(void *));
    free(ptr);
    mysize = newmysize;
    ptr = tmpptr;
  }
  ptr[idx] = p;
  idx++;
}

void MyGC::remPtr(void *p) {
  for (int i = mysize - 1; i >= 0; i--) {
    if (ptr[i] == p) {
      ptr[i] = NULL;
      break;
    }
  }
}

MyGC::~MyGC() {
  for (int i = 0; i < mysize; i++) {
    if (ptr[i] != NULL) {
      free(ptr[i]);
    }
  }
  free(ptr);

}



/*************************************************
 * GPUTYPE
 *************************************************/



/*************************************************
 * GPUMAT
 *************************************************/
/// gmGetGPUmat
/*
 * Returns pointer to GPUmat structure
 */
GPUmat * gmGetGPUmat() {
  // tmp
  mxArray *lhs[2];
  mexCallMATLAB(1, &lhs[0], 0, NULL, "GPUmodulesManager");
  GPUmat *gm = (GPUmat *) (UINTPTR mxGetScalar(lhs[0]));
  return gm;
}

/// gmgetModule
/**
 * Returns a pointer to the CUDA module
 */
CUmodule * gmGetModule(STRINGCONST char *modname) {
  mxArray *myfun;
  mxArray *tmplhs = mxCreateString(modname);
  mexCallMATLAB(1, &myfun, 1, &(tmplhs), "GPUgetUserModule");
  CUmodule *drvmod = (CUmodule*) (UINTPTR mxGetScalar(myfun));
  mxDestroyArray(myfun);
  return drvmod;
}

/// gmCheckGPUmat
/**
 * Check GPUmat consistency
 */

void gmCheckGPUmat(GPUmat *gm) {

  if (gm->mod.gpumat==0)
    mexErrMsgTxt("GPUmat modules error: NATIVE GPUmat functions not loaded");
  if (gm->mod.modules==0)
    mexErrMsgTxt("GPUmat modules error: MAIN module manager not loaded");
  if (gm->mod.numerics==0)
    mexErrMsgTxt("GPUmat modules error: NUMERICS module manager not loaded");
  if (gm->mod.rand==0)
      mexErrMsgTxt("GPUmat modules error: RAND module manager not loaded");


}



/*************************************************
 * UTILS
 *************************************************/

//Round a / b to nearest higher integer value
int iDivUp(int a, int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

/*************************************************
 * HOST DRIVERS
 *************************************************/
void hostGPUDRV(CUfunction drvfun, int N, int nrhs, hostdrv_pars_t *prhs) {


  unsigned int maxthreads = MAXTHREADS_STREAM;
  int nstreams = iDivUp(N, maxthreads*BLOCK_DIM1D);
  CUresult err = CUDA_SUCCESS;
  for (int str = 0; str < nstreams; str++) {
    int offset = str * maxthreads * BLOCK_DIM1D;
    int size = 0;
    if (str == (nstreams - 1))
      size = N - str * maxthreads * BLOCK_DIM1D;
    else
      size = maxthreads * BLOCK_DIM1D;


    int gridx = iDivUp(size, BLOCK_DIM1D); // number of x blocks

    // setup execution parameters

    if (CUDA_SUCCESS != (err = cuFuncSetBlockShape(drvfun, BLOCK_DIM1D, 1, 1))) {
      mexErrMsgTxt("Error in cuFuncSetBlockShape");
    }

    if (CUDA_SUCCESS != cuFuncSetSharedSize(drvfun, 0)) {
      mexErrMsgTxt("Error in cuFuncSetSharedSize");
    }


    // add parameters
    int poffset = 0;

    // CUDA kernels interface
    // N: number of elements
    // offset: used for streams
    ALIGN_UP(poffset, __alignof(size));
    if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, size)) {
      mexErrMsgTxt("Error in cuParamSeti");
    }
    poffset += sizeof(size);

    ALIGN_UP(poffset, __alignof(offset));
    if (CUDA_SUCCESS != cuParamSeti(drvfun, poffset, offset)) {
      mexErrMsgTxt("Error in cuParamSeti");
    }
    poffset += sizeof(offset);

    for (int p=0;p<nrhs;p++) {
      ALIGN_UP(poffset, prhs[p].align);
      if (CUDA_SUCCESS
          != cuParamSetv(drvfun, poffset, prhs[p].par, prhs[p].psize)) {
        mexErrMsgTxt("Error in cuParamSetv");
      }
      poffset += prhs[p].psize;
    }

    if (CUDA_SUCCESS != cuParamSetSize(drvfun, poffset)) {
      mexErrMsgTxt("Error in cuParamSetSize");
    }

    err = cuLaunchGridAsync(drvfun, gridx, 1, 0);
    if (CUDA_SUCCESS != err) {
      mexErrMsgTxt("Error running kernel");
    }
  }

}
