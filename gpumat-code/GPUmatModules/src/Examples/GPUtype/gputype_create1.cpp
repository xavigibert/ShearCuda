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



// static parameters
static int init = 0;
static GPUmat *gm;

/*
 * This function creates a GPUtype depending on the input
 * value. The GPUtype is initialized with '0'
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {



  // At least 1 argument expected
  if (nrhs!=1)
     mexErrMsgTxt("Wrong number of arguments");

  if (init == 0) {
    // Initialize function

    // load GPUmat
    gm = gmGetGPUmat();

    // load module
    // NOT REQUIRED

    // load float GPU function
    // NOT REQUIRED

    init = 1;
  }



  // mex parameters are:
  // type: integer

  gpuTYPE_t type = (gpuTYPE_t)( (int) mxGetScalar(prhs[0]));

  if ((type!=gpuFLOAT) && (type!=gpuCFLOAT) && (type!=gpuDOUBLE) && (type!=gpuCDOUBLE)) {
    mexErrMsgTxt("Wrong TYPE");
  }

  // create GPUtype, with given dimensions
  int mysize[] = {100,100};
  GPUtype R = gm->gputype.create(type,2,mysize, NULL);

  const void *gpuptr = gm->gputype.getGPUptr(R);   // pointer to GPU memory
  int numel = gm->gputype.getNumel(R);       // number of elements
  int datasize = gm->gputype.getDataSize(R); // bytes for each element

  cudaError_t cudastatus = cudaSuccess;
  cudastatus = cudaMemset((void *) gpuptr, 0, numel*datasize);
  if (cudastatus != cudaSuccess) {
    mexErrMsgTxt("Error in cudaMemset");
  }

  plhs[0] = gm->gputype.createMxArray(R);

}
