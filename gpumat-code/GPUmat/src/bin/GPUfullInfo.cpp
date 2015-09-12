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
#include "mex.h"

//#include "cutil.h"
#include "cublas.h"
#include "cuda_runtime.h"
#include "cuda.h"

#include "cudalib_common.h"
#include "cudalib_error.h"
#include "cudalib.h"

#define printf mexPrintf

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  //if (nrhs != 1)
  //  mexErrMsgTxt("Wrong number of arguments");

  //int dev = (int) mxGetScalar(prhs[0]);

  int deviceCount;

  cudaGetDeviceCount(&deviceCount);

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0)
    mexErrMsgTxt("There is no device supporting CUDA\n");

  int dev;

  int devStart = 0;
  int devStop = deviceCount;

  if (nrhs==1) {
    devStart = (int) mxGetScalar(prhs[0]);
    devStop = devStart+1;
    if (devStart >= deviceCount)
      mexErrMsgTxt("Please specify a valid GPU device.\n");
  }

  for (dev = devStart; dev < devStop; ++dev) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    if (dev == devStart) {
      // This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
      if (deviceProp.major == 9999 && deviceProp.minor == 9999)
        mexErrMsgTxt("There is no device supporting CUDA.\n");
      else if (deviceCount == 1)
        printf("There is 1 device supporting CUDA\n");
      else
        printf("There are %d devices supporting CUDA\n", deviceCount);
#if CUDART_VERSION >= 2020
      int driverVersion = 0, runtimeVersion = 0;
      cudaDriverGetVersion(&driverVersion);
      printf("CUDA Driver Version:                           %d.%d\n", driverVersion/1000, driverVersion%100);
      cudaRuntimeGetVersion(&runtimeVersion);
      printf("CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, runtimeVersion%100);
#endif

    }
    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

    printf("  CUDA Capability Major revision number:         %d\n",
      deviceProp.major);
    printf("  CUDA Capability Minor revision number:         %d\n",
      deviceProp.minor);

    printf("  Total amount of global memory:                 %u bytes\n",
      deviceProp.totalGlobalMem);
    //#if CUDART_VERSION >= 2000
    //    printf("  Number of multiprocessors:                     %d\n",
    //        deviceProp.multiProcessorCount);
    //    printf("  Number of cores:                               %d\n", 8
    //        * deviceProp.multiProcessorCount);
    //#endif
    printf("  Total amount of constant memory:               %u bytes\n",
      deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:       %u bytes\n",
      deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block: %d\n",
      deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n",
      deviceProp.warpSize);
    printf("  Maximum number of threads per block:           %d\n",
      deviceProp.maxThreadsPerBlock);
    printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
      deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
      deviceProp.maxThreadsDim[2]);
    printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
      deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
      deviceProp.maxGridSize[2]);
    printf("  Maximum memory pitch:                          %u bytes\n",
      deviceProp.memPitch);
    printf("  Texture alignment:                             %u bytes\n",
      deviceProp.textureAlignment);
    printf("  Clock rate:                                    %.2f GHz\n",
      deviceProp.clockRate * 1e-6f);
#if CUDART_VERSION >= 2000
    printf("  Concurrent copy and execution:                 %s\n",
      deviceProp.deviceOverlap ? "Yes" : "No");
#endif
#if CUDART_VERSION >= 2020
    printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
    printf("  Integrated:                                    %s\n", deviceProp.integrated ? "Yes" : "No");
    printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
    printf("  Compute mode:                                  %s\n", deviceProp.computeMode == cudaComputeModeDefault ?
      "Default (multiple host threads can use this device simultaneously)" :
    deviceProp.computeMode == cudaComputeModeExclusive ?
      "Exclusive (only one host thread at a time can use this device)" :
    deviceProp.computeMode == cudaComputeModeProhibited ?
      "Prohibited (no host thread can use this device)" :
    "Unknown");
#endif
  }


}
