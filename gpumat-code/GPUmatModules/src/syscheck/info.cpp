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

#include "cuda.h"
#include "cuda_runtime.h"

#define printf mexPrintf

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  if (nrhs != 0)
    mexErrMsgTxt("Wrong number of arguments");

  int deviceCount;

  cudaGetDeviceCount(&deviceCount);

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0)
    mexErrMsgTxt("There is no device supporting CUDA\n");

  int dev;

  int devStart = 0;
  int devStop = deviceCount;

  if (nrhs == 1) {
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
  }

}
