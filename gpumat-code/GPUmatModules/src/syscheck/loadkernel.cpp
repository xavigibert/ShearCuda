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


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
 
  CUdevice cuDevice;
  CUcontext cuContext;
  CUmodule cuModule;
  CUresult status;

  status = cuInit(0);
  if (status != CUDA_SUCCESS) 
    mexErrMsgTxt(
	"ERROR0: Unable to initialize driver.");
  
  if (nrhs != 1)
    mexErrMsgTxt("Wrong number of arguments");
  
  int size = 0;

  /* device */
  mexPrintf("- Getting device properties\n");
  int dev =  (int) mxGetScalar(prhs[0]);
  cuDevice = dev;

  cudaDeviceProp deviceProp;
  cudaError err = cudaGetDeviceProperties(&deviceProp, dev);

  int major = deviceProp.major;
  int minor = deviceProp.minor;
  mexPrintf("...done\n");

  /* load corresponding kernel */
  mexPrintf("- Loading kernel file '");
  char filename[256];
  sprintf(filename,"kernel%d%d.cubin",major,minor);
  mexPrintf(filename);
  mexPrintf("'\n");
  
  FILE *f = fopen(filename, "rb");
  char * fatbin;
  if (f == NULL) {
    fatbin = NULL;
    mexErrMsgTxt(
            "ERROR1: Error opening  KERNEL file.");
  }
  fseek(f, 0, SEEK_END);
  size = ftell(f);
  fseek(f, 0, SEEK_SET);
  fatbin = (char *) malloc((size + 1) * sizeof(char));
  if (size != fread(fatbin, sizeof(char), size, f)) {
    free(fatbin);
    mexErrMsgTxt(
            "ERROR2: Error reading  KERNEL  file.");
  }
  fclose(f);
  
  fatbin[size] = 0;
  size = size + 1; // adding null to the end of the string
  
  mexPrintf("...done\n");
  
  mexPrintf("- Creating context and loading kernel\n");
  /* create context */
  
  cuContext = 0;
  status = cuCtxCreate(&cuContext, 0, cuDevice);
  if (status != CUDA_SUCCESS) 
    mexErrMsgTxt(
            "ERROR3: Error creating  KERNEL context.");
  
  status = cuModuleLoadData(&cuModule, fatbin);
  if (CUDA_SUCCESS != status)
     mexErrMsgTxt(
            "ERROR3: Error loading  KERNEL.");
  
  free(fatbin);

  status = cuModuleUnload(cuModule);
  if (CUDA_SUCCESS != status)
    mexErrMsgTxt("ERROR4: Unable to unload CUDA module.");

  // cuCtxDetach is obsolete from CUDA 4.0
  status = cuCtxDestroy(cuContext);
  if (status != CUDA_SUCCESS) 
    mexErrMsgTxt("ERROR5: Error in CUDA context cleanup.");
  
  mexPrintf("...done\n");
  
  
  
}
