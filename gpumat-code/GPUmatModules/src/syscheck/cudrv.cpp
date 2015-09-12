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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  if (nrhs != 0)
    mexErrMsgTxt("Wrong number of arguments");

  mexPrintf("- Initializing CUDA driver\n");
  CUresult err = cuInit(0);
  if (err != CUDA_SUCCESS) {
    mexErrMsgTxt("ERROR1: Unable to initialize CUDA driver.");
  }

  int deviceCount = 0;
  err = cuDeviceGetCount(&deviceCount);
  if (err != CUDA_SUCCESS) {
    mexErrMsgTxt("ERROR2: Unable to find a CUDA compatible device.");
  }
  if (deviceCount == 0) {
    mexErrMsgTxt("ERROR2: Unable to find a CUDA compatible device.");
  }
  mexPrintf("...done\n");

}
