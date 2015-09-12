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

#include "GPUcommon.hh"
#include "GPUerror.hh"
#include "Queue.hh"
#include "GPUstream.hh"
#include "GPUmanager.hh"
#include "GPUtype.hh"
#include "GPUop.hh"

#include "GPUtypeMat.hh"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  if (nrhs != 1)
    mexErrMsgTxt("Wrong number of arguments");

  //int status;

  mxArray *lhs[2];

  int dev = (int) mxGetScalar(prhs[0]);

  CUdevice cuDevice = 0;
  int deviceCount = 0;
  CUresult err = cuInit(0);
  if (err != CUDA_SUCCESS) {
    mexErrMsgTxt("Unable to initialize CUDA driver.");
  }
  err = cuDeviceGetCount(&deviceCount);
  if (err != CUDA_SUCCESS) {
    mexErrMsgTxt("Unable to get the number of CUDA devices.");
  }
  if (deviceCount == 0) {
    mexErrMsgTxt("Unable to find a device supporting CUDA.");
  }

  if (dev > deviceCount - 1)
    dev = deviceCount - 1;
  cuDeviceGet(&cuDevice, dev);
  if (err != CUDA_SUCCESS) {
    mexErrMsgTxt("Unable to initialize the CUDA device.");
  }


}
