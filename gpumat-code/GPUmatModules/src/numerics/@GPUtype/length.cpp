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

#include "numerics.hh"

// static paramaters
static CUfunction drvfuns[4];
static int init = 0;
static GPUmat *gm;
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // At least 1 arguments expected
  // Input and result
  if (nrhs!=1)
     mexErrMsgTxt("Wrong number of arguments");
  if (init == 0) {
    // Initialize function
    //mexLock();
    // load GPUmat
    gm = gmGetGPUmat();
    gmCheckGPUmat(gm);
    init = 1;
  }
  // log
  gm->debug.log("> LENGTH\n",0);
  gm->debug.logPush();

  if (gm->comp.getCompileMode() == 1) {
    mexWarnMsgTxt(WARNING_NUMERICS_COMPNOTIMPLEMENTED);
  }

  // mex parameters are:
  // IN1
  GPUtype IN1  = gm->gputype.getGPUtype(prhs[0]);

  const int* size = gm->gputype.getSize(IN1);
  int ndims = gm->gputype.getNdims(IN1);

  int l = -10;
  for (int i = 0; i < ndims; i++) {
    int ll = size[i];
    if (ll > l)
      l = ll;

  }
  plhs[0] = mxCreateDoubleScalar((int) l);


  gm->debug.logPop();
}
