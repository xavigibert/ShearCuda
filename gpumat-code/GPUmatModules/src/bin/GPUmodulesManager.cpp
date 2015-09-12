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
static int init = 0;
static GPUmat gm;

/*
 * Initializes the modules. This function is called by gmGetGPUmat(),
 * and returns the pointer to the gm structure. The first time the gm
 * structure is initialized
 *
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  if (nrhs!=0)
     mexErrMsgTxt("Wrong number of arguments");

  if (init == 0) {
    // Initialize function
    mexLock();

    // load GPUmat interface
    mxArray *lhs[2];
    mexCallMATLAB(2, &lhs[0], 0, NULL, "GPUmanager");
    GPUmatInterface *gmat = (GPUmatInterface *) (UINTPTR mxGetScalar(lhs[1]));

    // check version
    int major = 0;
    int minor = 0;
    gmat->config.getMajorMinor(&major, &minor);
    if ((major != GPUMATVERSION_MAJOR)||(minor != GPUMATVERSION_MINOR)) {
      // error
      mexPrintf("GPUmodules version -> %d.%d\n", major,minor);
      mexPrintf("GPUmat version     -> %d.%d\n", GPUMATVERSION_MAJOR, GPUMATVERSION_MINOR);
      mexErrMsgTxt("Inconsistent versions.");
    }

#include "GPUmatNumericsInit.hh"
#include "GPUmatFFTInit.hh"
#include "GPUmatGPUtypeInit.hh"
#include "GPUmatCompilerInit.hh"

    // update mod flags
    gm.mod.gpumat = 1;  // native functions loaded
    gm.mod.modules = 1; // main module loaded

    // stores the pointer to native GPUmat functions
    gm.gmat = gmat;

    init = 1;
  }


  plhs[0] = mxCreateDoubleScalar(UINTPTR &gm);


}
