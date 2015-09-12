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
 * This function creates a GPUtype from a Matlab array, using GPUmat
 * function mxToGPUtype. Return the GPUtype to Matlab as a GPUmat object (GPUsingle, GPUdouble)
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {



  // At least 1 argument expected
  if (nrhs!=1)
     mexErrMsgTxt("Wrong number of arguments.");

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
  // IN: Matlab array

  GPUtype IN = gm->gputype.mxToGPUtype(prhs[0]);
  plhs[0] = gm->gputype.createMxArray(IN);

}
