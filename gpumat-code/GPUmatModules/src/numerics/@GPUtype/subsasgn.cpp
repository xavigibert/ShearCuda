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

static int init = 0;
static GPUmat *gm;



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  CUresult cudastatus = CUDA_SUCCESS;

  // simple garbage collection
  MyGCObj<Range> mygc1;

  // 3 arguments expected
  if (nrhs != 3)
    mexErrMsgTxt("Wrong number of arguments");

  if (init == 0) {
    // Initialize function
    //mexLock();

    // load GPUmat
    gm = gmGetGPUmat();

    // check gm
    gmCheckGPUmat(gm);

    // load module
    // NO MODULE REQUIRED

    // load float GPU function
    // NO FUNCTION REQUIRED

    init = 1;
  }

  // log
  gm->debug.log("> SUBSASGN\n",0);
  gm->debug.logPush();

  if (gm->comp.getCompileMode() == 1) {
    // not supported
    gm->comp.abort("SUBSASGN not supported in compilation mode.");
  }

  // mex parameters are:
  // LHS: GPUtype variable (left hand side)
  // SUBS: check Matlab manual for subsasgn for information about the argument
  // RHS: GPUtype variable (right hand side)
  //      or Matlab array automatically converted to GPUtype

  GPUtype LHS = gm->gputype.getGPUtype(prhs[0]);

  //GPUtype RHS = gm->gputype.getGPUtype(prhs[2]);
  GPUtype RHS;
  // convert Matlab array to GPUtype
  if ((mxGetClassID(prhs[2]) == mxDOUBLE_CLASS)||(mxGetClassID(prhs[2]) == mxSINGLE_CLASS)) {
    RHS = gm->gputype.mxToGPUtype(prhs[2]);
  } else {
    RHS = gm->gputype.getGPUtype(prhs[2]);
  }

  // rg is the Range we have to populate
  Range *rg;
  parseMxRange(gm->gputype.getNdims(LHS), prhs[1],&rg, gm, mygc1);

  // LHS and RHS must be of the same type
  // Automatic casting

  // Cast to complex if necessary
  int lhscpx =  gm->gputype.isComplex(LHS);
  int rhscpx =  gm->gputype.isComplex(RHS);

  if (lhscpx && !rhscpx) {
    // convert RHS to complex
    RHS = gm->gputype.realToComplex(RHS);
  } else if (!lhscpx && rhscpx) {
    // convert LHS to complex
    LHS = gm->gputype.realToComplex(LHS);
  }

  int lhsf =  gm->gputype.isFloat(LHS);
  int rhsf =  gm->gputype.isFloat(RHS);

  int lhsd =  gm->gputype.isDouble(LHS);
  int rhsd =  gm->gputype.isDouble(RHS);

  if (lhsf && rhsd) {
    // cast RHS to FLOAT
    RHS = gm->gputype.doubleToFloat(RHS);
  }

  if (lhsd && rhsf) {
    // cast RHS to DOUBLE
    RHS = gm->gputype.floatToDouble(RHS);
  }



  // After creating the Range, I can call mxAssign
  // mxAssign uses indexes starting from 1 (Fortran/Matlab)

  // last parameter is 'direction'. Direction = 1 means that range is applied to LHS
  gm->gputype.mxAssign(LHS, RHS,*rg, 1);

  // create Matlab output
  plhs[0] = gm->gputype.createMxArray(LHS);
  // Output is the same as input LHS
  //plhs[0] = (mxArray*)prhs[0];

  gm->debug.logPop();

}
