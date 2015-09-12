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

  // 2 arguments expected
  if (nrhs != 2)
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
  gm->debug.log("> SUBSREF\n",0);
  gm->debug.logPush();

  if (gm->comp.getCompileMode() == 1) {
    // not supported
    gm->comp.abort("SUBSREF not supported in compilation mode.");
  }

  // mex parameters are:
  // RHS: GPUtype variable (right hand side)
  // ...: check Matlab manual for subsref for information about the argument

  GPUtype RHS = gm->gputype.getGPUtype(prhs[0]);

  // rg is the Range we have to populate
  Range *rg;

  // we manage also the condition where the range has more dimensions
  // than the RHS vector, but set to '1'
  // For example:
  // A = GPUsingle(rand(10,10,10));
  // A(1,1,1,1)
  // we pass RHS dimensions to parseMxRange. This function will ignore
  // eventually the indexes that are greater than RHS dimensions
  parseMxRange(gm->gputype.getNdims(RHS), prhs[1],&rg, gm, mygc1);

  // After creating the Range, I can call mxSlice
  // mxSlice uses indexes starting from 1 (Fortran/Matlab)

  // we have to manage a particular condition. An empty range
  // generates an error in GPUmat, but in Matlab it means an
  // assignment like B = A(). So this condition is handled by
  // copying the RHS to the output and preserving the original
  // size.


  if (rg==NULL) {

    GPUtype OUT = gm->gputype.clone(RHS);

    // create Matlab output
    plhs[0] = gm->gputype.createMxArray(OUT);
  } else {


    GPUtype OUT = gm->gputype.mxSlice(RHS,*rg);

    // I have to handle a particular case here.
    // The final result should have the same dimensions
    // as the indexes array
    // For example:
    // slice(A,{[1 2;3 4]})
    // size(A) = [2 2]
    //
    mxArray * subs = mxGetField(prhs[1], 0, "subs");
    int subsdim = mxGetNumberOfElements(subs);
    if (subsdim==1) {
      mxArray *mx = mxGetCell(subs, 0);
      if (mxGetClassID(mx) == mxDOUBLE_CLASS) {
        const mwSize * mysize = mxGetDimensions(mx);
        int n = mxGetNumberOfDimensions(mx);
        gm->gputype.setSize(OUT, n, mysize);

      } else if ((mxIsClass(mx, "GPUdouble"))||(mxIsClass(mx, "GPUsingle"))) {
        GPUtype IN = gm->gputype.getGPUtype(mx);
        const int *mysize = gm->gputype.getSize(IN);
        int n = gm->gputype.getNdims(IN);
        gm->gputype.setSize(OUT, n, mysize);


      } else if (mxGetClassID(mx) == mxCHAR_CLASS) {
        const int *mysize = gm->gputype.getSize(OUT);
        int newsize[2];
        newsize[0] = mysize[1];
        newsize[1] = mysize[0];

        gm->gputype.setSize(OUT, 2, newsize);
      }
    }
    // create Matlab output
    plhs[0] = gm->gputype.createMxArray(OUT);
    // Garbage collector will be deleted and all pointers to Range as well
  }

  gm->debug.logPop();

}
