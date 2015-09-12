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
#ifndef MATLAB
#define MATLAB
#endif

#ifdef UNIX
#include <stdint.h>
#endif

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

#define BUFFERSIZE 300
#define CLEARBUFFER memset(buffer,0,BUFFERSIZE);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  // Garbage collector
  MyGC mygc = MyGC();

  char buffer[BUFFERSIZE];

  if (nrhs != 0)
    mexErrMsgTxt("Wrong number of arguments (1 argument expected)");

  // tmp
  mxArray *lhs[2];
  mxArray *rhs[2];

  mexCallMATLAB(2, &lhs[0], 0, NULL, "GPUmanager");
  GPUmanager *gm = (GPUmanager *) (UINTPTR mxGetScalar(lhs[0]));

  if (gm->getCompileMode() != 1) {
    mexErrMsgTxt("GPUfor can be used only in compilation mode.");
  }

  // check for-loop count
  if (gm->compGetFourCount()>0) {
    gm->compForCountDecrease();
  } else {
    mexWarnMsgTxt("Too many GPUend statements. Skipping the last one.");
    return;
  }

  try {


    CLEARBUFFER
      sprintf(buffer, "GPUFORSTOP");
    gm->compRegisterInstruction(buffer);

    gm->debugPopIndent();
    gm->debugLog("> END GPUfor\n", 0);


  } catch (GPUexception ex) {
    gm->compAbort(NULL);
    mexErrMsgTxt(ex.getError());
  }


}
