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

  if (nrhs != 1)
    mexErrMsgTxt("Wrong number of arguments (1 argument expected)");

  // tmp
  mxArray *lhs[2];
  mxArray *rhs[2];

  mexCallMATLAB(2, &lhs[0], 0, NULL, "GPUmanager");
  GPUmanager *gm = (GPUmanager *) (UINTPTR mxGetScalar(lhs[0]));

  if (gm->getCompileMode() != 1) {
    mexErrMsgTxt("GPUfor can be used only in compilation mode.");
  }

  double start = 0;
  double step = 0;
  double stop = 0;


  if (mxGetClassID(prhs[0])==mxDOUBLE_CLASS){
    // have to understand the limits in the iterator

    // Do the following:
    // 1) Check if indexes are consecutive by scanning the array
    // 2) If consecutive, store only start/last index and stride
    // 3) If not consecutive generate error

    mxArray *mx = (mxArray*) prhs[0];
    int n = (int) mxGetNumberOfElements(mx);
    double *tmpidx = mxGetPr(mx);


    int delta = 1;
    if (n > 1) {
      delta = (int) (tmpidx[1] - tmpidx[0]);
    }

    int consecutive = 1;
    for (int jj = 1; jj < n; jj++) {
      int newdelta = (int) (tmpidx[jj] - tmpidx[jj - 1]);
      consecutive = (newdelta == delta) && consecutive &&(delta!=0);
      delta = newdelta;
    }

    if (consecutive) {
      start = tmpidx[0];
      stop = tmpidx[n-1];
      step = delta;

    } else {
      gm->compAbort(NULL);
      mexErrMsgTxt(ERROR_GPUFOR_ITWRONG);
    }

  } else {
    gm->compAbort(NULL);
    mexErrMsgTxt(ERROR_GPUFOR_ITDOUBLE);
  }



  try {

    // update for-loop count
    gm->compForCountIncrease();

    gm->debugLog("> START GPUfor\n", 0);
    gm->debugPushIndent();

    CLEARBUFFER
      sprintf(buffer, "  START -> %f\n", start);
    gm->debugLog(buffer, 3);

    CLEARBUFFER
      sprintf(buffer, "  STEP  -> %f\n", step);
    gm->debugLog(buffer, 3);

    CLEARBUFFER
      sprintf(buffer, "  STOP  -> %f\n", stop);
    gm->debugLog(buffer, 3);

    // push iterator

    gm->compPush((mxArray *) prhs[0], STACKMX);
    int mx_id = gm->compGetContext((mxArray *) prhs[0], STACKMX);

    CLEARBUFFER
      if (start<=stop)
        sprintf(buffer, "GPUFORSTART(%d, %s, %f, %f, %f)", mx_id, "<=", start, step, stop);
      else
        sprintf(buffer, "GPUFORSTART(%d, %s, %f, %f, %f)", mx_id, ">=", start, step, stop);

    gm->compRegisterInstruction(buffer);

    // now generate code
    /*CLEARBUFFER
    sprintf(buffer, "DECLARE_MXNID(%d, %d)", mx_id, 1);
    gm->compRegisterInstruction(buffer);
    CLEARBUFFER
    sprintf(buffer, "DECLARE_MXID(%d, %d)", mx_id, 1);
    gm->compRegisterInstruction(buffer);*/
    CLEARBUFFER
      sprintf(buffer, "ASSIGN_MXID_DOUBLE(%d, %d, MXIT(%d))", mx_id, 0, mx_id);
    gm->compRegisterInstruction(buffer);

  } catch (GPUexception ex) {
    gm->compAbort(NULL);
    mexErrMsgTxt(ex.getError());
  }



}
