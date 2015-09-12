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

  // tmp
  mxArray *lhs[2];
  mexCallMATLAB(2, &lhs[0], 0, NULL, "GPUmanager");
  GPUmanager *gm = (GPUmanager *) (UINTPTR mxGetScalar(lhs[0]));

  try {
    gm->debugLog("> STOP compilation\n", 0);
    gm->debugPushIndent();

    if (gm->getCompileMode() != 1) {
      mexErrMsgTxt("GPUmat compilation was not started.");
    }

    // check if all for-loops are closed
    if (gm->compGetFourCount()!=0) {
      gm->compAbort("Missing GPUend statement.");
    }

    CLEARBUFFER
      sprintf(buffer, "Output arguments -> %d\n", nrhs);
    gm->debugLog(buffer, 0);
    gm->debugPushIndent();

    // retunred variables
    for (int i = 0; i < nrhs; i++) {
      if ((mxIsClass(prhs[i], "GPUsingle"))
        || (mxIsClass(prhs[i], "GPUdouble")) || (mxIsClass(prhs[i],
        "GPUint32"))) {

          GPUtype *p = mxToGPUtype(prhs[i], gm);

          int pidx = gm->compGetContext(p, STACKGPUTYPE);
          if (pidx == -1) {
            CLEARBUFFER
              sprintf(
              buffer,
              "The argument N. %d passed to GPUcompileStop function is not available in current compilation context.",
              i + 1);
            gm->compAbort(buffer);

          }
          // now generate code
          CLEARBUFFER
            sprintf(buffer, "GPUMAT_RETURNGPUTYPE(GPUTYPEID(%d), %d)", pidx,
            i);
          gm->compRegisterInstruction(buffer);

          CLEARBUFFER
            sprintf(buffer, "OUTPUT ARG%d -> GPUARG%d\n", i, pidx);
          gm->debugLog(buffer, 0);
      } else {
        // not implemented
        mexErrMsgTxt("Only GPUtype are supported.");
      }
    }
    gm->debugPopIndent();

    gm->compRegisterInstruction(
      "//*****************************************************************");
    gm->compRegisterInstruction("GPUMAT_COMP_END");
    gm->compRegisterInstruction(
      "//*****************************************************************\n");

    // Compile mex file
    char *tmpfilename = gm->compGetFilename();
    char *filename = (char *) Mymalloc(
      (strlen(tmpfilename) + 1) * sizeof(char), &mygc);
    memset(filename, 0, strlen(tmpfilename) + 1);
    strcpy(filename, tmpfilename);

    // flush file and stop
    gm->compFlush();
    gm->compStop();

    if (filename != NULL) {
      CLEARBUFFER
        sprintf(buffer, "GPUcompileMEX({'%s'})", filename);
      lhs[0] = mxCreateString(buffer);
      mexCallMATLAB(0, NULL, 1, &lhs[0], "eval");
    } else {
      mexErrMsgTxt("Unexpected error in compilation.");
    }

    gm->debugPopIndent();

  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }

}
