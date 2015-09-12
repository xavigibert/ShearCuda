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

  if (nrhs == 0)
    mexErrMsgTxt("Wrong number of arguments");

  // tmp
  mxArray *lhs[2];
  mxArray *rhs[2];

  mexCallMATLAB(2, &lhs[0], 0, NULL, "GPUmanager");
  GPUmanager *gm = (GPUmanager *) (UINTPTR mxGetScalar(lhs[0]));

  if (gm->getCompileMode() == 1) {
    mexErrMsgTxt("GPUmat compilation already started.");
  }

  // compilation parameters
  int force = 0;
  int verbose = 0;

  try {

    // file name should not have extension. This is the assumption. That's why we add
    // 3 characters
    int extsize = 4;
    STRINGCONST char *cppext =".cpp";
    STRINGCONST char *mext =".m";

    unsigned int strlen = mxGetM(prhs[0]);
    if (mxGetN(prhs[0]) > strlen)
      strlen = mxGetN(prhs[0]);
    char *name = (char*) Mymalloc((strlen + 1) * sizeof(char), &mygc);
    memset(name, 0, strlen + 1);
    mxGetString(prhs[0], name, strlen + 1);

    char *filename = (char*) Mymalloc((strlen + 1 + extsize) * sizeof(char), &mygc);
    memset(filename, 0, strlen + 1 + extsize);
    //mxGetString(prhs[0], filename, strlen + 1);
    sprintf(filename,"%s%s",name,cppext);

    // generate also the Matlab help file
    char *mfilename = (char*) Mymalloc((strlen + 1 + extsize) * sizeof(char), &mygc);
    memset(mfilename, 0, strlen + 1 + extsize);
    //mxGetString(prhs[0], mfilename, strlen + 1);
    sprintf(mfilename,"%s%s",name,mext);

    int nargs = 0;
    // I have to parse input arguments before starting compilation
    for (int i = 1; i < nrhs; i++) {
      if (mxGetClassID(prhs[i])==mxCHAR_CLASS){
        unsigned int strlen = mxGetM(prhs[i]);
        if (mxGetN(prhs[i]) > strlen)
          strlen = mxGetN(prhs[i]);
        char *str = (char*) Mymalloc((strlen + 1) * sizeof(char), &mygc);
        memset(str, 0, strlen + 1);
        mxGetString(prhs[i], str, strlen + 1);
        // now parse parameter
        if (strcmp(str,"-f")==0) {
          force = 1;
        } else if (strcmp(str,"-verbose0")==0) {
          verbose = 0;
        } else if (strcmp(str,"-verbose1")==0) {
          verbose = 1;
        } else if (strcmp(str,"-verbose2")==0) {
          verbose = 2;
        } else if (strcmp(str,"-verbose3")==0) {
          verbose = 3;
        } else if (strcmp(str,"-verbose4")==0) {
          verbose = 4;
        } else if (strcmp(str,"-verbose40")==0) {
          verbose = 40;
        } else {
          CLEARBUFFER
            sprintf(buffer, "Unrecognized option -> %s\n", str);
          mexErrMsgTxt(buffer);
        }
      } else {
        // update the number of input arguments
        nargs++;
      }

    }

    rhs[0] = (mxArray*)prhs[0];
    rhs[1] = mxCreateDoubleScalar(force);

    // check if file exists
    mexCallMATLAB(0, NULL, 2, &rhs[0], "GPUcompileCheckFile");

    // First write mfilename and then start the .cpp compilation
    gm->compStart(mfilename,0); // no header
    CLEARBUFFER
      sprintf(buffer, "%% %s - GPUmat compiled function", name);
    gm->compRegisterInstruction(buffer);

    gm->compRegisterInstruction("% SYNTAX");
    CLEARBUFFER
      sprintf(buffer, "%% %s ( ARGS ), where ARGS are:", name);
    gm->compRegisterInstruction(buffer);

    int argsindex = 0;

    for (int i = 1; i < nrhs; i++) {
      if ((mxIsClass(prhs[i], "GPUsingle"))
        || (mxIsClass(prhs[i], "GPUdouble")) || (mxIsClass(prhs[i],
        "GPUint32"))) {
          CLEARBUFFER
            sprintf(buffer, "%% ARGS(%d) - GPU variable (GPUdouble, GPUsingle, ...)", argsindex);
          gm->compRegisterInstruction(buffer);
          argsindex++;

      } else if (mxGetClassID(prhs[i])==mxCHAR_CLASS){

      } else {
        CLEARBUFFER
          sprintf(buffer, "%% ARGS(%d) - Matlab variable", argsindex);
        gm->compRegisterInstruction(buffer);
        argsindex++;
      }
    }

    // flush and stop
    gm->compFlush();
    gm->compStop();

    /////////////////////////////////////////////////////////////////////
    // Now write the .cpp file
    gm->compStart(filename);

    // set verbosity
    gm->debugSetVerbose(verbose);

    gm->debugLog("> START compilation\n", 0);
    gm->debugPushIndent();

    gm->compRegisterInstruction("// MATLAB mex",1);
    gm->compRegisterInstruction("#define MATLABMEX",1);

    gm->compRegisterInstruction("#include <stdio.h>",1);
    gm->compRegisterInstruction("#include <string.h>",1);
    gm->compRegisterInstruction("#include <stdarg.h>",1);
    gm->compRegisterInstruction("#ifdef UNIX",1);
    gm->compRegisterInstruction("#include <stdint.h>",1);
    gm->compRegisterInstruction("#endif",1);
    gm->compRegisterInstruction("#include \"mex.h\"",1);
    gm->compRegisterInstruction("#include \"cuda.h\"",1);
    gm->compRegisterInstruction("#include \"cuda_runtime.h\"",1);
    gm->compRegisterInstruction("#include \"GPUmat.hh\"",1);
    gm->compRegisterInstruction("#include \"GPUmatCompiler.hh\"",1);
    gm->compRegisterInstruction("GPUMAT_COMP_HEADER",1);

    CLEARBUFFER
      sprintf(buffer, "GPUMAT_COMP_MEX0(%d)", nargs);
    gm->compRegisterInstruction(buffer,1);

    CLEARBUFFER
      sprintf(buffer, "File name -> %s\n", filename);
    gm->debugLog(buffer, 0);

    // now push compilation variables
    CLEARBUFFER
      sprintf(buffer, "Input arguments -> %d\n", nargs);
    gm->debugLog(buffer, 0);
    gm->debugPushIndent();


    argsindex = 0;
    for (int i = 1; i < nrhs; i++) {
      if ((mxIsClass(prhs[i], "GPUsingle"))
        || (mxIsClass(prhs[i], "GPUdouble")) || (mxIsClass(prhs[i],
        "GPUint32"))) {

          GPUtype *p = mxToGPUtype(prhs[i], gm);


          gm->compPush(p, STACKGPUTYPE);
          int pidx = gm->compGetContext(p, STACKGPUTYPE);

          // now generate code
          CLEARBUFFER
            sprintf(buffer, "GPUMAT_READGPUTYPE(GPUTYPEID(%d), %d)", pidx, argsindex);
          gm->compRegisterInstruction(buffer);

          CLEARBUFFER
            sprintf(buffer, "INPUT ARG%d -> GPUARG%d\n", argsindex, pidx);
          gm->debugLog(buffer, 0);
          argsindex++;
      } else if (mxGetClassID(prhs[i])==mxCHAR_CLASS){
        // skip it already parsed
      } else {
        gm->compPush((mxArray *) prhs[i], STACKMX);
        int mx_id = gm->compGetContext((mxArray *) prhs[i], STACKMX);

        // now generate code
        //CLEARBUFFER
        //sprintf(buffer, "DECLARE_MXNID(%d, %d)", mx_id, 1);
        //gm->compRegisterInstruction(buffer);
        //CLEARBUFFER
        //sprintf(buffer, "DECLARE_MXID(%d, %d)", mx_id, 1);
        //gm->compRegisterInstruction(buffer);

        CLEARBUFFER
          sprintf(buffer, "GPUMAT_READMX(%d, 0, %d)", mx_id, argsindex);
        gm->compRegisterInstruction(buffer);
        CLEARBUFFER
          sprintf(buffer, "INPUT ARG%d -> MXARG%d\n", argsindex, mx_id);
        gm->debugLog(buffer, 0);
        argsindex++;
      }
    }
    gm->debugPopIndent();

  } catch (GPUexception ex) {
    gm->compAbort(NULL);
    mexErrMsgTxt(ex.getError());
  }

  gm->debugPopIndent();

}
