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

#ifdef UNIX
#include <stdint.h>
#endif

#include "mex.h"
#ifndef MATLAB
#define MATLAB
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


void
mexFunction( int nlhs, mxArray *plhs[],
            int nrhs,  const mxArray *prhs[] )
{

  // Garbage collector
  MyGC mygc = MyGC();

  if (nrhs!=2)
    mexErrMsgTxt("Wrong number of arguments");

  // tmp
  mxArray *lhs[2];

  mexCallMATLAB(1, &lhs[0], 0, &lhs[0], "GPUmanager");\
    GPUmanager * GPUman = (GPUmanager *) (UINTPTR mxGetScalar(lhs[0]));
  mxDestroyArray(lhs[0]);

  if (GPUman->getCompileMode()==1) {
    GPUman->compAbort(ERROR_GPUMANAGER_COMPNOTIMPLEMENTED);
  }


  unsigned int strlen = mxGetM(prhs[0]);
  if (mxGetN(prhs[0]) > strlen)
    strlen = mxGetN(prhs[0]);
  char *modname = (char*) Mymalloc((strlen+1)*sizeof(char),&mygc);
  memset(modname,0,strlen+1);
  mxGetString(prhs[0], modname, strlen+1);

  strlen = mxGetM(prhs[1]);
  if (mxGetN(prhs[1]) > strlen)
    strlen = mxGetN(prhs[1]);
  char *filename = (char*) Mymalloc((strlen+1)*sizeof(char),&mygc);
  memset(filename,0,strlen+1);
  mxGetString(prhs[1], filename, strlen+1);


  // try to load module

  try {
    GPUman->registerUserModule(modname, filename);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }

}
