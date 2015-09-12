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

  MyGC mgc = MyGC();

  gpuTYPE_t type;
  double *tmpsize;
  int    ndims;
  GPUmanager *GPUman;
  int * size;

  int i;
  if (nrhs!=4)
    mexErrMsgTxt("Wrong number of arguments");

  type      =  (gpuTYPE_t) ((int) mxGetScalar(prhs[0]));
  ndims     =  (int) mxGetScalar(prhs[1]);
  tmpsize      =  mxGetPr(prhs[2]);
  GPUman    =  (GPUmanager *) (UINTPTR mxGetScalar(prhs[3]));

  /* should convert size to int * */
  if (ndims > 0) {
    size = (int *) Mymalloc(ndims*sizeof(int), &mgc);
    for (i=0;i<ndims;i++) {
      size[i] = (int) tmpsize[i];
    }
  } else {
    size = NULL;
  }

  GPUtype *p;
  try {
    p = new GPUtype(type, ndims , size, GPUman);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }

  /* free size */
  //Myfree(size);

  plhs[0] = mxCreateDoubleScalar(UINTPTR p);


}
