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
#include "GPUnumeric.hh"

#include "MatlabTemplates.hh"

struct mxArray_tag {
  void    *reserved;
  int      reserved1[2];
  void    *reserved2;
  size_t  number_of_dims;
  unsigned int reserved3;
  struct {
    unsigned int    flag0 : 1;
    unsigned int    flag1 : 1;
    unsigned int    flag2 : 1;
    unsigned int    flag3 : 1;
    unsigned int    flag4 : 1;
    unsigned int    flag5 : 1;
    unsigned int    flag6 : 1;
    unsigned int    flag7 : 1;
    unsigned int    flag7a: 1;
    unsigned int    flag8 : 1;
    unsigned int    flag9 : 1;
    unsigned int    flag10 : 1;
    unsigned int    flag11 : 4;
    unsigned int    flag12 : 8;
    unsigned int    flag13 : 8;
  }   flags;
  size_t reserved4[2];
  union {
    struct {
      void  *pdata;
      void  *pimag_data;
      void  *reserved5;
      size_t reserved6[3];
    }   number_array;
  }   data;
};

void print(mxArray *tmp) {
  mexPrintf("%p\n",tmp->reserved);
  mexPrintf("%d\n",tmp->reserved1[0]);
  mexPrintf("%d\n",tmp->reserved1[1]);
  mexPrintf("%p\n",tmp->reserved2);
  mexPrintf("%d\n",tmp->number_of_dims);
  mexPrintf("%d\n",tmp->reserved3);
  mexPrintf("%d\n",tmp->reserved4[0]);
  mexPrintf("%d\n",tmp->reserved4[1]);

  mexPrintf("%p\n",tmp->data.number_array.pdata);
  mexPrintf("%p\n",tmp->data.number_array.pimag_data);
  mexPrintf("%p\n",tmp->data.number_array.reserved5);
  mexPrintf("%d\n",tmp->data.number_array.reserved6[0]);
  mexPrintf("%d\n",tmp->data.number_array.reserved6[1]);
  mexPrintf("%d\n",tmp->data.number_array.reserved6[2]);

  mexPrintf("%d\n",tmp->flags.flag0);
  mexPrintf("%d\n",tmp->flags.flag1);
  mexPrintf("%d\n",tmp->flags.flag2);
  mexPrintf("%d\n",tmp->flags.flag3);
  mexPrintf("%d\n",tmp->flags.flag4);
  mexPrintf("%d\n",tmp->flags.flag5);
  mexPrintf("%d\n",tmp->flags.flag6);
  mexPrintf("%d\n",tmp->flags.flag7);
  mexPrintf("%d\n",tmp->flags.flag8);
  mexPrintf("%d\n",tmp->flags.flag9);
  mexPrintf("%d\n",tmp->flags.flag10);
  mexPrintf("%d\n",tmp->flags.flag11);
  mexPrintf("%d\n",tmp->flags.flag12);
  mexPrintf("%d\n",tmp->flags.flag13);
}
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  if (nrhs != 5)
    mexErrMsgTxt("Wrong number of arguments");


  // 0. GPUmanager
  // 1. start
  // 2. stop
  // 2. GPUtype
  // 3. gpuTYPE_t

  GPUmanager *GPUman = (GPUmanager *) (UINTPTR mxGetScalar(prhs[0]));
  int start = (int) mxGetScalar(prhs[1]);
  int stop  = (int) mxGetScalar(prhs[2]);
  mxArray *mx = (mxArray *) prhs[3];
  gpuTYPE_t type = (gpuTYPE_t) ((int) mxGetScalar(prhs[4]));



  try {
    int newslot = -1;
    void *mxtmp = GPUman->extCacheGetFreeSlot(&newslot, gpuNOTDEF);
    //int nel = mxGetNumberOfElements(mx);
    for (int i = start; i < stop; i++) {
      mxArray *tmpmx = mxGetCell(mx, i);
      //if (newslot!=slot) {
      //  throw GPUexception(GPUmatError,ERROR_MXID_REGISTERGT);
      //}
      //print(tmpmx);
      GPUman->extCacheRegisterCachedPtrBySlot(i, mxID(tmpmx), tmpmx, type);
    }

  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }


}

