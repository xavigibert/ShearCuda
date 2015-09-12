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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef UNIX
#include <stdint.h>
#endif


//#include "cutil.h"
#include "cublas.h"
#include "cuda_runtime.h"
#include "cuda.h"
//extern "C" cudaError_t  cudaFree(void *devPtr);
#include "cudalib_common.h"
#include "cudalib_error.h"
#include "cudalib.h"

#include "GPUcommon.hh"
#include "GPUerror.hh"
#include "Queue.hh"
#include "GPUstream.hh"
#include "GPUmanager.hh"
#include "GPUtype.hh"


GPUmatResult_t
GPUstream::run() {
  GPUmatResult status = GPUmatSuccess;
  status =  (*func)(plhs,prhs,0);

  // must clean the pointers
	// don't know the type, only caller knows.
  /*for (int i=0;i<nlhs;i++) {
    delete this->plhs[i];
  }

  for (int i=0;i<nrhs;i++) {
    delete this->prhs[i];
  }*/
  return status;
}

GPUstream::GPUstream(GPUmatResult_t (*f)(void **, void **,int),
                     int nlhs, void ** lhs, int nrhs, void **rhs) {

  func = f;
  this->nlhs = nlhs;
  this->nrhs = nrhs;

  this->plhs = (void **) Mymalloc(nlhs*sizeof(void *));
  this->prhs = (void **) Mymalloc(nrhs*sizeof(void *));

  for (int i=0;i<nlhs;i++) {
    this->plhs[i] = lhs[i];
  }

  for (int i=0;i<nrhs;i++) {
    this->prhs[i] = rhs[i];
  }
}

void
GPUstream::print() {
  (*func)(plhs,prhs,1);

}




