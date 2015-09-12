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

#include "cublas.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  
  if (nrhs != 0)
    mexErrMsgTxt("Wrong number of arguments");
  
  cublasStatus status;
  int N = 100;
  float* d_A = 0;
  int n2 = N * N;
  
  /* Initialize CUBLAS */
  mexPrintf("- Initializing CUBLAS\n");
  
  status = cublasInit();
  if (status != CUBLAS_STATUS_SUCCESS) {
    mexErrMsgTxt( "ERROR1: CUBLAS initialization error\n");
  }
  mexPrintf("...done\n");
  
  mexPrintf("- Running CUBLAS test\n");
  /* Allocate device memory for the matrices */
  status = cublasAlloc(n2, sizeof(d_A[0]), (void**)&d_A);
  if (status != CUBLAS_STATUS_SUCCESS) {
    mexErrMsgTxt( "ERROR2: device memory allocation error\n");
  }
  
  status = cublasFree(d_A);
  if (status != CUBLAS_STATUS_SUCCESS) {
    mexErrMsgTxt( "ERROR3: memory free error\n");
  }
  
  status = cublasShutdown();
  if (status != CUBLAS_STATUS_SUCCESS) {
    mexErrMsgTxt( "ERROR4: shutdown error\n");
  }
  
  /* end */
  mexPrintf("...done\n");
  
  
}
