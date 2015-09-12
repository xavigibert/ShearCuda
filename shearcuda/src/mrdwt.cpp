/*
File Name: mrdwt.c
Last Modification Date:	%G%	%U%
Current Version: %M%	%I%
File Creation Date: Wed Oct 12 08:44:43 1994
Author: Markus Lang  <lang@jazz.rice.edu>

Copyright: All software, documentation, and related files in this distribution
           are Copyright (c) 1994  Rice University

Permission is granted for use and non-profit distribution providing that this
notice be clearly maintained. The right to distribute any portion for profit
or as part of any commercial product is specifically reserved for the author.

Change History:

        Changed functions fpconv() and MRDWT() to be template functions, so
        we can use the same code to process both single and double
        precission data
        Xavier Gibert Serra <gibert@umiacs.umd.edu> Wed Jan 16 2013

        Fixed code such that the result has the same dimension as the 
        input for 1D problems. Also, added some standard error checking.
		Jan Erik Odegard <odegard@ece.rice.edu> Wed Jun 14 1995
*/

#include <math.h>
/*#include <malloc.h>*/
#include <stdio.h>
#include "mex.h"
#include "matrix.h"
#include "mrdwt_r.h"

#define even(x)  ((x & 1) ? 0 : 1)
#define isint(x) ((x - floor(x)) > 0.0 ? 0 : 1)


void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])

{
  double *Lr;
  int m, n, h_col, h_row, lh, L, i, po2, j;
  double mtest, ntest;
  mxClassID mexDataType = mxDOUBLE_CLASS;
  

  /* check for correct # of input variables */
  if (nrhs>3){
    mexErrMsgTxt("There are at most 3 input parameters allowed!");
    return;
  }
  if (nrhs<2){
    mexErrMsgTxt("There are at least 2 input parameters required!");
    return;
  }
  
  // Check input data types
  if (mxIsClass(prhs[0], "single") && mxIsClass(prhs[1], "single"))
  {
    mexDataType = mxSINGLE_CLASS;
  }
  else if (mxIsClass(prhs[0], "double") && mxIsClass(prhs[1], "double"))
  {
    mexDataType = mxDOUBLE_CLASS;
  }
  else
  {
    mexErrMsgIdAndTxt("MATLAB:atrousc:rhs",
            "Input arguments are neither single nor double.");
  }
  
  n = mxGetN(prhs[0]); 
  m = mxGetM(prhs[0]); 
  h_col = mxGetN(prhs[1]); 
  h_row = mxGetM(prhs[1]); 
  if (h_col>h_row)
    lh = h_col;
  else  
    lh = h_row;
  if (nrhs == 3){
    if (!mxIsDouble(prhs[2]))
    {
      mexErrMsgTxt("Input arguments are not double.");
    }      
    L = (int) *mxGetPr(prhs[2]);
    if (L < 0)
      mexErrMsgTxt("The number of levels, L, must be a non-negative integer");
  }
  else /* Estimate L */ {
    i=n;j=0;
    while (even(i)){
      i=(i>>1);
      j++;
    }
    L=m;i=0;
    while (even(L)){
      L=(L>>1);
      i++;
    }
    if(min(m,n) == 1)
      L = max(i,j);
    else
      L = min(i,j);
    if (L==0){
      mexErrMsgTxt("Maximum number of levels is zero; no decomposition can be performed!");
      return;
    }
  }
  /* Check the ROW dimension of input */
  if(m > 1){
    mtest = (double) m/pow(2.0, (double) L);
    if (!isint(mtest))
      mexErrMsgTxt("The matrix row dimension must be of size m*2^(L)");
  }
  /* Check the COLUMN dimension of input */
  if(n > 1){
    ntest = (double) n/pow(2.0, (double) L);
    if (!isint(ntest))
      mexErrMsgTxt("The matrix column dimension must be of size n*2^(L)");
  }
  mwSize dims[2] = {m, n};
  plhs[0] = mxCreateNumericArray(2, dims, mexDataType, mxREAL);
  if (min(m,n) == 1) {
    dims[0] = m;
    dims[1] = L*n;
  }
  else {
    dims[0] = m;
    dims[1] = 3*L*n;
  }
  plhs[1] = mxCreateNumericArray(2, dims, mexDataType, mxREAL);
  plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
  Lr = mxGetPr(plhs[2]);
  *Lr = L;
  
  if (mexDataType == mxSINGLE_CLASS)
  {
    const float* x = (float*)mxGetData(prhs[0]);
    const float* h = (float*)mxGetData(prhs[1]);
  
    float* yh = (float*)mxGetData(plhs[1]);
    float* yl = (float*)mxGetData(plhs[0]);
  
    MRDWT<float>(x, m, n, h, lh, L, yl, yh);
  }
  else
  {
    const double* x = (double*)mxGetData(prhs[0]);
    const double* h = (double*)mxGetData(prhs[1]);
  
    double* yh = (double*)mxGetData(plhs[1]);
    double* yl = (double*)mxGetData(plhs[0]);
  
    MRDWT<double>(x, m, n, h, lh, L, yl, yh);      
  }
}
