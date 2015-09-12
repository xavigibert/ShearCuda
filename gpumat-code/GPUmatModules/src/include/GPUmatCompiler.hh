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

#if !defined(GPUMATCOMPILER_HH_)
#define GPUMATCOMPILER_HH_

// MATLAB compilation
#ifdef MATLABMEX

// NAMES
#define GPUTYPEID(N) gtP##N
#define MXNID(N)     mxNRHS##N
#define MXID(N)      mxP##N
#define MXIT(N)      mxIT##N


// GPUTYPE
#define DECLARE_GPUTYPEID(N)              GPUtype GPUTYPEID(N);

// MX
#define DECLARE_MXNID(N, D)             int MXNID(N) = D;
#define DECLARE_MXID(N, D)              mxArray *MXID(N)[D];
#define DECLARE_MXID_DOUBLEPTR_REAL(N, D)     double *mxP##N##_##D##_PR = mxGetPr(MXID(N)[D]);
#define DECLARE_MXID_DOUBLEPTR_IMAG(N, D)     double *mxP##N##_##D##_PI = mxGetPi(MXID(N)[D]);

#define DECLARE_MXID_SINGLEPTR_REAL(N, D)     float *mxP##N##_##D##_PR = (float *) mxGetPr(MXID(N)[D]);
#define DECLARE_MXID_SINGLEPTR_IMAG(N, D)     float *mxP##N##_##D##_PI = (float *) mxGetPi(MXID(N)[D]);

#define CREATE_MXID_CELL(N, I, D)             MXDIMS[1] = D;MXID(N)[I] = mxCreateCellArray(2, MXDIMS);
#define CREATE_MXID_DOUBLEARRAY(N, I, D, CPX) MXDIMS[1] = D;MXID(N)[I] = mxCreateNumericArray(2, MXDIMS, mxDOUBLE_CLASS, CPX);
#define CREATE_MXID_SINGLEARRAY(N, I, D, CPX) MXDIMS[1] = D;MXID(N)[I] = mxCreateNumericArray(2, MXDIMS, mxSINGLE_CLASS, CPX);

#define ASSIGN_MXID_MXID(N, I, M, J)     MXID(N)[I] = MXID(M)[J];

#define ASSIGN_MXID_DOUBLE(N, I, D)     MXID(N)[I] = mxCreateDoubleScalar(D);
#define ASSIGN_MXID_CHAR(N, I, D)       MXID(N)[I] = mxCreateString(D);

#define ASSIGN_MXID_CELL_DOUBLE(N, I, J, D) mxSetCell(MXID(N)[I], J, mxCreateDoubleScalar(D));
#define ASSIGN_MXID_CELL_CHAR(N, I, J, D)   mxSetCell(MXID(N)[I], J, mxCreateString(D));
#define ASSIGN_MXID_CELL_MXID(N, I, J, M, D)   mxSetCell(MXID(N)[I], J, MXID(M)[D]);

#define ASSIGN_MXID_SUBSREF_STRUCT(N,I,M,D)\
MXDIMS[1] = 1;\
MXID(N)[D] = mxCreateStructArray(2, MXDIMS, 2, SUBSREF_FIELDS);\
mxSetFieldByNumber(MXID(N)[D],0,0,mxCreateString("()"));\
mxSetFieldByNumber(MXID(N)[D],1,0,MXID(M)[D]);

#define ASSIGN_MXID_DOUBLEARRAY_IMAG(N, I, J, D) mxP##N##_##I##_PI[J] = D;
#define ASSIGN_MXID_DOUBLEARRAY_REAL(N, I, J, D) mxP##N##_##I##_PR[J] = D;

#define ASSIGN_MXID_SINGLEARRAY_IMAG(N, I, J, D) mxP##N##_##I##_PI[J] = D;
#define ASSIGN_MXID_SINGLEARRAY_REAL(N, I, J, D) mxP##N##_##I##_PR[J] = D;


// GPUFOR
#define GPUFORSTART(N, LT, START, STEP, STOP) for (double MXIT(N) = (START); MXIT(N) LT (STOP); MXIT(N) = MXIT(N) + (STEP)) {
#define GPUFORSTOP }

// GPUTYPE
#define GPUTYPE_EMPTY(T)   gm->gputype.create((gpuTYPE_t) T, 0, NULL, NULL)
#define CREATE_GPUTYPE_EMPTY(D,T)   GPUTYPEID(D) = gm->gputype.create((gpuTYPE_t) T, 0, NULL, NULL);

// HEADER
#define GPUMAT_COMP_HEADER static int init = 0; static GPUmat *gm;

// GPUMAT_COMP_MEX0
#define GPUMAT_COMP_MEX0(NPAR) \
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { \
  MyGC mygc; \
  mwSize MXDIMS[2] = {1,1};\
  const char *SUBSREF_FIELDS[] = {"type", "subs"};\
  if (nrhs != NPAR ) \
    mexErrMsgTxt("Wrong number of arguments");\
  if (init == 0) {\
    gm = gmGetGPUmat();\
    gmCheckGPUmat(gm);\
    init = 1;\
  }\
  if (gm->comp.getCompileMode() == 1) {\
    gm->comp.abort("Cannot execute GPUmat compile code in compilation mode.");\
  }

// GPUMAT_READGPUTYPE
#define GPUMAT_READGPUTYPE(GT, MXINDEX)\
GT = gm->gputype.getGPUtype(prhs[MXINDEX]);

// GPUMAT_RETURNGPUTYPE
#define GPUMAT_RETURNGPUTYPE(GT, MXINDEX)\
if (MXINDEX == 0) {\
  plhs[MXINDEX] = gm->gputype.createMxArray(GT);\
} else {\
  if (nlhs > MXINDEX)\
    plhs[MXINDEX] = gm->gputype.createMxArray(GT);\
}

// GPUMAT_READMX
#define GPUMAT_READMX(N, D, MXINDEX)\
MXID(N)[D] = (mxArray*) prhs[MXINDEX];

// COMPILATION END
#define GPUMAT_COMP_END }

// FUNCTION CONTEXT
#define GPUMAT_FUN_BEGIN {
#define GPUMAT_FUN_END }

#endif

// include numeric functions
#include "GPUmatCompilerNumerics.hh"

// Clone
#define GPUMAT_Clone(OUT, IN) \
OUT = gm->gputype.clone(IN);

// numel
#define GPUMAT_Numel(OUT, IN) \
OUT = mxCreateDoubleScalar(gm->gputype.getNumel(IN));

// mxMemCpyDtoD
#define GPUMAT_mxMemCpyDtoD(DST, SRC, NRHS, PRHS) \
gm->gputype.mxMemCpyDtoD(DST, SRC, NRHS, (const mxArray**) PRHS);

// mxMemCpyHtoD
#define GPUMAT_mxMemCpyHtoD(DST, NRHS, PRHS) \
gm->gputype.mxMemCpyHtoD(DST, NRHS, (const mxArray**) PRHS);

// mxRepmatDrv
#define GPUMAT_mxRepmatDrv(R, IN, NRHS, PRHS) \
R = gm->gputype.mxRepmatDrv(IN, NRHS, (const mxArray**) PRHS);

// mxPermuteDrv
#define GPUMAT_mxPermuteDrv(R, IN, NRHS, PRHS) \
R = gm->gputype.mxPermuteDrv(IN, NRHS, (const mxArray**) PRHS);

// mxEyeDrv
#define GPUMAT_mxEyeDrv(R, IN, NRHS, PRHS) \
R = gm->gputype.mxEyeDrv(IN, NRHS, (const mxArray**) PRHS);

// mxZerosDrv
#define GPUMAT_mxZerosDrv(R, IN, NRHS, PRHS) \
R = gm->gputype.mxZerosDrv(IN, NRHS, (const mxArray**) PRHS);

// Zeros
#define GPUMAT_Zeros(IN) \
gm->gputype.zeros(IN);

// mxOnesDrv
#define GPUMAT_mxOnesDrv(R, IN, NRHS, PRHS) \
R = gm->gputype.mxOnesDrv(IN, NRHS, (const mxArray**) PRHS);

// Ones
#define GPUMAT_Ones(IN) \
gm->gputype.ones(IN);

// GPUeye
#define GPUMAT_GPUeye(OUT) \
gm->gputype.eye(OUT);

// mxToGPUtype
#define GPUMAT_mxToGPUtype(R, PRHS) \
R = gm->gputype.mxToGPUtype(PRHS);

// mxAssign
#define GPUMAT_mxAssign(LHS, RHS, DIR, NRHS, PRHS) \
gm->aux.mxAssign(LHS, RHS, DIR, NRHS, (const mxArray**) PRHS);

// mxSliceDrv
#define GPUMAT_mxSliceDrv(OUT, RHS, NRHS, PRHS) \
OUT = gm->aux.mxSliceDrv(RHS, NRHS, (const mxArray**) PRHS);

// mxFill
#define GPUMAT_mxFill(DST, NRHS, PRHS) \
gm->gputype.mxFill(DST, NRHS, (const mxArray**) PRHS);

// Colon
#define GPUMAT_Colon(DST, TYPE, J, D, K) \
DST = gm->gputype.colon((gpuTYPE_t) TYPE, J, D, K);

// mxColon
#define GPUMAT_mxColonDrv(OUT, IN, NRHS, PRHS) \
OUT = gm->gputype.mxColonDrv(IN, NRHS, (const mxArray**) PRHS);

// ComplexDrv
#define GPUMAT_ComplexDrv(OUT, RE) \
OUT = gm->gputype.realToComplex(RE);

// ComplexDrv1
#define GPUMAT_ComplexDrv1(OUT, RE, IM) \
OUT = gm->gputype.realImagToComplex(RE, IM);

// GPUcomplex
#define GPUMAT_GPUcomplex(OUT, RE) \
gm->gputype.realimag(OUT, RE, RE, 0, 1);

// GPUcomplex1
#define GPUMAT_GPUcomplex1(OUT, RE, IM) \
gm->gputype.realimag(OUT, RE, IM, 0, 0);

// DoubleToFloat
#define GPUMAT_DoubleToFloat(OUT, IN) \
OUT = gm->gputype.doubleToFloat(IN);

// FloatToDouble
#define GPUMAT_FloatToDouble(OUT, IN) \
OUT = gm->gputype.floatToDouble(IN);

// FFT1Drv
#define GPUMAT_FFT1Drv(OUT, IN) \
OUT = gm->fft.FFT1Drv(IN);

// FFT2Drv
#define GPUMAT_FFT2Drv(OUT, IN) \
OUT = gm->fft.FFT2Drv(IN);

// FFT3Drv
#define GPUMAT_FFT3Drv(OUT, IN) \
OUT = gm->fft.FFT3Drv(IN);

// IFFT1Drv
#define GPUMAT_IFFT1Drv(OUT, IN) \
OUT = gm->fft.IFFT1Drv(IN);

// IFFT2Drv
#define GPUMAT_IFFT2Drv(OUT, IN) \
OUT = gm->fft.IFFT2Drv(IN);

// IFFT3Drv
#define GPUMAT_IFFT3Drv(OUT, IN) \
OUT = gm->fft.IFFT3Drv(IN);



// RAND
// mxRandDrv
#define GPUMAT_mxRandDrv(R, IN, NRHS, PRHS) \
R = gm->rand.mxRandDrv(IN, NRHS, (const mxArray**) PRHS);

// Rand
#define GPUMAT_Rand(IN) \
gm->rand.rand(IN);

// mxRandnDrv
#define GPUMAT_mxRandnDrv(R, IN, NRHS, PRHS) \
R = gm->rand.mxRandnDrv(IN, NRHS, (const mxArray**) PRHS);

// Randn
#define GPUMAT_Randn(IN) \
gm->rand.randn(IN);


#endif
