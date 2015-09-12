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

#if !defined(GPUOP_H_)
#define GPUOP_H_

#define RANGEBEGIN 0
#define RANGEEND   -1
typedef struct RangeS{
  int inf;
  int sup;
  int stride;
  int begin;
  int end;

  // we allow different type of indexes
  // iindx -> integer
  // findx -> float
  // dindx -> double

  int * iindx;
  float  *findx;
  double *dindx;

  void * gindx;
  void * gindxptr;

  RangeS *next;

} Range;



/* In this file all the operations that are specific to the GPU
* 1. These functions have in common the fact that can be streamed
* 2. They must return an error message using GPUmanager error functions
*/

/*************************************************************************
* GPUopMacro
*************************************************************************/
#define CGPUOP(GPUOPFUN, FUN, KER, INARG1, INARG2, OUTARG1)\
  GPUmatResult_t GPUop##GPUOPFUN(int N, INARG1 d_idata1, INARG2 d_idata2, OUTARG1 d_odata, GPUmanager* GPUman) {\
  GPUmatResult_t status = GPUmatSuccess;\
  CUDALIBResult cudalibres = FUN(N, UINTPTR d_idata1, UINTPTR d_idata2, UINTPTR d_odata, (GPUman->getCuFunction(N_##KER##_KERNEL)));\
  if (cudalibres != CUDALIBSuccess) {\
  char buffer[300];\
  sprintf(buffer,"Kernel execution error (%d)",cudalibres);\
  throw GPUexception(GPUmatError, buffer);\
  }\
  return status;\
}
#define HGPUOP(GPUOPFUN, FUN, INARG1, INARG2, OUTARG1)\
  GPUmatResult_t GPUop##GPUOPFUN(int N, INARG1 d_idata1, INARG2 d_idata2, OUTARG1 d_odata, GPUmanager*);

/*************************************************************************
* GPUop1Macro
*************************************************************************/
#define CGPUOP1(GPUOPFUN, FUN, KER, INARG1, OUTARG1)\
  GPUmatResult_t GPUop##GPUOPFUN(int N, INARG1 d_idata1, OUTARG1 d_odata, GPUmanager *GPUman) {\
  GPUmatResult_t status = GPUmatSuccess;\
  CUDALIBResult cudalibres = FUN(N, UINTPTR d_idata1, UINTPTR d_odata, (GPUman->getCuFunction(N_##KER##_KERNEL)));\
  if (cudalibres != CUDALIBSuccess) {\
  char buffer[300];\
  sprintf(buffer,"Kernel execution error (%d)",cudalibres);\
  throw GPUexception(GPUmatError, buffer);\
  }\
  return status;\
}
#define HGPUOP1(GPUOPFUN, FUN, INARG1, OUTARG1)\
  GPUmatResult_t GPUop##GPUOPFUN(int N, INARG1 d_idata1, OUTARG1 d_odata, GPUmanager *GPUman);
/*************************************************************************
* GPUop2Macro
*************************************************************************/
#define CGPUOP2(GPUOPFUN, FUN, KER, INARG1, INARG2, OUTARG1)\
  GPUmatResult_t GPUop##GPUOPFUN(int N, INARG1 d_idata1, INARG2 d_idata2, OUTARG1 d_odata, GPUmanager* GPUman) {\
  GPUmatResult_t status = GPUmatSuccess;\
  CUDALIBResult cudalibres = FUN(N, UINTPTR d_idata1, d_idata2, UINTPTR d_odata, (GPUman->getCuFunction(N_##KER##_KERNEL)));\
  if (cudalibres != CUDALIBSuccess) {\
  char buffer[300];\
  sprintf(buffer,"Kernel execution error (%d)",cudalibres);\
  throw GPUexception(GPUmatError, buffer);\
  }\
  return status;\
}
#define HGPUOP2(GPUOPFUN, FUN, INARG1, INARG2, OUTARG1)\
  GPUmatResult_t GPUop##GPUOPFUN(int N, INARG1 d_idata1, INARG2 d_idata2, OUTARG1 d_odata, GPUmanager*);

/*************************************************************************
* GPUop3Macro
*************************************************************************/
#define CGPUOP3(GPUOPFUN)\
  GPUtype * GPUop##GPUOPFUN##Drv(GPUtype &p) { \
  GPUmatResult_t status = GPUmatSuccess; \
  GPUtype *r; \
  GPUmanager *GPUman = p.getGPUmanager(); \
  if (GPUman->getCompileMode()==1) {\
  r = new GPUtype(GPUman);\
  GPUman->compPush(r,0);\
  GPUman->compFunctionStart("GPUMAT_"#GPUOPFUN"Drv");\
  GPUman->compFunctionSetParamGPUtype(r);\
  GPUman->compFunctionSetParamGPUtype(&p);\
  GPUman->compFunctionEnd();\
  } else {\
  r = arg1op_drv(NULL, p, (GPUmatResult_t(*)(GPUtype&, GPUtype&)) GPUop##GPUOPFUN);\
  }\
  return r;\
}\

#define HGPUOP3(GPUOPFUN)\
  GPUtype * GPUop##GPUOPFUN##Drv(GPUtype &p);\

/*************************************************************************
* GPUop4Macro
*************************************************************************/
#define CGPUOP4(GPUOPFUN, GPUKERNEL)\
  GPUmatResult_t GPUop##GPUOPFUN(GPUtype &p, GPUtype &r) {\
  GPUmatResult_t status = GPUmatSuccess;\
  cudaError_t cudastatus = cudaSuccess;\
  GPUmanager * GPUman = p.getGPUmanager();\
  if (GPUman->getCompileMode()==1) {\
  GPUman->compFunctionStart("GPUMAT_"#GPUOPFUN);\
  GPUman->compFunctionSetParamGPUtype(&r);\
  GPUman->compFunctionSetParamGPUtype(&p);\
  GPUman->compFunctionEnd();\
  } else {\
  status = arg1op_common(NULL, p, r,\
  N_##GPUKERNEL##F_KERNEL, N_##GPUKERNEL##C_KERNEL,\
  N_##GPUKERNEL##D_KERNEL, N_##GPUKERNEL##CD_KERNEL);\
  }\
  return status;\
}\

#define HGPUOP4( GPUOPFUN )\
  GPUmatResult_t GPUop##GPUOPFUN(GPUtype &p, GPUtype &r);\

/*************************************************************************
* GPUop5Macro
*************************************************************************/
/*#define CGPUOP5(GPUOPFUN)\
GPUtype * GPUop##GPUOPFUN##Drv(GPUtype &p, GPUtype &q) {\
GPUmatResult_t status = GPUmatSuccess;\
GPUtype *r;\
GPUmanager *GPUman = p.getGPUmanager();\
if (GPUman->executionDelayed()) {\
} else {\
r = arg2op_drv(0, p, q, (GPUmatResult_t(*)(GPUtype&, GPUtype&, GPUtype&)) GPUop##GPUOPFUN## );\
}\
return r;\
}\

#define HGPUOP5(GPUOPFUN)\
GPUtype * GPUop##GPUOPFUN##Drv(GPUtype &p, GPUtype &q);\*/

/*************************************************************************
* GPUop5Macro
*************************************************************************/
#define CGPUOP5(GPUOPFUN)\
  GPUtype * GPUop##GPUOPFUN##Drv(GPUtype &p, GPUtype &q) {\
  GPUmatResult_t status = GPUmatSuccess;\
  GPUtype *r;\
  GPUmanager *GPUman = p.getGPUmanager();\
  if (GPUman->getCompileMode()==1) {\
  r = new GPUtype(GPUman);\
  GPUman->compPush(r,0);\
  GPUman->compFunctionStart("GPUMAT_"#GPUOPFUN"Drv");\
  GPUman->compFunctionSetParamGPUtype(r);\
  GPUman->compFunctionSetParamGPUtype(&p);\
  GPUman->compFunctionSetParamGPUtype(&q);\
  GPUman->compFunctionEnd();\
  } else {\
  r = arg3op_drv(NULL, p, q, (GPUmatResult_t(*)(GPUtype&, GPUtype&, GPUtype&)) GPUop##GPUOPFUN );\
  }\
  return r;\
}\

#define HGPUOP5(GPUOPFUN)\
  GPUtype * GPUop##GPUOPFUN##Drv(GPUtype &p, GPUtype &q);\
  /*************************************************************************
* GPUop6Macro
*************************************************************************/
/*#define CGPUOP6(GPUOPFUN, GPUKERNEL)\
GPUmatResult_t GPUop##GPUOPFUN##(GPUtype &p, GPUtype &q, GPUtype &r) {\
GPUmatResult_t status = GPUmatSuccess;\
cudaError_t cudastatus = cudaSuccess;\
GPUmanager * GPUman = p.getGPUmanager();\
if (GPUman->executionDelayed()) {\
} else {\
status =  arg2op_common(0,p,q,r,\
N_##GPUKERNEL##F_KERNEL, N_##GPUKERNEL##F_SCALAR_KERNEL,\
N_##GPUKERNEL##C_KERNEL, N_##GPUKERNEL##C_SCALAR_KERNEL,\
N_##GPUKERNEL##D_KERNEL, N_##GPUKERNEL##D_SCALAR_KERNEL,\
N_##GPUKERNEL##CD_KERNEL, N_##GPUKERNEL##CD_SCALAR_KERNEL);\
}\
return status;\
}\

#define HGPUOP6(GPUOPFUN)\
GPUmatResult_t GPUop##GPUOPFUN##(GPUtype &p, GPUtype &q, GPUtype &r);\*/

/*************************************************************************
* GPUop6Macro
*************************************************************************/
/*#define CGPUOP6(GPUOPFUN, GPUKERNEL)\
GPUmatResult_t GPUop##GPUOPFUN##(GPUtype &p, GPUtype &q, GPUtype &r) {\
GPUmatResult_t status = GPUmatSuccess;\
cudaError_t cudastatus = cudaSuccess;\
GPUmanager * GPUman = p.getGPUmanager();\
if (GPUman->executionDelayed()) {\
} else {\
status =  arg2op2_common(0,p,q,r,\
N_##GPUKERNEL##F_KERNEL, \
N_##GPUKERNEL##C_KERNEL, \
N_##GPUKERNEL##D_KERNEL, \
N_##GPUKERNEL##CD_KERNEL);\
}\
return status;\
}\

#define HGPUOP6(GPUOPFUN)\
GPUmatResult_t GPUop##GPUOPFUN##(GPUtype &p, GPUtype &q, GPUtype &r);\*/

/*************************************************************************
* GPUop6Macro
*************************************************************************/
#define CGPUOP6(GPUOPFUN, GPUKERNEL)\
  GPUmatResult_t GPUop##GPUOPFUN(GPUtype &p, GPUtype &q, GPUtype &r) {\
  GPUmatResult_t status = GPUmatSuccess;\
  cudaError_t cudastatus = cudaSuccess;\
  GPUmanager * GPUman = p.getGPUmanager();\
  if (GPUman->getCompileMode()==1) {\
  GPUman->compFunctionStart("GPUMAT_"#GPUOPFUN);\
  GPUman->compFunctionSetParamGPUtype(&r);\
  GPUman->compFunctionSetParamGPUtype(&p);\
  GPUman->compFunctionSetParamGPUtype(&q);\
  GPUman->compFunctionEnd();\
  } else {\
  status =  arg3op2_common(NULL, p, q, r,\
  N_##GPUKERNEL##_F_F_KERNEL,\
  N_##GPUKERNEL##_F_C_KERNEL,\
  N_##GPUKERNEL##_F_D_KERNEL,\
  N_##GPUKERNEL##_F_CD_KERNEL,\
  N_##GPUKERNEL##_C_F_KERNEL,\
  N_##GPUKERNEL##_C_C_KERNEL,\
  N_##GPUKERNEL##_C_D_KERNEL,\
  N_##GPUKERNEL##_C_CD_KERNEL,\
  N_##GPUKERNEL##_D_F_KERNEL,\
  N_##GPUKERNEL##_D_C_KERNEL,\
  N_##GPUKERNEL##_D_D_KERNEL,\
  N_##GPUKERNEL##_D_CD_KERNEL,\
  N_##GPUKERNEL##_CD_F_KERNEL,\
  N_##GPUKERNEL##_CD_C_KERNEL,\
  N_##GPUKERNEL##_CD_D_KERNEL,\
  N_##GPUKERNEL##_CD_CD_KERNEL\
  );\
  }\
  return status;\
}\

#define HGPUOP6(GPUOPFUN)\
  GPUmatResult_t GPUop##GPUOPFUN(GPUtype &p, GPUtype &q, GPUtype &r);\


/* GPUopGPUallocVector */
GPUmatResult_t GPUopAllocVector(GPUtype &p);
GPUmatResult_t GPUopAllocVector(void **plhs, void **prhs, int print);

/* GPUopGPUallocVector */
GPUmatResult_t GPUopFree(GPUtype &p);
GPUmatResult_t GPUopFree(void **plhs, void **prhs, int print);

/* GPUopTransposeDrv
* GPUopTranspose */
HGPUOP3( Transpose )
HGPUOP4( Transpose )

/* GPUopCtransposeDrv
* GPUopCtranspose */
HGPUOP3(Ctranspose)
HGPUOP4(Ctranspose)

/* GPUopCudaMemcpy */
GPUmatResult_t GPUopCudaMemcpy(void* dst, const void* src, size_t count,
                               enum cudaMemcpyKind kind, GPUmanager *GPUman);
/* GPUopCudaMemcpyAsync */
GPUmatResult_t GPUopCudaMemcpyAsync(void* dst, const void* src, size_t count,
                                    enum cudaMemcpyKind kind, GPUmanager *GPUman);

/* GPUopCudaMemcpy2D */
GPUmatResult_t GPUopCudaMemcpy2D(void* dst, size_t dpitch, const void* src,
                                 size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind,
                                 GPUmanager *GPUman);


/* GPUopPackC2C */
GPUmatResult_t GPUopPackC2C(int re, GPUtype &d_re_idata,
                            GPUtype & d_im_idata, GPUtype & d_odata);

/* GPUopUnpackC2C */
GPUmatResult_t GPUopUnpackC2C(int mode, GPUtype& d_idata,
                              GPUtype& d_re_odata, GPUtype& d_im_odata);

/* GPUopAssign */
GPUtype *  GPUopAssign (GPUtype &p, GPUtype &q,  const Range &r, int dir, int ret, int fortranidx);

/* GPUopPermute */
GPUtype *  GPUopPermute (GPUtype &p, GPUtype &q,  const Range &r, int dir, int ret, int fortranidx, int*perm);

/* GPUopSubsindex */
GPUtype * GPUopSubsindexDrv(GPUtype &p, int subsdim, int *range);
GPUmatResult_t GPUopSubsindex(GPUtype &d_idata, const int idxshift,
                              GPUtype &d_pars, GPUtype &d_odata);

/* GPUopSubsindexf */
//GPUmatResult_t GPUopSubsindexf(GPUtype &d_idata, const int idxshift,
//		GPUtype &d_pars, GPUtype &d_odata);

/* GPUopSubsindexc */
//GPUmatResult_t GPUopSubsindexc(GPUtype &d_idata, const int idxshift,
//		GPUtype &d_pars, GPUtype &d_odata);

/* GPUopFFTSymm */
GPUmatResult_t GPUopFFTSymm(GPUtype &d_idata, int batch);


/* GPUopAbsDrv
* GPUopAbs */
HGPUOP3(Abs)
HGPUOP4(Abs)

/* GPUopAndDrv */
/* GPUopAnd */
HGPUOP5(And)
HGPUOP6(And)

/* GPUopAcosDrv
* GPUopAcos */
HGPUOP3(Acos)
HGPUOP4(Acos)

/* GPUopAcoshDrv
* GPUopAcosh */
HGPUOP3(Acosh)
HGPUOP4(Acosh)

/* GPUopAsinDrv
* GPUopAsin */
HGPUOP3(Asin)
HGPUOP4(Asin)

/* GPUopAsinhDrv
* GPUopAsinh */
HGPUOP3(Asinh)
HGPUOP4(Asinh)

/* GPUopAtanDrv
* GPUopAtan */
HGPUOP3(Atan)
HGPUOP4(Atan)

/* GPUopAtanhDrv
* GPUopAtanh */
HGPUOP3(Atanh)
HGPUOP4(Atanh)

/* GPUopCeilDrv
* GPUopCeil */
HGPUOP3(Ceil)
HGPUOP4(Ceil)

/* GPUopCosDrv
* GPUopCos */
HGPUOP3(Cos)
HGPUOP4(Cos)

/* GPUopCoshDrv
* GPUopCosh */
HGPUOP3(Cosh)
HGPUOP4(Cosh)

/* GPUopConjDrv
* GPUopConj */
HGPUOP3(Conj)
HGPUOP4(Conj)

/* GPUopEqDrv */
/* GPUopEq */
HGPUOP5(Eq)
HGPUOP6(Eq)

/* GPUopExpDrv
* GPUopExp */
HGPUOP3(Exp)
HGPUOP4(Exp)

/* GPUopFloorDrv
* GPUopFloor */
HGPUOP3(Floor)
HGPUOP4(Floor)

/* GPUopGeDrv */
/* GPUopGe */
HGPUOP5(Ge)
HGPUOP6(Ge)

/* GPUopGtDrv */
/* GPUopGt */
HGPUOP5(Gt)
HGPUOP6(Gt)

/* GPUopLdivideDrv */
/* GPUopLdivide */
HGPUOP5(Ldivide)
HGPUOP6(Ldivide)

/* GPUopLeDrv */
/* GPUopLe */
HGPUOP5(Le)
HGPUOP6(Le)

/* GPUopLog1pDrv
* GPUopLog1p */
HGPUOP3(Log1p)
HGPUOP4(Log1p)

/* GPUopLog2Drv
* GPUopLog2 */
HGPUOP3(Log2)
HGPUOP4(Log2)

/* GPUopLog10Drv
* GPUopLog10 */
HGPUOP3(Log10)
HGPUOP4(Log10)

/* GPUopLogDrv
* GPUopLog */
HGPUOP3(Log)
HGPUOP4(Log)

/* GPUopLtDrv */
/* GPUopLt */
HGPUOP5(Lt)
HGPUOP6(Lt)

/* GPUopMinusDrv */
/* GPUopMinus */
HGPUOP5(Minus)
HGPUOP6(Minus)

/* GPUopNeDrv */
/* GPUopNe */
HGPUOP5(Ne)
HGPUOP6(Ne)

/* GPUopNotDrv
* GPUopNot */
HGPUOP3(Not)
HGPUOP4(Not)

/* GPUopOrDrv */
/* GPUopOr */
HGPUOP5(Or)
HGPUOP6(Or)

/* GPUopPlusDrv */
/* GPUopPlus */
HGPUOP5(Plus)
HGPUOP6(Plus)

/* GPUopPowerDrv */
/* GPUopPower */
HGPUOP5(Power)
HGPUOP6(Power)

/* GPUopRealImag */
GPUmatResult_t GPUopRealImag(GPUtype& data, GPUtype &re, GPUtype &im, int dir, int mode);

/* GPUopRdivideDrv */
/* GPUopRdivide */
HGPUOP5(Rdivide)
HGPUOP6(Rdivide)

/* GPUopRoundDrv
* GPUopRound */
HGPUOP3(Round)
HGPUOP4(Round)

/* GPUopSinDrv
* GPUopSin */
HGPUOP3(Sin)
HGPUOP4(Sin)

/* GPUopSinhDrv
* GPUopSinh */
HGPUOP3(Sinh)
HGPUOP4(Sinh)

/* GPUopSqrtDrv
* GPUopSqrt */
HGPUOP3(Sqrt)
HGPUOP4(Sqrt)

/* GPUopTanDrv
* GPUopTan */
HGPUOP3(Tan)
HGPUOP4(Tan)

/* GPUopTanhDrv
* GPUopTanh */
HGPUOP3(Tanh)
HGPUOP4(Tanh)

/* GPUopTimesDrv */
/* GPUopTimes */
HGPUOP5(Times)
HGPUOP6(Times)

/* GPUopTimes2Drv */
/* GPUopTimes2 */
//HGPUOP5(Times2)
//HGPUOP6(Times2)

/* GPUopMtimesDrv */
/* GPUopMtimes */
HGPUOP5(Mtimes)
HGPUOP6(Mtimes)


/* GPUopUminusDrv
* GPUopUminus */
HGPUOP3(Uminus)
HGPUOP4(Uminus)

/* GPUopColon */
GPUtype * GPUopColonDrv(double J, double K, double D, GPUtype &p);
//GPUtype * GPUopColonDrv(double J, double K, double D, gpuTYPE_t type);
GPUmatResult_t GPUopColon(double j, double d, GPUtype &r);

/* GPUopFillVector */
GPUmatResult_t GPUopFillVector(double, double, GPUtype &r);

/* GPUopFillVector1 */
GPUmatResult_t GPUopFillVector1(double, double, GPUtype &r, int, int, int, int);


/* GPUopSum */
GPUmatResult_t GPUopSum(GPUtype &, int Nthread, int M, int GroupSize,
                        int GroupOffset, GPUtype &);

/* GPUopSum2 */
GPUmatResult_t GPUopSum2(GPUtype &, int Nthread, int M, int GroupSize,
                         int GroupOffset, GPUtype &);

/* GPUopFFT */
GPUmatResult_t GPUopFFT(GPUtype &p, GPUtype &r,
                        int direction, int batch, int dim);

/* GPUopFFTDrv */
GPUtype * GPUopFFTDrv(GPUtype &p, int dim, int dir);

/* GPUopRealDrv */
GPUtype * GPUopRealDrv(GPUtype &p);

/* GPUopReal */
GPUmatResult_t GPUopReal(GPUtype &p, GPUtype &r);

/* GPUopComplexDrv */
GPUtype * GPUopComplexDrv(GPUtype &p);
GPUtype * GPUopComplexDrv(GPUtype &p, GPUtype &im);

/* GPUopComplex */
GPUmatResult_t GPUopComplex(GPUtype &p, GPUtype &r);
GPUmatResult_t GPUopComplex(GPUtype &p, GPUtype &q, GPUtype &r);


/* GPUopImagDrv */
GPUtype * GPUopImagDrv(GPUtype &p);

/* GPUopImag */
GPUmatResult_t GPUopImag(GPUtype &p, GPUtype &r);

/* GPUopZerosDrv
* GPUopZeros */
HGPUOP3(Zeros)
HGPUOP4(Zeros)

/* GPUopOnesDrv
* GPUopOnes */
HGPUOP3(Ones)
HGPUOP4(Ones)

/* CASTING */

/* GPUopFloatToDoubleDrv */
GPUtype * GPUopFloatToDoubleDrv(GPUtype &p);

/* GPUopFloatToDouble */
GPUmatResult_t GPUopFloatToDouble(GPUtype &p, GPUtype &r);

/* GPUopDoubleToFloatDrv */
GPUtype * GPUopDoubleToFloatDrv(GPUtype &p);

/* GPUopDoubleToFloat */
GPUmatResult_t GPUopDoubleToFloat(GPUtype &p, GPUtype &r);

#endif

