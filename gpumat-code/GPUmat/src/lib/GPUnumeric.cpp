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

///#include "cutil.h"
#include "cublas.h"
#include "cufft.h"
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
#include "GPUop.hh"
#include "GPUnumeric.hh"
#include "util.hh"


/*************************************************************************
 * Utilities to check results consistency
 *************************************************************************/

/* result type of the operation between GPUtypes */
  /*    F C D CD I
   *  F
   *  C .
   *  D
   * CD
   *  I
   */
  gpuTYPE_t GPURESULTA[NGPUTYPE][NGPUTYPE] = {
      {gpuFLOAT  , gpuCFLOAT , gpuFLOAT  , gpuCFLOAT , gpuNOTDEF },
      {gpuCFLOAT , gpuCFLOAT , gpuCFLOAT , gpuCFLOAT , gpuNOTDEF },
      {gpuFLOAT  , gpuCFLOAT , gpuDOUBLE , gpuCDOUBLE, gpuNOTDEF},
      {gpuCFLOAT , gpuCFLOAT , gpuCDOUBLE, gpuCDOUBLE, gpuNOTDEF},
      {gpuNOTDEF , gpuNOTDEF , gpuNOTDEF , gpuNOTDEF , gpuINT32}
  };

  gpuTYPE_t GPURESULTB[NGPUTYPE] = {gpuFLOAT, gpuCFLOAT, gpuDOUBLE, gpuCDOUBLE, gpuINT32};

  STRINGCONST char * GPUTEXTA[NGPUTYPE] ={"GPUsingle","GPUsingle(COMPLEX)","GPUdouble","GPUdouble(COMPLEX)","GPUint32"};


void checkResult(GPUtype &p, GPUtype &q, GPUtype &r, gpuTYPE_t *type) {

  #define RES(i,j) TMPRES[(i) + (j) * NGPUTYPE]

  gpuTYPE_t ptype = p.getType();
  gpuTYPE_t qtype = q.getType();
  gpuTYPE_t rtype = r.getType();

  gpuTYPE_t *TMPRES;

  if (type==NULL)
    TMPRES = &(GPURESULTA[0][0]);
  else
    TMPRES = type;

  gpuTYPE_t restype  = RES(ptype,qtype);

  STRINGCONST char *t1 = GPUTEXTA[ptype];
  STRINGCONST char *t2 = GPUTEXTA[qtype];
  STRINGCONST char *t3 = GPUTEXTA[rtype];
  STRINGCONST char *t4 = GPUTEXTA[restype];

  if (rtype != restype) {
    char buffer[300];
        sprintf(
            buffer,
            "Returned variable expected to be of type '%s' instead of '%s'. (ERROR code 19)", t4,t3);
        throw GPUexception(GPUmatError, buffer);
  }
}

void checkResult(GPUtype &p, GPUtype &r, gpuTYPE_t *type) {


  gpuTYPE_t ptype = p.getType();
  gpuTYPE_t rtype = r.getType();

  gpuTYPE_t *TMPRES;

  if (type==NULL)
    TMPRES = &(GPURESULTB[0]);
  else
    TMPRES = type;

  gpuTYPE_t restype  = TMPRES[ptype];

  STRINGCONST char *t1 = GPUTEXTA[ptype];
  STRINGCONST char *t3 = GPUTEXTA[rtype];
  STRINGCONST char *t4 = GPUTEXTA[restype];

  if (rtype != restype) {
    char buffer[300];
        sprintf(
            buffer,
            "Returned variable expected to be of type '%s' instead of '%s'. (ERROR code 19)", t4,t3);
        throw GPUexception(GPUmatError, buffer);
  }
}




/*************************************************************************
 * mtimes_drv
 *************************************************************************/
/* The following function is used in almost all functions with 2 operands */
GPUtype*
mtimes_drv(GPUtype &p, GPUtype &q) {

	// garbage collector
  MyGCObj<GPUtype> mgc;

	// Use local copies. This way avoid to release GPUptr
	// on original pointers
	GPUtype ptmp = GPUtype(p, 0);
	GPUtype qtmp = GPUtype(q, 0);

	int ispcomplex = ptmp.isComplex();
	int isqcomplex = qtmp.isComplex();

	int opfloat  = ptmp.isFloat()  || qtmp.isFloat();
	int opdouble = ptmp.isDouble() || qtmp.isDouble();

	int opcomplex = ispcomplex || isqcomplex;

	/* Check input
	 check
	 Valid operations
	 scalar op GPUsingle
	 GPUsingle op scalar
	 GPUsingle op GPUsingle*/

	// All above cases can be either single or single complex (also scalars).
	// This function cannot be called with single arguments. This is the assumption
	if (ptmp.isScalar() || qtmp.isScalar()) {
		throw GPUexception(GPUmatError,
				ERROR_ARG_SCALARS);
	}

	// check operands, check like A * B
	if (ptmp.isEmpty() || (ptmp.getNdims() > 2) || (qtmp.isEmpty()) || (qtmp.getNdims()
			> 2))
		throw GPUexception(GPUmatError,ERROR_ARG2OP_2D);
	int *psize = ptmp.getSize();
	int *qsize = qtmp.getSize();
	if (psize[1] != qsize[0])
		throw GPUexception(GPUmatError,ERROR_ARG2OP_INNER);


	// allocate temp for results
	GPUtype * r;

	/* Manage complex numbers
	 */

	if (opcomplex && !isqcomplex) {
		GPUtype *qtmp1 = qtmp.REALtoCOMPLEX();
		qtmp = *qtmp1;
		delete qtmp1;
	}

	if (opcomplex && !ispcomplex) {
		GPUtype *ptmp1 = ptmp.REALtoCOMPLEX();
		ptmp = *ptmp1;
		delete ptmp1;
	}

	// Casting
	// At this point operands are both complex or real
	// Casting is done always to the element with lower accuracy for the final solution
	// but operations are done with higher accuracy

	if (ptmp.isFloat()&&opdouble) {
		GPUtype *qtmp1 = qtmp.DOUBLEtoFLOAT();
		qtmp = *qtmp1;
		delete qtmp1;
		/*GPUtype *ptmp1 = ptmp.FLOATtoDOUBLE();
		ptmp = *ptmp1;
		delete ptmp1;*/

	} else if (opdouble&&qtmp.isFloat()) {
		GPUtype *ptmp1 = ptmp.DOUBLEtoFLOAT();
		ptmp = *ptmp1;
		delete ptmp1;
		/*GPUtype *qtmp1 = qtmp.FLOATtoDOUBLE();
		qtmp = *qtmp1;
		delete qtmp1;*/
	}

  r = new GPUtype(ptmp, 1); // do not copy GPUptr
  mgc.setPtr(r);

	if (opcomplex)
		r->setComplex();

  // size
	int rsize[2];
	int *ptmpsize = ptmp.getSize();
	int *qtmpsize = qtmp.getSize();
	rsize[0] = ptmpsize[0];
	rsize[1] = qtmpsize[1];

	r->setSize(2, rsize);

	GPUopAllocVector(*r);

	// Run
	GPUopMtimes(ptmp, qtmp, *r);

	// now I have to cast to the original size
	/*if (opfloat&&r->isDouble()) {
		GPUtype *rtmp = r->DOUBLEtoFLOAT();
		mgc.remPtr(r);
		delete r;
		return rtmp;
	} else {*/
	  mgc.remPtr(r);
	  return r;
	//}

}

/*************************************************************************
 * mtimes
 *************************************************************************/
GPUmatResult_t
mtimes(GPUtype &p, GPUtype &q, GPUtype &r) {

	GPUmatResult_t status = GPUmatSuccess;

	GPUtype ptmp = GPUtype(p, 0);
	GPUtype qtmp = GPUtype(q, 0);

	/* result type of the operation between GPUtypes */
  /*    F C D CD I
   *  F
   *  C .
   *  D
   * CD
   *  I
   */
	gpuTYPE_t GPURESULT[5][5] = {
	        {gpuFLOAT  , gpuCFLOAT , gpuFLOAT  , gpuCFLOAT , gpuNOTDEF },
	        {gpuCFLOAT , gpuCFLOAT , gpuCFLOAT , gpuCFLOAT , gpuNOTDEF },
	        {gpuFLOAT  , gpuCFLOAT , gpuDOUBLE , gpuCDOUBLE, gpuNOTDEF},
	        {gpuCFLOAT , gpuCFLOAT , gpuCDOUBLE, gpuCDOUBLE, gpuNOTDEF},
	        {gpuNOTDEF , gpuNOTDEF , gpuNOTDEF , gpuNOTDEF , gpuINT32}
	    };


	gpuTYPE_t ptype = p.getType();
  gpuTYPE_t qtype = q.getType();
  gpuTYPE_t rtype = r.getType();


  gpuTYPE_t restype  = GPURESULT[ptype][qtype];

  STRINGCONST char *t1 = GPUTEXTA[ptype];
  STRINGCONST char *t2 = GPUTEXTA[qtype];
  STRINGCONST char *t3 = GPUTEXTA[rtype];
  STRINGCONST char *t4 = GPUTEXTA[restype];

  if (rtype != restype) {
    char buffer[300];
        sprintf(
            buffer,
            "Returned variable expected to be of type '%s' instead of '%s'. (ERROR code 19)", t4,t3);
        throw GPUexception(GPUmatError, buffer);
  }

	/* Check parameters */
	if (ptmp.isScalar() || qtmp.isScalar()) {
    throw GPUexception(GPUmatError,
        ERROR_ARG_SCALARS);
  }

  // check operands, check like A * B
  if (ptmp.isEmpty() || (ptmp.getNdims() > 2) || (qtmp.isEmpty()) || (qtmp.getNdims()
      > 2))
    throw GPUexception(GPUmatError,ERROR_ARG2OP_2D);
  int *psize = ptmp.getSize();
  int *qsize = qtmp.getSize();
  int *rsize = r.getSize();

  if (psize[1] != qsize[0])
    throw GPUexception(GPUmatError,ERROR_ARG2OP_INNER);

  // the following test is not performed by the driver function
  if (rsize[0] != psize[0])
        throw GPUexception(GPUmatError,ERROR_ARG2OP_INNER);

  if (rsize[1] != qsize[1])
        throw GPUexception(GPUmatError,ERROR_ARG2OP_INNER);

  if (ptype != qtype) {
    char buffer[300];
    sprintf(
        buffer,
        "Function arguments should be of the same type. Found '%s' and '%s'. (ERROR code 19)", t1,t2);
    throw GPUexception(GPUmatError, buffer);
  }
	/*********************/

	char transa = 'n';
	char transb = 'n';


	Complex alphaC = { 1.0, 0.0 };
	Complex betaC = { 0.0, 0.0 };
	float alphaF = 1.0;
	float betaF = 0.0;

	DoubleComplex alphaCD = { 1.0, 0.0 };
	DoubleComplex betaCD = { 0.0, 0.0 };
	double alphaD = 1.0;
	double betaD = 0.0;

	//  * lda    leading dimension of two-dimensional array used to store matrix A
	// using number of rows as leading dimension of matrix A always. If A is
	// trasposed then lda is number of columns of A.'
	int lda;
	int ldb;
	int ldc;

	// have to update ptmpsize, might have changed
	// I am doing this  explicitly because on Linux
	// I got very strange behavior
	int *ptmpsize = ptmp.getSize();
	int *qtmpsize = qtmp.getSize();
  //int *rsize    = r.getSize();

	lda = ptmpsize[0];
	ldb = qtmpsize[0];
	ldc = rsize[0];

	/*%  * m      number of rows of matrix op(A) and rows of matrix C
	 %  * n      number of columns of matrix op(B) and number of columns of C
	 %  * k      number of columns of matrix op(A) and number of rows of op(B)*/
	int m = ptmpsize[0];
	int n = qtmpsize[1];
	int k = ptmpsize[1];


	cublasStatus cublasstatus;
	if (ptmp.getType() == gpuCFLOAT) {
		cublasCgemm(transa, transb, m, n, k, (cuComplex) alphaC,
				(cuComplex*) ptmp.getGPUptr(), lda, (cuComplex*) qtmp.getGPUptr(), ldb,
				(cuComplex) betaC, (cuComplex*) r.getGPUptr(), ldc);

	} else if (ptmp.getType() == gpuFLOAT) {
		cublasSgemm(transa, transb, m, n, k, alphaF, (float*) ptmp.getGPUptr(),
				lda, (float*) qtmp.getGPUptr(), ldb, betaF, (float*) r.getGPUptr(),
				ldc);

	} else if (ptmp.getType() == gpuCDOUBLE) {
		cublasZgemm(transa, transb, m, n, k, (cuDoubleComplex) alphaCD,
				(cuDoubleComplex*) ptmp.getGPUptr(), lda, (cuDoubleComplex*) qtmp.getGPUptr(), ldb,
				(cuDoubleComplex) betaCD, (cuDoubleComplex*) r.getGPUptr(), ldc);

	} else if (ptmp.getType() == gpuDOUBLE) {
		cublasDgemm(transa, transb, m, n, k, alphaD, (double*) ptmp.getGPUptr(),
				lda, (double*) qtmp.getGPUptr(), ldb, betaD, (double*) r.getGPUptr(),
				ldc);
	}

	cublasstatus = cublasGetError();
	if (cublasstatus != CUBLAS_STATUS_SUCCESS) {
		char buffer[300];
		sprintf(buffer, "Error in cublasXgemm (%d)", cublasstatus);
		throw GPUexception(GPUmatError, buffer);
	}

	return status;

}

/*************************************************************************
 * arg1op_drv
 *************************************************************************/

GPUtype*
arg1op_drv(gpuTYPE_t *settype, GPUtype &p, GPUmatResult_t(*fun)(GPUtype&, GPUtype&)) {

	// garbage collector
	MyGCObj<GPUtype> mgc;

	// Use local copies. This way avoid to release GPUptr
	// on original pointers
	GPUtype ptmp = GPUtype(p, 0);

	// allocate temp for results
	GPUtype * r;
	r = new GPUtype(ptmp, 1); // do not copy GPUptr
	mgc.setPtr(r);

	// settype overrides previous command r.setType (REAL / COMPLEX)
	if (settype !=NULL) {
		r->setType(settype[p.getType()]);
	}


	GPUopAllocVector(*r);

	fun(ptmp, *r);

	mgc.remPtr(r);
	return r;

}

/*************************************************************************
 * arg1op_common
 *************************************************************************/

GPUmatResult_t arg1op_common(gpuTYPE_t *settype, GPUtype &p, GPUtype &r, int F_KERNEL, int CF_KERNEL, int D_KERNEL, int CD_KERNEL) {

	  CUDALIBResult cudalibres = CUDALIBSuccess;
		GPUmatResult_t status = GPUmatSuccess;

		GPUmanager * GPUman = p.getGPUmanager();

		/*********************************/
    // check output and input
    int numel = p.getNumel();
    if (r.getNumel()!=numel) {
      throw GPUexception(GPUmatError,
          ERROR_ARG2OP_ELEMENTS);
    }

    checkResult(p,r,settype);
    /*********************************/

		CUfunction *drvfun;
		void *op1Ptr;
		void *op2Ptr;

		unsigned int p1size;
		unsigned int p2size;
    
    size_t p1align;
    size_t p2align;

		op1Ptr = p.getGPUptrptr();
		op2Ptr = r.getGPUptrptr();
		p1size = sizeof(CUdeviceptr);
		p2size = sizeof(CUdeviceptr);
    p1align = __alignof(CUdeviceptr);
    p2align = __alignof(CUdeviceptr);


		int n = p.getNumel();

		if (p.getType()==gpuCFLOAT) {
			if (CF_KERNEL<0)
				throw GPUexception(GPUmatError, ERROR_NOTIMPL_CFLOAT);
			drvfun = GPUman->getCuFunction(CF_KERNEL);

		} else if (p.getType()==gpuFLOAT){
			if (F_KERNEL<0)
					throw GPUexception(GPUmatError, ERROR_NOTIMPL_FLOAT);
			drvfun = GPUman->getCuFunction(F_KERNEL);

		} else if (p.getType()==gpuCDOUBLE){
			if (CD_KERNEL<0)
				throw GPUexception(GPUmatError, ERROR_NOTIMPL_CDOUBLE);
			drvfun = GPUman->getCuFunction(CD_KERNEL);

		} else if (p.getType()==gpuDOUBLE){
			if (D_KERNEL<0)
				throw GPUexception(GPUmatError, ERROR_NOTIMPL_DOUBLE);
			drvfun = GPUman->getCuFunction(D_KERNEL);

		} /*else if (p.getType()==gpuINT32){
			if (I_KERNEL<0)
				throw GPUexception(GPUmatError, ERROR_NOTIMPL_INT32);
			drvfun = GPUman->getCuFunction(I_KERNEL);

		}*/




		// define kernel configuration
		gpukernelconfig_t * kconf = GPUman->getKernelConfig();
		hostdrv_pars_t pars[2];
		int nrhs = 2;

		pars[0].par =  op1Ptr;
		pars[0].psize = p1size;
    pars[0].align = p1align;
    
		pars[1].par = op2Ptr;
		pars[1].psize = p2size;
    pars[1].align = p2align;

		cudalibres = mat_HOSTDRV_A(n, kconf, nrhs, pars, drvfun);


		if (cudalibres != CUDALIBSuccess) {
			throw GPUexception(GPUmatError, "Kernel execution error.");
		}

		return status;
}

/*************************************************************************
 * arg2op_common
 *************************************************************************/
/* direction is used to determine if the operation is left or right (A + B or B + A) */

/*************************************************************************
 * GPUsetKernelTexture
 *************************************************************************/
// Use to function to set the texture for a GPU kernel.
GPUmatResult_t GPUsetKernelTextureA(GPUtype &p, CUfunction *drvfun, int nsize) {
  CUtexref *drvtexa;
	CUarray_format_enum drvtexformata;
	int drvtexnuma;

	GPUmanager * GPUman = p.getGPUmanager();

	gpuTYPE_t ptype = p.getType();

	if (ptype==gpuCFLOAT) {
			drvtexa = GPUman->getCuTexref(N_TEXREF_C1_A);
			drvtexformata = CU_AD_FORMAT_FLOAT;
			drvtexnuma = 2;
	} else if (ptype==gpuFLOAT){
			drvtexa = GPUman->getCuTexref(N_TEXREF_F1_A);
			drvtexformata = CU_AD_FORMAT_FLOAT;
			drvtexnuma = 1;
	} else if (ptype==gpuCDOUBLE){
			drvtexa = GPUman->getCuTexref(N_TEXREF_CD1_A);
			drvtexformata = CU_AD_FORMAT_SIGNED_INT32;
			drvtexnuma = 4;
	} else if (ptype==gpuDOUBLE){
			drvtexa = GPUman->getCuTexref(N_TEXREF_D1_A);
			drvtexformata = CU_AD_FORMAT_SIGNED_INT32;
			drvtexnuma = 2;
	}

	if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtexa, UINTPTR p.getGPUptr(), nsize)) {
		throw GPUexception(GPUmatError, "Kernel execution error1.");
	}
	if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT, *drvtexa)) {
		throw GPUexception(GPUmatError, "Kernel execution error3.");
	}
	if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtexa, drvtexformata, drvtexnuma)) {
		throw GPUexception(GPUmatError, "Kernel execution error2.");
	}
	GPUmatResult_t status = GPUmatSuccess;
	return status;

}

GPUmatResult_t GPUsetKernelTextureB(GPUtype &p, CUfunction *drvfun, int nsize) {
  CUtexref *drvtexa;
	CUarray_format_enum drvtexformata;
	int drvtexnuma;

	GPUmanager * GPUman = p.getGPUmanager();

	gpuTYPE_t ptype = p.getType();

	if (ptype==gpuCFLOAT) {
			drvtexa = GPUman->getCuTexref(N_TEXREF_C1_B);
			drvtexformata = CU_AD_FORMAT_FLOAT;
			drvtexnuma = 2;
	} else if (ptype==gpuFLOAT){
			drvtexa = GPUman->getCuTexref(N_TEXREF_F1_B);
			drvtexformata = CU_AD_FORMAT_FLOAT;
			drvtexnuma = 1;
	} else if (ptype==gpuCDOUBLE){
			drvtexa = GPUman->getCuTexref(N_TEXREF_CD1_B);
			drvtexformata = CU_AD_FORMAT_SIGNED_INT32;
			drvtexnuma = 4;
	} else if (ptype==gpuDOUBLE){
			drvtexa = GPUman->getCuTexref(N_TEXREF_D1_B);
			drvtexformata = CU_AD_FORMAT_SIGNED_INT32;
			drvtexnuma = 2;
	}

	if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtexa, UINTPTR p.getGPUptr(), nsize)) {
		throw GPUexception(GPUmatError, "Kernel execution error1.");
	}
	if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT, *drvtexa)) {
		throw GPUexception(GPUmatError, "Kernel execution error3.");
	}
	if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtexa, drvtexformata, drvtexnuma)) {
		throw GPUexception(GPUmatError, "Kernel execution error2.");
	}
	GPUmatResult_t status = GPUmatSuccess;
	return status;

}

/*************************************************************************
 * arg3op2_common
 *************************************************************************/
/* direction is used to determine if the operation is left or right (A + B or B + A) */
GPUmatResult_t arg3op2_common(gpuTYPE_t *settype, GPUtype &p,
		GPUtype &q, GPUtype &r,
		int F_F_KERNEL,
		int F_C_KERNEL,
		int F_D_KERNEL,
		int F_CD_KERNEL,
		int C_F_KERNEL,
		int C_C_KERNEL,
		int C_D_KERNEL,
		int C_CD_KERNEL,
		int D_F_KERNEL,
		int D_C_KERNEL,
		int D_D_KERNEL,
		int D_CD_KERNEL,
		int CD_F_KERNEL,
		int CD_C_KERNEL,
		int CD_D_KERNEL,
		int CD_CD_KERNEL
    )
{



	/*    F C D CD
	 *  F 0 1 2 3
	 *  C .
	 *  D
	 * CD
	 *
	 */
	int GPUKERNELSMAPPING[4][4] = {
			{F_F_KERNEL  , F_C_KERNEL  , F_D_KERNEL   , F_CD_KERNEL },
			{C_F_KERNEL  , C_C_KERNEL  , C_D_KERNEL   , C_CD_KERNEL },
			{D_F_KERNEL  , D_C_KERNEL  , D_D_KERNEL   , D_CD_KERNEL },
			{CD_F_KERNEL , CD_C_KERNEL , CD_D_KERNEL  , CD_CD_KERNEL }

	};

	GPUmanager * GPUman = p.getGPUmanager();

	GPUmatResult_t status = GPUmatSuccess;
	CUDALIBResult cudalibres = CUDALIBSuccess;

	// Check input arguments
  if (p.isScalar() && q.isScalar()) {
    throw GPUexception(GPUmatError,
        ERROR_ARG_SCALARS);
  }
  // dimensions must match.
    // check operands, check like A + B
  int numel = p.getNumel();
  if (q.getNumel()>numel)
    numel = q.getNumel();

  int dim = (p.getNdims() == q.getNdims() && (compareInt(p.getNdims(),p.getSize(), q.getSize())));
  if (dim == 1) {
  } else {
    if (p.isScalar() || q.isScalar()) {
      // OK
    } else {
      throw GPUexception(GPUmatError,
          ERROR_ARG2OP_DIMENSIONS);
    }
  }
  if (r.getNumel()!=numel) {
    throw GPUexception(GPUmatError,
        ERROR_ARG2OP_ELEMENTS);
  }


  checkResult(p,q,r,settype);

	// The GPU kernel has the following interface:
	// int n , int offset, INARG1 *idata1, INARG2 *idata2, INARG2 *dummy, OUTARG1 *odata, int right
	// int n , int offset, INARG1 *idata1, INARG2 idata2x, INARG2 idata2y, OUTARG1 *odata, int right <- scalar version
	// The dummy variable is created to have the same interface between scalar and non scalar. We use real and imaginayr part
	// separate because of an issue with DoubleComplex




	CUfunction *drvfun;
	/*CUtexref *drvtexa;
	CUtexref *drvtexb;
	CUarray_format_enum drvtexformata;
	CUarray_format_enum drvtexformatb;
	int drvtexnuma;
	int drvtexnumb;*/

	//void *op1Ptr;
	//void *op2Ptr;
	void *op3Ptr;

	//unsigned int p1size;
	//unsigned int p2size;
	unsigned int p3size;
  
  
	int i1 = -1;
	int i2 = -1;

	int n = 0;
	int opscalar = 0;

	int right = 1;
	// used to define a right or left operation: A + B or B + A

	op3Ptr = r.getGPUptrptr();
	p3size = sizeof(CUdeviceptr);
  size_t p3align = __alignof(CUdeviceptr);

	if (p.isScalar()) {

		n = q.getNumel();
		i1 = 0;

	} else if (q.isScalar()) {
		n = p.getNumel();
		i2 = 0;

	} else {
		n = p.getNumel();

	}
	//op1Ptr = p.getGPUptrptr();
	//p1size = sizeof(CUdeviceptr);
  

	//op2Ptr = q.getGPUptrptr();
	//p2size = sizeof(CUdeviceptr);
  

	// direction = 0: normal direction right = 1
	// direction = 1: left direction


	gpuTYPE_t ptype = p.getType();
  gpuTYPE_t qtype = q.getType();
  gpuTYPE_t rtype = r.getType();



  STRINGCONST char *t1 = GPUTEXTA[ptype];
  STRINGCONST char *t2 = GPUTEXTA[qtype];

	// calculate kernel offset
	int kerneln  = GPUKERNELSMAPPING[ptype][qtype];
	if (kerneln<0) {
		char buffer[300];
		sprintf(
				buffer,
				"Function not implemented for input arguments  of type '%s' and '%s'. (ERROR code 19)", t1, t2);
		throw GPUexception(GPUmatError, buffer);

		//throw GPUexception(GPUmatError, ERROR_NOTIMPL_GENERIC);
	}

	drvfun = GPUman->getCuFunction(kerneln);


	// define kernel configuration
	gpukernelconfig_t *kconf = GPUman->getKernelConfig();
	hostdrv_pars_t pars[3];
	int nrhs = 3;

	pars[0].par =  &i1;
	pars[0].psize = sizeof(i1);
  pars[0].align = __alignof(i1);

	pars[1].par = &i2;
	pars[1].psize = sizeof(i2);
  pars[1].align = __alignof(i2);

	pars[2].par =  op3Ptr;
	pars[2].psize = p3size;
  pars[2].align = p3align;



	// setup texture
	// setup texture
	GPUsetKernelTextureA(p, drvfun, p.getNumel()* p.getMySize());
	GPUsetKernelTextureB(q, drvfun, q.getNumel()* q.getMySize());

	/*if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtexa, UINTPTR p.getGPUptr(), p.getNumel()
			* p.getMySize())) {
		throw GPUexception(GPUmatError, "Kernel execution error1.");
	}
	if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT, *drvtexa)) {
		throw GPUexception(GPUmatError, "Kernel execution error3.");
	}
	if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtexa, drvtexformata, drvtexnuma)) {
		throw GPUexception(GPUmatError, "Kernel execution error2.");
	}

	if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtexb, UINTPTR q.getGPUptr(), q.getNumel()
			* q.getMySize())) {
		throw GPUexception(GPUmatError, "Kernel execution error1.");
	}
	if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT, *drvtexb)) {
		throw GPUexception(GPUmatError, "Kernel execution error3.");
	}
	if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtexb, drvtexformatb, drvtexnumb)) {
		throw GPUexception(GPUmatError, "Kernel execution error2.");
	}*/

	cudalibres = mat_HOSTDRV_A(n, kconf, nrhs, pars, drvfun);

	if (cudalibres != CUDALIBSuccess) {
		throw GPUexception(GPUmatError, "Kernel execution error.");
	}
	return status;
}

/*************************************************************************
 * arg2op_drv2
 *************************************************************************/
/* The following function is used in almost all functions with 2 operands */
GPUtype*
arg3op_drv(gpuTYPE_t *settype,  GPUtype &p, GPUtype &q, GPUmatResult_t(*fun)(GPUtype&, GPUtype&, GPUtype&)) {


	// garbage collector
  MyGCObj<GPUtype> mgc;

	// Use local copies. This way avoid to release GPUptr
	// on original pointers
	GPUtype ptmp = GPUtype(p, 0);
	GPUtype qtmp = GPUtype(q, 0);

	gpuTYPE_t ptype = ptmp.getType();
  gpuTYPE_t qtype = qtmp.getType();

	int ispcomplex = ptmp.isComplex();
	int isqcomplex = qtmp.isComplex();

	int opcomplex = ispcomplex || isqcomplex;

	/* Check input
	 check
	 Valid operations
	 scalar op GPUsingle
	 GPUsingle op scalar
	 GPUsingle op GPUsingle*/

	// All above cases can be either single or single complex (also scalars).
	if (ptmp.isScalar() && qtmp.isScalar()) {
		throw GPUexception(GPUmatError,
				ERROR_ARG_SCALARS);
	}
	// dimensions must match.
		// check operands, check like A + B
	int dim = (ptmp.getNdims() == qtmp.getNdims() && (compareInt(ptmp.getNdims(),
			ptmp.getSize(), qtmp.getSize())));
	if (dim == 1) {
	} else {
		if (ptmp.isScalar() || qtmp.isScalar()) {
			// OK
		} else {
			throw GPUexception(GPUmatError,
					ERROR_ARG2OP_DIMENSIONS);
		}
	}


	// allocate temp for results
	GPUtype * r;

  if (!ptmp.isScalar())
	  r = new GPUtype(ptmp, 1); // do not copy GPUptr
	else if (!qtmp.isScalar())
	  r = new GPUtype(qtmp, 1); // do not copy GPUptr
	else
	  r = new GPUtype(ptmp, 1); // do not copy GPUptr

  mgc.setPtr(r);

  gpuTYPE_t restype;
  if (settype!=NULL) {
    restype = settype[((int)ptype) + ((int)qtype) * NGPUTYPE];
  } else {
    restype = GPURESULTA[ptype][qtype];
  }

  if (restype == gpuNOTDEF) {
    // throw error
  	STRINGCONST char *t1 = GPUTEXTA[ptype];
 	  STRINGCONST char *t2 = GPUTEXTA[qtype];
		 char buffer[300];
				sprintf(
						buffer,
						"Function not implemented for input arguments  of type '%s' and '%s'. (ERROR code 19)", t1, t2);
				throw GPUexception(GPUmatError, buffer);

  }
  r->setType(restype);


	GPUopAllocVector(*r);

	// Run
	fun(ptmp, qtmp, *r);

	mgc.remPtr(r);
	return r;

}


GPUtype *
fftcommon(GPUtype &p, int direction, int batch) {

	GPUtype *r; // the result
	// Output
	// if p is complex also r will. setComplex does nothing if the GPUtype is
	// already complex

	r = new GPUtype(p, 1);
	r->setComplex();

	cufftType_t fftType = CUFFT_R2C;

	if (p.isComplex()) {
		fftType = CUFFT_C2C;

	} else {
		// this is necessary only in batch mode
		if (batch > 1) {
			int *rsize = r->getSize();
			rsize[0] = ((int) rsize[0] / 2) + 1;
		}
	}

	GPUopAllocVector(*r);

	/*% create plan
	 % For higher-dimensional transforms (2D and 3D), CUFFT performs
	 % FFTs in row-major or C order. For example, if the user requests a 3D
	 % transform plan for sizes X, Y, and Z, CUFFT transforms along Z, Y, and
	 % then X. The user can configure column-major FFTs by simply changing
	 % the order of the size parameters to the plan creation API functions*/

	cufftHandle plan;
	cufftResult_t status = cufftPlan1d(&plan, p.getNumel() / batch, fftType,
			batch);
	if (status != CUFFT_SUCCESS) {
		throw GPUexception(GPUmatError, "Error in cufftPlan1D.");
	}

	// Must pack into interleaved complex if p is real

	//GPUtype ptmp = GPUtype(p);

	/*if (p.isComplex() != 1) {
	 ptmp = GPUtype(p, 1);
	 ptmp.setComplex();
	 GPUopAllocVector(ptmp);
	 // pack data
	 GPUopPackfC2C(p.getNumel(), 1, p, p, ptmp); //1 is for onlyreal

	 }*/

	if (p.isComplex()) {
		status = cufftExecC2C(plan, (cufftComplex*) p.getGPUptr(),
				(cufftComplex*) r->getGPUptr(), direction);
	} else {
		status = cufftExecR2C(plan, (cufftReal*) p.getGPUptr(),
				(cufftComplex*) r->getGPUptr());
	}
	if (status != CUFFT_SUCCESS) {
		throw GPUexception(GPUmatError, "Error in cufftExecC2C.");
	}

	status = cufftDestroy(plan);
	if (status != CUFFT_SUCCESS) {
		throw GPUexception(GPUmatError, "Error in cufftDestroy.");
	}

	// if real I have to copy the symmetric coefficients
	if (!p.isComplex()) {
		if (batch > 1) {
			GPUtype *q = new GPUtype(p, 1);
			q->setComplex();
			GPUopAllocVector(*q);

			// copy r into q
			cudaError_t cudastatus = cudaSuccess;
			// width and height refer to src
			int *rsize = r->getSize();
			int *qsize = q->getSize();

			int width =  rsize[0]* r->getMySize();
			int height = rsize[1];

			int spitch = rsize[0] * r->getMySize();
			int dpitch = qsize[0] * q->getMySize();

			cudastatus = cudaMemcpy2D(q->getGPUptr(), dpitch, r->getGPUptr(),
					spitch, width, height, cudaMemcpyDeviceToDevice);
			if (cudastatus != cudaSuccess) {
				throw GPUexception(GPUmatError,"Error in memcpy2D");
			}

			delete r;
			GPUopFFTSymm(*q, 1);
			return q;
		} else {
			GPUopFFTSymm(*r, 0);
			return r;
		}

	} else {
		return r;
	}

}

GPUtype *
fftcommon2(GPUtype &p, int direction) {

	GPUtype *r; // the result
	// Output
	// if p is complex also r will. setComplex does nothing if the GPUtype is
	// already complex
	r = new GPUtype(p, 1);
	r->setComplex();
	GPUopAllocVector(*r);

	cufftType_t fftType = CUFFT_C2C;

	/*% create plan
	 % For higher-dimensional transforms (2D and 3D), CUFFT performs
	 % FFTs in row-major or C order. For example, if the user requests a 3D
	 % transform plan for sizes X, Y, and Z, CUFFT transforms along Z, Y, and
	 % then X. The user can configure column-major FFTs by simply changing
	 % the order of the size parameters to the plan creation API functions*/

	cufftHandle plan;
	int *mysize = p.getSize();
	cufftResult_t status = cufftPlan2d(&plan, mysize[1], mysize[0], fftType);
	if (status != CUFFT_SUCCESS) {
		throw GPUexception(GPUmatError, "Error in cufftPlan2D.");
	}

	// Must pack into interleaved complex if p is real

	GPUtype ptmp = GPUtype(p);

	if (p.isComplex() != 1) {
		ptmp = GPUtype(p, 1);
		ptmp.setComplex();
		GPUopAllocVector(ptmp);
		// pack data
		GPUopPackC2C(1, p, p, ptmp); //1 is for onlyreal

	}
	status = cufftExecC2C(plan, (cufftComplex*) ptmp.getGPUptr(),
			(cufftComplex*) r->getGPUptr(), direction);
	if (status != CUFFT_SUCCESS) {
		throw GPUexception(GPUmatError, "Error in cufftExecC2C.");
	}

	status = cufftDestroy(plan);
	if (status != CUFFT_SUCCESS) {
		throw GPUexception(GPUmatError, "Error in cufftDestroy.");
	}

	return r;

}
