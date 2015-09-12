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

// includes, system
#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include <math.h>

// includes, project
//#include "cutil.h"
#include "cuda_runtime.h"
#include "cuda.h"
#include "vector_types.h"

// definitions
//typedef float2 Complex;

#include "cudalib_common.h"
#include "cudalib_error.h"
#include "cudalib.h"

// These are GPU SIZE_OF
#define GPU_SIZE_OF_FLOAT   4
#define GPU_SIZE_OF_CFLOAT  8
#define GPU_SIZE_OF_INT     4
#define GPU_SIZE_OF_CHAR    1
#define GPU_SIZE_OF_CHAR2   2
#define GPU_SIZE_OF_DOUBLE  8
#define GPU_SIZE_OF_CDOUBLE 16

#define MAXTHREADSX 65000
		
#define ALIGN_UP(offset, alignment) \
      (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

/**************************************************************
 * Helper functions
 **************************************************************/

//Round a / b to nearest higher integer value
int iDivUp(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
int iAlignUp(int a, int b) {
	return (a % b != 0) ? (a - a % b + b) : a;
}

/**************************************************************
 * CUDA Runtime FUNCTIONS
 **************************************************************/

/**************************************************************
 * KERNELS
 **************************************************************/

// decrypt kernel
CUDALIBResult mat_CRYPT(const unsigned int N, CUdeviceptr d_idata,
		CUdeviceptr d_odata, CUdeviceptr d_ipos, CUfunction *drvfun) {
	int nstreams = iDivUp(N, MAXTHREADSX*BLOCK_DIM1D);
	CUresult err = CUDA_SUCCESS;
	for (int str = 0; str < nstreams; str++) {
		int offset = str * MAXTHREADSX * BLOCK_DIM1D;
		int size = 0;
		if (str == (nstreams - 1))
			size = N - str * MAXTHREADSX * BLOCK_DIM1D;
		else
			size = MAXTHREADSX * BLOCK_DIM1D;
		const unsigned int size_x = size;
		int gridx = size_x / BLOCK_DIM1D;
		float tx = fmod((float) size_x, (float) BLOCK_DIM1D);
		if (tx > 0)
			gridx++;

		// setup execution parameters

		if (CUDA_SUCCESS != (err = cuFuncSetBlockShape(*drvfun, BLOCK_DIM1D, 1, 1))) {
			return CUDALIBDrvInitError;
		}

		if (CUDA_SUCCESS != cuFuncSetSharedSize(*drvfun, 0)) {
			return CUDALIBDrvInitError;
		}

		// add parameters
		int poffset = 0;
		ALIGN_UP(poffset, __alignof(size_x));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, size_x)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(size_x);

		ALIGN_UP(poffset, __alignof(d_idata));
		CUdeviceptr tmp = (d_idata + offset	* GPU_SIZE_OF_CHAR2);
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &tmp, sizeof(tmp))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_idata);

		ALIGN_UP(poffset, __alignof(d_odata));
		CUdeviceptr tmp1 = (d_odata + offset * GPU_SIZE_OF_CHAR);
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &tmp1, sizeof(tmp1))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_odata);

		ALIGN_UP(poffset, __alignof(d_ipos));
		CUdeviceptr tmp2 = (d_ipos + offset* GPU_SIZE_OF_INT);
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &tmp2, sizeof(tmp2))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_ipos);

		if (CUDA_SUCCESS != cuParamSetSize(*drvfun, poffset)) {
			return CUDALIBDrvInitError;
		}

		err = cuLaunchGridAsync(*drvfun, gridx, 1, 0);
		if (CUDA_SUCCESS != err) {
			return CUDALIBDrvLunchError;
		}
	}
	return CUDALIBSuccess;
}

// decrypt kernel
CUDALIBResult mat_POS(const unsigned int N, const unsigned int M,
		CUdeviceptr d_ipos, CUfunction *drvfun) {
	int nstreams = iDivUp(N, MAXTHREADSX*BLOCK_DIM1D);
	CUresult err = CUDA_SUCCESS;
	for (int str = 0; str < nstreams; str++) {
		int offset = str * MAXTHREADSX * BLOCK_DIM1D;
		int size = 0;
		if (str == (nstreams - 1))
			size = N - str * MAXTHREADSX * BLOCK_DIM1D;
		else
			size = MAXTHREADSX * BLOCK_DIM1D;
		const unsigned int size_x = size;
		int gridx = size_x / BLOCK_DIM1D;
		float tx = fmod((float) size_x, (float) BLOCK_DIM1D);
		if (tx > 0)
			gridx++;

		// setup execution parameters

		if (CUDA_SUCCESS != (err = cuFuncSetBlockShape(*drvfun, BLOCK_DIM1D, 1, 1))) {
			return CUDALIBDrvInitError;
		}

		if (CUDA_SUCCESS != cuFuncSetSharedSize(*drvfun, 0)) {
			return CUDALIBDrvInitError;
		}

		// add parameters
		int poffset = 0;
		ALIGN_UP(poffset, __alignof(size_x));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, size_x)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(size_x);

		ALIGN_UP(poffset, __alignof(M));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, M)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(M);

		ALIGN_UP(poffset, __alignof(d_ipos));
		CUdeviceptr tmp = (d_ipos + offset * GPU_SIZE_OF_INT);
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &tmp, sizeof(tmp))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_ipos);

		if (CUDA_SUCCESS != cuParamSetSize(*drvfun, poffset)) {
			return CUDALIBDrvInitError;
		}

		err = cuLaunchGridAsync(*drvfun, gridx, 1, 0);
		if (CUDA_SUCCESS != err) {
			return CUDALIBDrvLunchError;
		}
	}
	return CUDALIBSuccess;
}

// Questo che segue � il kernel CGEN_FUN_1D_IN1 nuovo che usa stream
// e non � limitato ad un numero di thread
// Versione driver

CUDALIBResult mat_HOSTDRV_A(int N, gpukernelconfig *kconf, int nrhs, hostdrv_pars_t *prhs, CUfunction *drvfun) {


  //if (kconf->gpuexecute==0)
  //  return CUDALIBSuccess;

	unsigned int maxthreads = kconf->maxthreads;
	int nstreams = iDivUp(N, maxthreads*BLOCK_DIM1D);
	CUresult err = CUDA_SUCCESS;
	for (int str = 0; str < nstreams; str++) {
		int offset = str * maxthreads * BLOCK_DIM1D;
		int size = 0;
		if (str == (nstreams - 1))
			size = N - str * maxthreads * BLOCK_DIM1D;
		else
			size = maxthreads * BLOCK_DIM1D;
		const unsigned int size_x = size;

		int gridx = iDivUp(size_x, BLOCK_DIM1D); // number of x blocks

		// setup execution parameters

		if (CUDA_SUCCESS != (err = cuFuncSetBlockShape(*drvfun, BLOCK_DIM1D, 1, 1))) {
			return CUDALIBDrvInitError;
		}

		if (CUDA_SUCCESS != cuFuncSetSharedSize(*drvfun, 0)) {
			return CUDALIBDrvInitError;
		}


		// add parameters
		int poffset = 0;

		// CUDA kernels interface
		// N: number of elements
		// offset: used for streams
		ALIGN_UP(poffset, __alignof(size_x));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, size_x)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(size_x);

		ALIGN_UP(poffset, __alignof(offset));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, offset)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(offset);

		for (int p=0;p<nrhs;p++) {
		  ALIGN_UP(poffset, prhs[p].align);
			if (CUDA_SUCCESS
					!= cuParamSetv(*drvfun, poffset, prhs[p].par, prhs[p].psize)) {
				return CUDALIBDrvInitError;
			}
			poffset += prhs[p].psize;
		}

		if (CUDA_SUCCESS != cuParamSetSize(*drvfun, poffset)) {
			return CUDALIBDrvInitError;
		}

		err = cuLaunchGridAsync(*drvfun, gridx, 1, 0);
		if (CUDA_SUCCESS != err) {
			return CUDALIBDrvLunchError;
		}
	}
	return CUDALIBSuccess;
}

CUDALIBResult mat_HOSTDRV_TRANSPOSE(gpukernelconfig *kconf, const unsigned int M, const unsigned int N,
		CUdeviceptr d_odata, int complex, int mysize, CUfunction *drvfun) {

  //if (kconf->gpuexecute==0)
  //    return CUDALIBSuccess;

	int scale = 1;
	if (complex==1) {
		scale = 2;
	}

	unsigned int maxthreads = kconf->maxthreads;
	int nstreamsx = iDivUp(N*scale, maxthreads*BLOCK_DIM2D);
	int nstreamsy = iDivUp(M, maxthreads*BLOCK_DIM2D/scale);

	for (int strx = 0; strx < nstreamsx; strx++) {
		//int offsetx = strx * maxthreads * BLOCK_DIM2D;
		int offsetx = strx * maxthreads;

		int sizex = 0;
		if (strx == (nstreamsx - 1))
			sizex = N*scale - strx * maxthreads * BLOCK_DIM2D;
		else
			sizex = maxthreads * BLOCK_DIM2D;

		for (int stry = 0; stry < nstreamsy; stry++) {
			//int offsety = stry * maxthreads * BLOCK_DIM2D/scale;
			int offsety = stry * maxthreads;

			int sizey = 0;
			if (stry == (nstreamsy - 1))
				sizey = M - stry * maxthreads * BLOCK_DIM2D/scale;
			else
				sizey = maxthreads * BLOCK_DIM2D/scale;

			// setup execution parameters
			int gridx = iDivUp(sizex, BLOCK_DIM2D); // number of x blocks
			int gridy = iDivUp(sizey, BLOCK_DIM2D/scale); // number of y blocks


			// setup execution parameters
			if (CUDA_SUCCESS != cuFuncSetBlockShape(*drvfun, BLOCK_DIM2D, BLOCK_DIM2D/scale, 1)) {
				return CUDALIBDrvInitError;
			}
			if (CUDA_SUCCESS != cuFuncSetSharedSize(*drvfun, BLOCK_DIM2D*(BLOCK_DIM2D+1)
					* mysize)) {
				return CUDALIBDrvInitError;
			}
			// add parameters
			int poffset = 0;
		  ALIGN_UP(poffset, __alignof(d_odata));
			if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &d_odata, sizeof(d_odata))) {
				return CUDALIBDrvInitError;
			}
			poffset += sizeof(d_odata);

		  ALIGN_UP(poffset, __alignof(N));
			if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, N)) {
				return CUDALIBDrvInitError;
			}
			poffset += sizeof(N);

		  ALIGN_UP(poffset, __alignof(M));
			if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, M)) {
				return CUDALIBDrvInitError;
			}
			poffset += sizeof(M);

		  ALIGN_UP(poffset, __alignof(offsetx));
			if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, offsetx)) {
				return CUDALIBDrvInitError;
			}
			poffset += sizeof(offsetx);

		  ALIGN_UP(poffset, __alignof(offsety));
			if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, offsety)) {
				return CUDALIBDrvInitError;
			}
			poffset += sizeof(offsety);

			if (CUDA_SUCCESS != cuParamSetSize(*drvfun, poffset)) {
				return CUDALIBDrvInitError;
			}
			if (CUDA_SUCCESS != cuLaunchGrid(*drvfun, gridx, gridy)) {
				return CUDALIBDrvLunchError;
			}

			if (CUDA_SUCCESS != cuCtxSynchronize()) {
				return CUDALIBDrvLunchError;
			}




		}
	}
  return CUDALIBSuccess;
}


/*
 * mat_packfC2C
 */
#define WARP2 16
#define WARP4 8

CUDALIBResult mat_PACKFC2C(gpukernelconfig *kconf, const unsigned int N, int onlyreal,
		CUdeviceptr d_re_idata, CUdeviceptr d_im_idata, CUdeviceptr d_odata,
		CUfunction *drvfun) {


  //if (kconf->gpuexecute==0)
  //    return CUDALIBSuccess;


	int nstreams = iDivUp(N, MAXTHREADSX*BLOCK_DIM1D);
	for (int str = 0; str < nstreams; str++) {
		int offset = str * MAXTHREADSX * BLOCK_DIM1D;
		int size = 0;
		if (str == (nstreams - 1))
			size = N - str * MAXTHREADSX * BLOCK_DIM1D;
		else
			size = MAXTHREADSX * BLOCK_DIM1D;

		const unsigned int size_x = size;
		int gridx = size_x / BLOCK_DIM1D;
		float tx = fmod((float) size_x, (float) BLOCK_DIM1D);
		if (tx > 0)
			gridx++;

		// setup execution parameters
		if (CUDA_SUCCESS != cuFuncSetBlockShape(*drvfun, BLOCK_DIM1D, 1, 1)) {
			return CUDALIBDrvInitError;
		}
		if (CUDA_SUCCESS != cuFuncSetSharedSize(*drvfun, (BLOCK_DIM1D*2
				+ BLOCK_DIM1D / WARP4) * GPU_SIZE_OF_FLOAT)) {
			return CUDALIBDrvInitError;
		}
		// add parameters
		int poffset = 0;
		ALIGN_UP(poffset, __alignof(size_x));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, size_x)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(size_x);
		
		ALIGN_UP(poffset, __alignof(onlyreal));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, onlyreal)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(onlyreal);
		

		ALIGN_UP(poffset, __alignof(d_re_idata));
		CUdeviceptr tmp = (d_re_idata + offset * GPU_SIZE_OF_FLOAT);
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &tmp, sizeof(tmp))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_re_idata);
		
		ALIGN_UP(poffset, __alignof(d_im_idata));
		CUdeviceptr tmp1 = (d_im_idata + offset	* GPU_SIZE_OF_FLOAT);
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &tmp1, sizeof(tmp1))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_im_idata);


		ALIGN_UP(poffset, __alignof(d_odata));
		CUdeviceptr tmp2 = (d_odata + 2 * offset* GPU_SIZE_OF_FLOAT);
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &tmp2, sizeof(tmp2))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_odata);

		if (CUDA_SUCCESS != cuParamSetSize(*drvfun, poffset)) {
			return CUDALIBDrvInitError;
		}
		if (CUDA_SUCCESS != cuLaunchGridAsync(*drvfun, gridx, 1, 0)) {
			return CUDALIBDrvLunchError;
		}
	}
	return CUDALIBSuccess;

	//cudaThreadSynchronize();
}

/*
 * mat_unpackfC2C
 */

CUDALIBResult mat_UNPACKFC2C(gpukernelconfig *kconf, const unsigned int N, int onlyreal,
		CUdeviceptr d_idata, CUdeviceptr d_re_odata, CUdeviceptr d_im_odata,
		CUfunction *drvfun) {

  //if (kconf->gpuexecute==0)
  //    return CUDALIBSuccess;


	int nstreams = iDivUp(N, MAXTHREADSX*BLOCK_DIM1D);
	for (int str = 0; str < nstreams; str++) {
		int offset = str * MAXTHREADSX * BLOCK_DIM1D;
		int size = 0;
		if (str == (nstreams - 1))
			size = N - str * MAXTHREADSX * BLOCK_DIM1D;
		else
			size = MAXTHREADSX * BLOCK_DIM1D;

		const unsigned int size_x = size;

		int gridx = size_x / BLOCK_DIM1D;
		float tx = fmod((float) size_x, (float) BLOCK_DIM1D);
		if (tx > 0)
			gridx++;

		// setup execution parameters
		if (CUDA_SUCCESS != cuFuncSetBlockShape(*drvfun, BLOCK_DIM1D, 1, 1)) {
			return CUDALIBDrvInitError;
		}
		if (CUDA_SUCCESS != cuFuncSetSharedSize(*drvfun, (BLOCK_DIM1D*2
				+ BLOCK_DIM1D / WARP4) * GPU_SIZE_OF_FLOAT)) {
			return CUDALIBDrvInitError;
		}
		// add parameters
		int poffset = 0;
		ALIGN_UP(poffset, __alignof(size_x));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, size_x)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(size_x);
		
		
		ALIGN_UP(poffset, __alignof(onlyreal));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, onlyreal)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(onlyreal);

		ALIGN_UP(poffset, __alignof(d_idata));
		CUdeviceptr tmp = (d_idata + 2 * offset	* GPU_SIZE_OF_FLOAT);
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &tmp, sizeof(tmp))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_idata);

		ALIGN_UP(poffset, __alignof(d_re_odata));
    CUdeviceptr tmp1 = (d_re_odata + offset	* GPU_SIZE_OF_FLOAT);
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &tmp1, sizeof(tmp1))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_re_odata);

		ALIGN_UP(poffset, __alignof(d_im_odata));
		CUdeviceptr tmp2 = (d_im_odata + offset	* GPU_SIZE_OF_FLOAT);
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &tmp2, sizeof(tmp2))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_im_odata);

		if (CUDA_SUCCESS != cuParamSetSize(*drvfun, poffset)) {
			return CUDALIBDrvInitError;
		}
		if (CUDA_SUCCESS != cuLaunchGridAsync(*drvfun, gridx, 1, 0)) {
			return CUDALIBDrvLunchError;
		}
	}
	return CUDALIBSuccess;

}

#undef WARP2
#undef WARP4

/*
 * mat_packC2C
 */
#define WARP2 16
#define WARP4 8

CUDALIBResult mat_PACKC2C(gpukernelconfig *kconf, const unsigned int maxthreads, const unsigned int N, int onlyreal,
		                      CUdeviceptr d_re_idata, unsigned int d_re_idatasize,
													CUdeviceptr d_im_idata, unsigned int d_im_idatasize,
													CUdeviceptr d_odata,    unsigned int d_odatasize,
													CUfunction *drvfun) {

  //if (kconf->gpuexecute==0)
  //    return CUDALIBSuccess;

	int nstreams = iDivUp(N, maxthreads*BLOCK_DIM1D);
	// pack/unpack works as every array is of size float, double, etc. Never Complex, DoubleComplex
	// This is the reason why I create a variable gpusize and use it for the data size
	unsigned int gpusize = d_re_idatasize;

	for (int str = 0; str < nstreams; str++) {
		int offset = str * maxthreads * BLOCK_DIM1D;
		int size = 0;
		if (str == (nstreams - 1))
			size = N - str * maxthreads * BLOCK_DIM1D;
		else
			size = maxthreads * BLOCK_DIM1D;

		const unsigned int size_x = size;
		int gridx = size_x / BLOCK_DIM1D;
		float tx = fmod((float) size_x, (float) BLOCK_DIM1D);
		if (tx > 0)
			gridx++;

		// setup execution parameters
		if (CUDA_SUCCESS != cuFuncSetBlockShape(*drvfun, BLOCK_DIM1D, 1, 1)) {
			return CUDALIBDrvInitError;
		}
		if (CUDA_SUCCESS != cuFuncSetSharedSize(*drvfun, (BLOCK_DIM1D*2
				+ BLOCK_DIM1D / WARP4) * gpusize)) {
			return CUDALIBDrvInitError;
		}
		// add parameters
		int poffset = 0;
		ALIGN_UP(poffset, __alignof(size_x));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, size_x)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(size_x);
		
		ALIGN_UP(poffset, __alignof(onlyreal));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, onlyreal)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(onlyreal);

		ALIGN_UP(poffset, __alignof(d_re_idata));
		CUdeviceptr tmp = (d_re_idata + offset* gpusize);
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &tmp, sizeof(tmp))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_re_idata);

		ALIGN_UP(poffset, __alignof(d_im_idata));
		CUdeviceptr tmp1 = (d_im_idata + offset	* gpusize);
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &tmp1, sizeof(tmp1))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_im_idata);

		ALIGN_UP(poffset, __alignof(d_odata));
		CUdeviceptr tmp2 = (d_odata + 2 * offset* gpusize);
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &tmp2, sizeof(tmp2))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_odata);

		if (CUDA_SUCCESS != cuParamSetSize(*drvfun, poffset)) {
			return CUDALIBDrvInitError;
		}
		if (CUDA_SUCCESS != cuLaunchGridAsync(*drvfun, gridx, 1, 0)) {
			return CUDALIBDrvLunchError;
		}
	}
	return CUDALIBSuccess;

	//cudaThreadSynchronize();
}

/*
 * mat_unpackC2C
 */
CUDALIBResult mat_UNPACKC2C(gpukernelconfig *kconf, const unsigned int maxthreads, const unsigned int N, int onlyreal,
		                        CUdeviceptr d_idata,    unsigned int d_idatasize,
														CUdeviceptr d_re_odata, unsigned int d_re_odatasize,
														CUdeviceptr d_im_odata, unsigned int d_im_odatasize,
		                        CUfunction *drvfun) {

  //if (kconf->gpuexecute==0)
  //    return CUDALIBSuccess;

	// pack/unpack works as every array is of size float, double, etc. Never Complex, DoubleComplex
	// This is the reason why I create a variable gpusize and use it for the data size
	unsigned int gpusize = d_re_odatasize;

	int nstreams = iDivUp(N, maxthreads*BLOCK_DIM1D);
	for (int str = 0; str < nstreams; str++) {
		int offset = str * maxthreads * BLOCK_DIM1D;
		int size = 0;
		if (str == (nstreams - 1))
			size = N - str * maxthreads * BLOCK_DIM1D;
		else
			size = maxthreads * BLOCK_DIM1D;

		const unsigned int size_x = size;

		int gridx = size_x / BLOCK_DIM1D;
		float tx = fmod((float) size_x, (float) BLOCK_DIM1D);
		if (tx > 0)
			gridx++;

		// setup execution parameters
		if (CUDA_SUCCESS != cuFuncSetBlockShape(*drvfun, BLOCK_DIM1D, 1, 1)) {
			return CUDALIBDrvInitError;
		}
		if (CUDA_SUCCESS != cuFuncSetSharedSize(*drvfun, (BLOCK_DIM1D*2
				+ BLOCK_DIM1D / WARP4) * gpusize)) {
			return CUDALIBDrvInitError;
		}
		// add parameters
		int poffset = 0;
		ALIGN_UP(poffset, __alignof(size_x));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, size_x)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(size_x);
		
		
		ALIGN_UP(poffset, __alignof(onlyreal));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, onlyreal)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(onlyreal);

		ALIGN_UP(poffset, __alignof(d_idata));
		CUdeviceptr tmp = (d_idata + 2 * offset* gpusize);
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &tmp, sizeof(tmp))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_idata);

		ALIGN_UP(poffset, __alignof(d_re_odata));
		CUdeviceptr tmp1 = (d_re_odata + offset	* gpusize);
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &tmp1, sizeof(tmp1))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_re_odata);

		ALIGN_UP(poffset, __alignof(d_im_odata));
		CUdeviceptr tmp2 = (d_im_odata + offset	* gpusize);
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &tmp2, sizeof(tmp2))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_im_odata);

		if (CUDA_SUCCESS != cuParamSetSize(*drvfun, poffset)) {
			return CUDALIBDrvInitError;
		}
		if (CUDA_SUCCESS != cuLaunchGridAsync(*drvfun, gridx, 1, 0)) {
			return CUDALIBDrvLunchError;
		}
	}
	return CUDALIBSuccess;

}
#undef WARP2
#undef WARP4

/*
 * N - size of d_idata
 * M - size of d_odata
 * K - size of pars
 */


CUDALIBResult mat_SUBSINDEXF(gpukernelconfig *kconf, const unsigned int N, CUdeviceptr d_idata,
		const int idxshift, const unsigned int K, CUdeviceptr d_pars,
		const unsigned int M, CUdeviceptr d_odata, CUfunction *drvfun,
		CUtexref *drvtex_pars, CUtexref *drvtex_idata) {

  //if (kconf->gpuexecute==0)
  //    return CUDALIBSuccess;

	// setup texture
	// idata
	if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtex_idata, d_idata, N
			* sizeof(float))) {
		return CUDALIBDrvTextureError;
	}
	if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT,
			*drvtex_idata)) {
		return CUDALIBDrvTextureError;
	}
	if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtex_idata, CU_AD_FORMAT_FLOAT, 1)) {
		return CUDALIBDrvTextureError;
	}

	// pars

	if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtex_pars, d_pars, K
			* sizeof(float))) {
		return CUDALIBDrvTextureError;
	}
	if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT,
			*drvtex_pars)) {
		return CUDALIBDrvTextureError;
	}
	if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtex_pars,
			CU_AD_FORMAT_SIGNED_INT32, 1)) {
		return CUDALIBDrvTextureError;
	}

	int nstreams = iDivUp(M, MAXTHREADSX*BLOCK_DIM1D);
	for (int str = 0; str < nstreams; str++) {
		int offset = str * MAXTHREADSX * BLOCK_DIM1D;
		int size = 0;
		if (str == (nstreams - 1))
			size = M - str * MAXTHREADSX * BLOCK_DIM1D;
		else
			size = MAXTHREADSX * BLOCK_DIM1D;

		// define the threads configuration
		int gridx = iDivUp(size, BLOCK_DIM1D); // number of x blocks


		// setup execution parameters
		if (CUDA_SUCCESS != cuFuncSetBlockShape(*drvfun, BLOCK_DIM1D, 1, 1)) {
			return CUDALIBDrvInitError;
		}
		if (CUDA_SUCCESS != cuFuncSetSharedSize(*drvfun, 0)) {
			return CUDALIBDrvInitError;
		}
		// add parameters
		int poffset = 0;
		ALIGN_UP(poffset, __alignof(size));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, size)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(size);

		ALIGN_UP(poffset, __alignof(d_odata));
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &d_odata, sizeof(d_odata))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_odata);

		ALIGN_UP(poffset, __alignof(idxshift));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, idxshift)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(idxshift);

		ALIGN_UP(poffset, __alignof(offset));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, offset)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(offset);

		if (CUDA_SUCCESS != cuParamSetSize(*drvfun, poffset)) {
			return CUDALIBDrvInitError;
		}
		if (CUDA_SUCCESS != cuLaunchGrid(*drvfun, gridx, 1)) {
			return CUDALIBDrvLunchError;
		}

		if (CUDA_SUCCESS != cuCtxSynchronize()) {
			return CUDALIBDrvLunchError;
		}

	}

	return CUDALIBSuccess;

}

/*
 * N - size of d_idata
 * M - size of d_odata
 * K - size of pars
 */

CUDALIBResult mat_SUBSINDEXC(gpukernelconfig *kconf, const unsigned int N, CUdeviceptr d_idata,
		const int idxshift, const unsigned int K, CUdeviceptr d_pars,
		const unsigned int M, CUdeviceptr d_odata, CUfunction *drvfun,
		CUtexref *drvtex_pars, CUtexref *drvtex_idata) {

	// define the threads configuration
	//int gridx = iDivUp(M, BLOCK_DIM1D); // number of x blocks


  //if (kconf->gpuexecute==0)
  //    return CUDALIBSuccess;

	// setup texture
	// idata
	if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtex_idata, d_idata, N
			* sizeof(Complex))) {
		return CUDALIBDrvTextureError;
	}
	if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT,
			*drvtex_idata)) {
		return CUDALIBDrvTextureError;
	}
	if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtex_idata, CU_AD_FORMAT_FLOAT, 2)) {
		return CUDALIBDrvTextureError;
	}

	// pars
	if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtex_pars, d_pars, K
			* sizeof(float))) {
		return CUDALIBDrvTextureError;
	}
	if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT,
			*drvtex_pars)) {
		return CUDALIBDrvTextureError;
	}
	if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtex_pars,
			CU_AD_FORMAT_SIGNED_INT32, 1)) {
		return CUDALIBDrvTextureError;
	}

	int nstreams = iDivUp(M, MAXTHREADSX*BLOCK_DIM1D);
	for (int str = 0; str < nstreams; str++) {
		int offset = str * MAXTHREADSX * BLOCK_DIM1D;
		int size = 0;
		if (str == (nstreams - 1))
			size = M - str * MAXTHREADSX * BLOCK_DIM1D;
		else
			size = MAXTHREADSX * BLOCK_DIM1D;

		// define the threads configuration
		int gridx = iDivUp(size, BLOCK_DIM1D); // number of x blocks


		// setup execution parameters
		if (CUDA_SUCCESS != cuFuncSetBlockShape(*drvfun, BLOCK_DIM1D, 1, 1)) {
			return CUDALIBDrvInitError;
		}
		if (CUDA_SUCCESS != cuFuncSetSharedSize(*drvfun, 0)) {
			return CUDALIBDrvInitError;
		}
		// add parameters
		int poffset = 0;
		ALIGN_UP(poffset, __alignof(size));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, size)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(size);

		ALIGN_UP(poffset, __alignof(d_odata));
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &d_odata, sizeof(d_odata))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_odata);

		ALIGN_UP(poffset, __alignof(idxshift));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, idxshift)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(idxshift);

		ALIGN_UP(poffset, __alignof(offset));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, offset)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(offset);

		if (CUDA_SUCCESS != cuParamSetSize(*drvfun, poffset)) {
			return CUDALIBDrvInitError;
		}
		if (CUDA_SUCCESS != cuLaunchGrid(*drvfun, gridx, 1)) {
			return CUDALIBDrvLunchError;
		}

		if (CUDA_SUCCESS != cuCtxSynchronize()) {
			return CUDALIBDrvLunchError;
		}
	}

	return CUDALIBSuccess;

}

// Kernel to check texture


CUDALIBResult mat_CHECKTEXTURE(gpukernelconfig *kconf, const unsigned int N, CUdeviceptr d_idata,
		const unsigned int M, CUdeviceptr d_odata, const unsigned int offset,
		CUfunction *drvfun, CUtexref *drvtex_idata) {

  //if (kconf->gpuexecute==0)
  //    return CUDALIBSuccess;


	// setup texture
	// idata
	if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtex_idata, d_idata, N
			* sizeof(float))) {
		return CUDALIBDrvTextureError;
	}
	if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT,
			*drvtex_idata)) {
		return CUDALIBDrvTextureError;
	}
	if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtex_idata, CU_AD_FORMAT_FLOAT, 1)) {
		return CUDALIBDrvTextureError;
	}

	// define the threads configuration
	int gridx = iDivUp(M, BLOCK_DIM1D); // number of x blocks

	// setup execution parameters
	if (CUDA_SUCCESS != cuFuncSetBlockShape(*drvfun, BLOCK_DIM1D, 1, 1)) {
		return CUDALIBDrvInitError;
	}

	if (CUDA_SUCCESS != cuFuncSetSharedSize(*drvfun, 0)) {
		return CUDALIBDrvInitError;
	}

	// add parameters
	int poffset = 0;
  ALIGN_UP(poffset, __alignof(M));
	if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, M)) {
		return CUDALIBDrvInitError;
	}
	poffset += sizeof(M);

  ALIGN_UP(poffset, __alignof(d_odata));
	if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &d_odata, sizeof(d_odata))) {
		return CUDALIBDrvInitError;
	}
	poffset += sizeof(d_odata);

  ALIGN_UP(poffset, __alignof(offset));
	if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, offset)) {
		return CUDALIBDrvInitError;
	}
	poffset += sizeof(offset);

	if (CUDA_SUCCESS != cuParamSetSize(*drvfun, poffset)) {
		return CUDALIBDrvInitError;
	}
	if (CUDA_SUCCESS != cuLaunchGrid(*drvfun, gridx, 1)) {
		return CUDALIBDrvLunchError;
	}

	if (CUDA_SUCCESS != cuCtxSynchronize()) {
		return CUDALIBDrvLunchError;
	}

	return CUDALIBSuccess;

}

/*
 * N - size of d_idata
 * M - size of d_odata
 * K - size of pars
 */


CUDALIBResult mat_FFTSYMM(gpukernelconfig *kconf, int M, int N, int Q, CUdeviceptr d_idata,
		CUdeviceptr d_odata, int batch, CUfunction *drvfun, CUtexref *drvtex_idata) {

  //if (kconf->gpuexecute==0)
  //    return CUDALIBSuccess;


	// define the threads configuration
	//int gridx = iDivUp(M, BLOCK_DIM1D); // number of x blocks


	int Nthreads = M * N * Q;

	// setup texture
	// idata
	/*if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtex_idata, d_idata, Nthreads
			* sizeof(Complex))) {
		return CUDALIBDrvTextureError;
	}*/
	/*if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT,
			*drvtex_idata)) {
		return CUDALIBDrvTextureError;
	}*/
	/*if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtex_idata, CU_AD_FORMAT_FLOAT, 2)) {
		return CUDALIBDrvTextureError;
	}*/

	int nstreams = iDivUp(Nthreads, MAXTHREADSX*BLOCK_DIM1D);
	for (int str = 0; str < nstreams; str++) {
		int offset = str * MAXTHREADSX * BLOCK_DIM1D;
		int size = 0;
		if (str == (nstreams - 1))
			size = Nthreads - str * MAXTHREADSX * BLOCK_DIM1D;
		else
			size = MAXTHREADSX * BLOCK_DIM1D;

		// define the threads configuration
		int gridx = iDivUp(size, BLOCK_DIM1D); // number of x blocks


		// setup execution parameters
		if (CUDA_SUCCESS != cuFuncSetBlockShape(*drvfun, BLOCK_DIM1D, 1, 1)) {
			return CUDALIBDrvInitError;
		}
		if (CUDA_SUCCESS != cuFuncSetSharedSize(*drvfun, 0)) {
			return CUDALIBDrvInitError;
		}

		// add parameters
		int poffset = 0;
    ALIGN_UP(poffset, __alignof(size));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, size)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(size);

    ALIGN_UP(poffset, __alignof(M));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, M)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(M);

    ALIGN_UP(poffset, __alignof(N));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, N)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(N);

    ALIGN_UP(poffset, __alignof(Q));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, Q)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(Q);

    ALIGN_UP(poffset, __alignof(d_odata));
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &d_odata, sizeof(d_odata))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_odata);

    ALIGN_UP(poffset, __alignof(offset));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, offset)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(offset);

    ALIGN_UP(poffset, __alignof(batch));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, batch)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(batch);

		if (CUDA_SUCCESS != cuParamSetSize(*drvfun, poffset)) {
			return CUDALIBDrvInitError;
		}
		if (CUDA_SUCCESS != cuLaunchGrid(*drvfun, gridx, 1)) {
			return CUDALIBDrvLunchError;
		}

		if (CUDA_SUCCESS != cuCtxSynchronize()) {
			return CUDALIBDrvLunchError;
		}
	}

	return CUDALIBSuccess;

}

/*
 * mat_sumf
 *
 * n - number of elements (also size of odata also the number of threads)
 * m - number of times each thread adds
 * incr - increment when reading from memory
 * thrmap - each thread starts reading from xIndex*thrmap (check kernel)
 * texwidth - width of the 2D texture
 * odata - output vector
 *
 * Given N and M I know the total number of elements of d_idata. thrmap is used together
 * with incr to decide which schema I am using to do the addition.
 *
 * Performance considerations
 * I comare different solutions
 * a) array binding
 * b) linear memory binding
 * c) linear memory binding with pre-trasnposition
 *
 *           a    b
 * sum(A,1)
 * sum(A,2)
 * sum(A,3)
 */
CUDALIBResult mat_SUMF_TEX(gpukernelconfig *kconf, CUdeviceptr d_idata, const unsigned int Nthreads,
		const unsigned int M, const unsigned int GroupSize,
		const unsigned int GroupOffset, CUdeviceptr d_odata, CUfunction *drvfun,
		CUtexref *drvtex) {

  //if (kconf->gpuexecute==0)
  //    return CUDALIBSuccess;


	int gridx = iDivUp(Nthreads, BLOCK_DIM1D); // number of x blocks


	// setup execution parameters
	if (CUDA_SUCCESS != cuFuncSetBlockShape(*drvfun, BLOCK_DIM1D, 1, 1)) {
		return CUDALIBDrvInitError;
	}
	if (CUDA_SUCCESS != cuFuncSetSharedSize(*drvfun, 0)) {
		return CUDALIBDrvInitError;
	}

	// total number of elements in d_idata is M*N
	int NTOT = M * Nthreads;
	// size for array
	int array_x = iAlignUp((int) sqrt((float) NTOT), 16);
	//int array_y = iDivUp(NTOT, array_x);

	// setup texture
	if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtex, d_idata, NTOT
			* sizeof(float))) {
		return CUDALIBDrvTextureError;
	}
	if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT, *drvtex)) {
		return CUDALIBDrvTextureError;
	}
	if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtex, CU_AD_FORMAT_FLOAT, 1)) {
		return CUDALIBDrvTextureError;
	}

	// add parameters
	int poffset = 0;
  ALIGN_UP(poffset, __alignof(Nthreads));
	if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, Nthreads)) {
		return CUDALIBDrvInitError;
	}
	poffset += sizeof(Nthreads);
  
	ALIGN_UP(poffset, __alignof(M));
	if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, M)) {
		return CUDALIBDrvInitError;
	}
	poffset += sizeof(M);
  
	ALIGN_UP(poffset, __alignof(GroupSize));
	if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, GroupSize)) {
		return CUDALIBDrvInitError;
	}
	poffset += sizeof(GroupSize);
  
	ALIGN_UP(poffset, __alignof(GroupOffset));
	if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, GroupOffset)) {
		return CUDALIBDrvInitError;
	}
	poffset += sizeof(GroupOffset);
  
	ALIGN_UP(poffset, __alignof(array_x));
	if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, array_x)) {
		return CUDALIBDrvInitError;
	}
	poffset += sizeof(array_x);
  
	ALIGN_UP(poffset, __alignof(d_odata));
	if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &d_odata, sizeof(d_odata))) {
		return CUDALIBDrvInitError;
	}
	poffset += sizeof(d_odata);

	if (CUDA_SUCCESS != cuParamSetSize(*drvfun, poffset)) {
		return CUDALIBDrvInitError;
	}
	if (CUDA_SUCCESS != cuLaunchGrid(*drvfun, gridx, 1)) {
		return CUDALIBDrvLunchError;
	}

	if (CUDA_SUCCESS != cuCtxSynchronize()) {
		return CUDALIBDrvLunchError;
	}

	return CUDALIBSuccess;

}
#define MYBLOCK_DIM1D (BLOCK_DIM2D*BLOCK_DIM2D*2)

CUDALIBResult mat_SUM1F_TEX(gpukernelconfig *kconf, CUdeviceptr d_idata, const unsigned int Nthreads,
		const unsigned int M, const unsigned int GroupSize,
		const unsigned int GroupOffset, CUdeviceptr d_odata, CUfunction *drvfun,
		CUtexref *drvtex) {

  //if (kconf->gpuexecute==0)
  //    return CUDALIBSuccess;

	int gridx = iDivUp(Nthreads, MYBLOCK_DIM1D); // number of x blocks


	// setup execution parameters
	if (CUDA_SUCCESS != cuFuncSetBlockShape(*drvfun, BLOCK_DIM2D, BLOCK_DIM2D*2, 1)) {
		return CUDALIBDrvInitError;
	}
	if (CUDA_SUCCESS != cuFuncSetSharedSize(*drvfun, MYBLOCK_DIM1D * GPU_SIZE_OF_FLOAT)) {
		return CUDALIBDrvInitError;
	}

	// total number of elements in d_idata is M*N
	int NTOT = M * Nthreads;
	// size for array
	int array_x = iAlignUp((int) sqrt((float) NTOT), 16);
	//int array_y = iDivUp(NTOT, array_x);

	// setup texture
	if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtex, d_idata, NTOT
			* sizeof(float))) {
		return CUDALIBDrvTextureError;
	}
	if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT, *drvtex)) {
		return CUDALIBDrvTextureError;
	}
	if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtex, CU_AD_FORMAT_FLOAT, 1)) {
		return CUDALIBDrvTextureError;
	}

	// add parameters
	int poffset = 0;
  ALIGN_UP(poffset, __alignof(Nthreads));
	if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, Nthreads)) {
		return CUDALIBDrvInitError;
	}
	poffset += sizeof(Nthreads);


  ALIGN_UP(poffset, __alignof(M));
	if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, M)) {
		return CUDALIBDrvInitError;
	}
	poffset += sizeof(M);


  ALIGN_UP(poffset, __alignof(GroupSize));
	if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, GroupSize)) {
		return CUDALIBDrvInitError;
	}
	poffset += sizeof(GroupSize);
  
	ALIGN_UP(poffset, __alignof(GroupOffset));
	if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, GroupOffset)) {
		return CUDALIBDrvInitError;
	}
	poffset += sizeof(GroupOffset);

  ALIGN_UP(poffset, __alignof(d_odata));
	if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &d_odata, sizeof(d_odata))) {
		return CUDALIBDrvInitError;
	}
	poffset += sizeof(d_odata);

	if (CUDA_SUCCESS != cuParamSetSize(*drvfun, poffset)) {
		return CUDALIBDrvInitError;
	}
	if (CUDA_SUCCESS != cuLaunchGrid(*drvfun, gridx, 1)) {
		return CUDALIBDrvLunchError;
	}

	if (CUDA_SUCCESS != cuCtxSynchronize()) {
		return CUDALIBDrvLunchError;
	}

	return CUDALIBSuccess;

}


CUDALIBResult mat_SUMC_TEX(gpukernelconfig *kconf, CUdeviceptr d_idata, const unsigned int Nthreads,
		const unsigned int M, const unsigned int GroupSize,
		const unsigned int GroupOffset, CUdeviceptr d_odata, CUfunction *drvfun,
		CUtexref *drvtex) {

  //if (kconf->gpuexecute==0)
  //    return CUDALIBSuccess;


	int gridx = Nthreads / BLOCK_DIM1D;
	float tx = fmod((float) Nthreads, (float) BLOCK_DIM1D);
	if (tx > 0)
		gridx++;

	// setup execution parameters
	if (CUDA_SUCCESS != cuFuncSetBlockShape(*drvfun, BLOCK_DIM1D, 1, 1)) {
		return CUDALIBDrvInitError;
	}
	if (CUDA_SUCCESS != cuFuncSetSharedSize(*drvfun, 0)) {
		return CUDALIBDrvInitError;
	}

	// total number of elements in d_idata is M*N
	int NTOT = M * Nthreads;
	// size for array
	int array_x = iAlignUp((int) sqrt((float) NTOT), 16);
	//int array_y = iDivUp(NTOT, array_x);

	// setup texture
	if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtex, d_idata, NTOT
			* sizeof(Complex))) {
		return CUDALIBDrvTextureError;
	}
	if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT, *drvtex)) {
		return CUDALIBDrvTextureError;
	}
	if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtex, CU_AD_FORMAT_FLOAT, 2)) {
		return CUDALIBDrvTextureError;
	}

	// add parameters
	int poffset = 0;
  ALIGN_UP(poffset, __alignof(Nthreads));
	if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, Nthreads)) {
		return CUDALIBDrvInitError;
	}
	poffset += sizeof(Nthreads);

  ALIGN_UP(poffset, __alignof(M));
	if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, M)) {
		return CUDALIBDrvInitError;
	}
	poffset += sizeof(M);

  ALIGN_UP(poffset, __alignof(GroupSize));
	if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, GroupSize)) {
		return CUDALIBDrvInitError;
	}
	poffset += sizeof(GroupSize);

  ALIGN_UP(poffset, __alignof(GroupOffset));
	if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, GroupOffset)) {
		return CUDALIBDrvInitError;
	}
	poffset += sizeof(GroupOffset);

  ALIGN_UP(poffset, __alignof(array_x));
	if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, array_x)) {
		return CUDALIBDrvInitError;
	}
	poffset += sizeof(array_x);
  
	ALIGN_UP(poffset, __alignof(d_odata));
	if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &d_odata, sizeof(d_odata))) {
		return CUDALIBDrvInitError;
	}
	poffset += sizeof(d_odata);

	if (CUDA_SUCCESS != cuParamSetSize(*drvfun, poffset)) {
		return CUDALIBDrvInitError;
	}
	if (CUDA_SUCCESS != cuLaunchGrid(*drvfun, gridx, 1)) {
		return CUDALIBDrvLunchError;
	}

	if (CUDA_SUCCESS != cuCtxSynchronize()) {
		return CUDALIBDrvLunchError;
	}

	return CUDALIBSuccess;
}

/**************************************************************
 * UTILITIES KERNELS
 **************************************************************/
CUDALIBResult mat_LOADMODULE(gpukernelconfig *kconf, CUdevice dv, CUcontext *ctx, CUmodule *md,
		char *buffer) {

  //if (kconf->gpuexecute==0)
  //    return CUDALIBSuccess;


	CUresult status = CUDA_SUCCESS;
	status = cuCtxCreate(ctx, 0, dv);
	if (status != CUDA_SUCCESS) {
		return CUDALIBError;
	}

	status = cuModuleLoadData(md, buffer);
	if (CUDA_SUCCESS != status) {
		return CUDALIBError;
	}

	return CUDALIBSuccess;

}

CUDALIBResult mat_COPYMEMORY(gpukernelconfig *kconf, const unsigned int N, CUdeviceptr d_idata,
		CUdeviceptr d_odata, CUfunction *drvfun) {

  //if (kconf->gpuexecute==0)
  //    return CUDALIBSuccess;


	int nstreams = iDivUp(N, MAXTHREADSX*BLOCK_DIM1D);
	CUresult err = CUDA_SUCCESS;
	for (int str = 0; str < nstreams; str++) {
		int offset = str * MAXTHREADSX * BLOCK_DIM1D;
		int size = 0;
		if (str == (nstreams - 1))
			size = N - str * MAXTHREADSX * BLOCK_DIM1D;
		else
			size = MAXTHREADSX * BLOCK_DIM1D;
		const unsigned int size_x = size;
		int gridx = size_x / BLOCK_DIM1D;
		float tx = fmod((float) size_x, (float) BLOCK_DIM1D);
		if (tx > 0)
			gridx++;

		// setup execution parameters

		if (CUDA_SUCCESS != (err = cuFuncSetBlockShape(*drvfun, BLOCK_DIM1D, 1, 1))) {
			return CUDALIBDrvInitError;
		}

		if (CUDA_SUCCESS != cuFuncSetSharedSize(*drvfun, BLOCK_DIM1D )) {
			return CUDALIBDrvInitError;
		}

		// add parameters
		int poffset = 0;
    ALIGN_UP(poffset, __alignof(size_x));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, size_x)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(size_x);

    ALIGN_UP(poffset, __alignof(d_idata));
		CUdeviceptr tmp = (d_idata + offset* GPU_SIZE_OF_FLOAT);
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &tmp, sizeof(tmp))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_idata);

    ALIGN_UP(poffset, __alignof(d_odata));
		CUdeviceptr tmp1 = (d_odata + offset* GPU_SIZE_OF_FLOAT);
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &tmp1, sizeof(tmp1))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_odata);

		if (CUDA_SUCCESS != cuParamSetSize(*drvfun, poffset)) {
			return CUDALIBDrvInitError;
		}

		err = cuLaunchGridAsync(*drvfun, gridx, 1, 0);
		if (CUDA_SUCCESS != err) {
			return CUDALIBDrvLunchError;
		}
	}
	return CUDALIBSuccess;
}

/*
 * mat_fillVectorf
 */

CUDALIBResult mat_FILLVECTORF(gpukernelconfig *kconf, const unsigned int N, const int offs,
		const float incr, CUdeviceptr d_odata, CUfunction *drvfun) {

  //if (kconf->gpuexecute==0)
  //    return CUDALIBSuccess;


	int nstreams = iDivUp(N, MAXTHREADSX*BLOCK_DIM1D);
	for (int str = 0; str < nstreams; str++) {
		int offset = str * MAXTHREADSX * BLOCK_DIM1D;
		int size = 0;
		if (str == (nstreams - 1))
			size = N - str * MAXTHREADSX * BLOCK_DIM1D;
		else
			size = MAXTHREADSX * BLOCK_DIM1D;

		// define the threads configuration
		int gridx = iDivUp(size, BLOCK_DIM1D); // number of x blocks

		//const unsigned int size_x = N;
		//int gridx = size_x / BLOCK_DIM1D;
		//float tx = fmod((float) size_x, (float) BLOCK_DIM1D);
		//if (tx > 0)
		//	gridx++;

		// setup execution parameters
		if (CUDA_SUCCESS != cuFuncSetBlockShape(*drvfun, BLOCK_DIM1D, 1, 1)) {
			return CUDALIBDrvInitError;
		}
		if (CUDA_SUCCESS != cuFuncSetSharedSize(*drvfun, 0)) {
			return CUDALIBDrvInitError;
		}

		// add parameters
		int poffset = 0;
    ALIGN_UP(poffset, __alignof(size));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, size)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(size);
    
		ALIGN_UP(poffset, __alignof(offs));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, offs)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(offs);
  
		ALIGN_UP(poffset, __alignof(incr));
		if (CUDA_SUCCESS != cuParamSetf(*drvfun, poffset, incr)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(incr);
  
		ALIGN_UP(poffset, __alignof(d_odata));
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &d_odata, sizeof(d_odata))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(d_odata);

    ALIGN_UP(poffset, __alignof(offset));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, offset)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(offset);

		if (CUDA_SUCCESS != cuParamSetSize(*drvfun, poffset)) {
			return CUDALIBDrvInitError;
		}
		if (CUDA_SUCCESS != cuLaunchGrid(*drvfun, gridx, 1)) {
			return CUDALIBDrvLunchError;
		}
	}

	return CUDALIBSuccess;

}
/*
 * mat_REALIMAG
 * Driver for REALIMAG_KERNEL. Perform Complex to Real and Real to Complex
 * conversions
 * * Depending on mode and direction the kernel perform the following operations
 *
 * dir
 * 0 - REAL to COMPLEX
 * 1 - COMPLEX to REAL
 * mode
 * 0 - REAL, IMAG
 * 1 - REAL
 * 2 - IMAG
 *
 */


CUDALIBResult mat_REALIMAG(gpukernelconfig *kconf, const unsigned int N, CUdeviceptr data,
		CUdeviceptr re, CUdeviceptr im , int dir, int mode, CUfunction *drvfun) {


  //if (kconf->gpuexecute==0)
  //    return CUDALIBSuccess;

	int nstreams = iDivUp(N, MAXTHREADSX*BLOCK_DIM1D);
	for (int str = 0; str < nstreams; str++) {
		int offset = str * MAXTHREADSX * BLOCK_DIM1D;
		int size = 0;
		if (str == (nstreams - 1))
			size = N - str * MAXTHREADSX * BLOCK_DIM1D;
		else
			size = MAXTHREADSX * BLOCK_DIM1D;

		// define the threads configuration
		int gridx = iDivUp(size, BLOCK_DIM1D); // number of x blocks


		// setup execution parameters
		if (CUDA_SUCCESS != cuFuncSetBlockShape(*drvfun, BLOCK_DIM1D, 1, 1)) {
			return CUDALIBDrvInitError;
		}
		if (CUDA_SUCCESS != cuFuncSetSharedSize(*drvfun, 0)) {
			return CUDALIBDrvInitError;
		}
		// add parameters
		int poffset = 0;

		ALIGN_UP(poffset, __alignof(size));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, size)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(size);

		ALIGN_UP(poffset, __alignof(data));
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &data, sizeof(data))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(data);

		ALIGN_UP(poffset, __alignof(re));
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &re, sizeof(re))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(re);

		ALIGN_UP(poffset, __alignof(im));
		if (CUDA_SUCCESS != cuParamSetv(*drvfun, poffset, &im, sizeof(im))) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(im);

		ALIGN_UP(poffset, __alignof(dir));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, dir)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(dir);

		ALIGN_UP(poffset, __alignof(mode));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, mode)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(mode);


		ALIGN_UP(poffset, __alignof(offset));
		if (CUDA_SUCCESS != cuParamSeti(*drvfun, poffset, offset)) {
			return CUDALIBDrvInitError;
		}
		poffset += sizeof(offset);

		if (CUDA_SUCCESS != cuParamSetSize(*drvfun, poffset)) {
			return CUDALIBDrvInitError;
		}
		if (CUDA_SUCCESS != cuLaunchGridAsync(*drvfun, gridx, 1, 0)) {
			return CUDALIBDrvLunchError;
		}

	}

	return CUDALIBSuccess;

}

