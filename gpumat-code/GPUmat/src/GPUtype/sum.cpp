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
static int init = 0;
static GPUmanager *GPUman;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	MyGC mgc = MyGC();
	// garbage collector
	MyGCObj<GPUtype> mgc1;


	// tmp
	mxArray *lhs[2];

	//GPUtype * p;

	if (nrhs > 2)
		mexErrMsgTxt("Wrong number of arguments");

	// check input
	MATCHECKINPUT(prhs[0])

	// check input
  if (init == 0) {
    // Initialize function
    mexLock();
    // load GPUmanager
    mexCallMATLAB(2, &lhs[0], 0, NULL, "GPUmanager");
    GPUman = (GPUmanager *) (UINTPTR mxGetScalar(lhs[0]));
    mxDestroyArray(lhs[0]);
    init = 1;
  }
  GPUtype *p = mxToGPUtype(prhs[0], GPUman);

  if (GPUman->getCompileMode()==1) {
      GPUman->compAbort(ERROR_GPUMANAGER_COMPNOTIMPLEMENTED);
  }

	// if the input is 1D dim=2 if psize[0]==1
	int *psize = p->getSize();


	int dim = 1;
	if (nrhs == 2) {
		dim = (int) mxGetScalar(prhs[1]);
	} else {
		if (psize[0]==1)
		  dim =2;
	}

	if (dim < 0)
		mexErrMsgTxt(
				ERROR_SUM_POSINDEX);

	// check
	GPUtype *r;

	int ispcomplex = p->isComplex();
	int opcomplex = ispcomplex;


	try {

		if (dim > p->getNdims()) {
			// In this case don't do anything
			r = new GPUtype(*p, 1); // do not copy GPUptr
			mgc1.setPtr(r);

			GPUopAllocVector(*r);
			GPUopCudaMemcpy(r->getGPUptr(), p->getGPUptr(), p->getMySize()
					* p->getNumel(), cudaMemcpyDeviceToDevice, p->getGPUmanager());

		} else {
			// allocate result
			r = new GPUtype(*p, 1); // do not copy GPUptr
			mgc1.setPtr(r);

			int *rsize = r->getSize();
			rsize[dim - 1] = 1;
			// if dim is the last dimension, the result will have one dimension less
			int mydims = p->getNdims();
			if (mydims > 2 && dim == mydims)
				mydims--;
			r->setSize(mydims, rsize);
			GPUopAllocVector(*r);

			// 09/06/15
			// According to present kernel, if dim =1 or if I have 1D case,
			// I have to use cublasSdot cublasDdot for some particular cases (see below)
			int cond = 0;
			if (dim==1) {
        int N = psize[0]-1;
        int M = p->getNumel()/psize[0] -1;
        if (N>(M*47))
        	cond =1;
			}

			if (dim==2) {
				int N = psize[1]-1;
				int M = p->getNumel()/psize[1] -1;
				if (N>(M*80))
					cond =1;
			}


			//cond = 0;
			if (cond) {
				int status = CUBLAS_STATUS_SUCCESS;
				int sumstep = 1; // the step in the sum
				int sumnel  = 1; // the number of elements in the sum

				if (dim ==1 ) {
					sumstep = 1;
					sumnel = psize[0];
				}

				if (dim ==2 ) {
						sumstep = psize[0];
						sumnel = psize[1];
				}


				int ntot = p->getNumel()/sumnel;


				/*int cpx = 1;
				if (p->isComplex())
					cpx = 2;*/

				// temp for result
				float *tmpF;
				cuComplex *tmpC;
				double *tmpD;
				cuDoubleComplex *tmpCD;

				gpuTYPE_t ptype = p->getType();

				if (ptype == gpuFLOAT) {
				  tmpF = (float *) Mymalloc(ntot*sizeof(float),&mgc);
				} else if (ptype == gpuCFLOAT) {
				  tmpC = (cuComplex *) Mymalloc(ntot*sizeof(cuComplex),&mgc);
				} if (ptype == gpuDOUBLE) {
				  tmpD = (double *) Mymalloc(ntot*sizeof(double),&mgc);
				} if (ptype == gpuCDOUBLE) {
				  tmpCD = (cuDoubleComplex *) Mymalloc(ntot*sizeof(cuDoubleComplex),&mgc);
				}

				uintptr_t dst;

				// ones vector
				GPUtype ones = GPUtype(*r,1);
				int onessize[2];
				onessize[0] = 1;
				onessize[1] = sumnel;
				ones.setSize(2,onessize);
				GPUopAllocVector(ones);
				GPUopOnes(ones,ones);


				//for (int i=0;i<psize[1];i++) {
				for (int i=0;i<ntot;i++) {
					int incr = psize[0]*i;
					if (dim==2) {
						incr = (i % psize[0]) + ((int) i/psize[0])*psize[0]*psize[1];
					}
					dst = (UINTPTR p->getGPUptr()) + incr*p->getMySize();

					if (ptype == gpuFLOAT) {
						tmpF[i]= cublasSdot(sumnel,(float*) dst, sumstep, (float *) (UINTPTR ones.getGPUptr()), 1);
					} else if (ptype == gpuCFLOAT) {
						tmpC[i]= cublasCdotu(sumnel,(cuComplex*) dst, sumstep, (cuComplex *) (UINTPTR ones.getGPUptr()), 1);
					} if (ptype == gpuDOUBLE) {
						tmpD[i]= cublasDdot(sumnel,(double*) dst, sumstep, (double *) (UINTPTR ones.getGPUptr()), 1);
					} if (ptype == gpuCDOUBLE) {
						tmpCD[i]= cublasZdotu(sumnel,(cuDoubleComplex*) dst, sumstep, (cuDoubleComplex *) (UINTPTR ones.getGPUptr()), 1);
					}

					/*if (p->isFloat())
					  tmpF[i*cpx]= cublasSdot(sumnel,(float*) dst, cpx*sumstep, (float *) (UINTPTR ones.getGPUptr()), 1);
					if (p->isDouble())
					  tmpD[i*cpx]= cublasDdot(sumnel,(double*) dst, cpx*sumstep, (double*) (UINTPTR ones.getGPUptr()), 1);*/

					//tmp[i*cpx]= cublasSasum(sumnel,(float*) dst, cpx*sumstep);

					/*if (cpx==2) {
						dst = (UINTPTR p->getGPUptr()) + incr*p->getMySize()+p->getMySize()/2;
						if (p->isFloat())
						  tmpF[i*cpx+1]= cublasSdot(sumnel,(float*) dst, cpx*sumstep, (float*) (UINTPTR ones.getGPUptr()), 1);
						if (p->isDouble())
							tmpD[i*cpx+1]= cublasDdot(sumnel,(double*) dst, cpx*sumstep, (double*) (UINTPTR ones.getGPUptr()), 1);

						//tmp[i*cpx+1]= cublasSasum(sumnel,(float*) dst, cpx*sumstep);

					}*/

					status = cublasGetError();
					if (status!=CUBLAS_STATUS_SUCCESS) {
						mexErrMsgTxt("Error in cublasSasum");
					}
				}

				// copy back result
				cudaError_t cudastatus;
				if (ptype == gpuFLOAT) {
					cudastatus = cudaMemcpy(
					(void *)r->getGPUptr(),
					(void *)tmpF,
					r->getNumel()*r->getMySize(),
					cudaMemcpyHostToDevice);

				} else if (ptype == gpuCFLOAT) {
					cudastatus = cudaMemcpy(
					(void *)r->getGPUptr(),
					(void *)tmpC,
					r->getNumel()*r->getMySize(),
					cudaMemcpyHostToDevice);

				} else if (ptype == gpuDOUBLE) {
					cudastatus = cudaMemcpy(
					(void *)r->getGPUptr(),
					(void *)tmpD,
					r->getNumel()*r->getMySize(),
					cudaMemcpyHostToDevice);

				} else if (ptype == gpuCDOUBLE) {
					cudastatus = cudaMemcpy(
					(void *)r->getGPUptr(),
					(void *)tmpCD,
					r->getNumel()*r->getMySize(),
					cudaMemcpyHostToDevice);
				}

				if (cudastatus != cudaSuccess) {
					mexErrMsgTxt("Error in memcpy");
				}
			} else {

				// if the input vector is 2D and dim=0 then
				GPUtype ptmp = GPUtype(*p);
				if ((dim == 1) && (p->getNdims() == 2)) {
					if ((psize[0]==1)||(psize[1]==1)) {

					} else {
						ptmp = GPUtype(ptmp, 1); // using flag 1 the GPUptr pointer is not the same
						// allocate memory
						GPUopAllocVector(ptmp);
						GPUopTranspose(*p, ptmp);
						dim = 2;
						int *newsize = ptmp.getSize();
						int *oldsize = p->getSize();

						newsize[0] = oldsize[1];
						newsize[1] = oldsize[0];
					}

				}

				// RUN
				dim = dim - 1; // index start from zero. Original implementation done in Matlab
				// where index of arrays starts from 1
				psize = ptmp.getSize();
				int Nthread = r->getNumel();
				int M = psize[dim];
				int GroupSize = 1;
				for (int ii = 0; ii <= (dim - 1); ii++)
					GroupSize *= psize[ii];

				int GroupOffset = GroupSize * M;
				//mexPrintf("%d %d %d %d %d\n", dim, Nthread, M, GroupSize, GroupOffset);
				//GPUopSum(*p, Nthread, M, GroupSize, GroupOffset, *r);
				GPUopSum(ptmp, Nthread, M, GroupSize, GroupOffset, *r);
			}


		}
	} catch (GPUexception ex) {
		mexErrMsgTxt(ex.getError());
	}

	// remove useless ones ate the end of r
	int rdims = r->getNdims();
	int *rsize = r->getSize();
	if ((rdims>2) && (rsize[rdims-1] == 1)) {
		rdims--;
		r->setSize(rdims,rsize);
	}
 	mgc1.remPtr(r);
	plhs[0] = toMx(r);

}

