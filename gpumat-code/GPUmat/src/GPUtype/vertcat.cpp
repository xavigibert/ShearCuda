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

#ifndef MATLAB
#define MATLAB
#endif

#ifdef UNIX
#include <stdint.h>
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
#include "MatlabTemplates.hh"
static int init = 0;
static GPUmanager *GPUman;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	// My garbage collector
	MyGC mgc = MyGC();
	// garbage collector
	MyGCObj<GPUtype> mgc1;

	// tmp
	mxArray *lhs[2];

	if (nrhs == 0)
		mexErrMsgTxt("Wrong number of arguments");

  if (init == 0) {
    // Initialize function
    mexLock();
    // load GPUmanager
    mexCallMATLAB(2, &lhs[0], 0, NULL, "GPUmanager");
    GPUman = (GPUmanager *) (UINTPTR mxGetScalar(lhs[0]));
    mxDestroyArray(lhs[0]);
    init = 1;
  }

	int totrows = 0;
	int totcols = -1;

	GPUtype **ps = (GPUtype **) Mymalloc(nrhs * sizeof(GPUtype **),&mgc);

	int opcomplex = 0;

	// used to check that all elements are of the same type
	gpuTYPE_t mytype = gpuNOTDEF;

	for (int i = 0; i < nrhs; i++) {
		mxArray *p = (mxArray*) prhs[i];
		MATCHECKINPUT(p);
		//if (!(mxIsClass(p, "GPUsingle")))
		//	mexErrMsgTxt("Wrong argument. Expected a GPusingle.");
    ps[i] = mxToGPUtype(p, GPUman);
		
		GPUmanager *GPUman = ps[i]->getGPUmanager();
    if (GPUman->getCompileMode()==1) {
        GPUman->compAbort(ERROR_GPUMANAGER_COMPNOTIMPLEMENTED);
    }

		if (mytype==gpuNOTDEF) {
			mytype = ps[i]->getType();
		} else {
			int error = 0;
			/*if (mytype==gpuFLOAT) {
				if (!ps[i]->isFloat()) error = 1;
			} else if (mytype==gpuCFLOAT) {
				if (!ps[i]->isFloat()) error = 1;
			} else if (mytype==gpuDOUBLE) {
				if (!ps[i]->isDouble()) error = 1;
			} else if (mytype==gpuCDOUBLE) {
				if (!ps[i]->isDouble()) error = 1;
			}*/
			if (ps[i]->getType()!=mytype)
				error = 1;
			// check that type is the same
			if (error) {
				mexErrMsgTxt(ERROR_VERTCAT_SAMETYPE);
			}
		}

		int *mysize = ps[i]->getSize();
		if (ps[i]->getNdims() != 2)
			mexErrMsgTxt(ERROR_VERTCAT_DIM2);

		totrows = totrows + mysize[0];
		if (totcols == -1)
			totcols = mysize[1];
		else if (totcols != mysize[1])
			mexErrMsgTxt(ERROR_VERTCAT_DIMNOTCONSISTENT);

		// not supported for scalars
		if (ps[i]->isScalar())
			mexErrMsgTxt(ERROR_VERTCAT_SCALAR);

		//ispcomplex = not(isreal(p));
		opcomplex = opcomplex || ps[i]->isComplex();
	}

	int opsize;

	if ((ps[0]->getType()==gpuFLOAT)||(ps[0]->getType()==gpuCFLOAT)) {
	  opsize = GPU_SIZE_OF_FLOAT;
	  if (opcomplex)
			opsize = GPU_SIZE_OF_CFLOAT;

	} else if ((ps[0]->getType()==gpuDOUBLE)||(ps[0]->getType()==gpuCDOUBLE)) {
	  opsize = GPU_SIZE_OF_DOUBLE;
	  if (opcomplex)
			opsize = GPU_SIZE_OF_CDOUBLE;

	}

	// Create destination
	// create results
	GPUtype *r = new GPUtype(*(ps[0]), 1);
	r->setReal();
	r->setTrans(0);

	// set size
	int mysize[2] = { totrows, totcols };

	if (opcomplex)
		r->setComplex();
	r->setSize(2, mysize);

	try {
		// allocate vector
		GPUopAllocVector(*r);
    mgc1.setPtr(r); //update garbage collector


		// copy the first array
		// Remember that Matlab uses column-wise vectors
		GPUtype dst = GPUtype(*r);
		GPUtype *src = ps[0];
		int dst_w = totrows;
		int *s = src->getSize();
		int src_w = s[0];
		int src_h = s[1];

		int issrccomplex = src->isComplex();
		GPUtype srctmp = GPUtype(*src);
		//--delsrctmp = 0;

		for (int j=0;j<totcols;j++) {

			unsigned int prev_w = 0;
			src_w = 0;
			for (int i = 0; i < nrhs; i++) {
				src = ps[i];
				prev_w = prev_w + src_w;

				s = src->getSize();
				src_w = s[0];
				src_h = s[1];

				issrccomplex = src->isComplex();
				srctmp = GPUtype(*src);

				// Manage transpose
				int offsetdst = j*dst_w*opsize + prev_w*opsize;
				int offsetsrc = j*src_w*opsize;
				GPUopCudaMemcpy((void*) ((UINTPTR dst.getGPUptr())+offsetdst),
										(void*) ((UINTPTR srctmp.getGPUptr())+offsetsrc), src_w * opsize,
										cudaMemcpyDeviceToDevice, dst.getGPUmanager());



			}
		}






	} catch (GPUexception ex) {
		mexErrMsgTxt(ex.getError());
	}

	//Myfree(ps);

	// create output result
	mgc1.remPtr(r);
	plhs[0] = toMx(r);

}
