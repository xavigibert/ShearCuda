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
#include "MatlabTemplates.hh"
static int init = 0;
static GPUmanager *GPUman;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	// Number of arguments should be at least 2
	if (nrhs < 2)
		mexErrMsgTxt("Wrong number of arguments");
	// tmp
	mxArray *lhs[2];

	// check input
	MATCHECKINPUT(prhs[0])


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
	// Make the following test
	// 1) complex elements
	// 2) prhs[0] should be either a scalar or a vector with dimension 2
	// 3) If there are more arguments, each argument is checked to be scalar after wards

	// First element a GPUsingle

	int first = 1;

	for (int i = first; i < nrhs; i++) {
		if (mxIsComplex(prhs[i]))
			mexErrMsgTxt("Size vector must be a row vector with real elements.");
	}

	int *mysize = NULL;
	int ndims;

	if (nrhs == 2) {
		// reshape(A,1)
		if (mxGetNumberOfElements(prhs[first]) == 1) {
			// this condition is not allowed

			/*mysize = (int*) Mymalloc(2 * sizeof(int));
			 mysize[0] = (int) mxGetScalar(prhs[0]);
			 mysize[1] = mysize[0];
			 ndims = 2;*/
			mexErrMsgTxt("Size vector must have at least two elements.");
		} else {
			// have to avoid [M N; P Q]
			if ((mxGetNumberOfDimensions(prhs[first]) != 2) || (mxGetM(prhs[first])
					!= 1))
				mexErrMsgTxt("Size vector must be a row vector with real elements.");
			int n = mxGetNumberOfElements(prhs[first]);
			double *tmp = mxGetPr(prhs[first]);
			mysize = (int*) Mymalloc(n * sizeof(int));
			for (int i = 0; i < n; i++) {
				mysize[i] = (int) floor(tmp[i]);
			}
			ndims = n;

		}
	} else {
		// reshape(A,M,N,...)
		int n = nrhs - 1; // first element GPUsingle
		mysize = (int*) Mymalloc(n * sizeof(int));
		for (int i = 0; i < n; i++) {
			// The error message is obsolete
			//if (mxGetNumberOfElements(prhs[i]) == 1) {
			mysize[i] = (int) mxGetScalar(prhs[i+1]);
			//} else {
			//	mexErrMsgTxt("Input arguments must be scalar.");
			//}
		}
		ndims = n;
	}

	if (mysize == NULL)
		mexErrMsgTxt("Unexpected error in zeros");

	// Check all dimensions different from 0
	for (int i = 0; i < ndims; i++) {
		if (mysize[i] == 0)
			mexErrMsgTxt("Dimension cannot be zero.");

	}

	// define r only at this point, because
	// there are many error messages before and if somethign fails
	// r will remain in memory (memory leakage)
	GPUtype *r = new GPUtype(*p, 1); // need to clone from p to get the GPUmanager

	r->setSize(ndims, mysize);

	// clean up
	if (mysize != NULL)
		Myfree(mysize);

	// do some checks
	if (r->getNumel() != p->getNumel())
		mexErrMsgTxt("To RESHAPE the number of elements must not change.");

	try {
		GPUopAllocVector(*r);

		//copy from p to r
		GPUopCudaMemcpy(r->getGPUptr(), p->getGPUptr(), p->getMySize()
				* p->getNumel(), cudaMemcpyDeviceToDevice, p->getGPUmanager());

	} catch (GPUexception ex) {
		// if something goes wrong I have to delete the GPUtype
		delete r;
		mexErrMsgTxt(ex.getError());
	}

	// create output result
	MATEXPLIKEPART3
	/*mxArray *tmpr = toMxStruct(r);
	mexCallMATLAB(1, plhs, 1, &tmpr, "GPUsingle");

	mxDestroyArray(tmpr);*/

}

