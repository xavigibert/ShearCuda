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

#include "mex.h"

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
#include "GPUop.hh"

#define BUFFERSIZE 300
#define CLEARBUFFER memset(buffer,0,BUFFERSIZE);

/* toMxStruct */
/*mxArray *
toMxStruct(GPUtype *p) {
	const char *field_names[] = { "GPUtypePtr", };
	mxArray *r;
	mwSize dims[2] = { 1, 1 };
	mxArray *field_value;

	r = mxCreateStructArray(2, dims, 1, field_names);
	field_value = mxCreateDoubleScalar(UINTPTR p);
	mxSetFieldByNumber(r, 0, 0, field_value);
	return r;
}*/


struct mxArray_tag {
    void    *reserved;
    int      reserved1[2];
    void    *reserved2;
    size_t  number_of_dims;
    unsigned int reserved3;
    struct {
        unsigned int    flag0 : 1;
        unsigned int    flag1 : 1;
        unsigned int    flag2 : 1;
        unsigned int    flag3 : 1;
        unsigned int    flag4 : 1;
        unsigned int    flag5 : 1;
        unsigned int    flag6 : 1;
        unsigned int    flag7 : 1;
        unsigned int    flag7a: 1;
        unsigned int    flag8 : 1;
        unsigned int    flag9 : 1;
        unsigned int    flag10 : 1;
        unsigned int    flag11 : 4;
        unsigned int    flag12 : 8;
        unsigned int    flag13 : 8;
    }   flags;
    size_t reserved4[2];
    union {
        struct {
            void  *pdata;
            void  *pimag_data;
            void  *reserved5;
            size_t reserved6[3];
        }   number_array;
    }   data;
};

/*void printMx1(mxArray *tmp, int dir) {
  const char *inout = NULL;

  if (dir==0)
    inout = "<--";
  else
    inout = "-->";

  mexPrintf("%s %p\n",inout,tmp->reserved);
  mexPrintf("%s %d\n",inout,tmp->reserved1[0]);
  mexPrintf("%s %d\n",inout,tmp->reserved1[1]);
  mexPrintf("%s %p\n",inout,tmp->reserved2);
  mexPrintf("%s %d\n",inout,tmp->number_of_dims);
  mexPrintf("%s %d\n",inout,tmp->reserved3);
  mexPrintf("%s %d\n",inout,tmp->reserved4[0]);
  mexPrintf("%s %d\n",inout,tmp->reserved4[1]);

  mexPrintf("%s %p\n",inout,tmp->data.number_array.pdata);
  mexPrintf("%s %p\n",inout,tmp->data.number_array.pimag_data);
  mexPrintf("%s %p\n",inout,tmp->data.number_array.reserved5);
  mexPrintf("%s %d\n",inout,tmp->data.number_array.reserved6[0]);
  mexPrintf("%s %d\n",inout,tmp->data.number_array.reserved6[1]);
  mexPrintf("%s %d\n",inout,tmp->data.number_array.reserved6[2]);

  mexPrintf("%s %d\n",inout,tmp->flags.flag0);
  mexPrintf("%s %d\n",inout,tmp->flags.flag1);
  mexPrintf("%s %d\n",inout,tmp->flags.flag2);
  mexPrintf("%s %d\n",inout,tmp->flags.flag3);
  mexPrintf("%s %d\n",inout,tmp->flags.flag4);
  mexPrintf("%s %d\n",inout,tmp->flags.flag5);
  mexPrintf("%s %d\n",inout,tmp->flags.flag6);
  mexPrintf("%s %d\n",inout,tmp->flags.flag7);
  mexPrintf("%s %d\n",inout,tmp->flags.flag8);
  mexPrintf("%s %d\n",inout,tmp->flags.flag9);
  mexPrintf("%s %d\n",inout,tmp->flags.flag10);
  mexPrintf("%s %d\n",inout,tmp->flags.flag11);
  mexPrintf("%s %d\n",inout,tmp->flags.flag12);
  mexPrintf("%s %d\n",inout,tmp->flags.flag13);
}*/

/* mxID */
/* This function returns a unique value that identifies the mxArray. This is
 * based on the investigation of the mxArray data when it is considered as a mxArray_tag
 * (check mxArray.h).
 * Based on the investigation, the field 'pdata' stores a unique identifier.
 * reserved2 is used to store links to other pointers. In Matlab, when we type
 * a = b
 * the object 'a' is the same as 'b'. They are linked together with reserved2.
 * If you type
 * c = b
 * the 'a', 'b' and 'c' are all linked together.
 *
 */

void *
mxID (const mxArray *p) {
  void *ret = p->data.number_array.pdata;

  // the following checks that the identifier is unique
  // starting point is
  void * start= p->reserved2;
  void * stop = NULL;

  /*while (start != stop) {
    // get next pointer
    mxArray *next = (mxArray *) start;

    // check identifier
    void *id = next->data.number_array.pdata;
    if (id!=ret) {
      // error
      mexErrMsgTxt(ERROR_MXID_NOTCONSISTENT);
    }
    stop = next->reserved2;
  }*/

  return ret;

}

GPUtype *
mxToGPUtype (const mxArray *prhs, GPUmanager *GPUman) {
  ///mexPrintf("************** begin mx to gputype ****************\n");
  //GPUman->extCachePrint();
  mxArray * tmp = (mxArray*)prhs;
  //mexPrintf("-> %p\n",tmp->reserved2);
  //mexPrintf("-> %p\n",tmp->data.number_array.pdata);
  //printMx1(tmp, 1);
  GPUtype *p = (GPUtype *) GPUman->extCacheGetGPUtypePtr( mxID( prhs));
  if (p==NULL) {
    // have to get it with slot number
    GPUman->extCacheCacheMiss();
    mxArray *lhs[1];
    mexCallMATLAB(1, &lhs[0], 1, (mxArray**) &prhs, "struct");
    int slot = (int) mxGetScalar(mxGetFieldByNumber(lhs[0], 0, 0));
    //mexPrintf("-> slot %d\n",slot);

    mxDestroyArray(lhs[0]);
    p = (GPUtype *) GPUman->extCacheGetGPUtypePtrBySlot(slot);
    if (p==NULL)
      mexErrMsgTxt(ERROR_MXTOGPUTYPE);
  }
  //mexPrintf("************** end mx to gputype ****************\n");
  return p;
}

/* objToStruct */
////GPUtype *
////objToStruct(mxArray *p) {
	/* p contains the pointer to a GPUtype */
////	GPUtype *r;
////	r = (GPUtype *) (UINTPTR mxGetScalar(mxGetFieldByNumber(p, 0, 0)));

////	return r;
///}

/* toMx
 * Used to return a GPUtype to Matlab
 * If the GPUtype is scalar a Matlab scalar is returned
 * It assumes that the passed pointer should be deleted if scalar
 * at the end of the procedure
 *
 * */

mxArray * toMx(GPUtype *r, int isscalar=0) {
	// garbage collector
	//MyGCObj<GPUtype> mgc;
	//mgc.setPtr(r);

	mxArray *plhs[1];
	mwSize dims[2] = {1,1};

	if ((r->getNumel() == 1)||(isscalar==1)) {
		if (r->isComplex()) {
			if (r->getType()==gpuCFLOAT) {
				Complex tmpr;
				try {

					GPUopCudaMemcpy(&tmpr, r->getGPUptr(), GPU_SIZE_OF_CFLOAT * 1,
							cudaMemcpyDeviceToHost, r->getGPUmanager());

				} catch (GPUexception ex) {
					mexErrMsgTxt(ex.getError());
				}
				//plhs[0] = mxCreateDoubleMatrix(1, 1, mxCOMPLEX);
				plhs[0] = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxCOMPLEX);
				memcpy (mxGetPr(plhs[0]), &tmpr.x, sizeof(float) );
				memcpy (mxGetPi(plhs[0]), &tmpr.y, sizeof(float) );
				//*mxGetPr(plhs[0]) = tmpr.x;
				//*mxGetPi(plhs[0]) = tmpr.y;
			} else if (r->getType()==gpuCDOUBLE){
				DoubleComplex tmpr;
				try {

					GPUopCudaMemcpy(&tmpr, r->getGPUptr(), GPU_SIZE_OF_CDOUBLE * 1,
							cudaMemcpyDeviceToHost, r->getGPUmanager());

				} catch (GPUexception ex) {
					mexErrMsgTxt(ex.getError());
				}
				//plhs[0] = mxCreateDoubleMatrix(1, 1, mxCOMPLEX);
				plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxCOMPLEX);
				memcpy (mxGetPr(plhs[0]), &tmpr.x, sizeof(double) );
				memcpy (mxGetPi(plhs[0]), &tmpr.y, sizeof(double) );

				//*mxGetPr(plhs[0]) = tmpr.x;
				//*mxGetPi(plhs[0]) = tmpr.y;
			}
		} else {
			if (r->getType()==gpuFLOAT) {
				float tmpr = 0.0;
				try {

					GPUopCudaMemcpy(&tmpr, r->getGPUptr(), GPU_SIZE_OF_FLOAT * 1,
							cudaMemcpyDeviceToHost, r->getGPUmanager());

				} catch (GPUexception ex) {
					mexErrMsgTxt(ex.getError());
				}
				//plhs[0] = mxCreateDoubleScalar(tmpr);
				plhs[0] = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
				memcpy (mxGetPr(plhs[0]), &tmpr, sizeof(float) );
				//*mxGetPr(plhs[0]) = tmpr;
			} else if (r->getType()==gpuDOUBLE) {
				double tmpr = 0.0;
				try {

					GPUopCudaMemcpy(&tmpr, r->getGPUptr(), GPU_SIZE_OF_DOUBLE * 1,
							cudaMemcpyDeviceToHost, r->getGPUmanager());

				} catch (GPUexception ex) {
					mexErrMsgTxt(ex.getError());
				}
				//plhs[0] = mxCreateDoubleScalar(tmpr);
				plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
				memcpy (mxGetPr(plhs[0]), &tmpr, sizeof(double) );
				//*mxGetPr(plhs[0]) = tmpr;
			}
		}

		// remove first from Garbage collector
		//mgc.remPtr(r);

		// must delete r
		delete r;
	} else {
	  GPUmanager *gman = r->getGPUmanager();
	  int slot = 0;
	  // request a free slot in cache
		gpuTYPE_t mxtype = gpuNOTDEF;
		
		// first I search for a cached element. mxtype is the tyep of cached
		// element I am looking for. 
		// If the cached element is not found, I have to create a new GPUsingle/GPUdouble
		// which is not registered to the cache. 

		if (r->isFloat())
			mxtype = gpuFLOAT; // complex as well
		if (r->isDouble())
			mxtype = gpuDOUBLE; // complex as well

		// NPN yet implemented
	  //void *mxtmp = gman->extCacheGetFreeSlot(&slot, mxtype);
	  void *mxtmp = gman->extCacheGetFreeSlot(&slot, gpuNOTDEF);

	  if (slot<0) {
	    // internal error
	    mexErrMsgTxt(ERROR_MXID_CACHEINTERNAL);
	  }

	  if (mxtmp==NULL) {
	    // the mxArray is not in cache
      mxArray *prhs[2];
      prhs[0] = mxCreateDoubleScalar(slot);
      prhs[1] = prhs[0];

      //mxArray *tmpr = toMxStruct(r);

      if (r->getType()==gpuCFLOAT) {
        mexCallMATLAB(1, plhs, 2, prhs, "GPUsingle");
      } else if (r->getType()==gpuFLOAT){
        mexCallMATLAB(1, plhs, 2, prhs, "GPUsingle");
      } else if (r->getType()==gpuCDOUBLE){
        mexCallMATLAB(1, plhs, 2, prhs, "GPUdouble");
      } else if (r->getType()==gpuDOUBLE){
        mexCallMATLAB(1, plhs, 2, prhs, "GPUdouble");
      }

      gman->extCacheRegisterPtrBySlot(slot, mxID(plhs[0]), plhs[0], r, gpuNOTDEF);

      //mexPrintf("%p -> %p\n", plhs[0], r);

      // clean up
      mxDestroyArray(prhs[0]);

	  } else {
	    // shoud no be here, not impl.
	    mexErrMsgTxt(ERROR_MXID_CACHEINTERNAL);
	    plhs[0] = (mxArray *) mxtmp;
	    gman->extCacheRegisterPtrBySlot(slot, mxID(plhs[0]), NULL, r, mxtype);
	  }
	  //mexPrintf("************** begin return mx ****************\n");
	  //gman->extCachePrint();

	}

	// remove from gc
	//mgc.remPtr(r);
	mxArray * tmp = plhs[0];
	//printMx1(tmp, 0);
	//mexPrintf("************** end return mx ****************\n");
	//mexPrintf("<- %p\n",tmp->reserved2);
  //mexPrintf("<- %p\n",tmp->data.number_array.pdata);

  return plhs[0];

}

/* GPUtypeToMxNumericArray
 * Converts a GPUtype to a Matlab numeric array
 */

mxArray * GPUtypeToMxNumericArray(GPUtype &p) {
	mxClassID cls;
	mxComplexity cpx;
	gpuTYPE_t ptype = p.getType();
	if (ptype==gpuFLOAT) {
		cls = mxSINGLE_CLASS;
		cpx = mxREAL;
	} if (ptype==gpuCFLOAT) {
		cls = mxSINGLE_CLASS;
		cpx = mxCOMPLEX;
	} if (ptype==gpuDOUBLE) {
		cls = mxDOUBLE_CLASS;
		cpx = mxREAL;
	} if (ptype==gpuCDOUBLE) {
		cls = mxDOUBLE_CLASS;
		cpx = mxCOMPLEX;
	}

	mxArray *res = mxCreateNumericArray(p.getNdims(), p.getSize(), cls, cpx);

	if (p.isComplex()) {
		//
		GPUtype re = GPUtype(p, 1);
		re.setReal();
		GPUopAllocVector(re);

		GPUtype im = GPUtype(p, 1);
		im.setReal();
		GPUopAllocVector(im);

		GPUopRealImag(p,re,im,1,0);

		try {
			GPUopCudaMemcpy(mxGetPr(res), re.getGPUptr(), re.getMySize()
							* re.getNumel(), cudaMemcpyDeviceToHost, re.getGPUmanager());
			GPUopCudaMemcpy(mxGetPi(res), im.getGPUptr(), im.getMySize()
													* im.getNumel(), cudaMemcpyDeviceToHost, im.getGPUmanager());

		} catch (GPUexception ex) {
			mexErrMsgTxt(ex.getError());
		}
	} else {
		try {
			GPUopCudaMemcpy(mxGetPr(res), p.getGPUptr(), p.getMySize()
							* p.getNumel(), cudaMemcpyDeviceToHost, p.getGPUmanager());
		} catch (GPUexception ex) {
			mexErrMsgTxt(ex.getError());
		}
	}
	return res;
}

/* GPUtypeToMxNumericArray
 * Converts a GPUtype to a Matlab numeric array
 */

void mxNumericArrayToGPUtype(mxArray *res, GPUtype *p) {

  GPUmanager *GPUman = p->getGPUmanager();
	// garbage collector
	MyGCObj<GPUtype> mgc;
  if (p->getNumel()==0)
    return;

	if (p->isComplex()) {
		//
		GPUtype re = GPUtype(*p, 1);
		re.setReal();
		GPUopAllocVector(re);

		GPUtype im = GPUtype(*p, 1);
		im.setReal();
		GPUopAllocVector(im);


		// double precision scalars are automatically casted
		try {
		  if ((mxIsDouble(res))&&(GPUman->getCudaCapability()<13)&&(p->isScalar())) {
        float tmpre = (float) *mxGetPr(res);
        float tmpim = (float) *mxGetPi(res);

		    GPUopCudaMemcpy(re.getGPUptr(), &tmpre , re.getMySize()
                * re.getNumel(), cudaMemcpyHostToDevice, re.getGPUmanager());
        GPUopCudaMemcpy(im.getGPUptr(), &tmpim, im.getMySize()
                            * im.getNumel(), cudaMemcpyHostToDevice, im.getGPUmanager());

      } else {
        GPUopCudaMemcpy(re.getGPUptr(), mxGetPr(res) , re.getMySize()
                * re.getNumel(), cudaMemcpyHostToDevice, re.getGPUmanager());
        GPUopCudaMemcpy(im.getGPUptr(), mxGetPi(res), im.getMySize()
                            * im.getNumel(), cudaMemcpyHostToDevice, im.getGPUmanager());
      }
		} catch (GPUexception ex) {
			mexErrMsgTxt(ex.getError());
		}
		GPUopRealImag(*p,re,im,0,0);

	} else {
		try {
		  if ((mxIsDouble(res))&&(GPUman->getCudaCapability()<13)&&(p->isScalar())) {
		    float tmpre = (float) *mxGetPr(res);
		    GPUopCudaMemcpy(p->getGPUptr(), &tmpre,  p->getMySize()
		                * p->getNumel(), cudaMemcpyHostToDevice, p->getGPUmanager());

		  } else {
			  GPUopCudaMemcpy(p->getGPUptr(), mxGetPr(res),  p->getMySize()
							* p->getNumel(), cudaMemcpyHostToDevice, p->getGPUmanager());
		  }
		} catch (GPUexception ex) {
			mexErrMsgTxt(ex.getError());
		}
	}

}

/* GPUtypeToMxNumericArray
 * Converts a GPUtype to a Matlab numeric array
 */

GPUtype * mxNumericArrayToGPUtype(mxArray *res, GPUmanager *GPUman) {

	// garbage collector
	MyGCObj<GPUtype> mgc;

	// log
  GPUman->debugLog("> MXTOGPUTYPE\n",0);
  GPUman->debugPushIndent();

	gpuTYPE_t ptype = gpuFLOAT;
	if (mxIsSingle(res)) {
    ptype = gpuFLOAT;
    if (mxIsComplex(res)) {
    	ptype = gpuCFLOAT;
    }
	} else if (mxIsDouble(res)) {
	  // automatic casting for scalars only
	  int isscalar = 0;
	  if (mxGetNumberOfDimensions(res)==2) {
	    int *tmpsize = (int *)mxGetDimensions(res);
	    if ((tmpsize[0]==1)&&(tmpsize[1]==1))
	      isscalar = 1;
	  }
	  if ((GPUman->getCudaCapability()<13)&&(isscalar)) {
	    ptype = gpuFLOAT;
      if (mxIsComplex(res)) {
        ptype = gpuCFLOAT;
      }
	  } else {
      ptype = gpuDOUBLE;
      if (mxIsComplex(res)) {
        ptype = gpuCDOUBLE;
      }
	  }
	} else {
		mexErrMsgTxt("Wrong argument.");
	}
	GPUtype *p = new GPUtype(ptype, mxGetNumberOfDimensions(res) , mxGetDimensions(res), GPUman);
	mgc.setPtr(p);

  if (p->getNumel()>0) {
    GPUopAllocVector(*p);
    mxNumericArrayToGPUtype(res,p);
  }

	GPUman->debugPopIndent();
	mgc.remPtr(p);
	return p;
}


/* mxCreateGPUtype
 * Creates a GPUtype using a Matalb-like way to create variables.
 * Example (in Matlab)
 * zeros(2,3,4)
 * zeros([2 3 4])
 *
 */

GPUtype * mxCreateGPUtype(gpuTYPE_t type, GPUmanager *GPUman, int nrhs, const mxArray *prhs[]) {

	// My garbage collector
	MyGC mgc = MyGC();    // Garbage collector for Malloc
	MyGCObj<GPUtype> mgcobj; // Garbage collector for GPUtype

	// Make the following test
	// 1) complex elements
	// 2)  should be either a scalar or a vector with dimension 2
	// 3) If there are more arguments, each argument is checked to be scalar afterwards

	int nrhsstart = 0; // first 2 arguments are type and GPUmanager
	int nrhseff = nrhs-0;

	for (int i = nrhsstart; i < nrhs; i++) {
		if (mxIsComplex(prhs[i]) || (mxGetNumberOfDimensions(prhs[i]) != 2)
				|| (mxGetM(prhs[i]) != 1))
			mexErrMsgTxt("Size vector must be a row vector with real elements.");
	}

	int *mysize = NULL;
	int ndims;

	if (nrhseff == 1) {
		if (mxGetNumberOfElements(prhs[nrhsstart]) == 1) {
			mysize = (int*) Mymalloc(2 * sizeof(int),&mgc);
			mysize[0] = (int) mxGetScalar(prhs[nrhsstart]);
			mysize[1] = mysize[0];
			ndims = 2;
		} else {
			int n = mxGetNumberOfElements(prhs[nrhsstart]);
			double *tmp = mxGetPr(prhs[nrhsstart]);
			mysize = (int*) Mymalloc(n * sizeof(int),&mgc);
			for (int i = 0; i < n; i++) {
				mysize[i] = (int) floor(tmp[i]);
			}
			ndims = n;

		}
	} else {
		int n = nrhseff;
		mysize = (int*) Mymalloc(n * sizeof(int),&mgc);
		for (int i = nrhsstart; i < nrhs; i++) {
			if (mxGetNumberOfElements(prhs[i]) == 1) {
				mysize[i-nrhsstart] = (int) mxGetScalar(prhs[i]);
			} else {
				mexErrMsgTxt("Input arguments must be scalar.");
			}
		}
		ndims = n;
	}

	if (mysize == NULL)
		mexErrMsgTxt("Unexpected error in GPUtypeNew.");

	// Check all dimensions different from 0
	for (int i = 0; i < ndims; i++) {
		if (mysize[i] == 0)
			mexErrMsgTxt("Dimension cannot be zero.");

	}

	// remove any one at the end
	int finalndims = ndims;
	for (int i = ndims - 1; i > 1; i--) {
		if (mysize[i] == 1)
			finalndims--;
		else
			break;
	}
	ndims = finalndims;

	GPUtype *r = new GPUtype(type, ndims , mysize, GPUman);
	mgcobj.setPtr(r); // should delete this pointer
	r->setSize(ndims, mysize);


	try {
		 GPUopAllocVector(*r);
	} catch (GPUexception ex) {
		mexErrMsgTxt(ex.getError());
	}


	// create output result

	mgcobj.remPtr(r);
	return r;

}


