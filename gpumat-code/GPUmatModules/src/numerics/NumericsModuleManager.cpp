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
#include <stdarg.h>

#ifdef UNIX
#include <stdint.h>
#endif

#include "mex.h"

// CUDA
#include "cuda.h"
#include "cuda_runtime.h"

#include "GPUmat.hh"
#include "numerics.hh"

// static paramaters
static int init = 0;
static GPUmat *gm;

// GPU kernels
static CUfunction EYEdrvfuns[4];

#define BUFFERSIZE 300
#define CLEARBUFFER memset(buffer,0,BUFFERSIZE);
char buffer[BUFFERSIZE];

//*******************************************************************
// NUMERICS FUNCTIONS
//*******************************************************************


/// GPUmxFill
void GPUmxFill(const GPUtype &DST, int nrhs, const mxArray *prhs[]) {

  if (nrhs != 6)
      mexErrMsgTxt("Wrong number of arguments.");

  double offset = mxGetScalar(prhs[0]);
  double incr   = mxGetScalar(prhs[1]);
  int m         = (int) mxGetScalar(prhs[2]);
  int p         = (int) mxGetScalar(prhs[3]);
  int offsetp   = (int) mxGetScalar(prhs[4]);
  int type      = (int) mxGetScalar(prhs[5]);

  int dst_numel = gm->gputype.getNumel(DST);


  if ((type!=0)&&(type!=1)&&(type!=2))
    mexErrMsgTxt("Wrong type. Allowed values are 0,1,2");

  if (m<=0)
    m = dst_numel;
  if (p<=0)
      p = 1;

  gm->gputype.fill(DST, offset, incr, m, p, offsetp, type);


}

/// GPUmxColon
GPUtype GPUmxColon(const GPUtype &IN, int nrhs, const mxArray *prhs[]) {
  // J:K  is the same as [J, J+1, ..., K].
  // J:K  is empty if J > K.
  // J:D:K  is the same as [J, J+D, ..., J+m*D] where m = fix((K-J)/D).
  // J:D:K  is empty if D == 0, if D > 0 and J > K, or if D < 0 and J < K.

  gpuTYPE_t type = gm->gputype.getType(IN);



  double j=0.0;
  double k=0.0;
  double d=0.0;

  // Number of arguments should be at least 2
  if (nrhs < 2)
    mexErrMsgTxt("Wrong number of arguments");

  if (nrhs == 2) {
    mxArray * J = (mxArray*)prhs[0];
    mxArray * K = (mxArray*)prhs[1];
    d = 1;

    // Operands must be scalars
    if ((mxGetNumberOfElements(J) != 1) || (mxGetNumberOfElements(K) != 1))
      mexWarnMsgTxt(WARNING_MXCOLON_OPREALSCALARS);

    j = mxGetScalar(J);
    k = mxGetScalar(K);

  } else if (nrhs == 3) {
    mxArray * J = (mxArray*)prhs[0];
    mxArray * K = (mxArray*)prhs[2];
    mxArray * D = (mxArray*)prhs[1];

    // Operands must be scalars
    if ((mxGetNumberOfElements(J) != 1) || (mxGetNumberOfElements(K) != 1)
        || (mxGetNumberOfElements(D) != 1))
      mexWarnMsgTxt(WARNING_MXCOLON_OPREALSCALARS);

    j = *mxGetPr(prhs[0]);
    k = *mxGetPr(prhs[2]);
    d = *mxGetPr(prhs[1]);

  } else {
    mexErrMsgTxt("Wrong number of arguments.");
  }

  return gm->gputype.colon(type, j, d, k);


}



/// GPUmxSlice
GPUtype GPUmxSliceDrv(const GPUtype &RHS, int nrhs,
    const mxArray *prhs[]) {

  // simple garbage collection
  MyGCObj<Range> mygc1;

  GPUtype OUT;

  gpuTYPE_t trhs = gm->gputype.getType(RHS);


  // rg is the Range we have to populate
  Range *rg;
  parseRange(nrhs,&prhs[0],&rg, mygc1);

  // After creating the Range, I can call mxSlice
  // mxSlice uses indexes starting from 1 (Fortran/Matlab)
  OUT = gm->gputype.mxSlice(RHS,*rg);

  // I have to handle some particular cases here.
  // 1) The final result should have the same dimensions
  // as the indexes array
  // For example:
  // slice(A,{[1 2;3 4]})
  // size(A) = [2 2]
  //
  // slice(A,:) should generate a Nx1 vector (and not 1xN)
  //
  if (nrhs==1) {
    if (mxGetClassID(prhs[0]) == mxCELL_CLASS) {
      mxArray *mx = mxGetCell(prhs[0], 0);
      const mwSize * mysize = mxGetDimensions(mx);
      int n = mxGetNumberOfDimensions(mx);
      gm->gputype.setSize(OUT, n, mysize);

    } else if (mxGetClassID(prhs[0]) == mxCHAR_CLASS) {
      const int *mysize = gm->gputype.getSize(OUT);
      int newsize[2];
      newsize[0] = mysize[1];
      newsize[1] = mysize[0];

      gm->gputype.setSize(OUT, 2, newsize);
    }
  }

  return OUT;

}


/// GPUmxAssign
void GPUmxAssign(const GPUtype &LHS, const GPUtype &RHS, int dir, int nrhs,
    const mxArray *prhs[]) {

  // simple garbage collection
  MyGCObj<Range> mygc1;


  // rg is the Range we have to populate
  Range *rg;
  parseRange(nrhs, &prhs[0], &rg, mygc1);

  // After creating the Range, I can call
  gm->gputype.mxAssign(LHS, RHS, *rg, dir);

}

/// GPUmxPermuteDrv
GPUtype GPUmxPermuteDrv(const GPUtype &RHS, int nrhs,
    const mxArray *prhs[]) {

  // Garbage collector
  MyGC mygc;
  MyGCObj<Range> mygc1;

  int rhs_ndims = gm->gputype.getNdims(RHS);
  gpuTYPE_t rhs_type = gm->gputype.getType(RHS);
  const int* rhs_size = gm->gputype.getSize(RHS);



  // populate range
  Range *rg = new Range(0); // dummy start
  mygc1.setPtr(rg);
  Range * tmprg = rg; // used in loop

  // Loop through remaining parameters and construct the range
  for (int i = 0; i < rhs_ndims; i++) {
    tmprg->next = new Range(0, 1, END); //same as 1:end
    // register Range in GC
    mygc1.setPtr(tmprg->next);
    tmprg = tmprg->next;
  }

  double *dperm = mxGetPr(prhs[0]);

  // create integer vector
  int nperm = mxGetNumberOfElements(prhs[0]);
  int *iperm = (int *) malloc(sizeof(int) * nperm);
  mygc.setPtr(iperm);

  for (int i=0;i<nperm;i++)
    iperm[i] = (int) dperm[i];

  // perform so additional check
  if (nperm != rhs_ndims) {
    mexErrMsgTxt(ERROR_PERMUTE_INVALIDPERM);
  }


  // create output
  int *r_size = (int *) malloc(sizeof(int) * nperm);
  mygc.setPtr(r_size);

  // set size
  for (int i=0;i<nperm;i++) {
    // iperm is in Matlab format (index starts with 1)
    if ((iperm[i]<1)||(iperm[i]>nperm))
      mexErrMsgTxt(ERROR_PERMUTE_INVALIDPERM);
    iperm[i]=iperm[i]-1;
    r_size[i] = rhs_size[iperm[i]];
  }

  GPUtype R = gm->gputype.create(rhs_type, rhs_ndims, r_size, NULL);

  if (rg->next==0)
    mexErrMsgTxt("Unexpected error");

  gm->gputype.permute(R, RHS, *(rg->next), 0, iperm);
  return R;

}




/// eye function
void GPUeye(const GPUtype &OUT) {

  // number of elements
  int nout = gm->gputype.getNumel(OUT);

  gpuTYPE_t tout = gm->gputype.getType(OUT);

  CUdeviceptr d_OUT = (CUdeviceptr)(UINTPTR gm->gputype.getGPUptr(OUT));

// The GPU kernel depends on the type of input/output
  CUfunction drvfun;

  /* Load kernel depending on type */
  /* EYEdrvfuns[N_EYEF] float kernel
   * EYEdrvfuns[N_EYEC] Complex kernel
   * EYEdrvfuns[N_EYED] double kernel
   * EYEdrvfuns[N_EYEDC] DoubleComplex kernel
   */

  if (tout == gpuFLOAT) {
    drvfun = EYEdrvfuns[N_EYEF];
  } else if (tout == gpuCFLOAT) {
    drvfun = EYEdrvfuns[N_EYEC];
  } else if (tout == gpuDOUBLE) {
    drvfun = EYEdrvfuns[N_EYED];
  } else if (tout == gpuCDOUBLE) {
    drvfun = EYEdrvfuns[N_EYEDC];
  }

  const int * outsize = gm->gputype.getSize(OUT);
  int step = outsize[0]+1;
  int maxindex = (outsize[0]-1)*step;
  hostdrv_pars_t gpuprhs[3];
  int gpunrhs = 3;
  gpuprhs[0] = hostdrv_pars(&d_OUT,sizeof(d_OUT),__alignof(d_OUT));
  gpuprhs[1] = hostdrv_pars(&maxindex,sizeof(maxindex),__alignof(maxindex));
  gpuprhs[2] = hostdrv_pars(&step,sizeof(step),__alignof(step));

  hostGPUDRV(drvfun, nout, gpunrhs, gpuprhs);

}

/// zeros function
void GPUzeros(const GPUtype &OUT) {
  gm->gputype.fill(OUT, 0, 0, gm->gputype.getNumel(OUT), 1, 0, 0);

}

/// ones function
void GPUones(const GPUtype &OUT) {
  gm->gputype.fill(OUT, 1, 0, gm->gputype.getNumel(OUT), 1, 0, 0);

}

/// zeros driver function
GPUtype GPUmxZerosDrv(const GPUtype &IN, int nrhs, const mxArray *prhs[]) {
  gpuTYPE_t tin = gm->gputype.getType(IN);

  // result is created anyway
  GPUtype R = gm->gputype.createMx(tin, nrhs, prhs);
  gm->gputype.fill(R, 0, 0, gm->gputype.getNumel(R), 1, 0, 0);
  return R;
}

/// ones driver function
GPUtype GPUmxOnesDrv(const GPUtype &IN, int nrhs, const mxArray *prhs[]) {
  gpuTYPE_t tin = gm->gputype.getType(IN);

  // result is created anyway
  GPUtype R = gm->gputype.createMx(tin, nrhs, prhs);
  gm->gputype.fill(R, 1, 0, gm->gputype.getNumel(R), 1, 0, 0);
  return R;
}


/// eye driver function
GPUtype GPUmxEyeDrv(const GPUtype &IN, int nrhs, const mxArray *prhs[]) {
  gpuTYPE_t tin = gm->gputype.getType(IN);

  // result is created anyway
  GPUtype R = gm->gputype.createMx(tin, nrhs, prhs);
  GPUeye(R);
  return R;
}

/// mxMemCpyDtoD
void GPUmxMemCpyHtoD(const GPUtype &DST, int nrhs, const mxArray *prhs[]) {

  // dst_index is given in Matlab format, starting from 1
  // we convert to C
  int dst_index = (int) mxGetScalar(prhs[1]) - 1;
  int count    = (int) mxGetScalar(prhs[2]);

  // retrieve DST properties
  gpuTYPE_t dst_type = gm->gputype.getType(DST);
  const void * dst_gpuptr =  gm->gputype.getGPUptr(DST);
  int dst_numel = gm->gputype.getNumel(DST);
  int dst_dsize = gm->gputype.getDataSize(DST);

  // retrieve SRC properties
  gpuTYPE_t src_type;
  if (mxIsSingle(prhs[0])) {
    src_type = gpuFLOAT;
    if (mxIsComplex(prhs[0])) {
      //src_type = gpuCFLOAT;
      mexErrMsgTxt(ERROR_MEMCPYHTOD_MXCOMPLEXNOTSUPP);
    }
  } else if (mxIsDouble(prhs[0])) {
    src_type = gpuDOUBLE;
    if (mxIsComplex(prhs[0])) {
      //src_type = gpuCDOUBLE;
      mexErrMsgTxt(ERROR_MEMCPYHTOD_MXCOMPLEXNOTSUPP);
    }
  }

  const void * src_gpuptr =  mxGetPr(prhs[0]);
  int src_numel = mxGetNumberOfElements(prhs[0]);
  int src_dsize = mxGetElementSize(prhs[0]);


  // do some checks
  if ((dst_index>=dst_numel)||(dst_index<0))
    mexErrMsgTxt(ERROR_MEMCPYHTOD_WRONGDSTINDEX);

  if ((dst_index+count)>dst_numel)
    mexErrMsgTxt(ERROR_MEMCPYHTOD_TOOMANYEL);

  if (src_type != dst_type)
    mexErrMsgTxt(ERROR_MEMCPYHTOD_DSTSRCSAME);


  // calculate destination
  int offsetdst = dst_dsize*dst_index;
  void *dst = (void*) ((UINTPTR dst_gpuptr)+offsetdst);

  // source
  const void *src = src_gpuptr;

  cudaError_t cudastatus = cudaSuccess;
  cudastatus = cudaMemcpyAsync(dst, src, count*dst_dsize, cudaMemcpyHostToDevice, 0);
  if (cudastatus != cudaSuccess) {
    mexErrMsgTxt("cudaMemcpy error");
  }

}

/// mxMemCpyDtoD
void GPUmxMemCpyDtoD(const GPUtype &DST, const GPUtype &SRC, int nrhs, const mxArray *prhs[]) {


  // dst_index is given in Matlab format, starting from 1
  // we convert to C
  int dst_index = (int) mxGetScalar(prhs[0]) - 1;

  int count    = (int) mxGetScalar(prhs[1]);

  // retrieve DST properties
  gpuTYPE_t dst_type = gm->gputype.getType(DST);
  const void * dst_gpuptr =  gm->gputype.getGPUptr(DST);
  int dst_numel = gm->gputype.getNumel(DST);
  int dst_dsize = gm->gputype.getDataSize(DST);

  // retrieve SRC properties
  gpuTYPE_t src_type = gm->gputype.getType(SRC);
  const void * src_gpuptr =  gm->gputype.getGPUptr(SRC);
  int src_numel = gm->gputype.getNumel(SRC);
  int src_dsize = gm->gputype.getDataSize(SRC);


  // do some checks
  if ((dst_index>=dst_numel)||(dst_index<0))
    mexErrMsgTxt(ERROR_MEMCPYDTOD_WRONGDSTINDEX);

  if ((dst_index+count)>dst_numel)
    mexErrMsgTxt(ERROR_MEMCPYDTOD_TOOMANYEL);

  if (src_type != dst_type)
    mexErrMsgTxt(ERROR_MEMCPYDTOD_DSTSRCSAME);

  // calculate destination
  int offsetdst = dst_dsize*dst_index;
  void *dst = (void*) ((UINTPTR dst_gpuptr)+offsetdst);

  // source
  const void *src = src_gpuptr;

  cudaError_t cudastatus = cudaSuccess;
  cudastatus = cudaMemcpyAsync(dst, src, count*dst_dsize, cudaMemcpyDeviceToDevice, 0);
  if (cudastatus != cudaSuccess) {
    mexErrMsgTxt("cudaMemcpy error");
  }

}

/// repmat
GPUtype GPUmxRepmatDrv(const GPUtype &IN, int nrhs, const mxArray *prhs[]) {

  // Garbage collector
  MyGC mygc1;
  MyGCObj<Range> mygc2;

  gpuTYPE_t tin = gm->gputype.getType(IN);


  // we have to parse the dimensions array.
  // We have the following possibilities:
  // B = repmat(A,M,N)
  // B = REPMAT(A,[M N])

  // GPUmat  implements the  function createMX  which allows  to parse
  // this  kind of  input.  We use  this  function to  create a  dummy
  // GPUtype with the parsed  dimensions. It is dummy because actually
  // we  dont  need  it, we  just  need  its  dimensions. It  will  be
  // automatically deleted at the end by the garbage collector

  // nrhs-1 because first argument is a GPUtype
  // DIM is the returned dummy GPUtype with parsed
  GPUtype DIM = gm->gputype.createMx(tin, nrhs, &prhs[0]);
  int dim_ndims = gm->gputype.getNdims(DIM);
  const int *dim_size = gm->gputype.getSize(DIM);

  const int *in_size = gm->gputype.getSize(IN);
  int in_ndims = gm->gputype.getNdims(IN);
  gpuTYPE_t in_type = gm->gputype.getType(IN);

  int *inbackup_size = NULL;
  int inbackup_ndims = in_ndims;

  if (dim_ndims > in_ndims) {
    int *tmp_size = (int*) malloc(sizeof(int) * dim_ndims);
    mygc1.setPtr(tmp_size);

    for (int i = 0; i < dim_ndims; i++)
      tmp_size[i] = 1;

    // make a backup and update tmp_size
    int *inbackup_size = (int*) malloc(sizeof(int) * in_ndims);
    mygc1.setPtr(inbackup_size);
    for (int i = 0; i < in_ndims; i++) {
      inbackup_size[i] = in_size[i];
      tmp_size[i] = in_size[i];
    }

    // update IN
    in_ndims = dim_ndims;
    gm->gputype.setSize(IN, in_ndims, tmp_size);
    in_size = gm->gputype.getSize(IN);

  }

  int ind_ndims = in_ndims;
  int **ind = (int **) malloc(sizeof(int *) * ind_ndims);
  mygc1.setPtr(ind);

  Range *ind_range = new Range(0);
  mygc2.setPtr(ind_range);
  Range *rg = ind_range;

  for (int k = 0; k < ind_ndims; k++) {
    int dimk = (k < dim_ndims) ? dim_size[k] : 1;
    ind[k] = (int*) malloc(sizeof(int) * dimk * in_size[k]);
    mygc1.setPtr(ind[k]);
    int *tmp = ind[k];
    for (int j = 0; j < dimk; j++) {
      for (int i = 0; i < in_size[k]; i++) {
        tmp[i + j * in_size[k]] = i;
      }
    }
    rg->next = new Range(dimk * in_size[k] - 1, ind[k]);
    mygc2.setPtr(rg->next);
    rg = rg->next;
  }

  GPUtype R = gm->gputype.slice(IN, *(ind_range->next));

  // restore size in IN
  if (inbackup_size != NULL) {
    gm->gputype.setSize(IN, inbackup_ndims, inbackup_size);
  }

  return R;


}

/*
 * Initializes numerics MODULE.
 * 1) Load GPU kernels
 * 2) Register functions in GPUmat structure
 *
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  if (nrhs != 0)
    mexErrMsgTxt("Wrong number of arguments");

  if (init == 0) {
    // Initialize function
    mexLock();

    // load GPUmat
    gm = gmGetGPUmat();

    // load module
    CUmodule *drvmod = gmGetModule("numerics");

    //************************************************************************
    // EYE GPU KERNELS
    //************************************************************************

    // load float GPU function
    CUresult status = cuModuleGetFunction(&EYEdrvfuns[N_EYEF], *drvmod, "EYEF");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    // load complex GPU function
    status = cuModuleGetFunction(&EYEdrvfuns[N_EYEC], *drvmod, "EYEC");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    // load double GPU function
    status = cuModuleGetFunction(&EYEdrvfuns[N_EYED], *drvmod, "EYED");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    // load complex GPU function
    status = cuModuleGetFunction(&EYEdrvfuns[N_EYEDC], *drvmod, "EYEDC");
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to load user function.");
    }

    //************************************************************************
    // REGISTER FUNCTION IN GPUMAT STRUCTURE
    //************************************************************************

    // put here functions to be registered
    gm->gputype.mxRepmatDrv = GPUmxRepmatDrv;
    gm->gputype.mxPermuteDrv = GPUmxPermuteDrv;
    gm->gputype.mxEyeDrv = GPUmxEyeDrv;
    gm->gputype.eye = GPUeye;

    gm->gputype.mxZerosDrv = GPUmxZerosDrv;
    gm->gputype.zeros = GPUzeros;

    gm->gputype.mxOnesDrv = GPUmxOnesDrv;
    gm->gputype.ones = GPUones;

    gm->gputype.mxFill = GPUmxFill;

    gm->gputype.mxColonDrv = GPUmxColon;

    gm->gputype.mxMemCpyDtoD = GPUmxMemCpyDtoD;
    gm->gputype.mxMemCpyHtoD = GPUmxMemCpyHtoD;



    // aux
    gm->aux.mxAssign = GPUmxAssign;

    gm->aux.mxSliceDrv = GPUmxSliceDrv;


    //************************************************************************
    // UPDATE FLAGS
    //************************************************************************
    gm->mod.numerics = 1; // numerics module was loaded


    init = 1;
  }

}
