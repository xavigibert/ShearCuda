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

//#include "cutil.h"
#include "cublas.h"
#include "cufft.h"
#include "cuda_runtime.h"
#include "cuda.h"

#include "cudalib_common.h"
#include "cudalib_error.h"
#include "cudalib.h"

//extern "C" cudaError_t  cudaFree(void *devPtr);
#include "GPUcommon.hh"
#include "GPUerror.hh"
#include "Queue.hh"
#include "GPUstream.hh"
#include "GPUmanager.hh"
#include "GPUtype.hh"
#include "GPUop.hh"

#include "GPUnumeric.hh"



//#define UINTPTR (uintptr_t)
#include "kernelnames.h"

#define BUFFERSIZE 300
#define CLEARBUFFER memset(buffer,0,BUFFERSIZE);
char buffer[BUFFERSIZE];


#define NGPUTYPE 5
gpuTYPE_t GPURESULTC[NGPUTYPE][NGPUTYPE] = {
      {gpuFLOAT  , gpuFLOAT , gpuFLOAT  , gpuFLOAT , gpuNOTDEF },
      {gpuFLOAT  , gpuFLOAT , gpuFLOAT  , gpuFLOAT , gpuNOTDEF },
      {gpuFLOAT  , gpuFLOAT , gpuFLOAT  , gpuFLOAT , gpuNOTDEF},
      {gpuFLOAT  , gpuFLOAT , gpuFLOAT  , gpuFLOAT , gpuNOTDEF},
      {gpuNOTDEF , gpuNOTDEF , gpuNOTDEF , gpuNOTDEF , gpuINT32}
  };

gpuTYPE_t GPURESULTD[NGPUTYPE] = {gpuFLOAT, gpuFLOAT, gpuFLOAT, gpuFLOAT, gpuNOTDEF};

gpuTYPE_t GPURESULTE[NGPUTYPE] = {gpuFLOAT, gpuFLOAT, gpuDOUBLE, gpuDOUBLE, gpuNOTDEF};


/*************************************************************************
 * Util
 *************************************************************************/

/*************************************************************************
 * GPUopAllocVector
 *************************************************************************/

GPUmatResult_t GPUopAllocVector(GPUtype &p) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUmanager *GPUman = p.getGPUmanager();

  if (GPUman->executionDelayed()) {

  } else {
    cudaError_t cudastatus = cudaSuccess;
    CUresult cudrvstatus = CUDA_SUCCESS;
    // allocate on temporary pointer
    //void * GPUptr;
    int numel = p.getNumel();
    int mysize = p.getMySize();

    if (numel > 0) {

      // check if memory cache is available
      GPUtype *pcache = (GPUtype*) GPUman->cacheRequestPtr(numel*mysize);
      if (pcache!=0) {

      // check if memory is available
        p.releaseGPUptr();
        p.acquireGPUptr(pcache);

      } else {

        unsigned int req_mem = numel * mysize;
        size_t ava_mem;
        size_t tot_mem;
        CUresult err = cuMemGetInfo(&ava_mem, &tot_mem);
        if (err != CUDA_SUCCESS) {
          throw GPUexception(GPUmatError,
              "Unable to retrieve device information.");
        }

        if (req_mem > ava_mem) {
          // first try, clean cache
          //GPUman->extCacheClean();
          GPUman->cacheClean();

          err = cuMemGetInfo(&ava_mem, &tot_mem);
          if (err != CUDA_SUCCESS) {
            throw GPUexception(GPUmatError,
                "Unable to retrieve device information.");
          }
          if (req_mem > ava_mem) {
            char buffer[300];
            sprintf(
                buffer,
                "Device memory allocation error. Available memory is %d KB, required %d KB",
                ava_mem / 1024, req_mem / 1024);
            throw GPUexception(GPUmatError, buffer);
          }
        }


        //int cublasstatus = cublasAlloc(numel,mysize,p.getGPUptrptr());
        cudastatus = cudaMalloc(p.getGPUptrptr(), numel * mysize);
        //cudastatus = cudaGetLastError();
        //cudrvstatus = cuMemAlloc((CUdeviceptr *)p.getGPUptrptr(), numel * mysize);

        //if (cudrvstatus != CUDA_SUCCESS) {
        if (cudastatus != cudaSuccess) {
          //GPUman->extCacheClean();
          GPUman->cacheClean();
          cudastatus = cudaMalloc(p.getGPUptrptr(), numel * mysize);

        }

        if (cudastatus != cudaSuccess) {
          //if (cublasstatus != CUDA_SUCCESS) {
          // try to recover before giving up

          throw GPUexception(GPUmatError,
              "Unable to allocate memory using cudaMalloc");

        } else {
          // OK, I can update my GPUtype
          //p.setGPUptr(GPUptr);
  #ifdef DEBUG
          FILE *fout=fopen("gpumalloc.dat","a+");
          fprintf(fout,"%p\n",p.getGPUptr());
          fclose(fout);
  #endif

          status = GPUmatSuccess;
        }

        // now write to cache
        GPUman->cacheRegisterPtr(&p);
      }
    } else {
      status = GPUmatSuccess;
    }
  }

  return status;

}


/*************************************************************************
 * GPUopFree
 *************************************************************************/

// GPUopFree cannot have as input argument a GPUtype. The reason
// is that itsCounter is shared among several objects, and its
// value (GPUptr) can change. The safest way is to take a snapshot
// of what should be scheduled and this cannot change.

GPUmatResult_t GPUopFree(GPUtype &p) {

  GPUmatResult_t status = GPUmatSuccess;
  GPUmanager *GPUman = p.getGPUmanager();

  if (GPUman->executionDelayed()) {

  } else {
    //p.print();
    cudaError_t cudastatus = cudaSuccess;
    void * GPUptr = p.getGPUptr();

    if (GPUptr) {

      // pointer should be set to NULL here, although might not be consistent
      // with cudaFree, I mean maybe cudaFree can fail. If cudaFree fails there is
      // no reason anyway to keep the pointer valid, so it is better to NULL it anyway


      cudastatus = cudaFree(GPUptr);

      //free(GPUptr);
      if (cudastatus != cudaSuccess) {
        throw GPUexception(GPUmatError,
            "Unable to free GPU memory using cudaFree");

      }

      p.setGPUptr(NULL);


#ifdef DEBUG
      FILE *fout=fopen("gpufree.dat","a+");
      fprintf(fout,"%p\n",GPUptr);
      fclose(fout);
#endif
    } else {
      status = GPUmatError;
    }
  }

  return status;

}


/*************************************************************************
 * GPUopTransposeDrv
 *************************************************************************/
GPUtype * GPUopTransposeDrv(GPUtype &p) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUtype *r;
  GPUmanager *GPUman = p.getGPUmanager();

  if (p.isEmpty() || p.getNdims() != 2)
    throw GPUexception(GPUmatError,ERROR_TRANSPOSE_INPUT2D);


  r = arg1op_drv(NULL, p, (GPUmatResult_t(*)(GPUtype&, GPUtype&)) GPUopTranspose);
  // change size
  int tmpsize[2];
  int *psize = p.getSize();
  tmpsize[0] = psize[1];
  tmpsize[1] = psize[0];
  r->setSize(2, tmpsize);

  return r;
}

/*************************************************************************
 * GPUopTranspose
 *************************************************************************/

GPUmatResult_t GPUopTranspose(GPUtype &p, GPUtype &r) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUmanager *GPUman = p.getGPUmanager();


  int * size = p.getSize();
  int m = size[0];
  int n = size[1];

  if ((m==1)||(n==1)) {
    // transpose is a banal reshape
    // copy into r the input
    //copy from p to r
    GPUopCudaMemcpy(r.getGPUptr(), p.getGPUptr(), p.getMySize()
        * p.getNumel(), cudaMemcpyDeviceToDevice, p.getGPUmanager());

  } else {

    CUDALIBResult cudalibres = CUDALIBSuccess;

    CUfunction *drvfun;
    CUtexref *drvtex;
    int complex = 0;
    int scale = 1;
    CUarray_format_enum drvtexformat;
    int drvtexnum;

    // this kernel interanlly uses only float, double..
    // complex is treated non as Complex but as 2xfloat
    int mysize = GPU_SIZE_OF_FLOAT;

    if (p.getType() == gpuFLOAT) {
      drvfun = GPUman->getCuFunction(N_TRANSPOSEF_TEX_KERNEL);
      drvtex = GPUman->getCuTexref(N_TEXREF_F1_A);
      mysize = GPU_SIZE_OF_FLOAT;
      drvtexformat = CU_AD_FORMAT_FLOAT;
      drvtexnum = 1;
      scale = 1;

    } else if (p.getType() == gpuCFLOAT) {
      drvfun = GPUman->getCuFunction(N_TRANSPOSEC_TEX_KERNEL);
      drvtex = GPUman->getCuTexref(N_TEXREF_F1_A);
      mysize = GPU_SIZE_OF_FLOAT;
      complex = 1;
      drvtexformat = CU_AD_FORMAT_FLOAT;
      drvtexnum = 1;
      scale = 2;

    } else if (p.getType() == gpuDOUBLE) {
      drvfun = GPUman->getCuFunction(N_TRANSPOSED_TEX_KERNEL);
      drvtex = GPUman->getCuTexref(N_TEXREF_D1_A);
      mysize = GPU_SIZE_OF_DOUBLE;
      drvtexformat = CU_AD_FORMAT_SIGNED_INT32;
      drvtexnum = 2;
      scale = 1;

    } else if (p.getType() == gpuCDOUBLE) {
      drvfun = GPUman->getCuFunction(N_TRANSPOSECD_TEX_KERNEL);
      drvtex = GPUman->getCuTexref(N_TEXREF_D1_A);
      mysize = GPU_SIZE_OF_DOUBLE;
      complex = 1;
      drvtexformat = CU_AD_FORMAT_SIGNED_INT32;
      drvtexnum = 2;
      scale = 2;
    }

    // setup texture
    if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtex, UINTPTR p.getGPUptr(), scale*m*n
        * mysize)) {
      throw GPUexception(GPUmatError, "Kernel execution error (texture).");
    }
    if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT, *drvtex)) {
      throw GPUexception(GPUmatError, "Kernel execution error (texture).");
    }
    if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtex, drvtexformat, drvtexnum)) {
      throw GPUexception(GPUmatError, "Kernel execution error (texture).");
    }

    gpukernelconfig_t *kconf = GPUman->getKernelConfig();

    cudalibres = mat_HOSTDRV_TRANSPOSE(kconf, n, m,
          UINTPTR r.getGPUptr(), complex, mysize, drvfun);

    if (cudalibres != CUDALIBSuccess) {
      throw GPUexception(GPUmatError, "Kernel execution error.");
    }
  }
  return status;


}

/*************************************************************************
 * GPUopCtransposeDrv
 *************************************************************************/
GPUtype * GPUopCtransposeDrv(GPUtype &p) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUtype *r;
  GPUmanager *GPUman = p.getGPUmanager();

  if (p.isEmpty() || p.getNdims() != 2)
    throw GPUexception(GPUmatError,ERROR_TRANSPOSE_INPUT2D);


  r = arg1op_drv(NULL, p, (GPUmatResult_t(*)(GPUtype&, GPUtype&)) GPUopCtranspose);
  // change size
  int tmpsize[2];
  int *psize = p.getSize();
  tmpsize[0] = psize[1];
  tmpsize[1] = psize[0];
  r->setSize(2, tmpsize);

  return r;
}
/*************************************************************************
 * GPUopCtranspose
 *************************************************************************/

GPUmatResult_t GPUopCtranspose(GPUtype &p, GPUtype &r) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUmanager *GPUman = p.getGPUmanager();


  status = GPUopTranspose(p,r);
  if (p.isComplex()) {
    status = GPUopConj(r,r);
  }
  return status;


}

/*************************************************************************
 * GPUopCudaMemcpy
 *************************************************************************/

GPUmatResult_t GPUopCudaMemcpy(void* dst, const void* src, size_t count,
    enum cudaMemcpyKind kind, GPUmanager *GPUman) {
  GPUmatResult_t status = GPUmatSuccess;


  cudaError_t cudastatus = cudaSuccess;

  cudastatus = cudaMemcpy(dst, src, count, kind);
  //cudastatus = cudaGetLastError();
  if (cudastatus != cudaSuccess) {
    char buffer[300];
    sprintf(buffer, "cudaMemcpy error(%p,%p)", dst, src);
    throw GPUexception(GPUmatError, buffer);
    //throw GPUexception(GPUmatError, "cudaMemcpy error.");
  }


  return status;

}

/*************************************************************************
 * GPUopCudaMemcpyAsync
 *************************************************************************/

GPUmatResult_t GPUopCudaMemcpyAsync(void* dst, const void* src, size_t count,
    enum cudaMemcpyKind kind, GPUmanager *GPUman) {
  GPUmatResult_t status = GPUmatSuccess;


  cudaError_t cudastatus = cudaSuccess;

  cudastatus = cudaMemcpyAsync(dst, src, count, kind, 0);
  //cudastatus = cudaGetLastError();
  if (cudastatus != cudaSuccess) {
    char buffer[300];
    sprintf(buffer, "cudaMemcpy error(%p,%p)", dst, src);
    throw GPUexception(GPUmatError, buffer);
    //throw GPUexception(GPUmatError, "cudaMemcpy error.");
  }


  return status;

}

/*************************************************************************
 * GPUopCudaMemcpy2D
 *************************************************************************/

GPUmatResult_t GPUopCudaMemcpy2D(void* dst, size_t dpitch, const void* src,
    size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind,
    GPUmanager *GPUman) {
  GPUmatResult_t status = GPUmatSuccess;


  cudaError_t cudastatus = cudaSuccess;

  cudastatus = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
  //cudastatus = cudaGetLastError();
  if (cudastatus != cudaSuccess)
    throw GPUexception(GPUmatError, "cudaMemcpy error.");

  return status;

}

/*************************************************************************
 * GPUopPackC2C
 **************************************************************************/

GPUmatResult_t GPUopPackC2C(int re, GPUtype &d_re_idata,
    GPUtype & d_im_idata, GPUtype & d_odata) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUmanager *GPUman = d_re_idata.getGPUmanager();


  CUfunction *drvfun;

  if (d_odata.getType() == gpuCDOUBLE) {
    drvfun = GPUman->getCuFunction(N_PACKDC2C_KERNEL);
  } else if (d_odata.getType() == gpuCFLOAT) {
    drvfun = GPUman->getCuFunction(N_PACKFC2C_KERNEL);
  } else {
    throw GPUexception(GPUmatError, "Unexpected.");
  }

  gpukernelconfig_t *kconf = GPUman->getKernelConfig();
  CUDALIBResult cudalibres = mat_PACKC2C(kconf, GPUman->getKernelMaxthreads(),
                                          d_re_idata.getNumel(),
                                          re,
                                          UINTPTR d_re_idata.getGPUptr(), d_re_idata.getMySize(),
                                          UINTPTR d_im_idata.getGPUptr(), d_im_idata.getMySize(),
                                          UINTPTR d_odata.getGPUptr(), d_odata.getMySize(),
                                          drvfun );
  if (cudalibres != CUDALIBSuccess) {
    throw GPUexception(GPUmatError, "Kernel execution error.");
  }

  return status;
}

/*************************************************************************
 * GPUopUnpackC2C
 * mode
 * 0 - REAL, IMAG
 * 1 - REAL
 * 2 - IMAG
 **************************************************************************/

GPUmatResult_t GPUopUnpackC2C(int mode, GPUtype& d_idata,
    GPUtype& d_re_odata, GPUtype& d_im_odata) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUmanager *GPUman = d_idata.getGPUmanager();



  CUfunction *drvfun;

  if (d_idata.getType() == gpuCDOUBLE) {
    drvfun = GPUman->getCuFunction(N_UNPACKDC2C_KERNEL);
  } else if (d_idata.getType() == gpuCFLOAT) {
    drvfun = GPUman->getCuFunction(N_UNPACKFC2C_KERNEL);
  } else {
    throw GPUexception(GPUmatError, "Unexpected.");
  }

  gpukernelconfig_t *kconf = GPUman->getKernelConfig();
  CUDALIBResult cudalibres = mat_UNPACKC2C(kconf, GPUman->getKernelMaxthreads(),
                                   d_idata.getNumel(), mode,
                                   UINTPTR d_idata.getGPUptr(), d_idata.getMySize(),
                                   UINTPTR d_re_odata.getGPUptr(), d_re_odata.getMySize(),
                                   UINTPTR d_im_odata.getGPUptr(), d_im_odata.getMySize(),
                                   drvfun);
  if (cudalibres != CUDALIBSuccess) {
    throw GPUexception(GPUmatError, "Kernel execution error.");
  }

  return status;
}


/*************************************************************************
 * GPUopRealImag
 * dir
 * 0 - REAL to COMPLEX
 * 1 - COMPLEX to REAL
 * mode
 * 0 - REAL, IMAG
 * 1 - REAL
 * 2 - IMAG
 **************************************************************************/

GPUmatResult_t GPUopRealImag(GPUtype& data, GPUtype &re, GPUtype &im, int dir, int mode)
{
GPUmatResult_t status = GPUmatSuccess;
GPUmanager *GPUman = data.getGPUmanager();

  CUfunction *drvfun;

  if (data.getType() == gpuCDOUBLE) {
    drvfun = GPUman->getCuFunction(N_REALIMAGD_KERNEL);
  } else if (data.getType() == gpuCFLOAT) {
    drvfun = GPUman->getCuFunction(N_REALIMAGF_KERNEL);
  } else {
    throw GPUexception(GPUmatError, "Unexpected.");
  }

  gpukernelconfig_t *kconf = GPUman->getKernelConfig();
  CUDALIBResult cudalibres = mat_REALIMAG(
      kconf,
      data.getNumel(),
      //UINTPTR data.getGPUptr(),
      //UINTPTR re.getGPUptr(),
      //UINTPTR im.getGPUptr(),
      (CUdeviceptr) data.getGPUptr(),
      (CUdeviceptr) re.getGPUptr(),
      (CUdeviceptr) im.getGPUptr(),
      dir,
      mode,
      drvfun);
    if (cudalibres != CUDALIBSuccess) {
      throw GPUexception(GPUmatError, "Kernel execution error.");
    }


return status;
}

/*************************************************************************
* GPUopFillVector
*************************************************************************/

GPUmatResult_t GPUopFillVector(double offset, double incr, GPUtype &r) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUmanager *GPUman = r.getGPUmanager();

  // garbage collector
  MyGCObj<GPUtype> mgc;



  int n = r.getNumel();



  CUfunction *drvfun;
  CUtexref *drvtexa;
  CUtexref *drvtexb;
  CUarray_format_enum drvtexformata;
  CUarray_format_enum drvtexformatb;
  int drvtexnuma;
  int drvtexnumb;

  gpuTYPE_t rtype = r.getType();

  // For some unknown reason I cannot pass directly a double to
  // GPU kernel. So I have to create a GPUtype and use a texture
  // for the incr.

  GPUtype *incrg;
  GPUtype *offsg;

  if (rtype==gpuCFLOAT) {
      drvfun = GPUman->getCuFunction(N_FILLVECTORC_KERNEL);
      drvtexa = GPUman->getCuTexref(N_TEXREF_F1_A);
      drvtexformata = CU_AD_FORMAT_FLOAT;
      drvtexnuma = 1;

      drvtexb = GPUman->getCuTexref(N_TEXREF_F1_B);
      drvtexformatb = CU_AD_FORMAT_FLOAT;
      drvtexnumb = 1;

      incrg = new GPUtype((float) incr,GPUman);
      mgc.setPtr(incrg);
      offsg = new GPUtype((float) offset,GPUman);
      mgc.setPtr(offsg);


  } else if (rtype==gpuFLOAT){
      drvfun = GPUman->getCuFunction(N_FILLVECTORF_KERNEL);

      drvtexa = GPUman->getCuTexref(N_TEXREF_F1_A);
      drvtexformata = CU_AD_FORMAT_FLOAT;
      drvtexnuma = 1;

      drvtexb = GPUman->getCuTexref(N_TEXREF_F1_B);
      drvtexformatb = CU_AD_FORMAT_FLOAT;
      drvtexnumb = 1;

      incrg = new GPUtype((float) incr,GPUman);
      mgc.setPtr(incrg);
      offsg = new GPUtype((float) offset,GPUman);
      mgc.setPtr(offsg);

  } else if (rtype==gpuCDOUBLE){
      drvfun = GPUman->getCuFunction(N_FILLVECTORCD_KERNEL);

      drvtexa = GPUman->getCuTexref(N_TEXREF_D1_A);
      drvtexformata = CU_AD_FORMAT_SIGNED_INT32;
      drvtexnuma = 2;

      drvtexb = GPUman->getCuTexref(N_TEXREF_D1_B);
      drvtexformatb = CU_AD_FORMAT_SIGNED_INT32;
      drvtexnumb = 2;

      incrg = new GPUtype(incr,GPUman);
      mgc.setPtr(incrg);
      offsg = new GPUtype(offset,GPUman);
      mgc.setPtr(offsg);


  } else if (rtype==gpuDOUBLE){
      drvfun = GPUman->getCuFunction(N_FILLVECTORD_KERNEL);

      drvtexa = GPUman->getCuTexref(N_TEXREF_D1_A);
      drvtexformata = CU_AD_FORMAT_SIGNED_INT32;
      drvtexnuma = 2;

      drvtexb = GPUman->getCuTexref(N_TEXREF_D1_B);
      drvtexformatb = CU_AD_FORMAT_SIGNED_INT32;
      drvtexnumb = 2;

      incrg = new GPUtype(incr,GPUman);
      mgc.setPtr(incrg);
      offsg = new GPUtype(offset,GPUman);
      mgc.setPtr(offsg);

  }



  // setup texture
  if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtexa, UINTPTR incrg->getGPUptr(), incrg->getNumel()
      * incrg->getMySize())) {
    throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
  }
  if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT, *drvtexa)) {
    throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
  }
  if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtexa, drvtexformata, drvtexnuma)) {
    throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
  }

  if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtexb, UINTPTR offsg->getGPUptr(), offsg->getNumel()
      * offsg->getMySize())) {
    throw GPUexception(GPUmatError, "Kernel execution error1.");
  }
  if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT, *drvtexb)) {
    throw GPUexception(GPUmatError, "Kernel execution error3.");
  }
  if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtexb, drvtexformatb, drvtexnumb)) {
    throw GPUexception(GPUmatError, "Kernel execution error2.");
  }

  // define kernel configuration
  gpukernelconfig_t *kconf = GPUman->getKernelConfig();
  hostdrv_pars_t pars[1];
  int nrhs = 1;

  pars[0].par =  r.getGPUptrptr();
  pars[0].psize = sizeof(CUdeviceptr);
  pars[0].align = __alignof(CUdeviceptr);

  CUDALIBResult cudalibres = mat_HOSTDRV_A(n, kconf, nrhs, pars, drvfun);

  if (cudalibres != CUDALIBSuccess) {
    throw GPUexception(GPUmatError, "Kernel execution error.");
  }



  return status;

}

/*************************************************************************
* GPUopFillVector1
*************************************************************************/

GPUmatResult_t GPUopFillVector1(double offset, double incr,  GPUtype &r, int m, int p, int offsetp, int type) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUmanager *GPUman = r.getGPUmanager();

  // garbage collector
  MyGCObj<GPUtype> mgc;



  int n = r.getNumel();



  CUfunction *drvfun;
  CUtexref *drvtexa;
  CUtexref *drvtexb;
  CUarray_format_enum drvtexformata;
  CUarray_format_enum drvtexformatb;
  int drvtexnuma;
  int drvtexnumb;

  gpuTYPE_t rtype = r.getType();

  // For some unknown reason I cannot pass directly a double to
  // GPU kernel. So I have to create a GPUtype and use a texture
  // for the incr.

  GPUtype *incrg;
  GPUtype *offsg;

  if (rtype==gpuCFLOAT) {
      drvfun = GPUman->getCuFunction(N_FILLVECTOR1C_KERNEL);
      drvtexa = GPUman->getCuTexref(N_TEXREF_F1_A);
      drvtexformata = CU_AD_FORMAT_FLOAT;
      drvtexnuma = 1;

      drvtexb = GPUman->getCuTexref(N_TEXREF_F1_B);
      drvtexformatb = CU_AD_FORMAT_FLOAT;
      drvtexnumb = 1;

      incrg = new GPUtype((float) incr,GPUman);
      mgc.setPtr(incrg);
      offsg = new GPUtype((float) offset,GPUman);
      mgc.setPtr(offsg);


  } else if (rtype==gpuFLOAT){
      drvfun = GPUman->getCuFunction(N_FILLVECTOR1F_KERNEL);

      drvtexa = GPUman->getCuTexref(N_TEXREF_F1_A);
      drvtexformata = CU_AD_FORMAT_FLOAT;
      drvtexnuma = 1;

      drvtexb = GPUman->getCuTexref(N_TEXREF_F1_B);
      drvtexformatb = CU_AD_FORMAT_FLOAT;
      drvtexnumb = 1;

      incrg = new GPUtype((float) incr,GPUman);
      mgc.setPtr(incrg);
      offsg = new GPUtype((float) offset,GPUman);
      mgc.setPtr(offsg);

  } else if (rtype==gpuCDOUBLE){
      drvfun = GPUman->getCuFunction(N_FILLVECTOR1CD_KERNEL);

      drvtexa = GPUman->getCuTexref(N_TEXREF_D1_A);
      drvtexformata = CU_AD_FORMAT_SIGNED_INT32;
      drvtexnuma = 2;

      drvtexb = GPUman->getCuTexref(N_TEXREF_D1_B);
      drvtexformatb = CU_AD_FORMAT_SIGNED_INT32;
      drvtexnumb = 2;

      incrg = new GPUtype(incr,GPUman);
      mgc.setPtr(incrg);
      offsg = new GPUtype(offset,GPUman);
      mgc.setPtr(offsg);


  } else if (rtype==gpuDOUBLE){
      drvfun = GPUman->getCuFunction(N_FILLVECTOR1D_KERNEL);

      drvtexa = GPUman->getCuTexref(N_TEXREF_D1_A);
      drvtexformata = CU_AD_FORMAT_SIGNED_INT32;
      drvtexnuma = 2;

      drvtexb = GPUman->getCuTexref(N_TEXREF_D1_B);
      drvtexformatb = CU_AD_FORMAT_SIGNED_INT32;
      drvtexnumb = 2;

      incrg = new GPUtype(incr,GPUman);
      mgc.setPtr(incrg);
      offsg = new GPUtype(offset,GPUman);
      mgc.setPtr(offsg);

  }



  // setup texture
  if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtexa, UINTPTR incrg->getGPUptr(), incrg->getNumel()
      * incrg->getMySize())) {
    throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
  }
  if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT, *drvtexa)) {
    throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
  }
  if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtexa, drvtexformata, drvtexnuma)) {
    throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
  }

  if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtexb, UINTPTR offsg->getGPUptr(), offsg->getNumel()
      * offsg->getMySize())) {
    throw GPUexception(GPUmatError, "Kernel execution error1.");
  }
  if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT, *drvtexb)) {
    throw GPUexception(GPUmatError, "Kernel execution error3.");
  }
  if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtexb, drvtexformatb, drvtexnumb)) {
    throw GPUexception(GPUmatError, "Kernel execution error2.");
  }

  // define kernel configuration
  gpukernelconfig_t *kconf = GPUman->getKernelConfig();
  hostdrv_pars_t pars[5];
  int nrhs = 5;

  int tmpm = m;
  int tmpp = p;
  int tmpoffsetp = offsetp;
  int tmptype = type;

  pars[0].par =  r.getGPUptrptr();
  pars[0].psize = sizeof(CUdeviceptr);
  pars[0].align = __alignof(CUdeviceptr);

  pars[1].par =  &tmpm;
  pars[1].psize = sizeof(tmpm);
  pars[1].align = __alignof(tmpm);

  pars[2].par =  &tmpp;
  pars[2].psize = sizeof(tmpp);
  pars[2].align = __alignof(tmpp);

  pars[3].par =  &tmpoffsetp;
  pars[3].psize = sizeof(tmpoffsetp);
  pars[3].align = __alignof(tmpoffsetp);

  pars[4].par =  &tmptype;
  pars[4].psize = sizeof(tmptype);
  pars[4].align = __alignof(tmptype);


  CUDALIBResult cudalibres = mat_HOSTDRV_A(n, kconf, nrhs, pars, drvfun);

  if (cudalibres != CUDALIBSuccess) {
    throw GPUexception(GPUmatError, "Kernel execution error.");
  }


  return status;

}


/*************************************************************************
 * GPUopAssign
 *************************************************************************/
GPUtype * GPUopAssign (GPUtype &p, GPUtype &q,  const Range &r, int dir, int ret, int fortranidx) {
  // My garbage collector
  MyGC mgc = MyGC();
  // garbage collector
  MyGCObj<GPUtype> mgc1;

  GPUmanager * GPUman = p.getGPUmanager();



  // I have to construct the following arrays to be passed to the GPU kernel
  // int nsiz. number dims. of siz
  // int * siz. Index array's size.
  // int * idx. Index array
  // int * nd. Dimensions array. Should correspond to the variable
  //           where the slice is applied


  // scalar is 1 if RHS is a scalar and LHS is not
  int scalar = 0;

  // create index array from range
  int n = 0;

  // nidx should be calculated browsing the Range
  int nidxtmp = 0;

  Range *tmp = (Range*) &r;
  while (tmp) {
    n++;
    if ((tmp->iindx==0)&&(tmp->gindx==0)&&(tmp->findx==0)&&(tmp->dindx==0)) {
      nidxtmp ++;
    } else {
      if (tmp->gindx==0) {
        nidxtmp +=tmp->sup+1;
      } else {
        GPUtype *gg = (GPUtype*) tmp->gindxptr;
        nidxtmp +=gg->getNumel();
      }
    }
    tmp = tmp->next;
  }

  int subsdim = n; // nsiz

  int ndq = q.getNdims();
  int ndp = p.getNdims();

  // do some checks
  // check input
  // 1. same type
  // 2. subsdim should not exceed dim
  gpuTYPE_t ptype = p.getType();
  gpuTYPE_t qtype = q.getType();

  if (ptype!=qtype) {
    throw GPUexception(GPUmatError,ERROR_SUBSASGN_SAMETYPE);
  }

  if (dir==0)
    if (subsdim > ndq)
      throw GPUexception(GPUmatError,ERROR_SUBSREF_INDEXMAXDIM);

  if (dir==1)
    if (subsdim > ndp)
      throw GPUexception(GPUmatError,ERROR_SUBSREF_INDEXMAXDIM);



  int nsiz = subsdim;
  //int nidx = subsdim;
  int nidx = nidxtmp;

  int nnd  = subsdim+1;
  int nncons = subsdim;




  // I have to construct the dimensions array
  // nd should have the first element always set to 1 (check GPU kernel
  // implementation). We allocate then subsdim + 1 elements
  //int *nd = (int*) malloc((subsdim+1)*sizeof(int)); // nd array
  int *nd = (int*) Mymalloc((nnd)*sizeof(int),&mgc); // nd array
  nd[0] = 1;

  //int *siz = (int*) malloc(subsdim*sizeof(int)); // siz array
  //int *idx = (int*) malloc(subsdim*sizeof(int)); // idx array

  int *siz = (int*) Mymalloc(nsiz*sizeof(int),&mgc); // siz array
  int *idx = (int*) Mymalloc(nidx*sizeof(int),&mgc); // idx array

  // ncons is used to store a flag that tells us if the
  // indexes are consecutive. If yes, the GPU kernel expects
  // that the index start and stride are specified, otherwise all
  // indexes are given to the kernel. Stride is specified in ncons
  int *ncons = (int*) Mymalloc(nncons*sizeof(int),&mgc);
  //int *ncons = (int*) malloc(subsdim*sizeof(int));



  // construct siz , nd
  tmp = &(Range &)r;
  int i = 0;

  // block, used in cudaMemcpy
  int block = 0; // elements in block
  int idxblock = 0; // index for the copy
  int ndcum = 1; // cumulative prod of nd

  /* A Range of type [inf:stride:sup] has always 1 element
   * in the idx[] vector, because we store the first element,
   * the stride and number of indexes
   * A range of type iindx or gindx is different because has
   * a certain number of explicit indexes
   */
  int idxoffs = 0; // offset in idx[] array

  while (tmp) {
    /* Range r can have different information
     * 1) a range of type inf:stride:sup
     * 2) The array of indexes, either GPU or CPU memory
     */

      // J:D:K  is the same as [J, J+D, ..., J+m*D] where m = fix((K-J)/D).
      // J:D:K  is empty if D == 0, if D > 0 and J > K, or if D < 0 and J < K.

      // In this procedure I have to use Matlab notation
      // index should be added 1
    if ((tmp->iindx==0)&&(tmp->gindx==0)&&(tmp->findx==0)&&(tmp->dindx==0)) {
      int J  = tmp->inf+1-fortranidx; // Matlab notation internal
      int D  = tmp->stride;
      int K  = tmp->sup+1-fortranidx;

      // if K = -1 then I have to calculate the end
      //if ((K-1)==RANGEEND) {
      if ((K-1+fortranidx)<0) {
        int tmpK = -1 - (K - 1 + fortranidx);
        K = (dir==0) ? q.getEnd(i, subsdim):p.getEnd(i, subsdim);
        K = K - tmpK + 1; // Matlab notation in this procedure
      }

      // if J = -1 then I have to calculate the end
      //if ((J-1)==RANGEEND) {
      if ((J-1+fortranidx)<0) {
        int tmpJ = -1 - (J - 1 + fortranidx);
        J = (dir==0) ? q.getEnd(i, subsdim):p.getEnd(i, subsdim);
        J = J - tmpJ + 1; // Matlab notation in this procedure
      }

      // m is the number of elements for this dimension
      nd[i+1] = 0;
      if (D == 0) { // 1 element only
        siz[i] = 1;
      } else if ((D > 0 && J > K) || (D < 0 && J < K)) {
        // empty
        throw GPUexception(GPUmatError, ERROR_SUBSREF_EMPTYRESULT);
      } else {


        double m = floor((double) (K - J) / D);
        if (m < 0)
          m = 0;
        siz[i] = (int)  (m + 1); // siz[i] is the number of indexes I have on idx[i]
      }

      // maxel in the maximum index I can have on i dimension
      int maxel = (dir==0) ? q.getEnd(i, subsdim):p.getEnd(i, subsdim);
      maxel++;

      nd[i+1] = maxel; // actual dimensions on i


      // when doing the following test we consider J and K in Matlab
      // format (+1)
      if (J > maxel) {
        throw GPUexception(GPUmatError,ERROR_SUBSREF_INDEXMAXDIM);
      }
      if (K > maxel) {
        throw GPUexception(GPUmatError,ERROR_SUBSREF_INDEXMAXDIM);
      }
      if (J < 1) {
        throw GPUexception(GPUmatError,ERROR_SUBSREF_INDEXMAXDIM);
      }
      if (K < 1) {
        throw GPUexception(GPUmatError,ERROR_SUBSREF_INDEXMAXDIM);
      }

      ncons[i] = D;
      //idx[i] = J - 1; // to the GPU we want the index 0 based
      idx[idxoffs] = J - 1; // to the GPU we want the index 0 based

      // update block
      // First dimension:
      // if D==1 and J==1 and K==maxel then I have consecutive indexes
      // If I don't have consecutive on first dimension then I cannot use
      // block copy

      if (i==0) {
        if ((D==1)||(D==0)) {
          //if ((J==1)&&(K==maxel)) {
            block = siz[i];
            idxblock = idxblock + ndcum*(J-1);
          //} else {
          //  block = 0;
          //}
        } else {
          block = 0;
        }
      } else if (i==1) {
        if (D==1) {
          if ((J==1)&&(K==maxel)) {
            // this is only valid is 0-dimension
            // is full index (0:end), which is the
            // same as block = ndcum (cumulative product of dimensions)
            if (block==ndcum)
              block *= siz[i];
            else
              block = 0;
          } else {
            block = 0;
          }
        } else if (D==0) {
          idxblock = idxblock + ndcum*(J-1);
        } else {
          block = 0;
        }

      } else {
        if (D==0) {
          idxblock = idxblock + ndcum*(J-1);
        } else {
          block = 0;
        }
      }

      ndcum *=maxel;
      idxoffs ++ ;
    } else {
      int J  = tmp->inf;
      int D  = tmp->stride;
      int K  = tmp->sup; // last index in iindx or gindx

      // manage GPUtyep index
      GPUtype *gg;
      gpuTYPE_t ggt;
      double *ggd = NULL;
      float *ggf = NULL;
      if (tmp->gindx!=0) {
        gg = (GPUtype*) tmp->gindxptr;
        if (gg->isComplex())
          throw GPUexception(GPUmatError,ERROR_SUBSREF_INDEXREAL);

        K = gg->getNumel() - 1; // K is the last element index
        ggt = gg->getType();
        if (ggt == gpuFLOAT) {
          ggf = (float*) Mymalloc((K+1)*sizeof(float),&mgc);
          // transfer to CPU, why?
          // I have to check for index out of range and currently this is done
          // on CPU
          GPUopCudaMemcpy(ggf, gg->getGPUptr(), sizeof(float) * (K+1),
                          cudaMemcpyDeviceToHost, gg->getGPUmanager());

        } else if (ggt == gpuDOUBLE) {
          ggd = (double*) Mymalloc((K+1)*sizeof(double),&mgc);
          GPUopCudaMemcpy(ggd, gg->getGPUptr(), sizeof(double) * (K+1),
                                 cudaMemcpyDeviceToHost, gg->getGPUmanager());
        }
      }

      // maxel in the maximum index I can have on i dimension
      int maxel = (dir==0) ? q.getEnd(i, subsdim):p.getEnd(i, subsdim);

      //int *tmpindx = tmp->iindx;
      // have to check indexes in iindx or gindx
      for (int jj=0;jj<=K;jj++) {
        int idxtmp;
        if (tmp->iindx!=0)
          idxtmp = tmp->iindx[jj];
        if (tmp->findx!=0)
          idxtmp = (int) tmp->findx[jj];
        if (tmp->dindx!=0)
          idxtmp = (int) tmp->dindx[jj];
        if (ggd!=0)
          idxtmp = (int) ggd[jj];
        if (ggf!=0)
          idxtmp = (int) ggf[jj];

        idxtmp = idxtmp-fortranidx;

        if (idxtmp > maxel) {
          throw GPUexception(GPUmatError,ERROR_SUBSREF_INDEXMAXDIM);
        }
        idx[idxoffs+jj] = idxtmp;
      }

      siz[i] = K+1;
      nd[i+1] = maxel+1; // actual dimensions on i
      ncons[i] = 0; // no consecutive indexes
      // update idx[vector]
      //memcpy(&idx[idxoffs], tmpindx, siz[i]*sizeof(int));

      ndcum *=(maxel+1);
      block = 0; // when using this type of index block is alwys 0
                 // meaning that I don't know if I can use block transfers.
      idxoffs += siz[i] ;

    }

    tmp = tmp->next;
    i++;
  }


  // create output if required by parameter ret
  GPUtype *res = &p;
  if (ret==1) {
    res = new GPUtype(q, 1); // need it the same type as q
    mgc1.setPtr(res);

    if (subsdim == 1) {
      int tmpsize[] = {1, siz[0]};
      res->setSize(2, tmpsize);
    } else {
      // remove any ones at the end
      int effsubsdim = subsdim;
      for (int uu = subsdim - 1; uu > 1; uu--) {
        if (siz[uu] == 1)
          effsubsdim--;
        else
          break;
      }
      res->setSize(effsubsdim, siz);
    }
    GPUopAllocVector(*res);
  }

  // check dimensions
  // In an assignment  A(:) = B, the number of elements in A and B
  // must be the same
  int lhstot = 1;
  for (int i = 0; i < subsdim; i++)
    lhstot *= siz[i];

  if (dir==0) {
    // this case A = B(1,1:end) index applied to q
    // Manage also scalars -> A = 1 or A = B(1,2,3)
    if (lhstot != res->getNumel()) {
      if (lhstot==1) {
        scalar = 1;
        // no block mode in scalar mode
        block = 0;
      } else {
        throw GPUexception(GPUmatError, ERROR_SUBSASGN_SAMEEL);
      }
    }

  } else {
    // this case B(1,1:end) = A index applied to res
    // Manage also scalars -> A(1:end) = 1
    if (lhstot != q.getNumel()) {
      if (q.getNumel()==1) {
        scalar = 1;
        // no block mode in scalar mode
        block = 0;
      } else {
        throw GPUexception(GPUmatError, ERROR_SUBSASGN_SAMEEL);
      }
    }
  }

  // count non-singleton dimensions on left and right hand side
  int lhsns = 0;
  int rhsns = 0;
  int *effsize;
  int ndeff;

  if (dir==0) {
    effsize = res->getSize();
    ndeff = res->getNdims();
  } else {
    effsize = q.getSize();
    ndeff = q.getNdims();
  }

  for (int i = 0; i < subsdim; i++) {
    if (siz[i] > 1)
      lhsns += 1;
  }

  for (int i = 0; i < ndeff; i++) {
    if (effsize[i] > 1)
      rhsns += 1;
  }

  if (lhsns != rhsns) {
    if (scalar==1) {

    } else {
      throw GPUexception(GPUmatError, ERROR_SUBSASGN_NONSING);
    }
  }


  // do not perform the following test for scalar operations
  if (scalar==0) {
    int jstart = 0;
    for (int i = 0; i < subsdim; i++) {
      // search the first non-scalar dim in q and compare
      if (siz[i] > 1) {
        for (int j = jstart; j < ndeff; j++) {
          if (effsize[j] > 1) {
            // compare with nidx[i]
            if (siz[i] != effsize[j]) {
              throw GPUexception(GPUmatError,ERROR_SUBSASGN_DIMMIS);
            } else {
              jstart = j + 1;
              break;
            }
          }
        }
      }
    }
  }



  // create GPU parameters to pass to the kernel
  // should store
  // int   * siz,
  // int   * idx,
  // int   * nd,
  // int   * ncons,


  //if (1) {
  if (block==0) {
#ifdef DEBUG
    mexPrintf("NON BLOCK ASSIGN\n");
#endif
    int npars = nsiz + nidx + nnd + nncons;

    GPUtype GPUpars = GPUtype(*res, 1);
    GPUpars.setType(gpuFLOAT);
    int tmpsize[2];
    tmpsize[0] = 1;
    tmpsize[1] = npars;
    GPUpars.setSize(2, tmpsize);
    GPUopAllocVector(GPUpars);

    int *kpars = (int*) Mymalloc(npars*sizeof(int),&mgc);
    memcpy(kpars, siz, nsiz*sizeof(int));
    memcpy(kpars+nsiz, idx, nidx*sizeof(int));
    memcpy(kpars+(nsiz+nidx), nd, nnd*sizeof(int));
    memcpy(kpars+(nsiz+nidx+nnd), ncons, nncons*sizeof(int));

    // copy parameters to GPUpars
    GPUopCudaMemcpy(GPUpars.getGPUptr(), (void*) kpars,
                  GPUpars.getMySize() *npars , cudaMemcpyHostToDevice,
                  GPUman);


    CUfunction *drvfun;
    CUtexref *drvtexa;
    CUtexref *drvtexb;
    CUarray_format_enum drvtexformata;
    CUarray_format_enum drvtexformatb;
    int drvtexnuma;
    int drvtexnumb;

    gpuTYPE_t ptype = res->getType();

    if (ptype==gpuCFLOAT) {
      drvfun = GPUman->getCuFunction(N_SUBSINDEX1C_KERNEL);
      drvtexa = GPUman->getCuTexref(N_TEXREF_C1_A);
      drvtexformata = CU_AD_FORMAT_FLOAT;
      drvtexnuma = 2;
    } else if (ptype==gpuFLOAT){
      drvfun = GPUman->getCuFunction(N_SUBSINDEX1F_KERNEL);
      drvtexa = GPUman->getCuTexref(N_TEXREF_F1_A);
      drvtexformata = CU_AD_FORMAT_FLOAT;
      drvtexnuma = 1;
    } else if (ptype==gpuCDOUBLE){
      drvfun = GPUman->getCuFunction(N_SUBSINDEX1CD_KERNEL);
      drvtexa = GPUman->getCuTexref(N_TEXREF_CD1_A);
      drvtexformata = CU_AD_FORMAT_SIGNED_INT32;
      drvtexnuma = 4;
    } else if (ptype==gpuDOUBLE){
      drvfun = GPUman->getCuFunction(N_SUBSINDEX1D_KERNEL);
      drvtexa = GPUman->getCuTexref(N_TEXREF_D1_A);
      drvtexformata = CU_AD_FORMAT_SIGNED_INT32;
      drvtexnuma = 2;
    }

    drvtexb = GPUman->getCuTexref(N_TEXREF_I1_A);
    drvtexformatb = CU_AD_FORMAT_SIGNED_INT32;
    drvtexnumb = 1;

    /*__global__ void  SUBSINDEX1F_KERNEL(unsigned int n, int offset,
              float *odata, int nsiz, int nidx, int nnd, int nncons, int dir) {*/
    // define kernel configuration
    gpukernelconfig_t *kconf = GPUman->getKernelConfig();
    hostdrv_pars_t pars[8];
    int nrhs = 8;

    pars[0].par =  res->getGPUptrptr();
    pars[0].psize = sizeof(CUdeviceptr);
    pars[0].align = __alignof(CUdeviceptr);


    pars[1].par =  q.getGPUptrptr();
    pars[1].psize = sizeof(CUdeviceptr);
    pars[1].align = __alignof(CUdeviceptr);

    pars[2].par = &nsiz;
    pars[2].psize = sizeof(nsiz);
    pars[2].align = __alignof(nsiz);


    pars[3].par = &nidx;
    pars[3].psize = sizeof(nidx);
    pars[3].align = __alignof(nidx);

    pars[4].par = &nnd;
    pars[4].psize = sizeof(nnd);
    pars[4].align = __alignof(nnd);

    pars[5].par = &nncons;
    pars[5].psize = sizeof(nncons);
    pars[5].align = __alignof(nncons);
    

    pars[6].par = &dir;
    pars[6].psize = sizeof(dir);
    pars[6].align = __alignof(dir);
   
    pars[7].par = &scalar;
    pars[7].psize = sizeof(scalar);
    pars[7].align = __alignof(scalar);
   

    // setup texture
    if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtexa, UINTPTR q.getGPUptr(), q.getNumel()
        * q.getMySize())) {
      throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
    }
    if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT, *drvtexa)) {
      throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
    }
    if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtexa, drvtexformata, drvtexnuma)) {
      throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
    }

    if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtexb, UINTPTR GPUpars.getGPUptr(), GPUpars.getNumel()
        * GPUpars.getMySize())) {
      throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
    }
    if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT, *drvtexb)) {
      throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
    }
    if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtexb, drvtexformatb, drvtexnumb)) {
      throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
    }

    int N;
    if (dir == 0) {
      N = res->getNumel();
    } else {
      if (scalar==0) {
        N = q.getNumel();
      } else {
        N = 1;
        for (int i=0;i<nsiz;i++)
          N = N*siz[i];
      }
    }

    CUDALIBResult cudalibres = mat_HOSTDRV_A(N, kconf, nrhs, pars, drvfun);

    /*if (CUDA_SUCCESS != cuCtxSynchronize()) {
      throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
    }*/

    if (cudalibres != CUDALIBSuccess) {
      throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
    }
  } else {
    // internal check
    // the block I am copying cannot be bigger than destination
    if (block>res->getNumel())
      throw GPUexception(GPUmatError, "GPUopAssign internal error. Please report to gp-you@gp-you.org.");

    //mexPrintf("Mem copy\n");
    if (dir == 0) {
      if (idxblock>=q.getNumel())
         throw GPUexception(GPUmatError, "GPUopAssign internal error. Please report to gp-you@gp-you.org.");

      void *src = (void*) ((UINTPTR q.getGPUptr())+idxblock*q.getMySize());
      GPUopCudaMemcpy(res->getGPUptr(), src,
                      q.getMySize() *block , cudaMemcpyDeviceToDevice,
                      GPUman);
    } else {
      if (idxblock>=res->getNumel())
          throw GPUexception(GPUmatError, "GPUopAssign internal error. Please report to gp-you@gp-you.org.");

      void * dst = (void*) ((UINTPTR res->getGPUptr())+idxblock*res->getMySize());
      GPUopCudaMemcpy(dst, q.getGPUptr(),
                          res->getMySize() * block , cudaMemcpyDeviceToDevice,
                          GPUman);
    }


  }

  if (ret==1) {
    mgc1.remPtr(res);
    return res;
  } else {
    return NULL;
  }



}

/*************************************************************************
 * GPUopPermute
 *************************************************************************/
GPUtype * GPUopPermute (GPUtype &p, GPUtype &q,  const Range &r, int dir, int ret, int fortranidx, int*perm) {
  // My garbage collector
  MyGC mgc = MyGC();
  // garbage collector
  MyGCObj<GPUtype> mgc1;

  GPUmanager * GPUman = p.getGPUmanager();


  // I have to construct the following arrays to be passed to the GPU kernel
  // int nsiz. number dims. of siz
  // int * siz. Index array's size.
  // int * idx. Index array
  // int * nd. Dimensions array. Should correspond to the variable
  //           where the slice is applied


  // scalar is 1 if RHS is a scalar and LHS is not
  int scalar = 0;

  // create index array from range
  int n = 0;

  // nidx should be calculated browsing the Range
  int nidxtmp = 0;

  Range *tmp = (Range*) &r;
  while (tmp) {
    n++;
    if ((tmp->iindx==0)&&(tmp->gindx==0)&&(tmp->findx==0)&&(tmp->dindx==0)) {
      nidxtmp ++;
    } else {
      if (tmp->gindx==0) {
        nidxtmp +=tmp->sup+1;
      } else {
        GPUtype *gg = (GPUtype*) tmp->gindxptr;
        nidxtmp +=gg->getNumel();
      }
    }
    tmp = tmp->next;
  }

  int subsdim = n; // nsiz

  int ndq = q.getNdims();
  int ndp = p.getNdims();

  // do some checks
  // check input
  // 1. same type
  // 2. subsdim should not exceed dim
  gpuTYPE_t ptype = p.getType();
  gpuTYPE_t qtype = q.getType();

  if (ptype!=qtype) {
    throw GPUexception(GPUmatError,ERROR_SUBSASGN_SAMETYPE);
  }

  if (dir==0)
    if (subsdim > ndq)
      throw GPUexception(GPUmatError,ERROR_SUBSREF_INDEXMAXDIM);

  if (dir==1)
    if (subsdim > ndp)
      throw GPUexception(GPUmatError,ERROR_SUBSREF_INDEXMAXDIM);



  int nsiz = subsdim;
  //int nidx = subsdim;
  int nidx = nidxtmp;

  int nnd  = subsdim+1;
  int nncons = subsdim;




  // I have to construct the dimensions array
  // nd should have the first element always set to 1 (check GPU kernel
  // implementation). We allocate then subsdim + 1 elements
  //int *nd = (int*) malloc((subsdim+1)*sizeof(int)); // nd array
  int *nd = (int*) Mymalloc((nnd)*sizeof(int),&mgc); // nd array
  nd[0] = 1;

  //int *siz = (int*) malloc(subsdim*sizeof(int)); // siz array
  //int *idx = (int*) malloc(subsdim*sizeof(int)); // idx array

  int *siz = (int*) Mymalloc(nsiz*sizeof(int),&mgc); // siz array
  int *idx = (int*) Mymalloc(nidx*sizeof(int),&mgc); // idx array

  // ncons is used to store a flag that tells us if the
  // indexes are consecutive. If yes, the GPU kernel expects
  // that the index start and stride are specified, otherwise all
  // indexes are given to the kernel. Stride is specified in ncons
  int *ncons = (int*) Mymalloc(nncons*sizeof(int),&mgc);
  //int *ncons = (int*) malloc(subsdim*sizeof(int));



  // construct siz , nd
  tmp = &(Range &)r;
  int i = 0;

  // block, used in cudaMemcpy
  int block = 0; // elements in block
  int idxblock = 0; // index for the copy
  int ndcum = 1; // cumulative prod of nd

  /* A Range of type [inf:stride:sup] has always 1 element
   * in the idx[] vector, because we store the first element,
   * the stride and number of indexes
   * A range of type iindx or gindx is different because has
   * a certain number of explicit indexes
   */
  int idxoffs = 0; // offset in idx[] array

  while (tmp) {
    /* Range r can have different information
     * 1) a range of type inf:stride:sup
     * 2) The array of indexes, either GPU or CPU memory
     */

      // J:D:K  is the same as [J, J+D, ..., J+m*D] where m = fix((K-J)/D).
      // J:D:K  is empty if D == 0, if D > 0 and J > K, or if D < 0 and J < K.

      // In this procedure I have to use Matlab notation
      // index should be added 1
    if ((tmp->iindx==0)&&(tmp->gindx==0)&&(tmp->findx==0)&&(tmp->dindx==0)) {
      int J  = tmp->inf+1-fortranidx; // Matlab notation internal
      int D  = tmp->stride;
      int K  = tmp->sup+1-fortranidx;

      // if K = -1 then I have to calculate the end
      //if ((K-1)==RANGEEND) {
      if ((K-1)<0) {
        int tmpK = -1 - (K - 1 + fortranidx);
        K = (dir==0) ? q.getEnd(i, subsdim):p.getEnd(i, subsdim);
        K = K - tmpK + 1; // Matlab notation in this procedure
      }

      // if J = -1 then I have to calculate the end
      //if ((J-1)==RANGEEND) {
      if ((J-1)<0) {
        int tmpJ = -1 - (J - 1 + fortranidx);
        J = (dir==0) ? q.getEnd(i, subsdim):p.getEnd(i, subsdim);
        J = J - tmpJ + 1; // Matlab notation in this procedure
      }

      // m is the number of elements for this dimension
      nd[i+1] = 0;
      if (D == 0) { // 1 element only
        siz[i] = 1;
      } else if ((D > 0 && J > K) || (D < 0 && J < K)) {
        // empty
        throw GPUexception(GPUmatError, ERROR_SUBSREF_EMPTYRESULT);
      } else {


        double m = floor((double) (K - J) / D);
        if (m < 0)
          m = 0;
        siz[i] = (int)  (m + 1); // siz[i] is the number of indexes I have on idx[i]
      }

      // maxel in the maximum index I can have on i dimension
      int maxel = (dir==0) ? q.getEnd(i, subsdim):p.getEnd(i, subsdim);
      maxel++;

      nd[i+1] = maxel; // actual dimensions on i


      // when doing the following test we consider J and K in Matlab
      // format (+1)
      if (J > maxel) {
        throw GPUexception(GPUmatError,ERROR_SUBSREF_INDEXMAXDIM);
      }
      if (K > maxel) {
        throw GPUexception(GPUmatError,ERROR_SUBSREF_INDEXMAXDIM);
      }
      if (J < 1) {
        throw GPUexception(GPUmatError,ERROR_SUBSREF_INDEXMAXDIM);
      }
      if (K < 1) {
        throw GPUexception(GPUmatError,ERROR_SUBSREF_INDEXMAXDIM);
      }

      ncons[i] = D;
      //idx[i] = J - 1; // to the GPU we want the index 0 based
      idx[idxoffs] = J - 1; // to the GPU we want the index 0 based

      // update block
      // First dimension:
      // if D==1 and J==1 and K==maxel then I have consecutive indexes
      // If I don't have consecutive on first dimension then I cannot use
      // block copy

      if (i==0) {
        if ((D==1)||(D==0)) {
          //if ((J==1)&&(K==maxel)) {
            block = siz[i];
            idxblock = idxblock + ndcum*(J-1);
          //} else {
          //  block = 0;
          //}
        } else {
          block = 0;
        }
      } else if (i==1) {
        if (D==1) {
          if ((J==1)&&(K==maxel)) {
            // this is only valid is 0-dimension
            // is full index (0:end), which is the
            // same as block = ndcum (cumulative product of dimensions)
            if (block==ndcum)
              block *= siz[i];
            else
              block = 0;
          } else {
            block = 0;
          }
        } else if (D==0) {
          idxblock = idxblock + ndcum*(J-1);
        } else {
          block = 0;
        }

      } else {
        if (D==0) {
          idxblock = idxblock + ndcum*(J-1);
        } else {
          block = 0;
        }
      }

      ndcum *=maxel;
      idxoffs ++ ;
    } else {
      int J  = tmp->inf;
      int D  = tmp->stride;
      int K  = tmp->sup; // last index in iindx or gindx

      // manage GPUtyep index
      GPUtype *gg;
      gpuTYPE_t ggt;
      double *ggd = NULL;
      float *ggf = NULL;
      if (tmp->gindx!=0) {
        gg = (GPUtype*) tmp->gindxptr;
        if (gg->isComplex())
          throw GPUexception(GPUmatError,ERROR_SUBSREF_INDEXREAL);

        K = gg->getNumel() - 1; // K is the last element index
        ggt = gg->getType();
        if (ggt == gpuFLOAT) {
          ggf = (float*) Mymalloc((K+1)*sizeof(float),&mgc);
          // transfer to CPU, why?
          // I have to check for index out of range and currently this is done
          // on CPU
          GPUopCudaMemcpy(ggf, gg->getGPUptr(), sizeof(float) * (K+1),
                          cudaMemcpyDeviceToHost, gg->getGPUmanager());

        } else if (ggt == gpuDOUBLE) {
          ggd = (double*) Mymalloc((K+1)*sizeof(double),&mgc);
          GPUopCudaMemcpy(ggd, gg->getGPUptr(), sizeof(double) * (K+1),
                                 cudaMemcpyDeviceToHost, gg->getGPUmanager());
        }
      }

      // maxel in the maximum index I can have on i dimension
      int maxel = (dir==0) ? q.getEnd(i, subsdim):p.getEnd(i, subsdim);

      //int *tmpindx = tmp->iindx;
      // have to check indexes in iindx or gindx
      for (int jj=0;jj<=K;jj++) {
        int idxtmp;
        if (tmp->iindx!=0)
          idxtmp = tmp->iindx[jj];
        if (tmp->findx!=0)
          idxtmp = (int) tmp->findx[jj];
        if (tmp->dindx!=0)
          idxtmp = (int) tmp->dindx[jj];
        if (ggd!=0)
          idxtmp = (int) ggd[jj];
        if (ggf!=0)
          idxtmp = (int) ggf[jj];

        idxtmp = idxtmp-fortranidx;

        if (idxtmp > maxel) {
          throw GPUexception(GPUmatError,ERROR_SUBSREF_INDEXMAXDIM);
        }
        idx[idxoffs+jj] = idxtmp;
      }

      siz[i] = K+1;
      nd[i+1] = maxel+1; // actual dimensions on i
      ncons[i] = 0; // no consecutive indexes
      // update idx[vector]
      //memcpy(&idx[idxoffs], tmpindx, siz[i]*sizeof(int));

      ndcum *=(maxel+1);
      block = 0; // when using this type of index block is alwys 0
                 // meaning that I don't know if I can use block transfers.
      idxoffs += siz[i] ;

    }

    tmp = tmp->next;
    i++;
  }


  // create output if required by parameter ret
  GPUtype *res = &p;
  if (ret==1) {
    res = new GPUtype(q, 1); // need it the same type as q
    mgc1.setPtr(res);

    if (subsdim == 1) {
      int tmpsize[] = {1, siz[0]};
      res->setSize(2, tmpsize);
    } else {
      // remove any ones at the end
      int effsubsdim = subsdim;
      for (int uu = subsdim - 1; uu > 1; uu--) {
        if (siz[uu] == 1)
          effsubsdim--;
        else
          break;
      }
      res->setSize(effsubsdim, siz);
    }
    GPUopAllocVector(*res);
  }

  // check dimensions
  // In an assignment  A(:) = B, the number of elements in A and B
  // must be the same
  int lhstot = 1;
  for (int i = 0; i < subsdim; i++)
    lhstot *= siz[i];

  if (dir==0) {
    // this case A = B(1,1:end) index applied to q
    // Manage also scalars -> A = 1 or A = B(1,2,3)
    if (lhstot != res->getNumel()) {
      if (lhstot==1) {
        scalar = 1;
        // no block mode in scalar mode
        block = 0;
      } else {
        throw GPUexception(GPUmatError, ERROR_SUBSASGN_SAMEEL);
      }
    }

  } else {
    // this case B(1,1:end) = A index applied to res
    // Manage also scalars -> A(1:end) = 1
    if (lhstot != q.getNumel()) {
      if (q.getNumel()==1) {
        scalar = 1;
        // no block mode in scalar mode
        block = 0;
      } else {
        throw GPUexception(GPUmatError, ERROR_SUBSASGN_SAMEEL);
      }
    }
  }

  // count non-singleton dimensions on left and right hand side
  int lhsns = 0;
  int rhsns = 0;
  int *effsize;
  int ndeff;

  if (dir==0) {
    effsize = res->getSize();
    ndeff = res->getNdims();
  } else {
    effsize = q.getSize();
    ndeff = q.getNdims();
  }

  for (int i = 0; i < subsdim; i++) {
    if (siz[i] > 1)
      lhsns += 1;
  }

  for (int i = 0; i < ndeff; i++) {
    if (effsize[i] > 1)
      rhsns += 1;
  }

  if (lhsns != rhsns) {
    if (scalar==1) {

    } else {
      throw GPUexception(GPUmatError, ERROR_SUBSASGN_NONSING);
    }
  }


  // check permutation vector
  // permtmp is used to understand in the permutation vector is consistent
  int *permtmp = (int*) Mymalloc(nsiz*sizeof(int),&mgc); //
  for (int i=0;i<nsiz;i++)
    permtmp[i] = 0;

  // if the permutation is not set, then create one
  if (perm==NULL) {
    perm = (int*) Mymalloc(nsiz*sizeof(int),&mgc);
    for (int i=0;i<nsiz;i++)
      perm[i] = i;
  }
  for (int i=0;i<nsiz;i++) {
    if (perm[i]<0)
      throw GPUexception(GPUmatError, ERROR_PERMUTE_INVALIDPERM);
    if (perm[i]>=nsiz)
      throw GPUexception(GPUmatError, ERROR_PERMUTE_INVALIDPERM);
    permtmp[perm[i]]++;
  }

  // at the end of the above loop, permtmp should have all entries to 1
  for (int i=0;i<nsiz;i++) {
    if (permtmp[i]!=1)
      throw GPUexception(GPUmatError, ERROR_PERMUTE_INVALIDORDER);
  }
  // update with fortran index
  for (int i=0;i<nsiz;i++) {
      perm[i] = perm[i] - fortranidx;
  }



  // do not perform the following test for scalar operations
  if (scalar==0) {
    int jstart = 0;
    for (int i = 0; i < subsdim; i++) {
      // search the first non-scalar dim in q and compare
      if (siz[perm[i]] > 1) {
        for (int j = jstart; j < ndeff; j++) {
          if (effsize[j] > 1) {
            // compare with nidx[i]
            if (siz[perm[i]] != effsize[j]) {
              throw GPUexception(GPUmatError,ERROR_SUBSASGN_DIMMIS);
            } else {
              jstart = j + 1;
              break;
            }
          }
        }
      }
    }
  }



  // create GPU parameters to pass to the kernel
  // should store
  // int   * siz,
  // int   * idx,
  // int   * nd,
  // int   * ncons,

  block = 0;
  //if (1) {
  if (block==0) {
#ifdef DEBUG
    mexPrintf("NON BLOCK ASSIGN\n");
#endif
    int npars = nsiz + nsiz + nidx + nnd + nncons;

    GPUtype GPUpars = GPUtype(*res, 1);
    GPUpars.setType(gpuFLOAT);
    int tmpsize[2];
    tmpsize[0] = 1;
    tmpsize[1] = npars;
    GPUpars.setSize(2, tmpsize);
    GPUopAllocVector(GPUpars);

    int *kpars = (int*) Mymalloc(npars*sizeof(int),&mgc);
    memcpy(kpars, siz,  nsiz*sizeof(int));
    memcpy(kpars+nsiz, perm, nsiz*sizeof(int));
    memcpy(kpars+nsiz+nsiz, idx, nidx*sizeof(int));
    memcpy(kpars+(nsiz+nsiz+nidx), nd, nnd*sizeof(int));
    memcpy(kpars+(nsiz+nsiz+nidx+nnd), ncons, nncons*sizeof(int));

    // copy parameters to GPUpars
    GPUopCudaMemcpy(GPUpars.getGPUptr(), (void*) kpars,
                  GPUpars.getMySize() *npars , cudaMemcpyHostToDevice,
                  GPUman);


    CUfunction *drvfun;
    CUtexref *drvtexa;
    CUtexref *drvtexb;
    CUarray_format_enum drvtexformata;
    CUarray_format_enum drvtexformatb;
    int drvtexnuma;
    int drvtexnumb;

    gpuTYPE_t ptype = res->getType();

    if (ptype==gpuCFLOAT) {
      drvfun = GPUman->getCuFunction(N_PERMSUBSINDEX1C_KERNEL);
      drvtexa = GPUman->getCuTexref(N_TEXREF_C1_A);
      drvtexformata = CU_AD_FORMAT_FLOAT;
      drvtexnuma = 2;
    } else if (ptype==gpuFLOAT){
      drvfun = GPUman->getCuFunction(N_PERMSUBSINDEX1F_KERNEL);
      drvtexa = GPUman->getCuTexref(N_TEXREF_F1_A);
      drvtexformata = CU_AD_FORMAT_FLOAT;
      drvtexnuma = 1;
    } else if (ptype==gpuCDOUBLE){
      drvfun = GPUman->getCuFunction(N_PERMSUBSINDEX1CD_KERNEL);
      drvtexa = GPUman->getCuTexref(N_TEXREF_CD1_A);
      drvtexformata = CU_AD_FORMAT_SIGNED_INT32;
      drvtexnuma = 4;
    } else if (ptype==gpuDOUBLE){
      drvfun = GPUman->getCuFunction(N_PERMSUBSINDEX1D_KERNEL);
      drvtexa = GPUman->getCuTexref(N_TEXREF_D1_A);
      drvtexformata = CU_AD_FORMAT_SIGNED_INT32;
      drvtexnuma = 2;
    }

    drvtexb = GPUman->getCuTexref(N_TEXREF_I1_A);
    drvtexformatb = CU_AD_FORMAT_SIGNED_INT32;
    drvtexnumb = 1;

    /*__global__ void  SUBSINDEX1F_KERNEL(unsigned int n, int offset,
              float *odata, int nsiz, int nidx, int nnd, int nncons, int dir) {*/
    // define kernel configuration
    gpukernelconfig_t *kconf = GPUman->getKernelConfig();
    hostdrv_pars_t pars[8];
    int nrhs = 8;

    pars[0].par =  res->getGPUptrptr();
    pars[0].psize = sizeof(CUdeviceptr);
    pars[0].align = __alignof(CUdeviceptr);


    pars[1].par =  q.getGPUptrptr();
    pars[1].psize = sizeof(CUdeviceptr);
    pars[1].align = __alignof(CUdeviceptr);


    pars[2].par = &nsiz;
    pars[2].psize = sizeof(nsiz);
    pars[2].align = __alignof(nsiz);

    pars[3].par = &nidx;
    pars[3].psize = sizeof(nidx);
    pars[3].align = __alignof(nidx);


    pars[4].par = &nnd;
    pars[4].psize = sizeof(nnd);
    pars[4].align = __alignof(nnd);

    pars[5].par = &nncons;
    pars[5].psize = sizeof(nncons);
    pars[5].align = __alignof(nncons);

    pars[6].par = &dir;
    pars[6].psize = sizeof(dir);
    pars[6].align = __alignof(dir);

    pars[7].par = &scalar;
    pars[7].psize = sizeof(scalar);
    pars[7].align = __alignof(scalar);


    // setup texture
    if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtexa, UINTPTR q.getGPUptr(), q.getNumel()
        * q.getMySize())) {
      throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
    }
    if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT, *drvtexa)) {
      throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
    }
    if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtexa, drvtexformata, drvtexnuma)) {
      throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
    }

    if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtexb, UINTPTR GPUpars.getGPUptr(), GPUpars.getNumel()
        * GPUpars.getMySize())) {
      throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
    }
    if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT, *drvtexb)) {
      throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
    }
    if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtexb, drvtexformatb, drvtexnumb)) {
      throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
    }

    int N;
    if (dir == 0) {
      N = res->getNumel();
    } else {
      if (scalar==0) {
        N = q.getNumel();
      } else {
        N = 1;
        for (int i=0;i<nsiz;i++)
          N = N*siz[i];
      }
    }

    CUDALIBResult cudalibres = mat_HOSTDRV_A(N, kconf, nrhs, pars, drvfun);

    /*if (CUDA_SUCCESS != cuCtxSynchronize()) {
      throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
    }*/

    if (cudalibres != CUDALIBSuccess) {
      throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
    }
  } else {
    // internal check
    // the block I am copying cannot be bigger than destination
    if (block>res->getNumel())
      throw GPUexception(GPUmatError, "GPUopAssign internal error. Please report to gp-you@gp-you.org.");

    //mexPrintf("Mem copy\n");
    if (dir == 0) {
      if (idxblock>=q.getNumel())
         throw GPUexception(GPUmatError, "GPUopAssign internal error. Please report to gp-you@gp-you.org.");

      void *src = (void*) ((UINTPTR q.getGPUptr())+idxblock*q.getMySize());
      GPUopCudaMemcpy(res->getGPUptr(), src,
                      q.getMySize() *block , cudaMemcpyDeviceToDevice,
                      GPUman);
    } else {
      if (idxblock>=res->getNumel())
          throw GPUexception(GPUmatError, "GPUopAssign internal error. Please report to gp-you@gp-you.org.");

      void * dst = (void*) ((UINTPTR res->getGPUptr())+idxblock*res->getMySize());
      GPUopCudaMemcpy(dst, q.getGPUptr(),
                          res->getMySize() * block , cudaMemcpyDeviceToDevice,
                          GPUman);
    }


  }

  if (ret==1) {
    mgc1.remPtr(res);
    return res;
  } else {
    return NULL;
  }



}
/*************************************************************************
 * GPUopSubsindexDrv
 *************************************************************************/
#define MAXDIM 5
#define FIRSTPARS 15

GPUtype * GPUopSubsindexDrv(GPUtype &p, int subsdim, int *range) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUmanager *GPUman = p.getGPUmanager();

  // My garbage collector
  MyGC mgc = MyGC();
  // garbage collector
  MyGCObj<GPUtype> mgc1;

  GPUtype *r; // output


  if (subsdim > MAXDIM)
    throw GPUexception(GPUmatError, ERROR_SUBSREF_MAXIND);

  if (subsdim > p.getNdims())
    throw GPUexception(GPUmatError, ERROR_SUBSREF_INDEXC);

  // nd stores the dimension of current GPUtype
  int nd[] = { 1, 1, 1, 1, 1 };

  // start stores the starting index for each dimension
  int start[] = { 1, 1, 1, 1, 1 };

  // nidx stores the dimension of each index
  // For example
  // A(1:2,1)
  // nidx[0] = 2;
  // nidx[1] = 1;
  int nidx[] = { 1, 1, 1, 1, 1 };

  // ncons used as a flag for consecutive indexes
  int ncons[] = { 0, 0, 0, 0, 0 };

  // update nd with actual dimensions
  int *psize = p.getSize();
  for (int i = 0; i < p.getNdims(); i++) {
    nd[i] = psize[i];
  }


  for (int i = 0; i < subsdim; i++) {
    // J:D:K  is the same as [J, J+D, ..., J+m*D] where m = fix((K-J)/D).
    // J:D:K  is empty if D == 0, if D > 0 and J > K, or if D < 0 and J < K.

    int index = i*3;
    int J  = range[index+0]+1; // Matlab notation internal
    int D  = range[index+1];
    int K  = range[index+2]+1;

    // m is the number of elements for this dimension
    nidx[i] = 0;
    if (D == 0) {
      nidx[i] = 1;
    } else if	((D > 0 && J > K) || (D < 0 && J < K)) {
      // empty
      throw GPUexception(GPUmatError, ERROR_SUBSREF_EMPTYRESULT);
    } else {
      double m = floor((double) (K - J) / D);
      if (m < 0)
        m = 0;
      nidx[i] = (int)  (m + 1);
    }

    // maxel in the maximum index I can have on i dimension
    int maxel = p.getEnd(i, subsdim)+1;

    if (J > maxel) {
      throw GPUexception(GPUmatError, ERROR_SUBSREF_INDEXMAXDIM);
    }
    if (K > maxel) {
      throw GPUexception(GPUmatError, ERROR_SUBSREF_INDEXMAXDIM);
    }
    if (J < 1) {
      throw GPUexception(GPUmatError, ERROR_SUBSREF_SUBSINTEGER);
    }
    if (K < 1) {
      throw GPUexception(GPUmatError, ERROR_SUBSREF_SUBSINTEGER);
    }

    ncons[i] = D;
    start[i] = J;
  }


  // nidxtot is total number of indexes I have. If the indexes
  // are consecutive I store only the first index, so I add nidxtot 1.
  // Otherwise I have to add the number of indexes.

  int nidxtot = MAXDIM;

  // now I have to allocate the pars that should be passed to the host function
  // The vector has the following structure
  // first 10 elements to store nd and nidx
  // nidxtot more elements to store idx0 to idx4


  int pars[20];

  pars[0] = nd[0];
  pars[1] = nd[1];
  pars[2] = nd[2];
  pars[3] = nd[3];
  pars[4] = nd[4];

  pars[5] = nidx[0];
  pars[6] = nidx[1];
  pars[7] = nidx[2];
  pars[8] = nidx[3];
  pars[9] = nidx[4];

  pars[10] = ncons[0];
  pars[11] = ncons[1];
  pars[12] = ncons[2];
  pars[13] = ncons[3];
  pars[14] = ncons[4];

  pars[15] = start[0];
  pars[16] = start[1];
  pars[17] = start[2];
  pars[18] = start[3];
  pars[19] = start[4];


  // create GPU pars
  GPUtype GPUpars = GPUtype(p, 1);
  GPUpars.setType(gpuFLOAT);
  int tmpsize[2];
  tmpsize[0] = 1;
  tmpsize[1] = nidxtot + FIRSTPARS;
  GPUpars.setSize(2, tmpsize);
  GPUopAllocVector(GPUpars);

  // create output
  // output has different size respect to p but
  // same type of elements
  r = new GPUtype(p, 1);
  mgc1.setPtr(r);

  // subsdim is the number of indexes passed to this function
  // subsdim can be less than 5
  if (subsdim == 1) {
    //
    int rsize[5];
    rsize[0] = 1;
    rsize[1] = nidx[0];
    r->setSize(2, rsize);

  }  else {
    int rsize[5];
    rsize[0] = nidx[0];
    rsize[1] = nidx[1];
    rsize[2] = nidx[2];
    rsize[3] = nidx[3];
    rsize[4] = nidx[4];
    r->setSize(subsdim, rsize);
  }

  GPUopAllocVector(*r);

  GPUopCudaMemcpy(GPUpars.getGPUptr(), (void*) pars,
            GPUpars.getMySize() * (MAXDIM+FIRSTPARS), cudaMemcpyHostToDevice,
            GPUman);

  GPUopSubsindex(p, -1, GPUpars, *r);

  // remove 1's at the end
  int ndims = r->getNdims();
  int finalndims = ndims;
  int *tmpdims = r->getSize();
  int *tmprsize = (int*) Mymalloc(ndims * sizeof(int), &mgc);

  for (int uu = 0; uu < ndims; uu++) {
    tmprsize[uu] = (int) tmpdims[uu];
  }

  // remove any one at the end
  for (int uu = ndims - 1; uu > 1; uu--) {
    if (tmprsize[uu] == 1)
      finalndims--;
    else
      break;
  }

  r->setSize(finalndims, tmprsize);

  mgc1.remPtr(r);
  return r;

}

#undef MAXDIM
#undef FIRSTPARS







/*************************************************************************
 * GPUopSubsindex
 *************************************************************************/

GPUmatResult_t GPUopSubsindex(GPUtype &d_idata, const int idxshift,
    GPUtype &d_pars, GPUtype &d_odata) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUmanager *GPUman = d_idata.getGPUmanager();



  CUfunction *drvfun;
  CUtexref *drvtexa;
  CUtexref *drvtexb;
  CUarray_format_enum drvtexformata;
  CUarray_format_enum drvtexformatb;
  int drvtexnuma;
  int drvtexnumb;

  gpuTYPE_t ptype = d_idata.getType();

  if (ptype==gpuCFLOAT) {
      drvfun = GPUman->getCuFunction(N_SUBSINDEXC_KERNEL);
      drvtexa = GPUman->getCuTexref(N_TEXREF_C1_A);
      drvtexformata = CU_AD_FORMAT_FLOAT;
      drvtexnuma = 2;
  } else if (ptype==gpuFLOAT){
      drvfun = GPUman->getCuFunction(N_SUBSINDEXF_KERNEL);
      drvtexa = GPUman->getCuTexref(N_TEXREF_F1_A);
      drvtexformata = CU_AD_FORMAT_FLOAT;
      drvtexnuma = 1;
  } else if (ptype==gpuCDOUBLE){
    drvfun = GPUman->getCuFunction(N_SUBSINDEXCD_KERNEL);
      drvtexa = GPUman->getCuTexref(N_TEXREF_CD1_A);
      drvtexformata = CU_AD_FORMAT_SIGNED_INT32;
      drvtexnuma = 4;
  } else if (ptype==gpuDOUBLE){
    drvfun = GPUman->getCuFunction(N_SUBSINDEXD_KERNEL);
      drvtexa = GPUman->getCuTexref(N_TEXREF_D1_A);
      drvtexformata = CU_AD_FORMAT_SIGNED_INT32;
      drvtexnuma = 2;
  }

  drvtexb = GPUman->getCuTexref(N_TEXREF_I1_A);
  drvtexformatb = CU_AD_FORMAT_SIGNED_INT32;
  drvtexnumb = 1;

  // define kernel configuration
  gpukernelconfig_t *kconf = GPUman->getKernelConfig();
  hostdrv_pars_t pars[2];
  int nrhs = 2;

  pars[0].par =  d_odata.getGPUptrptr();
  pars[0].psize = sizeof(CUdeviceptr);
  pars[0].align = __alignof(CUdeviceptr);


  int idxshiftlocal = idxshift;
  pars[1].par = &idxshiftlocal;
  pars[1].psize = sizeof(idxshift);
  pars[1].align = __alignof(idxshift);


  // setup texture
  if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtexa, UINTPTR d_idata.getGPUptr(), d_idata.getNumel()
      * d_idata.getMySize())) {
    throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
  }
  if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT, *drvtexa)) {
    throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
  }
  if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtexa, drvtexformata, drvtexnuma)) {
    throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
  }

  if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtexb, UINTPTR d_pars.getGPUptr(), d_pars.getNumel()
      * d_pars.getMySize())) {
    throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
  }
  if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT, *drvtexb)) {
    throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
  }
  if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtexb, drvtexformatb, drvtexnumb)) {
    throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
  }

  CUDALIBResult cudalibres = mat_HOSTDRV_A(d_odata.getNumel(), kconf, nrhs, pars, drvfun);

  /*if (CUDA_SUCCESS != cuCtxSynchronize()) {
    throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
  }*/

  if (cudalibres != CUDALIBSuccess) {
    throw GPUexception(GPUmatError, ERROR_GPU_KERNELEXECUTION);
  }


  return status;

}

#ifdef XXX
/*************************************************************************
 * GPUopSubsindexf
 *************************************************************************/

GPUmatResult_t GPUopSubsindexf(GPUtype &d_idata, const int idxshift,
    GPUtype &d_pars, GPUtype &d_odata) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUmanager *GPUman = d_idata.getGPUmanager();

  if (GPUman->executionDelayed()) {
    /* streams */
    // TODO
  } else {
    unsigned int N = d_idata.getNumel();
    unsigned int K = d_pars.getNumel();
    unsigned int M = d_odata.getNumel();

    gpukernelconfig_t *kconf = GPUman->getKernelConfig();

    CUDALIBResult cudalibres = mat_SUBSINDEXF(kconf, N, UINTPTR d_idata.getGPUptr(),
        idxshift, K, UINTPTR d_pars.getGPUptr(), M,
        UINTPTR d_odata.getGPUptr(),
        GPUman->getCuFunction(N_SUBSINDEXF_KERNEL), GPUman->getCuTexref(
            N_TEXREF_I1_A), GPUman->getCuTexref(N_TEXREF_F1_A));
    if (cudalibres != CUDALIBSuccess) {
      throw GPUexception(GPUmatError, "Kernel execution error.");
    }

  }

  return status;

}

/*************************************************************************
 * GPUopSubsindexc
 *************************************************************************/

GPUmatResult_t GPUopSubsindexc(GPUtype &d_idata, const int idxshift,
    GPUtype &d_pars, GPUtype &d_odata) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUmanager *GPUman = d_idata.getGPUmanager();

  if (GPUman->executionDelayed()) {
    /* streams */
    // TODO
  } else {
    unsigned int N = d_idata.getNumel();
    unsigned int K = d_pars.getNumel();
    unsigned int M = d_odata.getNumel();

    gpukernelconfig_t *kconf = GPUman->getKernelConfig();
    CUDALIBResult cudalibres = mat_SUBSINDEXC(kconf, N, UINTPTR d_idata.getGPUptr(),
        idxshift, K, UINTPTR d_pars.getGPUptr(), M,
        UINTPTR d_odata.getGPUptr(),
        GPUman->getCuFunction(N_SUBSINDEXC_KERNEL), GPUman->getCuTexref(
            N_TEXREF_I1_A), GPUman->getCuTexref(N_TEXREF_C1_A));
    if (cudalibres != CUDALIBSuccess) {
      throw GPUexception(GPUmatError, "Kernel execution error.");
    }

  }

  return status;

}
#endif
/*************************************************************************
 * GPUopSum
 *************************************************************************/

GPUmatResult_t GPUopSum(GPUtype &p, int Nthread, int M, int GroupSize,
    int GroupOffset, GPUtype &r) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUmanager *GPUman = p.getGPUmanager();


  CUDALIBResult cudalibres;
  CUfunction *drvfun;

  gpuTYPE_t ptype = p.getType();
  if (ptype==gpuFLOAT) {
    drvfun = GPUman->getCuFunction(N_SUMF_TEX_KERNEL);
  } else if (ptype==gpuCFLOAT) {
    drvfun = GPUman->getCuFunction(N_SUMC_TEX_KERNEL);
  } else if (ptype==gpuDOUBLE) {
    drvfun = GPUman->getCuFunction(N_SUMD_TEX_KERNEL);
  } else if (ptype==gpuCDOUBLE) {
    drvfun = GPUman->getCuFunction(N_SUMCD_TEX_KERNEL);
  }

  // setup texture
  GPUsetKernelTextureA(p,drvfun,Nthread*M*p.getMySize());

  int n = Nthread;


  int i0 = Nthread;
  int i1 = M;
  int i2 = GroupSize;
  int i3 = GroupOffset;

  void *op4Ptr;
  unsigned int p4size;

  op4Ptr = r.getGPUptrptr();
  p4size = sizeof(CUdeviceptr);
  size_t p4align = __alignof(CUdeviceptr);

  // define kernel configuration
  gpukernelconfig_t * kconf = GPUman->getKernelConfig();
  hostdrv_pars_t pars[5];
  int nrhs = 5;

  pars[0].par =  &i0;
  pars[0].psize = sizeof(i0);
  pars[0].align = __alignof(i0);

  pars[1].par =  &i1;
  pars[1].psize = sizeof(i1);
  pars[1].align = __alignof(i1);

  pars[2].par =  &i2;
  pars[2].psize = sizeof(i2);
  pars[2].align = __alignof(i2);

  pars[3].par =  &i3;
  pars[3].psize = sizeof(i3);
  pars[3].align = __alignof(i3);

  pars[4].par = op4Ptr;
  pars[4].psize = p4size;
  pars[4].align = p4align;


  cudalibres = mat_HOSTDRV_A(n, kconf, nrhs, pars, drvfun);

  if (cudalibres != CUDALIBSuccess) {
    throw GPUexception(GPUmatError, "Kernel execution error.");
  }


  return status;

}

/*
 * GPUmatResult_t GPUopSum(GPUtype &p, int Nthread, int M, int GroupSize,
    int GroupOffset, GPUtype &r) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUmanager *GPUman = p.getGPUmanager();

  if (GPUman->executionDelayed()) {
    // TODO
  } else {
    CUDALIBResult cudalibres;
    if (p.isComplex()) {
      cudalibres = mat_SUMC_TEX(UINTPTR p.getGPUptr(), Nthread, M, GroupSize,
          GroupOffset, UINTPTR r.getGPUptr(), GPUman->getCuFunction(
              N_SUMC_TEX_KERNEL), GPUman->getCuTexref(N_TEXREF_C1_A));
    } else {
      cudalibres = mat_SUMF_TEX(UINTPTR p.getGPUptr(), Nthread, M, GroupSize,
          GroupOffset, UINTPTR r.getGPUptr(), GPUman->getCuFunction(
              N_SUMF_TEX_KERNEL), GPUman->getCuTexref(N_TEXREF_F1_A));
    }
    if (cudalibres != CUDALIBSuccess) {
      throw GPUexception(GPUmatError, "Kernel execution error.");
    }

  }

  return status;

}*/


/*************************************************************************
 * GPUopSum
 *************************************************************************/

GPUmatResult_t GPUopSum2(GPUtype &p, int Nthread, int M, int GroupSize,
    int GroupOffset, GPUtype &r) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUmanager *GPUman = p.getGPUmanager();


  CUDALIBResult cudalibres;
  if (p.isComplex()) {
    /*cudalibres = mat_SUMC_TEX(UINTPTR p.getGPUptr(), Nthread, M, GroupSize,
        GroupOffset, UINTPTR r.getGPUptr(), GPUman->getCuFunction(
            N_SUMC_TEX_KERNEL), GPUman->getCuTexref(N_TEXREF_C1_A));*/
  } else {
    gpukernelconfig_t *kconf = GPUman->getKernelConfig();
    cudalibres = mat_SUM1F_TEX(kconf, UINTPTR p.getGPUptr(), Nthread, M, GroupSize,
        GroupOffset, UINTPTR r.getGPUptr(), GPUman->getCuFunction(
            N_SUM1F_TEX_KERNEL), GPUman->getCuTexref(N_TEXREF_F1_A));
  }
  if (cudalibres != CUDALIBSuccess) {
    throw GPUexception(GPUmatError, "Kernel execution error.");
  }

  return status;

}

/*************************************************************************
 * GPUopAbsDrv
 *************************************************************************/
GPUtype * GPUopAbsDrv(GPUtype &p) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUtype *r;
  GPUmanager *GPUman = p.getGPUmanager();

  // garbage collector
  MyGCObj<GPUtype> mgc;



  // the result of the Abs is always REAL, even if the argument is complex
  // allocate temp for results
  r = new GPUtype(p, 1); // do not copy GPUptr
  mgc.setPtr(r);
  r->setReal(); // result always real
  GPUopAllocVector(*r);
  GPUopAbs(p, *r);

  mgc.remPtr(r);
  return r;
}

/*************************************************************************
 * GPUopAbs
 *************************************************************************/
//CGPUOP4(Abs,ABS)
GPUmatResult_t GPUopAbs(GPUtype &p, GPUtype &r) {
  GPUmatResult_t status = GPUmatSuccess;
  cudaError_t cudastatus = cudaSuccess;
  GPUmanager * GPUman = p.getGPUmanager();


  status = arg1op_common(&(GPURESULTE[0]), p, r,
      N_ABSF_KERNEL, N_ABSC_KERNEL,
      N_ABSD_KERNEL, N_ABSCD_KERNEL);

  return status;
}

/*************************************************************************
 * GPUopAndDrv
 * GPUopAnd
 *************************************************************************/
//CGPUOP5(And)
GPUtype * GPUopAndDrv(GPUtype &p, GPUtype &q) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUtype *r;
  GPUmanager *GPUman = p.getGPUmanager();

  r = arg3op_drv(&(GPURESULTC[0][0]), p, q, (GPUmatResult_t(*)(GPUtype&, GPUtype&, GPUtype&)) GPUopAnd);

  return r;
}
//CGPUOP6(And, AND)
GPUmatResult_t GPUopAnd(GPUtype &p, GPUtype &q, GPUtype &r) {
  GPUmatResult_t status = GPUmatSuccess;
  cudaError_t cudastatus = cudaSuccess;
  GPUmanager * GPUman = p.getGPUmanager();

  status =  arg3op2_common(&(GPURESULTC[0][0]), p, q, r,
      N_AND_F_F_KERNEL,
      N_AND_F_C_KERNEL,
      N_AND_F_D_KERNEL,
      N_AND_F_CD_KERNEL,
      N_AND_C_F_KERNEL,
      N_AND_C_C_KERNEL,
      N_AND_C_D_KERNEL,
      N_AND_C_CD_KERNEL,
      N_AND_D_F_KERNEL,
      N_AND_D_C_KERNEL,
      N_AND_D_D_KERNEL,
      N_AND_D_CD_KERNEL,
      N_AND_CD_F_KERNEL,
      N_AND_CD_C_KERNEL,
      N_AND_CD_D_KERNEL,
      N_AND_CD_CD_KERNEL    );

  return status;
}


/*************************************************************************
 * GPUopAcosDrv
 * GPUopAcos
 *************************************************************************/
CGPUOP3(Acos)
CGPUOP4(Acos, ACOS)

/*************************************************************************
 * GPUopAcoshDrv
 * GPUopAcosh
 *************************************************************************/
CGPUOP3(Acosh)
CGPUOP4(Acosh, ACOSH)

/*************************************************************************
 * GPUopAsinDrv
 * GPUopAsin
 *************************************************************************/
CGPUOP3(Asin)
CGPUOP4(Asin, ASIN)

/*************************************************************************
 * GPUopAsinhDrv
 * GPUopAsinh
 *************************************************************************/
CGPUOP3(Asinh)
CGPUOP4(Asinh, ASINH)

/*************************************************************************
 * GPUopAtanDrv
 * GPUopAtan
 *************************************************************************/
CGPUOP3(Atan)
CGPUOP4(Atan, ATAN)

/*************************************************************************
 * GPUopAtanhDrv
 * GPUopAtanh
 *************************************************************************/
CGPUOP3(Atanh)
CGPUOP4(Atanh, ATANH)

/*************************************************************************
 * GPUopCeilDrv
 * GPUopCeil
 *************************************************************************/
CGPUOP3(Ceil)
CGPUOP4(Ceil, CEIL)

/*************************************************************************
 * GPUopCosDrv
 * GPUopCos
 *************************************************************************/
CGPUOP3(Cos)
CGPUOP4(Cos, COS)

/*************************************************************************
 * GPUopCoshDrv
 * GPUopCosh
 *************************************************************************/
CGPUOP3(Cosh)
CGPUOP4(Cosh, COSH)

/*************************************************************************
 * GPUopConjDrv
 * GPUopConj
 *************************************************************************/
CGPUOP3(Conj)
CGPUOP4(Conj, CONJUGATE)

/*************************************************************************
 * GPUopEqDrv
 * GPUopEq
 *************************************************************************/
GPUtype * GPUopEqDrv(GPUtype &p, GPUtype &q) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUtype *r;
  GPUmanager *GPUman = p.getGPUmanager();

  r = arg3op_drv(&(GPURESULTC[0][0]), p, q, (GPUmatResult_t(*)(GPUtype&, GPUtype&, GPUtype&)) GPUopEq);

  return r;
}

//CGPUOP6(Eq, EQ)
GPUmatResult_t GPUopEq(GPUtype &p, GPUtype &q, GPUtype &r) {
  GPUmatResult_t status = GPUmatSuccess;
  cudaError_t cudastatus = cudaSuccess;
  GPUmanager * GPUman = p.getGPUmanager();
  status =  arg3op2_common(&(GPURESULTC[0][0]), p, q, r,
        N_EQ_F_F_KERNEL,
        N_EQ_F_C_KERNEL,
        N_EQ_F_D_KERNEL,
        N_EQ_F_CD_KERNEL,
        N_EQ_C_F_KERNEL,
        N_EQ_C_C_KERNEL,
        N_EQ_C_D_KERNEL,
        N_EQ_C_CD_KERNEL,
        N_EQ_D_F_KERNEL,
        N_EQ_D_C_KERNEL,
        N_EQ_D_D_KERNEL,
        N_EQ_D_CD_KERNEL,
        N_EQ_CD_F_KERNEL,
        N_EQ_CD_C_KERNEL,
        N_EQ_CD_D_KERNEL,
        N_EQ_CD_CD_KERNEL   );

  return status;
}



/*************************************************************************
 * GPUopPlusDrv
 *************************************************************************/
GPUtype * GPUopPlusDrv(GPUtype &p, GPUtype &q) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUtype *r;
  GPUmanager *GPUman = p.getGPUmanager();

  r = arg3op_drv(NULL, p, q, (GPUmatResult_t(*)(GPUtype&, GPUtype&, GPUtype&)) GPUopPlus);

  return r;
}
//CGPUOP5(Plus)

/*************************************************************************
 * GPUopPlus
 *************************************************************************/

/*GPUmatResult_t GPUopPlus(GPUtype &p, GPUtype &q, GPUtype &r) {
  GPUmatResult_t status = GPUmatSuccess;
  cudaError_t cudastatus = cudaSuccess;
  GPUmanager * GPUman = p.getGPUmanager();
  if (GPUman->executionDelayed()) {

  } else {
    status =  arg3op2_common(p,q,r,
        N_PLUS_F_F_KERNEL,
        N_PLUS_F_C_KERNEL,
        N_PLUS_F_D_KERNEL,
        N_PLUS_F_CD_KERNEL,
        N_PLUS_C_F_KERNEL,
        N_PLUS_C_C_KERNEL,
        N_PLUS_C_D_KERNEL,
        N_PLUS_C_CD_KERNEL,
        N_PLUS_D_F_KERNEL,
        N_PLUS_D_C_KERNEL,
        N_PLUS_D_D_KERNEL,
        N_PLUS_D_CD_KERNEL,
        N_PLUS_CD_F_KERNEL,
        N_PLUS_CD_C_KERNEL,
        N_PLUS_CD_D_KERNEL,
        N_PLUS_CD_CD_KERNEL
    );
  }
  return status;
}*/
CGPUOP6(Plus, PLUS)

/*************************************************************************
 * GPUopMtimesDrv
 * GPUopMtimes
 *************************************************************************/
GPUtype * GPUopMtimesDrv(GPUtype &p, GPUtype &q) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUtype *r;
  GPUmanager *GPUman = p.getGPUmanager();


  if (p.isScalar() || q.isScalar()) {
    r = GPUopTimesDrv(p,q);
  } else {
    r = mtimes_drv(p, q);
  }


  return r;
}

GPUmatResult_t GPUopMtimes(GPUtype &p, GPUtype &q, GPUtype &r) {
  GPUmatResult_t status = GPUmatSuccess;
  cudaError_t cudastatus = cudaSuccess;
  GPUmanager * GPUman = p.getGPUmanager();

  // If the operations involves scalar then I must use GPUopTimes
  if (p.isScalar() || q.isScalar()) {
    status = GPUopTimes(p,q,r);
  } else {
    status = mtimes(p,q,r);
  }

  return status;
}

/*************************************************************************
 * GPUopExpDrv
 *************************************************************************/
/*GPUtype * GPUopExpDrv(GPUtype &p) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUtype *r;
  GPUmanager *GPUman = p.getGPUmanager();

  if (GPUman->executionDelayed()) {
   // TODO
  } else {
    r = arg1op_drv(gpuNOTDEF, p, (GPUmatResult_t(*)(GPUtype&, GPUtype&)) GPUopExp);
  }
  return r;
}*/
GPUtype * GPUopExpDrv(GPUtype &p) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUtype *r;
  GPUmanager *GPUman = p.getGPUmanager();

  r = arg1op_drv(NULL, p, (GPUmatResult_t(*)(GPUtype&, GPUtype&)) GPUopExp);

  return r;
}
//CGPUOP3(Exp)

/*************************************************************************
 * GPUopExp
 *************************************************************************/

/*GPUmatResult_t GPUopExp(GPUtype &p, GPUtype &r) {
  GPUmatResult_t status = GPUmatSuccess;
  cudaError_t cudastatus = cudaSuccess;
  GPUmanager * GPUman = p.getGPUmanager();
  if (GPUman->executionDelayed()) {

  } else {
    status = arg1op_common(p,r,
        N_EXPF_KERNEL, N_EXPC_KERNEL,
        N_EXPD_KERNEL, N_EXPCD_KERNEL);
  }
  return status;
}*/
CGPUOP4(Exp, EXP)



/*************************************************************************
 * GPUopFloorDrv
 * GPUopFloor
 *************************************************************************/
CGPUOP3(Floor)
CGPUOP4(Floor, FLOOR)

/*************************************************************************
 * GPUopFFTSymm
 *************************************************************************/

GPUmatResult_t GPUopFFTSymm(GPUtype &d_idata, int batch) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUmanager *GPUman = d_idata.getGPUmanager();

  int ndims = d_idata.getNdims();
  int M = 1;
  int N = 1;
  int Q = 1;

  int *psize = d_idata.getSize();
  M = psize[0];

  if (ndims == 2) {
    if (M == 1) {
      N = 1;
      M = psize[1];
    } else {
      N = psize[1];
    }
  }

  if (ndims == 3) {
    N = psize[1];
    Q = psize[2];
  }

  CUfunction *drvfun;
  CUtexref *drvtex;
  CUarray_format_enum drvtexformata;
  CUarray_format_enum drvtexformatb;
  int drvtexnuma;
  int drvtexnumb;

  if (d_idata.isFloat()) {
    drvfun =  GPUman->getCuFunction(N_FFTSYMMC_KERNEL);
    drvtex = GPUman->getCuTexref(N_TEXREF_C1_A);
    drvtexformata = CU_AD_FORMAT_FLOAT;
    drvtexnuma = 2;
  } else if (d_idata.isDouble()) {
    drvfun =  GPUman->getCuFunction(N_FFTSYMMCD_KERNEL);
    drvtex = GPUman->getCuTexref(N_TEXREF_CD1_A);
    drvtexformata = CU_AD_FORMAT_SIGNED_INT32;
    drvtexnuma = 4;
  }


  if (CUDA_SUCCESS != cuTexRefSetAddress(NULL, *drvtex, UINTPTR d_idata.getGPUptr(), M*N*Q* d_idata.getMySize())) {
    throw GPUexception(GPUmatError, "Kernel execution error1.");
  }

  if (CUDA_SUCCESS != cuParamSetTexRef(*drvfun, CU_PARAM_TR_DEFAULT, *drvtex)) {
    throw GPUexception(GPUmatError, "Kernel execution error3.");
  }
  if (CUDA_SUCCESS != cuTexRefSetFormat(*drvtex, drvtexformata, drvtexnuma)) {
    throw GPUexception(GPUmatError, "Kernel execution error2.");
  }

  CUDALIBResult cudalibres;
  gpukernelconfig_t *kconf = GPUman->getKernelConfig();
  if (d_idata.isFloat()) {
    cudalibres =  mat_FFTSYMM(kconf, M, N, Q, UINTPTR d_idata.getGPUptr(),
          UINTPTR d_idata.getGPUptr(), batch, drvfun, drvtex);

  } else if (d_idata.isDouble()) {
    cudalibres =  mat_FFTSYMM(kconf, M, N, Q, UINTPTR d_idata.getGPUptr(),
          UINTPTR d_idata.getGPUptr(), batch, drvfun, drvtex);
  }
  if (cudalibres != CUDALIBSuccess) {
    throw GPUexception(GPUmatError, "Kernel execution error.");
  }



  return status;

}

/*************************************************************************
 * GPUopGeDrv
 * GPUopGe
 *************************************************************************/
GPUtype * GPUopGeDrv(GPUtype &p, GPUtype &q) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUtype *r;
  GPUmanager *GPUman = p.getGPUmanager();

  r = arg3op_drv(&(GPURESULTC[0][0]), p, q, (GPUmatResult_t(*)(GPUtype&, GPUtype&, GPUtype&)) GPUopGe);

  return r;
}

//CGPUOP6(Ge, GE)
GPUmatResult_t GPUopGe(GPUtype &p, GPUtype &q, GPUtype &r) {\
  GPUmatResult_t status = GPUmatSuccess;\
  cudaError_t cudastatus = cudaSuccess;\
  GPUmanager * GPUman = p.getGPUmanager();\

    status =  arg3op2_common(&(GPURESULTC[0][0]), p, q, r,\
        N_GE_F_F_KERNEL,\
        N_GE_F_C_KERNEL,\
        N_GE_F_D_KERNEL,\
        N_GE_F_CD_KERNEL,\
        N_GE_C_F_KERNEL,\
        N_GE_C_C_KERNEL,\
        N_GE_C_D_KERNEL,\
        N_GE_C_CD_KERNEL,\
        N_GE_D_F_KERNEL,\
        N_GE_D_C_KERNEL,\
        N_GE_D_D_KERNEL,\
        N_GE_D_CD_KERNEL,\
        N_GE_CD_F_KERNEL,\
        N_GE_CD_C_KERNEL,\
        N_GE_CD_D_KERNEL,\
        N_GE_CD_CD_KERNEL   );\

  return status;\
}\

/*************************************************************************
 * GPUopGtDrv
 * GPUopGt
 *************************************************************************/
GPUtype * GPUopGtDrv(GPUtype &p, GPUtype &q) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUtype *r;
  GPUmanager *GPUman = p.getGPUmanager();


  r = arg3op_drv(&(GPURESULTC[0][0]), p, q, (GPUmatResult_t(*)(GPUtype&, GPUtype&, GPUtype&)) GPUopGt);

  return r;
}

//CGPUOP6(Gt, GT)
GPUmatResult_t GPUopGt(GPUtype &p, GPUtype &q, GPUtype &r) {\
  GPUmatResult_t status = GPUmatSuccess;\
  cudaError_t cudastatus = cudaSuccess;\
  GPUmanager * GPUman = p.getGPUmanager();\

    status =  arg3op2_common(&(GPURESULTC[0][0]), p, q, r,\
        N_GT_F_F_KERNEL,\
        N_GT_F_C_KERNEL,\
        N_GT_F_D_KERNEL,\
        N_GT_F_CD_KERNEL,\
        N_GT_C_F_KERNEL,\
        N_GT_C_C_KERNEL,\
        N_GT_C_D_KERNEL,\
        N_GT_C_CD_KERNEL,\
        N_GT_D_F_KERNEL,\
        N_GT_D_C_KERNEL,\
        N_GT_D_D_KERNEL,\
        N_GT_D_CD_KERNEL,\
        N_GT_CD_F_KERNEL,\
        N_GT_CD_C_KERNEL,\
        N_GT_CD_D_KERNEL,\
        N_GT_CD_CD_KERNEL   );\

  return status;\
}\


/*************************************************************************
 * GPUopLdivideDrv
 * GPUopLdivide
 *************************************************************************/
CGPUOP5(Ldivide)
CGPUOP6(Ldivide, LDIVIDE)

/*************************************************************************
 * GPUopLeDrv
 * GPUopLe
 *************************************************************************/
GPUtype * GPUopLeDrv(GPUtype &p, GPUtype &q) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUtype *r;
  GPUmanager *GPUman = p.getGPUmanager();

  r = arg3op_drv(&(GPURESULTC[0][0]), p, q, (GPUmatResult_t(*)(GPUtype&, GPUtype&, GPUtype&)) GPUopLe);

  return r;
}

//CGPUOP6(Le, LE)
GPUmatResult_t GPUopLe(GPUtype &p, GPUtype &q, GPUtype &r) {\
  GPUmatResult_t status = GPUmatSuccess;\
  cudaError_t cudastatus = cudaSuccess;\
  GPUmanager * GPUman = p.getGPUmanager();\

    status =  arg3op2_common(&(GPURESULTC[0][0]), p, q, r,\
        N_LE_F_F_KERNEL,\
        N_LE_F_C_KERNEL,\
        N_LE_F_D_KERNEL,\
        N_LE_F_CD_KERNEL,\
        N_LE_C_F_KERNEL,\
        N_LE_C_C_KERNEL,\
        N_LE_C_D_KERNEL,\
        N_LE_C_CD_KERNEL,\
        N_LE_D_F_KERNEL,\
        N_LE_D_C_KERNEL,\
        N_LE_D_D_KERNEL,\
        N_LE_D_CD_KERNEL,\
        N_LE_CD_F_KERNEL,\
        N_LE_CD_C_KERNEL,\
        N_LE_CD_D_KERNEL,\
        N_LE_CD_CD_KERNEL   );\

  return status;\
}\


/*************************************************************************
 * GPUopLog1pDrv
 * GPUoplog1p
 *************************************************************************/
CGPUOP3(Log1p)
CGPUOP4(Log1p, LOG1P)

/*************************************************************************
 * GPUopLog2Drv
 * GPUopLog2
 *************************************************************************/
CGPUOP3(Log2)
CGPUOP4(Log2, LOG2)

/*************************************************************************
 * GPUopLog10Drv
 * GPUopLog10
 *************************************************************************/
CGPUOP3(Log10)
CGPUOP4(Log10, LOG10)

/*************************************************************************
 * GPUopLogDrv
 * GPUopLog
 *************************************************************************/
CGPUOP3(Log)
CGPUOP4(Log, LOG)

/*************************************************************************
 * GPUopLtDrv
 * GPUopLt
 *************************************************************************/
GPUtype * GPUopLtDrv(GPUtype &p, GPUtype &q) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUtype *r;
  GPUmanager *GPUman = p.getGPUmanager();

  r = arg3op_drv(&(GPURESULTC[0][0]), p, q, (GPUmatResult_t(*)(GPUtype&, GPUtype&, GPUtype&)) GPUopLt);

  return r;
}

//CGPUOP6(Lt, LT)
GPUmatResult_t GPUopLt(GPUtype &p, GPUtype &q, GPUtype &r) {\
  GPUmatResult_t status = GPUmatSuccess;\
  cudaError_t cudastatus = cudaSuccess;\
  GPUmanager * GPUman = p.getGPUmanager();\

    status =  arg3op2_common(&(GPURESULTC[0][0]), p, q, r,\
        N_LT_F_F_KERNEL,\
        N_LT_F_C_KERNEL,\
        N_LT_F_D_KERNEL,\
        N_LT_F_CD_KERNEL,\
        N_LT_C_F_KERNEL,\
        N_LT_C_C_KERNEL,\
        N_LT_C_D_KERNEL,\
        N_LT_C_CD_KERNEL,\
        N_LT_D_F_KERNEL,\
        N_LT_D_C_KERNEL,\
        N_LT_D_D_KERNEL,\
        N_LT_D_CD_KERNEL,\
        N_LT_CD_F_KERNEL,\
        N_LT_CD_C_KERNEL,\
        N_LT_CD_D_KERNEL,\
        N_LT_CD_CD_KERNEL   );\

  return status;\
}\


/*************************************************************************
 * GPUopMinusDrv
 * GPUopMinus
 *************************************************************************/
CGPUOP5(Minus)
CGPUOP6(Minus, MINUS)

/*************************************************************************
 * GPUopNeDrv
 * GPUopNe
 *************************************************************************/
GPUtype * GPUopNeDrv(GPUtype &p, GPUtype &q) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUtype *r;
  GPUmanager *GPUman = p.getGPUmanager();

  r = arg3op_drv(&(GPURESULTC[0][0]), p, q, (GPUmatResult_t(*)(GPUtype&, GPUtype&, GPUtype&)) GPUopNe);

  return r;
}

//CGPUOP6(Ne, NE)
GPUmatResult_t GPUopNe(GPUtype &p, GPUtype &q, GPUtype &r) {\
  GPUmatResult_t status = GPUmatSuccess;\
  cudaError_t cudastatus = cudaSuccess;\
  GPUmanager * GPUman = p.getGPUmanager();\

    status =  arg3op2_common(&(GPURESULTC[0][0]), p, q, r,\
        N_NE_F_F_KERNEL,\
        N_NE_F_C_KERNEL,\
        N_NE_F_D_KERNEL,\
        N_NE_F_CD_KERNEL,\
        N_NE_C_F_KERNEL,\
        N_NE_C_C_KERNEL,\
        N_NE_C_D_KERNEL,\
        N_NE_C_CD_KERNEL,\
        N_NE_D_F_KERNEL,\
        N_NE_D_C_KERNEL,\
        N_NE_D_D_KERNEL,\
        N_NE_D_CD_KERNEL,\
        N_NE_CD_F_KERNEL,\
        N_NE_CD_C_KERNEL,\
        N_NE_CD_D_KERNEL,\
        N_NE_CD_CD_KERNEL   );\

  return status;\
}\



/*************************************************************************
 * GPUopNotDrv
 * GPUopNot
 *************************************************************************/
GPUtype * GPUopNotDrv(GPUtype &p) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUtype *r;
  GPUmanager *GPUman = p.getGPUmanager();

  r = arg1op_drv(&(GPURESULTD[0]), p, (GPUmatResult_t(*)(GPUtype&, GPUtype&)) GPUopNot);

  return r;
}
//CGPUOP4(Not, NOT)
GPUmatResult_t GPUopNot(GPUtype &p, GPUtype &r) {
  GPUmatResult_t status = GPUmatSuccess;
  cudaError_t cudastatus = cudaSuccess;
  GPUmanager * GPUman = p.getGPUmanager();
  gpuTYPE_t GPURESULT[5] = {gpuFLOAT  , gpuFLOAT, gpuFLOAT  , gpuFLOAT, gpuNOTDEF};

  status = arg1op_common(&(GPURESULTD[0]), p, r,
        N_NOTF_KERNEL, N_NOTC_KERNEL,
        N_NOTD_KERNEL, N_NOTCD_KERNEL);

  return status;
}


/*************************************************************************
 * GPUopOrDrv
 * GPUopOr
 *************************************************************************/
//CGPUOP5(Or)
GPUtype * GPUopOrDrv(GPUtype &p, GPUtype &q) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUtype *r;
  GPUmanager *GPUman = p.getGPUmanager();

  r = arg3op_drv(&(GPURESULTC[0][0]), p, q, (GPUmatResult_t(*)(GPUtype&, GPUtype&, GPUtype&)) GPUopOr);

  return r;
}
//CGPUOP6(Or, OR)
GPUmatResult_t GPUopOr(GPUtype &p, GPUtype &q, GPUtype &r) {
  GPUmatResult_t status = GPUmatSuccess;
  cudaError_t cudastatus = cudaSuccess;
  GPUmanager * GPUman = p.getGPUmanager();

    status =  arg3op2_common(&(GPURESULTC[0][0]), p, q, r,
        N_OR_F_F_KERNEL,
        N_OR_F_C_KERNEL,
        N_OR_F_D_KERNEL,
        N_OR_F_CD_KERNEL,
        N_OR_C_F_KERNEL,
        N_OR_C_C_KERNEL,
        N_OR_C_D_KERNEL,
        N_OR_C_CD_KERNEL,
        N_OR_D_F_KERNEL,
        N_OR_D_C_KERNEL,
        N_OR_D_D_KERNEL,
        N_OR_D_CD_KERNEL,
        N_OR_CD_F_KERNEL,
        N_OR_CD_C_KERNEL,
        N_OR_CD_D_KERNEL,
        N_OR_CD_CD_KERNEL   );

  return status;
}


/*************************************************************************
 * GPUopPowerDrv
 * GPUopPower
 *************************************************************************/
CGPUOP5(Power)
CGPUOP6(Power, POWER)

/*************************************************************************
 * GPUopRdivideDrv
 * GPUopRdivide
 *************************************************************************/
CGPUOP5(Rdivide)
CGPUOP6(Rdivide, RDIVIDE)

/*************************************************************************
 * GPUopRoundDrv
 * GPUopRound
 *************************************************************************/
CGPUOP3(Round)
CGPUOP4(Round, ROUND)

/*************************************************************************
 * GPUopSinDrv
 * GPUopSin
 *************************************************************************/
CGPUOP3(Sin)
CGPUOP4(Sin, SIN)

/*************************************************************************
 * GPUopSinhDrv
 * GPUopSinh
 *************************************************************************/
CGPUOP3(Sinh)
CGPUOP4(Sinh, SINH)

/*************************************************************************
 * GPUopSqrtDrv
 * GPUopSqrt
 *************************************************************************/
CGPUOP3(Sqrt)
CGPUOP4(Sqrt, SQRT)

/*************************************************************************
 * GPUopTanDrv
 * GPUopTan
 *************************************************************************/
CGPUOP3(Tan)
CGPUOP4(Tan, TAN)

/*************************************************************************
 * GPUopTanhDrv
 * GPUopTanh
 *************************************************************************/
CGPUOP3(Tanh)
CGPUOP4(Tanh, TANH)

/*************************************************************************
 * GPUopTimesDrv
 * GPUopTimes
 *************************************************************************/
CGPUOP5(Times)
CGPUOP6(Times, TIMES)

/*************************************************************************
 * GPUopTimes2Drv
 * GPUopTimes2
 *************************************************************************/
//CGPUOP5(Times2)
//CGPUOP6(Times2, TIMES2)

/*************************************************************************
 * GPUopUminusDrv
 * GPUopUminus
 *************************************************************************/
CGPUOP3(Uminus)
CGPUOP4(Uminus, UMINUS)

/*************************************************************************
 * GPUopRealDrv
 *************************************************************************/

GPUtype * GPUopRealDrv(GPUtype &p) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUmanager *GPUman = p.getGPUmanager();

  // garbage collector
  MyGCObj<GPUtype> mgc;

  GPUtype *r = new GPUtype(p, 1);
  mgc.setPtr(r);
  r->setReal(); // real anyway



  GPUopAllocVector(*r);

  status = GPUopReal(p, *r);

  mgc.remPtr(r);
  return r;
}

/*************************************************************************
 * GPUopReal
 *************************************************************************/
GPUmatResult_t GPUopReal(GPUtype &p, GPUtype &r) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUmanager * GPUman = p.getGPUmanager();


  /* check arguments */
  int numel = p.getNumel();
  if (r.getNumel()!=numel) {
    throw GPUexception(GPUmatError,
        ERROR_ARG2OP_ELEMENTS);
  }
  // output should be real
  if (r.isComplex()) {
    throw GPUexception(GPUmatError,
        ERROR_REAL_COMPLEXOUT);
  }


  gpuTYPE_t GPURESULT[NGPUTYPE] = {gpuFLOAT, gpuFLOAT, gpuDOUBLE, gpuDOUBLE, gpuNOTDEF};
  checkResult(p,r,&(GPURESULT[0]));

  if (p.isComplex()) {
    /* mode for GPUopUnpackfC2C
     * 0 - REAL, IMAG
     * 1 - REAL
     * 2 - IMAG*/
    //status = GPUopUnpackC2C(1, p,
    //    r, r);
    status = GPUopRealImag(p, r, r, 1, 1);

  } else {
    // returns the same vector
    cudaError_t status;
    status = cudaMemcpy(r.getGPUptr(),p.getGPUptr(),p.getNumel()*p.getMySize(),cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess) {
      throw GPUexception(GPUmatError, "Error in cudaMemcpy.");
    }
  }
  return status;
}

/*************************************************************************
 * GPUopComplexDrv
 *************************************************************************/

GPUtype * GPUopComplexDrv(GPUtype &p) {
  GPUmatResult_t status = GPUmatSuccess;

  // garbage collector
  MyGCObj<GPUtype> mgc;
  GPUtype *r;
  GPUmanager * GPUman = p.getGPUmanager();

  r = p.REALtoCOMPLEX();

  return r;
}

GPUtype * GPUopComplexDrv(GPUtype &p, GPUtype &im) {
  GPUmatResult_t status = GPUmatSuccess;

  // garbage collector
  MyGCObj<GPUtype> mgc;
  GPUtype *r;
  GPUmanager * GPUman = p.getGPUmanager();

  r = p.REALtoCOMPLEX(im);

  return r;
}


/*************************************************************************
 * GPUopComplex
 *************************************************************************/

GPUmatResult_t GPUopComplex(GPUtype &p, GPUtype &r) {
  GPUmatResult_t status = GPUmatSuccess;
  cudaError_t cudastatus = cudaSuccess;
  GPUmanager * GPUman = p.getGPUmanager();

  if (r.getNumel()!= p.getNumel()) {
    throw GPUexception(GPUmatError,  ERROR_GPUOPCOMPLEX_WRONGNEL);
  }

  if (p.getType() == gpuFLOAT) {
    if (r.getType()!= gpuCFLOAT) {
      throw GPUexception(GPUmatError,  ERROR_GPUOPCOMPLEX_WRONGOUT);
    }

    GPUopRealImag(r, p, p, 0, 1);
  } else if (p.getType() == gpuCFLOAT) {
    throw GPUexception(GPUmatError,
                      ERROR_GPUTYPE_FLOATTOCOMPLEX_COMPLEX);

  } else if (p.getType() == gpuDOUBLE) {
    if (r.getType()!= gpuCDOUBLE) {
      throw GPUexception(GPUmatError,  ERROR_GPUOPCOMPLEX_WRONGOUT);
    }

    GPUopRealImag(r, p, p, 0, 1);

  } else if (p.getType() == gpuCDOUBLE) {
    throw GPUexception(GPUmatError,
                      ERROR_GPUTYPE_FLOATTOCOMPLEX_COMPLEX);
  }

  return status;
}

GPUmatResult_t GPUopComplex(GPUtype &p, GPUtype &q, GPUtype &r) {
  GPUmatResult_t status = GPUmatSuccess;
  cudaError_t cudastatus = cudaSuccess;
  GPUmanager * GPUman = p.getGPUmanager();


  if (r.getNumel()!= p.getNumel()) {
    throw GPUexception(GPUmatError,  ERROR_GPUOPCOMPLEX_WRONGNEL);
  }
  if (r.getNumel()!= q.getNumel()) {
    throw GPUexception(GPUmatError,  ERROR_GPUOPCOMPLEX_WRONGNEL);
  }

  if (p.getType() == gpuFLOAT) {
    if (r.getType()!= gpuCFLOAT) {
      throw GPUexception(GPUmatError,  ERROR_GPUOPCOMPLEX_WRONGOUT);
    }
    if (q.getType()!= gpuFLOAT) {
      throw GPUexception(GPUmatError,  ERROR_GPUOPCOMPLEX_WRONGIN);
    }

    GPUopRealImag(r, p, q, 0, 0);
  } else if (p.getType() == gpuCFLOAT) {
    throw GPUexception(GPUmatError,
                      ERROR_GPUTYPE_FLOATTOCOMPLEX_COMPLEX);

  } else if (p.getType() == gpuDOUBLE) {
    if (r.getType()!= gpuCDOUBLE) {
      throw GPUexception(GPUmatError,  ERROR_GPUOPCOMPLEX_WRONGOUT);
    }

    if (q.getType()!= gpuDOUBLE) {
      throw GPUexception(GPUmatError,  ERROR_GPUOPCOMPLEX_WRONGIN);
    }
    GPUopRealImag(r, p, q, 0, 0);

  } else if (p.getType() == gpuCDOUBLE) {
    throw GPUexception(GPUmatError,
                      ERROR_GPUTYPE_FLOATTOCOMPLEX_COMPLEX);
  }

  return status;
}




/*************************************************************************
 * GPUopImagDrv
 *************************************************************************/

GPUtype * GPUopImagDrv(GPUtype &p) {
  GPUmatResult_t status = GPUmatSuccess;

  GPUmanager *GPUman = p.getGPUmanager();

  // garbage collector
  MyGCObj<GPUtype> mgc;

  GPUtype *r = new GPUtype(p, 1);
  mgc.setPtr(r);
  r->setReal(); // real anyway

  GPUopAllocVector(*r);
  status = GPUopImag(p, *r);

  mgc.remPtr(r);
  return r;

}

/*************************************************************************
 * GPUopImag
 *************************************************************************/
GPUmatResult_t GPUopImag(GPUtype &p, GPUtype &r) {
  GPUmatResult_t status = GPUmatSuccess;

  GPUmanager *GPUman = p.getGPUmanager();


  /* check arguments */
  int numel = p.getNumel();
  if (r.getNumel()!=numel) {
    throw GPUexception(GPUmatError,
        ERROR_ARG2OP_ELEMENTS);
  }

  gpuTYPE_t GPURESULT[NGPUTYPE] = {gpuFLOAT, gpuFLOAT, gpuDOUBLE, gpuDOUBLE, gpuNOTDEF};
  checkResult(p,r,&(GPURESULT[0]));

  if (p.isComplex()) {
    /* mode for GPUopUnpackfC2C
     * 0 - REAL, IMAG
     * 1 - REAL
     * 2 - IMAG*/
    //status = GPUopUnpackC2C(2, p,
    //    r, r);
    status = GPUopRealImag(p, r, r, 1, 2);


  } else {
    // returns a vector of zeros
    cudaError_t status;
    status = cudaMemset(r.getGPUptr(), 0, r.getNumel() * r.getMySize());
    if (status != cudaSuccess) {
      throw GPUexception(GPUmatError, "Error in cudaMemset.");
    }
  }
  return status;
}

/*************************************************************************
 * GPUopFFTDrv
 *************************************************************************/
/* dim - dimension
 *       1 - FFT 1D
 *       2 - FFT 2D
 *       3 - FFT 3D
 */
#define CUPLAN1DMAXEL 8000000
GPUtype * GPUopFFTDrv(GPUtype &p, int dim, int dir) {
  GPUmatResult_t status = GPUmatSuccess;

  // garbage collector
  MyGCObj<GPUtype> mgc;

  // FFT usually requires a lot of memory
  // clean up with some euristics
  GPUmanager * GPUman = p.getGPUmanager();
//  GPUman->cacheClean();


  //check input argument
  // ndims is length(size), but here we want to use 1d fft for arrays that
  // have size = [1 n] or [n 1]
  int *size = p.getSize();
  int nd = 0;
  for (int i = 0; i < p.getNdims(); i++) {
    if (size[i] > 1)
      nd++;
  }

  int batch = 1;

  if (dim == 1) {
    if (nd > 3) {
      throw GPUexception(GPUmatError, ERROR_FFT_MAXDIM);
    }
    // calculate batch
    if (nd > 1) {
      batch = p.getNumel() / size[0];
    }

    if (p.getNumel() > CUPLAN1DMAXEL) {
      throw GPUexception(GPUmatError, ERROR_FFT_MAXEL1D);
    }
  } else if (dim==2) {
    // allow also 3rd dimension in that case work in batch
    //if ((nd != 2) || (p.getNdims() != 2))
    if ((nd < 2) || (p.getNdims() < 2))
      throw GPUexception(GPUmatError, ERROR_FFT_ONLY2D);

    if (nd==2) {
      // nothing
    } else if (nd==3) {
      batch = size[2];
    } else {
      throw GPUexception(GPUmatError, ERROR_FFT_ONLY2D);
    }

  } else if (dim==3) {
    if ((nd != 3) || (p.getNdims() != 3))
        throw GPUexception(GPUmatError, ERROR_FFT_ONLY3D);

  } else {
    throw GPUexception(GPUmatError,
              "Wrong dim parameter.");
  }


  GPUtype *r = new GPUtype(p, 1);
  r->setComplex();

  cufftType_t fftType;

  if (p.isFloat()) {
    fftType = CUFFT_R2C;
  } else if (p.isDouble()) {
#ifndef NODOUBLE
    fftType = CUFFT_D2Z;
#endif
  }

  GPUtype ptmp = GPUtype(p);


  if ((p.isComplex())||(dir==CUFFT_INVERSE)) {
    if (p.isFloat()) {
      fftType = CUFFT_C2C;
    } else if (p.isDouble()) {
#ifndef NODOUBLE
      fftType = CUFFT_Z2Z;
#endif
    }
  } else {
    // this is necessary only in batch mode
    if ((nd > 1)) {
      int *rsize = r->getSize();
      rsize[0] = ((int) rsize[0] / 2) + 1;
    }
  }

  //GPUopAllocVector(*r);

  mgc.setPtr(r);
  // register pointer. Must unregister before return

  if ((!p.isComplex())&&(dir==CUFFT_INVERSE)) {
    // complex
    ptmp = GPUtype(p, 1);
    ptmp.setComplex();
    GPUopAllocVector(ptmp);
    // pack data
    GPUopPackC2C(1, p, p, ptmp); //1 is for onlyreal
  }

  status = GPUopFFT(ptmp, *r, dir, batch, dim);

  // scale the inverse
  if (dir == CUFFT_INVERSE) {
    if (r->isFloat()) {
      Complex scale;
      scale.x = (float) 1.0 / (p.getNumel()/batch);
      scale.y = 0.0;

      GPUtype *scaleGPU = new GPUtype(scale, r->getGPUmanager());
      mgc.setPtr(scaleGPU);
      GPUopTimes(*r, *scaleGPU, *r);
    } else if (r->isDouble()) {
      DoubleComplex scale;
      scale.x = (double) 1.0 / (p.getNumel()/batch);
      scale.y = 0.0;

      GPUtype *scaleGPU = new GPUtype(scale, r->getGPUmanager());
      mgc.setPtr(scaleGPU);
      GPUopTimes(*r, *scaleGPU, *r);
    }
  }

  // if real I have to copy the symmetric coefficients
  // The following code make sense only for FFT (CUFFT_FORWARD)
  if ((!p.isComplex())&&(dir = CUFFT_FORWARD)) {
    if ((batch > 1)||(dim!=1)) {
      GPUtype *q = new GPUtype(p, 1);
      q->setComplex();
      GPUopAllocVector(*q);

      mgc.setPtr(q);

      // copy r into q
      cudaError_t cudastatus = cudaSuccess;
      // width and height refer to src
      int *rsize = r->getSize();
      int *qsize = q->getSize();

      int rdim = r->getNdims();
      int tdim = 1;
      if (rdim == 3)
        tdim = rsize[2];

      // define parameters for 2d mem copy

      int width = rsize[0] * r->getMySize();
      int height = rsize[1];

      int spitch = rsize[0] * r->getMySize();
      int dpitch = qsize[0] * q->getMySize();

      // loop on 3d dimension
      for (int i = 0; i < tdim; i++) {

        uintptr_t dst = (UINTPTR q->getGPUptr()) + i * qsize[0] * qsize[1]
            * q->getMySize();
        uintptr_t src = (UINTPTR r->getGPUptr()) + i * rsize[0] * rsize[1]
            * r->getMySize();

        cudastatus = cudaMemcpy2D((void*) dst, dpitch, (void*) src, spitch,
            width, height, cudaMemcpyDeviceToDevice);
        if (cudastatus != cudaSuccess) {
          throw GPUexception(GPUmatError, ERROR_FFT_MEMCOPY);
        }
      }

      // must also unregister
      mgc.remPtr(r);
      delete r;

      if (dim==1) {
        GPUopFFTSymm(*q, 1);
      } else if (dim==2) {
        if (batch>1)
          GPUopFFTSymm(*q, 2);
        else
          GPUopFFTSymm(*q, 0);
      } else {
        GPUopFFTSymm(*q, 0);
      }

      mgc.remPtr(q);
      return q;
    } else {
      GPUopFFTSymm(*r, 0);

      mgc.remPtr(r);
      return r;
    }

  } else {

    mgc.remPtr(r);
    return r;
  }

}
#undef CUPLAN1DMAXEL

/*************************************************************************
 * GPUopFFT
 *************************************************************************/
GPUmatResult_t GPUopFFT(GPUtype &p, GPUtype &r, int direction, int batch, int dim) {


  GPUmatResult_t status = GPUmatSuccess;
  GPUmanager * GPUman = p.getGPUmanager();



  cufftType_t fftType;

  if (p.isFloat()) {
    fftType = CUFFT_R2C;
  } else if (p.isDouble()) {
#ifndef NODOUBLE
    fftType = CUFFT_D2Z;
#endif
  }

  if (p.isComplex()) {
    if (p.isFloat()) {
      fftType = CUFFT_C2C;
    } else if (p.isDouble()) {
#ifndef NODOUBLE
      fftType = CUFFT_Z2Z;
#endif
    }
  }

  /*% create plan
   % For higher-dimensional transforms (2D and 3D), CUFFT performs
   % FFTs in row-major or C order. For example, if the user requests a 3D
   % transform plan for sizes X, Y, and Z, CUFFT transforms along Z, Y, and
   % then X. The user can configure column-major FFTs by simply changing
   % the order of the size parameters to the plan creation API functions*/

  // batch mode in two just perform a loop and transforms
  cufftHandle plan;
  cufftResult_t cufftstatus;
  int *psize = p.getSize();
  int *rsize = r.getSize();

  if (dim==1) {
    cufftstatus = cufftPlan1d(&plan, p.getNumel() / batch, fftType, batch);
  } else if(dim==2) {
    cufftstatus = cufftPlan2d(&plan, psize[1], psize[0], fftType);
  } else if (dim==3) {
    cufftstatus = cufftPlan3d(&plan, psize[2], psize[1], psize[0], fftType);
  } else {
    throw GPUexception(GPUmatError, "Wrong dim argument.");
  }

  // try to free cache if I had problems with planning

  if (cufftstatus != CUFFT_SUCCESS) {
    GPUman->cacheClean();

    if (dim==1) {
      cufftstatus = cufftPlan1d(&plan, p.getNumel() / batch, fftType, batch);
    } else if(dim==2) {
      cufftstatus = cufftPlan2d(&plan, psize[1], psize[0], fftType);
    } else if (dim==3) {
      cufftstatus = cufftPlan3d(&plan, psize[2], psize[1], psize[0], fftType);
    } else {
      throw GPUexception(GPUmatError, "Wrong dim argument.");
    }
  }

  if (cufftstatus != CUFFT_SUCCESS) {
    throw GPUexception(GPUmatError, "Error in cufftPlan.");
  }

  // there is a possibility that we didn't allocate yet r
  // we decided to allocate after the planning which is really consuming
  // resources
  if (r.getGPUptr()==NULL) {
    try {
      GPUopAllocVector(r);
    } catch (GPUexception ex) {
      cufftstatus = cufftDestroy(plan);
      char buffer[300];
      sprintf(
          buffer,
          "Error allocating result for the FFT (%s)", ex.getError());
      throw GPUexception(GPUmatError, buffer);
    }
  }
  if (p.isComplex()) {
    if (dim==1) {
      if (p.isFloat()) {
      // single precision
        cufftstatus = cufftExecC2C(plan, (cufftComplex*) p.getGPUptr(),
          (cufftComplex*) r.getGPUptr(), direction);
      if (cufftstatus != CUFFT_SUCCESS) {
        cufftstatus = cufftExecC2C(plan, (cufftComplex*) p.getGPUptr(),
          (cufftComplex*) r.getGPUptr(), direction);
      }

      // double precision
      } else if (p.isDouble()) {
#ifndef NODOUBLE
        cufftstatus = cufftExecZ2Z(plan, (cufftDoubleComplex*) p.getGPUptr(),
          (cufftDoubleComplex*) r.getGPUptr(), direction);
        if (cufftstatus != CUFFT_SUCCESS) {
          cufftstatus = cufftExecZ2Z(plan, (cufftDoubleComplex*) p.getGPUptr(),
            (cufftDoubleComplex*) r.getGPUptr(), direction);
        }
#endif
      }

    } else {
      for (int i=0;i<batch;i++) {

        if (p.isFloat()) {
          cufftComplex * src = (cufftComplex*) p.getGPUptr();
          cufftComplex * dst = (cufftComplex*) r.getGPUptr();
          // single precision
          cufftstatus = cufftExecC2C(plan, src+i* psize[1]*psize[0],
                dst + i* rsize[1]*rsize[0], direction);
          if (cufftstatus != CUFFT_SUCCESS) {
            cufftstatus = cufftExecC2C(plan, src+i* psize[1]*psize[0],
                dst + i* rsize[1]*rsize[0], direction);
          }

        } else if (p.isDouble()) {
#ifndef NODOUBLE
          cufftDoubleComplex * src = (cufftDoubleComplex*) p.getGPUptr();
          cufftDoubleComplex * dst = (cufftDoubleComplex*) r.getGPUptr();
          // double precision
          cufftstatus = cufftExecZ2Z(plan, src+i* psize[1]*psize[0],
                dst + i* rsize[1]*rsize[0], direction);
          if (cufftstatus != CUFFT_SUCCESS) {
            cufftstatus = cufftExecZ2Z(plan, src+i* psize[1]*psize[0],
                dst + i* rsize[1]*rsize[0], direction);
          }
#endif
        }
      }
    }
  } else {
    if (dim==1) {
      if (p.isFloat()) {
        cufftstatus = cufftExecR2C(plan, (cufftReal*) p.getGPUptr(),
          (cufftComplex*) r.getGPUptr());
        if (cufftstatus != CUFFT_SUCCESS) {
          cufftstatus = cufftExecR2C(plan, (cufftReal*) p.getGPUptr(),
          (cufftComplex*) r.getGPUptr());
        }

      } else if (p.isDouble()) {
#ifndef NODOUBLE
        cufftstatus = cufftExecD2Z(plan, (cufftDoubleReal*) p.getGPUptr(),
          (cufftDoubleComplex*) r.getGPUptr());
        if (cufftstatus != CUFFT_SUCCESS) {
          cufftstatus = cufftExecD2Z(plan, (cufftDoubleReal*) p.getGPUptr(),
            (cufftDoubleComplex*) r.getGPUptr());
        }
#endif
      }
    } else {
      for (int i=0;i<batch;i++) {
        if (p.isFloat()) {
          // single precision
          cufftReal * src = (cufftReal*) p.getGPUptr();
          cufftComplex * dst = (cufftComplex*) r.getGPUptr();
          cufftstatus = cufftExecR2C(plan, src+i* psize[1]*psize[0],
                    dst+ i* rsize[1]*rsize[0]);
          if (cufftstatus != CUFFT_SUCCESS) {
            cufftstatus = cufftExecR2C(plan, src+i* psize[1]*psize[0],
                    dst+ i* rsize[1]*rsize[0]);
          }

        } else if (p.isDouble()) {
#ifndef NODOUBLE
          // double precision
          cufftDoubleReal * src = (cufftDoubleReal*) p.getGPUptr();
          cufftDoubleComplex * dst = (cufftDoubleComplex*) r.getGPUptr();
          cufftstatus = cufftExecD2Z(plan, src+i* psize[1]*psize[0],
                    dst+ i* rsize[1]*rsize[0]);
          if (cufftstatus != CUFFT_SUCCESS) {
            cufftstatus = cufftExecD2Z(plan, src+i* psize[1]*psize[0],
                    dst+ i* rsize[1]*rsize[0]);
          }
#endif
        }
      }
    }
  }
  if (cufftstatus != CUFFT_SUCCESS) {
    cufftstatus = cufftDestroy(plan);
    throw GPUexception(GPUmatError, "Error in cufftExec.");
  }

  cufftstatus = cufftDestroy(plan);
  if (cufftstatus != CUFFT_SUCCESS) {
    throw GPUexception(GPUmatError, "Error in cufftDestroy.");
  }

  return status;

}

/*************************************************************************
 * GPUopZerosDrv
 * GPUopZeros
 *************************************************************************/
CGPUOP3(Zeros)
CGPUOP4(Zeros, ZEROS)

/*************************************************************************
 * GPUopOnesDrv
 * GPUopOnes
 *************************************************************************/
CGPUOP3(Ones)
CGPUOP4(Ones, ONES)

/*************************************************************************
 * GPUopColonDrv
 * GPUopColon
 *************************************************************************/
// Colon. Equivalent to the Matlab colon :
// J:K  is the same as [J, J+1, ..., K].
// J:K  is empty if J > K.
// J:D:K  is the same as [J, J+D, ..., J+m*D] where m = fix((K-J)/D).
// J:D:K  is empty if D == 0, if D > 0 and J > K, or if D < 0 and J < K.
// Can have two forms, with 2 or 3 arguments
// GPUtype p is used to clone and create r

//GPUtype * GPUopColonDrv(double j, double k, double d, GPUtype &p) {
GPUtype * GPUopColonDrv(double j, double k, double d, GPUtype &p) {

  GPUmatResult_t status = GPUmatSuccess;
  GPUtype *r;
  GPUmanager *GPUman = p.getGPUmanager();

  // garbage collector
  MyGCObj<GPUtype> mgc;

  r = new GPUtype(p, 1);
  mgc.setPtr(r);




  //double d = 1.0;
  if ((d == 0) || (d > 0 && j > k) || (d < 0 && j < k)) {
    // empty result
  } else {
    // trick here to avoid precision problems
    double m = 0.0;
    /*for (int uu=1;uu<10;uu++) {
      double off = uu*10.0;
      double mtemp = floor(((k*off) - (j*off)) / (d*off));
      if (mtemp>m)
        m=mtemp;
    }*/
    double eps = 1e-12;
    m = floor(((k+eps) - j) / d);

    if (m < 0)
      m = 0;

    int mysize[2];
    mysize[0] = 1;
    mysize[1] = (int) (m + 1);
    r->setSize(2, mysize);

    GPUopAllocVector(*r);
    GPUopColon(j, d, *r);
    //GPUopFillVectorf((int) j, (float) d, *r);

  }


  mgc.remPtr(r);
  return r;
}

GPUmatResult_t GPUopColon(double j, double d, GPUtype &r) {
  GPUmatResult_t status = GPUmatSuccess;
  cudaError_t cudastatus = cudaSuccess;
  GPUmanager * GPUman = r.getGPUmanager();
  GPUopFillVector( j, d, r);

  return status;
}


/* CASTING */

/*************************************************************************
 * GPUopFloatToDoubleDrv
 *************************************************************************/
GPUtype * GPUopFloatToDoubleDrv(GPUtype &p) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUtype *r;
  GPUmanager *GPUman = p.getGPUmanager();



  if ((p.getType() == gpuFLOAT) || (p.getType() == gpuCFLOAT)) {
  } else {
    throw GPUexception(GPUmatError, ERROR_CAST_WRONG_ARG);
  }

  // allocate temp for results
  r = p.FLOATtoDOUBLE();

  return r;
}

/*************************************************************************
 * GPUopFloatToDouble
 *************************************************************************/

GPUmatResult_t GPUopFloatToDouble(GPUtype &p, GPUtype &r) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUmanager * GPUman = p.getGPUmanager();

  CUDALIBResult cudalibres = CUDALIBSuccess;

  int n = 0;
  // should change type
  if (!r.isComplex()) {
    n = r.getNumel();
  } else {
    n = r.getNumel()*GPU_SIZE_OF_CDOUBLE/GPU_SIZE_OF_DOUBLE;
  }

  void *op1Ptr;
  void *op2Ptr;

  unsigned int p1size;
  unsigned int p2size;

  op1Ptr = p.getGPUptrptr();
  op2Ptr = r.getGPUptrptr();
  p1size = sizeof(CUdeviceptr);
  p2size = sizeof(CUdeviceptr);

  CUfunction *drvfun = GPUman->getCuFunction(N_FLOAT_TO_DOUBLE_KERNEL);
  // define kernel configuration
  gpukernelconfig_t * kconf = GPUman->getKernelConfig();
  hostdrv_pars_t pars[2];
  int nrhs = 2;

  pars[0].par =  op1Ptr;
  pars[0].psize = p1size;
  pars[0].align = __alignof(CUdeviceptr);

  pars[1].par = op2Ptr;
  pars[1].psize = p2size;
  pars[1].align = __alignof(CUdeviceptr);

  cudalibres = mat_HOSTDRV_A(n, kconf, nrhs, pars, drvfun);

  if (cudalibres != CUDALIBSuccess) {
    throw GPUexception(GPUmatError, "Kernel execution error.");
  }

  return status;
}

/*************************************************************************
 * GPUopDoubleToFloatDrv
 *************************************************************************/
GPUtype * GPUopDoubleToFloatDrv(GPUtype &p) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUtype *r;
  GPUmanager *GPUman = p.getGPUmanager();

  if ((p.getType() == gpuDOUBLE) || (p.getType() == gpuCDOUBLE)) {
  } else {
    throw GPUexception(GPUmatError, ERROR_CAST_WRONG_ARG);
  }

  // allocate temp for results
  r = p.DOUBLEtoFLOAT();

  return r;
}

/*************************************************************************
 * GPUopDoubleToFloat
 *************************************************************************/

GPUmatResult_t GPUopDoubleToFloat(GPUtype &p, GPUtype &r) {
  GPUmatResult_t status = GPUmatSuccess;
  GPUmanager * GPUman = p.getGPUmanager();

  CUDALIBResult cudalibres = CUDALIBSuccess;

  int n = 0;
  // should change type
  if (!r.isComplex()) {
    n = r.getNumel();
  } else {
    n = r.getNumel()*GPU_SIZE_OF_CDOUBLE/GPU_SIZE_OF_DOUBLE;
  }

  void *op1Ptr;
  void *op2Ptr;

  unsigned int p1size;
  unsigned int p2size;

  op1Ptr = p.getGPUptrptr();
  op2Ptr = r.getGPUptrptr();
  p1size = sizeof(CUdeviceptr);
  p2size = sizeof(CUdeviceptr);

  CUfunction *drvfun = GPUman->getCuFunction(N_DOUBLE_TO_FLOAT_KERNEL);
  // define kernel configuration
  gpukernelconfig_t * kconf = GPUman->getKernelConfig();
  hostdrv_pars_t pars[2];
  int nrhs = 2;

  pars[0].par =  op1Ptr;
  pars[0].psize = p1size;
  pars[0].align = __alignof(CUdeviceptr);

  pars[1].par = op2Ptr;
  pars[1].psize = p2size;
  pars[1].align = __alignof(CUdeviceptr);


  cudalibres = mat_HOSTDRV_A(n, kconf, nrhs, pars, drvfun);

  if (cudalibres != CUDALIBSuccess) {
    throw GPUexception(GPUmatError, "Kernel execution error.");
  }

  return status;
}



