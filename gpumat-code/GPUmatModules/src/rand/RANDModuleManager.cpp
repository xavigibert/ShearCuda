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
#include <time.h>

#ifdef UNIX
#include <stdint.h>
#endif

#include "mex.h"

// CUDA
#include "cuda.h"
#include "curand.h"

#include "GPUmat.hh"
#include "rand.hh"

// static paramaters
static int init = 0;
static GPUmat *gm;
curandGenerator_t gen;

time_t seed;

#define MAXNUM 10000

#define BUFFERSIZE 300
#define CLEARBUFFER memset(buffer,0,BUFFERSIZE);
char buffer[BUFFERSIZE];

//*******************************************************************
// RANDOM functions
//*******************************************************************
/* GPUrand */
void GPUrand(const GPUtype &OUT) {

  curandStatus_t status;

  gpuTYPE_t type = gm->gputype.getType(OUT);

  gm->gmat->control.cacheClean();


  const void *gpuptr = gm->gputype.getGPUptr(OUT); // pointer to GPU memory
  int numel = gm->gputype.getNumel(OUT);           // number of elements
  int datasize = gm->gputype.getDataSize(OUT);     // bytes for each element


  gen = 0;
  // implement recovery procedure
  // try and if error try again

  // init curand
  if (curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT)!=CURAND_STATUS_SUCCESS) {
    mexErrMsgTxt(ERROR_CURAND_INIT);
  }
  //if (curandCreateGenerator(&gen,CURAND_RNG_QUASI_DEFAULT)!=CURAND_STATUS_SUCCESS) {
  //  mexErrMsgTxt(ERROR_CURAND_INIT);
  //}

  // seed
  seed++;
  if (curandSetPseudoRandomGeneratorSeed(gen, time(NULL)+seed)!=CURAND_STATUS_SUCCESS) {
    mexErrMsgTxt(ERROR_CURAND_SEED);
  }

  if (type == gpuFLOAT) {
    status = curandGenerateUniform(gen, (float *) gpuptr, numel);
  } else if (type == gpuCFLOAT) {
    status = curandGenerateUniform(gen, (float *) gpuptr, numel*2);
  } else if (type == gpuDOUBLE) {
    status = curandGenerateUniformDouble(gen, (double *) gpuptr, numel);
  } else if (type == gpuCDOUBLE) {
    status = curandGenerateUniformDouble(gen, (double *) gpuptr, numel*2);
  }

  if (status!=CURAND_STATUS_SUCCESS) {
    curandDestroyGenerator(gen);
    mexErrMsgTxt(ERROR_CURAND_GEN);
  }


  // destroy
  if (curandDestroyGenerator(gen)!=CURAND_STATUS_SUCCESS) {
    mexErrMsgTxt(ERROR_CURAND_DESTROY);
  }


}

/* GPUrandn */
void GPUrandn(const GPUtype &OUT) {

  curandStatus_t status;

  gpuTYPE_t type = gm->gputype.getType(OUT);


  gm->gmat->control.cacheClean();



  const void *gpuptr = gm->gputype.getGPUptr(OUT); // pointer to GPU memory
  int numel = gm->gputype.getNumel(OUT);           // number of elements
  int datasize = gm->gputype.getDataSize(OUT);     // bytes for each element


  gen = 0;
  // implement recovery procedure
  // try and if error try again

  // init curand
  if (curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT)!=CURAND_STATUS_SUCCESS) {
    mexErrMsgTxt(ERROR_CURAND_INIT);
  }
  //if (curandCreateGenerator(&gen,CURAND_RNG_QUASI_DEFAULT)!=CURAND_STATUS_SUCCESS) {
  //  mexErrMsgTxt(ERROR_CURAND_INIT);
  //}

  // randn requires even numbers
  // we split the execution in 2 parts (overlap if not even)

  // seed
  seed++;
  if (curandSetPseudoRandomGeneratorSeed(gen, time(NULL)+seed)!=CURAND_STATUS_SUCCESS) {
    mexErrMsgTxt(ERROR_CURAND_SEED);
  }

  unsigned int n = 0;

  if (type == gpuFLOAT) {
    n = numel;
  } else if (type == gpuCFLOAT) {
    n = numel*2;
  } else if (type == gpuDOUBLE) {
    n = numel;
  } else if (type == gpuCDOUBLE) {
    n = numel*2;
  }

  unsigned int even = (n%2) == 0;

  unsigned int offset = 0;
  unsigned int mysize = 0;


  unsigned int iter = 1;
  if (!even) {
    n = n-1;
    iter = 2;
  }

  if (type == gpuFLOAT) {
    float mean = 0.0;
    float std = 1.0;
    status = curandGenerateNormal(gen, (float *) gpuptr, n, mean, std);

    if (!even) {
      float *devData;
      if((cudaMalloc((void **)&devData, 4 * sizeof(float))) != cudaSuccess) {
        status = CURAND_STATUS_LAUNCH_FAILURE;
      } else {
        status = curandGenerateNormal(gen, devData, 4, mean, std);
        if (status==CURAND_STATUS_SUCCESS) {
          void *dst = (void *) ((UINTPTR gpuptr)+n*datasize);
          if (cudaMemcpy(dst, (void *) devData, datasize, cudaMemcpyDeviceToDevice)!=cudaSuccess) {
            status = CURAND_STATUS_LAUNCH_FAILURE;
          }
        }
        if(cudaFree(devData) != cudaSuccess) {
          status = CURAND_STATUS_LAUNCH_FAILURE;
        }
      }
    }

  } else if (type == gpuCFLOAT) {
    float mean = 0.0;
    float std = 1.0;
    status = curandGenerateNormal(gen, (float *) gpuptr, n, mean, std);
  } else if (type == gpuDOUBLE) {
    double mean = 0.0;
    double std = 1.0;
    status = curandGenerateNormalDouble(gen, (double *) gpuptr, n, mean, std);
    if (!even) {
      double *devData;
      if((cudaMalloc((void **)&devData, 4 * sizeof(double))) != cudaSuccess) {
        status = CURAND_STATUS_LAUNCH_FAILURE;
      } else {
        status = curandGenerateNormalDouble(gen, devData, 4, mean, std);
        if (status==CURAND_STATUS_SUCCESS) {
          void *dst = (void *) ((UINTPTR gpuptr)+n*datasize);
          if (cudaMemcpy(dst, (void *) devData, datasize, cudaMemcpyDeviceToDevice)!=cudaSuccess) {
            status = CURAND_STATUS_LAUNCH_FAILURE;
          }
        }
        if(cudaFree(devData) != cudaSuccess) {
          status = CURAND_STATUS_LAUNCH_FAILURE;
        }
      }
    }


  } else if (type == gpuCDOUBLE) {
    double mean = 0.0;
    double std = 1.0;
    status = curandGenerateNormalDouble(gen, (double *) gpuptr, n, mean, std);

  }

  if (status!=CURAND_STATUS_SUCCESS) {
    curandDestroyGenerator(gen);
    mexErrMsgTxt(ERROR_CURAND_GEN);
  }


  // destroy
  if (curandDestroyGenerator(gen)!=CURAND_STATUS_SUCCESS) {
    mexErrMsgTxt(ERROR_CURAND_DESTROY);
  }


}

/// rand driver function
GPUtype GPUmxRandDrv(const GPUtype &IN, int nrhs, const mxArray *prhs[]) {
  gpuTYPE_t tin = gm->gputype.getType(IN);

  // result is created anyway
  GPUtype R = gm->gputype.createMx(tin, nrhs, prhs);
  GPUrand(R);
  return R;
}

/// randn driver function
GPUtype GPUmxRandnDrv(const GPUtype &IN, int nrhs, const mxArray *prhs[]) {
  gpuTYPE_t tin = gm->gputype.getType(IN);

  // result is created anyway
  GPUtype R = gm->gputype.createMx(tin, nrhs, prhs);
  GPUrandn(R);
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
    // nothing to load

    //************************************************************************
    // INIT
    //************************************************************************

    seed = 0;
    // init curand
    //if (curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT)!=CURAND_STATUS_SUCCESS) {
    //  mexErrMsgTxt(ERROR_CURAND_INIT);
    //}


    //************************************************************************
    // REGISTER FUNCTION IN GPUMAT STRUCTURE
    //************************************************************************

    // put here functions to be registered
    gm->rand.rand      = GPUrand;
    gm->rand.mxRandDrv = GPUmxRandDrv;

    gm->rand.randn      = GPUrandn;
    gm->rand.mxRandnDrv = GPUmxRandnDrv;


    //************************************************************************
    // UPDATE FLAGS
    //************************************************************************
    gm->mod.rand = 1; // rand module was loaded


    init = 1;
  }

}
