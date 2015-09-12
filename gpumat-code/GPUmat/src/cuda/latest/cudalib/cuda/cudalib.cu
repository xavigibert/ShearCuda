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


#ifndef _CUDALIB_KERNEL_H_
#define _CUDALIB_KERNEL_H_


#include <stdio.h>
#include "cudalib_common.h"
#include "kernelnames.h"

/// TEXTURES
// Texture declaration must go here
 
texture<float, 1>          texref_f1_a;
texture<float, 1>          texref_f1_b;
texture<Complex, 1>        texref_c1_a;
texture<Complex, 1>        texref_c1_b;
texture<int2, 1>           texref_d1_a;
texture<int2, 1>           texref_d1_b;
texture<int4, 1>           texref_cd1_a;
texture<int4, 1>           texref_cd1_b;
texture<int, 1>            texref_i1_a;
//texture<int, 1>            texref_i1_b;


texture<float, 2>          texref_f2_a;
texture<Complex, 2>        texref_c2_a;
texture<int2, 2>           texref_d2_a;
texture<DoubleComplex, 2>  texref_cd2_a;
texture<int, 2>            texref_i2_a;

// texture aliases
#define TEXREF_float_A texref_f1_a
#define TEXREF_float_B texref_f1_b

#define TEXREF_Complex_A texref_c1_a
#define TEXREF_Complex_B texref_c1_b

#define TEXREF_double_A texref_d1_a
#define TEXREF_double_B texref_d1_b

#define TEXREF_DoubleComplex_A texref_cd1_a
#define TEXREF_DoubleComplex_B texref_cd1_b

#define TEXREF_int_A texref_i1_a
#define TEXREF_int_B texref_i1_b


#define tex1Dfetch_float   tex1Dfetch
#define tex1Dfetch_Complex tex1Dfetch

static __inline__ __device__ double tex1Dfetch_double(texture<int2, 1> t, int i)
{
    int2 v = tex1Dfetch(t,i);
#if !defined(CUDA_NO_SM_13_DOUBLE_INTRINSICS)    
    return __hiloint2double(v.y, v.x);
#else
    return 0.0;
#endif
}

static __inline__ __device__ DoubleComplex tex1Dfetch_DoubleComplex(texture<int4, 1> t, int i)
{
    int4 v = tex1Dfetch(t,i);
#if !defined(CUDA_NO_SM_13_DOUBLE_INTRINSICS)    
    return make_double2(__hiloint2double(v.y, v.x),__hiloint2double(v.w, v.z));
#else
    return make_double2(0.0,0.0);
#endif    
}

/* Casting */
#define todouble toDouble
#define tofloat  toFloat

// toDouble
__inline__ __device__ double toDouble(float x) {
  return (double)x;
}
__inline__ __device__ double toDouble(double x) {
  return x;
}
__inline__ __device__ double toDouble(int x) {
  return (double)x;
}

// toFloat
__inline__ __device__ float toFloat(float x) {
  return x;
}
__inline__ __device__ float toFloat(double x) {
  return (float)x;
}
__inline__ __device__ float toFloat(int x) {
  return (float)x;
}

// toComplex
__inline__ __device__ Complex toComplex(float x) {
  return make_float2(x,0.0);
}
__inline__ __device__ Complex toComplex(Complex x) {
  return x;
}
__inline__ __device__ Complex toComplex(double x) {
  return make_float2((float)x,0.0);
}
__inline__ __device__ Complex toComplex(DoubleComplex x) {
  return make_float2((float)x.x,(float)x.y);
}

// toDoubleComplex
__inline__ __device__ DoubleComplex toDoubleComplex(float x) {
  return make_double2((double)x,0.0);
}
__inline__ __device__ DoubleComplex toDoubleComplex(Complex x) {
  return make_double2((double)x.x,(double)x.y);
}
__inline__ __device__ DoubleComplex toDoubleComplex(double x) {
  return make_double2(x,0.0);
}
__inline__ __device__ DoubleComplex toDoubleComplex(DoubleComplex x) {
  return x;
}

#if __DEVICE_EMULATION__
// Interface for bank conflict checker
#define CUT_BANK_CHECKER( array, index)                                      \
    (cutCheckBankAccess( threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x,  \
    blockDim.y, blockDim.z,                                                  \
    __FILE__, __LINE__, #array, index ),                                     \
    array[index])
#else
#define CUT_BANK_CHECKER( array, index)  array[index]
#endif




#define GEN_KERNEL_1D_IN1(KERNAME, INARG1, OUTARG1, FUNNAME) \
extern "C" {__global__ void KERNAME (int n,  int offset, INARG1 idata1, OUTARG1 odata)  \
{  \
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset; \
    if ((xIndex - offset) < n) \
        odata[xIndex] = FUNNAME((idata1[xIndex])); \
}\
}\

/* The following kernel allows to perform operations also on scalars. If i1
 * or i2 is > 0, then this value is taken  as index. Doing this way it is possible
 * to do operations between arrays and scalars
 */

#define GEN_KERNEL_1D_IN1_IN2(KERNAME, INARG1, INARG2, OUTARG1, FUNNAME) \
extern "C" {__global__ void KERNAME (int n , int offset, int i1, int i2, OUTARG1 *odata)\
{\
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;\
    int xIndex1 = (i1<0)?xIndex:i1; \
    int xIndex2 = (i2<0)?xIndex:i2; \
    if ((xIndex - offset) < n) {\
        INARG1 d1 = tex1Dfetch_##INARG1 (TEXREF_##INARG1##_A, xIndex1);\
        INARG2 d2 = tex1Dfetch_##INARG2 (TEXREF_##INARG2##_B, xIndex2);\
        odata[xIndex] = ##FUNNAME (d1,d2);\
    }\
}\
}\

#define GEN1_KERNEL_1D_IN1_IN2(KERNAME, INARG1, INARG2, OUTARG1, FUNNAME, FUNNAMEINARG) \
extern "C" {__global__ void KERNAME (int n , int offset, int i1, int i2, OUTARG1 *odata)\
{\
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;\
    int xIndex1 = (i1<0)?xIndex:i1; \
    int xIndex2 = (i2<0)?xIndex:i2; \
    if ((xIndex - offset) < n) {\
        INARG1 d1 = tex1Dfetch_##INARG1 (TEXREF_##INARG1##_A, xIndex1);\
        INARG2 d2 = tex1Dfetch_##INARG2 (TEXREF_##INARG2##_B, xIndex2);\
        odata[xIndex] = to##OUTARG1 ( FUNNAME (to##FUNNAMEINARG (d1),to##FUNNAMEINARG (d2)));\
    }\
}\
}\



/*#define GEN_KERNEL_1D_IN1_IN2_ALL(KERNAME, kername)   \
GEN_KERNEL_1D_IN1_IN2( KERNAME##F_KERNEL, float , float , float , kername##f);\
GEN_KERNEL_1D_IN1_IN2( KERNAME##C_KERNEL, Complex , Complex , Complex , kername##c);\
GEN_KERNEL_1D_IN1_IN2( KERNAME##D_KERNEL, double , double , double , kername##d);\
GEN_KERNEL_1D_IN1_IN2( KERNAME##CD_KERNEL, DoubleComplex , DoubleComplex , DoubleComplex , kername##cd);\

#define GEN_KERNEL_1D_IN1_IN2_REAL(KERNAME, kername)   \
GEN_KERNEL_1D_IN1_IN2( KERNAME##F_KERNEL, float , float , float , kername##f);\
GEN_KERNEL_1D_IN1_IN2( KERNAME##D_KERNEL, double , double , double , kername##d);\*/

#define GEN1_KERNEL_1D_IN1_IN2_ALL(KERNAME, kername)   \
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_F_F_KERNEL, float  , float   , float   , kername , float);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_F_C_KERNEL, float  , Complex , Complex , kername , Complex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_F_D_KERNEL, float  , double  , float   , kername , double);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_F_CD_KERNEL, float , DoubleComplex  , Complex  , kername , DoubleComplex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_C_F_KERNEL, Complex, float   , Complex , kername , Complex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_C_C_KERNEL, Complex , Complex , Complex , kername , Complex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_C_D_KERNEL, Complex  , double  , Complex   , kername , DoubleComplex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_C_CD_KERNEL, Complex       , DoubleComplex  , Complex   , kername , DoubleComplex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_D_F_KERNEL, double , float   , float   , kername , double);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_D_C_KERNEL, double   , Complex , Complex   , kername , DoubleComplex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_D_D_KERNEL, double  , double  , double  , kername , double);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_D_CD_KERNEL,  double        , DoubleComplex , DoubleComplex , kername , DoubleComplex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_CD_F_KERNEL, DoubleComplex  , float          , Complex  , kername , DoubleComplex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_CD_C_KERNEL, DoubleComplex , Complex        , Complex   , kername , DoubleComplex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_CD_D_KERNEL,  DoubleComplex , double        , DoubleComplex , kername , DoubleComplex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_CD_CD_KERNEL, DoubleComplex , DoubleComplex , DoubleComplex , kername , DoubleComplex);\

#define GEN1_KERNEL_1D_IN1_IN2_REAL(KERNAME, kername)   \
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_F_F_KERNEL, float   , float   , float   , kername , float);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_F_D_KERNEL, float  , double  , float   , kername , double);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_D_F_KERNEL, double , float   , float   , kername , double);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_D_D_KERNEL, double  , double  , double  , kername , double);\

// for some kernels the returned type is always REAL
#define GEN1_KERNEL_1D_IN1_IN2_OUTFLOAT_ALL(KERNAME, kername)   \
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_F_F_KERNEL, float  , float   , float   , kername , float);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_F_C_KERNEL, float  , Complex , float , kername , Complex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_F_D_KERNEL, float  , double  , float   , kername , double);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_F_CD_KERNEL, float , DoubleComplex  , float  , kername , DoubleComplex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_C_F_KERNEL, Complex, float   , float , kername , Complex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_C_C_KERNEL, Complex , Complex , float , kername , Complex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_C_D_KERNEL, Complex  , double  , float   , kername , DoubleComplex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_C_CD_KERNEL, Complex       , DoubleComplex  , float   , kername , DoubleComplex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_D_F_KERNEL, double , float   , float   , kername , double);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_D_C_KERNEL, double   , Complex , float   , kername , DoubleComplex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_D_D_KERNEL, double  , double  , float  , kername , double);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_D_CD_KERNEL,  double        , DoubleComplex , float , kername , DoubleComplex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_CD_F_KERNEL, DoubleComplex  , float          , float  , kername , DoubleComplex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_CD_C_KERNEL, DoubleComplex , Complex        , float   , kername , DoubleComplex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_CD_D_KERNEL,  DoubleComplex , double        , float , kername , DoubleComplex);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_CD_CD_KERNEL, DoubleComplex , DoubleComplex , float , kername , DoubleComplex);\

#define GEN1_KERNEL_1D_IN1_IN2_OUTFLOAT_REAL(KERNAME, kername)   \
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_F_F_KERNEL, float  , float   , float   , kername , float);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_F_D_KERNEL, float  , double  , float   , kername , double);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_D_F_KERNEL, double , float   , float   , kername , double);\
GEN1_KERNEL_1D_IN1_IN2( KERNAME##_D_D_KERNEL, double  , double  , float  , kername , double);\
//extern "C" {

/* transposef_kernel */
extern "C" {__global__ void TRANSPOSEF_KERNEL(float *odata, float *idata, int width, int height)
{
  __shared__ float block[BLOCK_DIM2D][BLOCK_DIM2D+1];

  // read the matrix tile into shared memory
  unsigned int xIndex = blockIdx.x * BLOCK_DIM2D + threadIdx.x;
  unsigned int yIndex = blockIdx.y * BLOCK_DIM2D + threadIdx.y;
  if ((xIndex < width) && (yIndex < height))
  {
    unsigned int index_in = yIndex * width + xIndex;
    block[threadIdx.y][threadIdx.x] = idata[index_in];
  }

  __syncthreads();

  // write the transposed matrix tile to global memory
  xIndex = blockIdx.y * BLOCK_DIM2D + threadIdx.x;
  yIndex = blockIdx.x * BLOCK_DIM2D + threadIdx.y;
  if ((xIndex < height) && (yIndex < width))
  {
    unsigned int index_out = yIndex * height + xIndex;
    odata[index_out] = block[threadIdx.x][threadIdx.y];
  }
}

/* transpose_kernel */
__global__ void TRANSPOSEF_TEX_KERNEL(float *odata, int width, int height, int offsetx, int offsety)
{
  __shared__ float block[BLOCK_DIM2D][BLOCK_DIM2D+1];

  // read the matrix tile into shared memory
  unsigned int xIndex = (blockIdx.x + offsetx) * BLOCK_DIM2D + threadIdx.x;
  unsigned int yIndex = (blockIdx.y + offsety) * BLOCK_DIM2D + threadIdx.y;
  //unsigned int xIndex = blockIdx.x * BLOCK_DIM2D + threadIdx.x + offsetx;
  //unsigned int yIndex = blockIdx.y * BLOCK_DIM2D + threadIdx.y + offsety;
  if ((xIndex < width) && (yIndex < height))
  {
    unsigned int index_in = yIndex * width + xIndex;
    block[threadIdx.y][threadIdx.x] = tex1Dfetch(texref_f1_a,index_in);
  }

  __syncthreads();

  // write the transposed matrix tile to global memory
  xIndex = (blockIdx.y + offsety) * BLOCK_DIM2D + threadIdx.x;
  yIndex = (blockIdx.x + offsetx) * BLOCK_DIM2D + threadIdx.y;
  //xIndex = blockIdx.y * BLOCK_DIM2D + threadIdx.x + offsety;
  //yIndex = blockIdx.x * BLOCK_DIM2D + threadIdx.y + offsetx;
  if ((xIndex < height) && (yIndex < width))
  {
    unsigned int index_out = yIndex * height + xIndex;
    odata[index_out] = block[threadIdx.x][threadIdx.y];
  }
}

__global__ void TRANSPOSED_TEX_KERNEL(double *odata, int width, int height, int offsetx, int offsety)
{
  __shared__ double block[BLOCK_DIM2D][BLOCK_DIM2D+1];

  // read the matrix tile into shared memory
  unsigned int xIndex = (blockIdx.x + offsetx) * BLOCK_DIM2D + threadIdx.x;
  unsigned int yIndex = (blockIdx.y + offsety) * BLOCK_DIM2D + threadIdx.y;
  //unsigned int xIndex = blockIdx.x * BLOCK_DIM2D + threadIdx.x + offsetx;
  //unsigned int yIndex = blockIdx.y * BLOCK_DIM2D + threadIdx.y + offsety;
  if ((xIndex < width) && (yIndex < height))
  {
    unsigned int index_in = yIndex * width + xIndex;
    block[threadIdx.y][threadIdx.x] = tex1Dfetch_double(texref_d1_a,index_in);
  }

  __syncthreads();

  // write the transposed matrix tile to global memory
  xIndex = (blockIdx.y + offsety) * BLOCK_DIM2D + threadIdx.x;
  yIndex = (blockIdx.x + offsetx) * BLOCK_DIM2D + threadIdx.y;
  //xIndex = blockIdx.y * BLOCK_DIM2D + threadIdx.x + offsety;
  //yIndex = blockIdx.x * BLOCK_DIM2D + threadIdx.y + offsetx;
  if ((xIndex < height) && (yIndex < width))
  {
    unsigned int index_out = yIndex * height + xIndex;
    odata[index_out] = block[threadIdx.x][threadIdx.y];
  }
}
#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define block(i,j) CUT_BANK_CHECKER(((float*)&block[0]), ((i)*(BLOCK_DIM2D+1) + (j)))
#else
#define block(i,j) block[(i)][(j)]
#endif

__global__ void TRANSPOSEC_TEX_KERNEL(float *odata, int width, int height, int offsetx, int offsety)
{
  __shared__ float block[BLOCK_DIM2D][BLOCK_DIM2D+1];

  // read the matrix tile into shared memory
  unsigned int xIndex = (blockIdx.x + offsetx) * BLOCK_DIM2D + threadIdx.x;
  unsigned int yIndex = (blockIdx.y + offsety) * BLOCK_DIM2D_HALF + threadIdx.y;
  //unsigned int xIndex = blockIdx.x * BLOCK_DIM2D + threadIdx.x +offsetx;
  //unsigned int yIndex = blockIdx.y * BLOCK_DIM2D_HALF + threadIdx.y +offsety;
  if ((xIndex < (width*2)) && (yIndex < height))
  {
    unsigned int index_in = yIndex * (width*2) + xIndex;
    
    block(threadIdx.y*2,threadIdx.x) = tex1Dfetch(texref_f1_a,index_in);
  }

  __syncthreads();

  
  /* x c x c x c 
     .   .   .
     x c x c x c

     becomes
  
     x . x . x .
     c . c . c .
   */
    
  
  block(threadIdx.y*2+1,threadIdx.x) = block(threadIdx.y*2,threadIdx.x+1); 

  __syncthreads();

  // write the transposed matrix tile to global memory
  xIndex = (blockIdx.y + offsety) * BLOCK_DIM2D + threadIdx.x;
  yIndex = (blockIdx.x + offsetx) * BLOCK_DIM2D_HALF + threadIdx.y;
  //xIndex = blockIdx.y * BLOCK_DIM2D + threadIdx.x + offsety;
  //yIndex = blockIdx.x * BLOCK_DIM2D_HALF + threadIdx.y + offsetx;
  if ((xIndex < (height*2)) && (yIndex < width))
  {
    unsigned int index_out = yIndex * (height*2) + xIndex;
    odata[index_out] = block(threadIdx.x,threadIdx.y*2);
  }
}

__global__ void TRANSPOSECD_TEX_KERNEL(double *odata, int width, int height, int offsetx, int offsety)
{
  __shared__ double block[BLOCK_DIM2D][BLOCK_DIM2D+1];

  // read the matrix tile into shared memory
  unsigned int xIndex = (blockIdx.x + offsetx) * BLOCK_DIM2D + threadIdx.x;
  unsigned int yIndex = (blockIdx.y + offsety) * BLOCK_DIM2D_HALF + threadIdx.y;
  //unsigned int xIndex = blockIdx.x * BLOCK_DIM2D + threadIdx.x + offsetx;
  //unsigned int yIndex = blockIdx.y * BLOCK_DIM2D_HALF + threadIdx.y + offsety;
  if ((xIndex < (width*2)) && (yIndex < height))
  {
    unsigned int index_in = yIndex * (width*2) + xIndex;
    
    block(threadIdx.y*2,threadIdx.x) = tex1Dfetch_double(texref_d1_a,index_in);
  }

  __syncthreads();

  
  /* x c x c x c 
     .   .   .
     x c x c x c

     becomes
  
     x . x . x .
     c . c . c .
   */
    
  
  block(threadIdx.y*2+1,threadIdx.x) = block(threadIdx.y*2,threadIdx.x+1); 

  __syncthreads();

  // write the transposed matrix tile to global memory
  xIndex = (blockIdx.y + offsety) * BLOCK_DIM2D + threadIdx.x;
  yIndex = (blockIdx.x + offsetx) * BLOCK_DIM2D_HALF + threadIdx.y;
  //xIndex = blockIdx.y * BLOCK_DIM2D + threadIdx.x + offsety;
  //yIndex = blockIdx.x * BLOCK_DIM2D_HALF + threadIdx.y + offsetx;
  if ((xIndex < (height*2)) && (yIndex < width))
  {
    unsigned int index_out = yIndex * (height*2) + xIndex;
    odata[index_out] = block(threadIdx.x,threadIdx.y*2);
  }
}


__global__ void TRANSPOSEI_TEX_KERNEL(float *odata, int width, int height)
{
  __shared__ int block[BLOCK_DIM2D][BLOCK_DIM2D+1];

  // read the matrix tile into shared memory
  unsigned int xIndex = blockIdx.x * BLOCK_DIM2D + threadIdx.x;
  unsigned int yIndex = blockIdx.y * BLOCK_DIM2D + threadIdx.y;
  if ((xIndex < width) && (yIndex < height))
  {
    unsigned int index_in = yIndex * width + xIndex;
    block[threadIdx.y][threadIdx.x] = tex1Dfetch(texref_i1_a,index_in);
  }

  __syncthreads();

  // write the transposed matrix tile to global memory
  xIndex = blockIdx.y * BLOCK_DIM2D + threadIdx.x;
  yIndex = blockIdx.x * BLOCK_DIM2D + threadIdx.y;
  if ((xIndex < height) && (yIndex < width))
  {
    unsigned int index_out = yIndex * height + xIndex;
    odata[index_out] = block[threadIdx.x][threadIdx.y];
  }
}
}
/* NOTE:
 * For most of the functions I use intrinsic functions that are faster
 * but less accurate
 */

/* expf_kernel */
/* expc_kernel */
/* expd_kernel */
/* expcd_kernel */
__device__ inline Complex expc(Complex data1) {
  Complex c = make_float2(0.0,0.0);
  c.x = (expf(data1.x))*(cosf(data1.y));
  c.y = (expf(data1.x))*(sinf(data1.y));
  return c;
}
__device__ inline DoubleComplex expcd(DoubleComplex data1) {
  DoubleComplex c = make_double2(0.0,0.0);
  c.x = (exp(data1.x))*(cos(data1.y));
  c.y = (exp(data1.x))*(sin(data1.y));
  return c;
}

/*__global__ void EXPF_KERNEL(int n,  int offset, float *idata1, float *odata)
{  
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset; 
    if ((xIndex - offset) < n) 
        odata[xIndex] = expf((idata1[xIndex])); 
}*/  

GEN_KERNEL_1D_IN1(EXPF_KERNEL, float *, float *, expf);
GEN_KERNEL_1D_IN1(EXPC_KERNEL, Complex *, Complex *, expc);
GEN_KERNEL_1D_IN1(EXPD_KERNEL, double *, double *, exp);
GEN_KERNEL_1D_IN1(EXPCD_KERNEL, DoubleComplex *, DoubleComplex *, expcd);

/* sqrt_kernel */
GEN_KERNEL_1D_IN1(SQRTF_KERNEL, float *, float *, sqrtf);
GEN_KERNEL_1D_IN1(SQRTD_KERNEL, double *, double *, sqrt);
/* abs_kernel */
__device__ inline float fabs(Complex data1) {
  return sqrtf(data1.x*data1.x + data1.y*data1.y);
}
__device__ inline double fabs(DoubleComplex data1) {
  return sqrt(data1.x*data1.x + data1.y*data1.y);
}
__device__ inline int fabs(int data1) {
  return (data1>=0)?data1:-(data1);
}

GEN_KERNEL_1D_IN1(ABSF_KERNEL, float *, float *, fabsf);
GEN_KERNEL_1D_IN1(ABSC_KERNEL, Complex *, float *, fabs);
GEN_KERNEL_1D_IN1(ABSD_KERNEL, double *, double *, fabs);
GEN_KERNEL_1D_IN1(ABSCD_KERNEL, DoubleComplex *, double *, fabs);
GEN_KERNEL_1D_IN1(ABSI_KERNEL, int *, int *, fabs);

/* log_kernel */
/* z = x+i*y */
/* log(z) = log(abs(z)) + i*atan2(y,x) */
__device__ inline float mylog(float data1) {
  return logf(fabsf(data1));
}
__device__ inline double mylog(double data1) {
  return log(fabs(data1));
}
__device__ inline Complex mylog(Complex data1) {
  Complex c = make_float2(0.0,0.0);
	c.x = logf(fabs(data1));
	c.y = atan2f(data1.y,data1.x);
  return c;
}
__device__ inline DoubleComplex mylog(DoubleComplex data1) {
  DoubleComplex c = make_double2(0.0,0.0);
	c.x = log(fabs(data1));
	c.y = atan2(data1.y,data1.x);
  return c;
}

GEN_KERNEL_1D_IN1(LOGF_KERNEL, float *, float *, mylog);
GEN_KERNEL_1D_IN1(LOGC_KERNEL, Complex *, Complex *, mylog);
GEN_KERNEL_1D_IN1(LOGD_KERNEL, double *, double *, mylog);
GEN_KERNEL_1D_IN1(LOGCD_KERNEL, DoubleComplex *, DoubleComplex *, mylog);

/* log2_kernel */
GEN_KERNEL_1D_IN1(LOG2F_KERNEL, float *, float *, log2f);
GEN_KERNEL_1D_IN1(LOG2D_KERNEL, double *, double *, log2);

/* log10_kernel */
__device__ inline float mylog10(float data1) {
  return logf(fabsf(data1))/logf(10.0);
}
__device__ inline double mylog10(double data1) {
  return log(fabs(data1))/log(10.0);
}
__device__ inline Complex mylog10(Complex data1) {
  Complex c = make_float2(0.0,0.0);
	c.x = logf(fabs(data1))/logf(10.0);
	c.y = atan2f(data1.y,data1.x)/logf(10.0);
  return c;
}
__device__ inline DoubleComplex mylog10(DoubleComplex data1) {
  DoubleComplex c = make_double2(0.0,0.0);
	c.x = log(fabs(data1))/log(10.0);
	c.y = atan2(data1.y,data1.x)/log(10.0);
  return c;
}
GEN_KERNEL_1D_IN1(LOG10F_KERNEL, float *, float *, mylog10);
GEN_KERNEL_1D_IN1(LOG10C_KERNEL, Complex *, Complex *, mylog10);
GEN_KERNEL_1D_IN1(LOG10D_KERNEL, double *, double *, mylog10);
GEN_KERNEL_1D_IN1(LOG10CD_KERNEL, DoubleComplex *, DoubleComplex *, mylog10);

/* log1p_kernel */
GEN_KERNEL_1D_IN1(LOG1PF_KERNEL, float *, float *, log1pf);
GEN_KERNEL_1D_IN1(LOG1PD_KERNEL, double *, double *, log1p);

/* sin_kernel */
GEN_KERNEL_1D_IN1(SINF_KERNEL, float *, float *, sinf);
GEN_KERNEL_1D_IN1(SIND_KERNEL, double *, double *, sin);

/* cos_kernel */
GEN_KERNEL_1D_IN1(COSF_KERNEL, float *, float *, cosf);
GEN_KERNEL_1D_IN1(COSD_KERNEL, double *, double *, cos);

/* tan_kernel */
GEN_KERNEL_1D_IN1(TANF_KERNEL, float *, float *, tanf);
GEN_KERNEL_1D_IN1(TAND_KERNEL, double *, double *, tan);

/* asin_kernel */
GEN_KERNEL_1D_IN1(ASINF_KERNEL, float *, float *, asinf);
GEN_KERNEL_1D_IN1(ASIND_KERNEL, double *, double *, asin);

/* acos_kernel */
GEN_KERNEL_1D_IN1(ACOSF_KERNEL, float *, float *, acosf);
GEN_KERNEL_1D_IN1(ACOSD_KERNEL, double *, double *, acos);

/* atan_kernel */
GEN_KERNEL_1D_IN1(ATANF_KERNEL, float *, float *, atanf);
GEN_KERNEL_1D_IN1(ATAND_KERNEL, double *, double *, atan);


/* sinh_kernel */
GEN_KERNEL_1D_IN1(SINHF_KERNEL, float *, float *, sinhf);
GEN_KERNEL_1D_IN1(SINHD_KERNEL, double *, double *, sinh);

/* cosh_kernel */
GEN_KERNEL_1D_IN1(COSHF_KERNEL, float *, float *, coshf);
GEN_KERNEL_1D_IN1(COSHD_KERNEL, double *, double *, cosh);

/* tanh_kernel */
GEN_KERNEL_1D_IN1(TANHF_KERNEL, float *, float *, tanhf);
GEN_KERNEL_1D_IN1(TANHD_KERNEL, double *, double *, tanh);

/* asinh_kernel */
GEN_KERNEL_1D_IN1(ASINHF_KERNEL, float *, float *, asinhf);
GEN_KERNEL_1D_IN1(ASINHD_KERNEL, double *, double *, asinh);

/* acosh_kernel */
GEN_KERNEL_1D_IN1(ACOSHF_KERNEL, float *, float *, acoshf);
GEN_KERNEL_1D_IN1(ACOSHD_KERNEL, double *, double *, acosh);

/* atanh_kernel */
GEN_KERNEL_1D_IN1(ATANHF_KERNEL, float *, float *, atanhf);
GEN_KERNEL_1D_IN1(ATANHD_KERNEL, double *, double *, atanh);


/* round (int) simply copies the input to the output */
__device__ inline float round(int data1) {
  return data1;
}
/* round_kernel */
GEN_KERNEL_1D_IN1(ROUNDF_KERNEL, float *, float *, roundf);
GEN_KERNEL_1D_IN1(ROUNDD_KERNEL, double *, double *, round);
GEN_KERNEL_1D_IN1(ROUNDI_KERNEL, int *, int *, round);

/* ceil(int) simply copies the input to the output */
__device__ inline float ceil(int data1) {
  return data1;
}
/* ceil_kernel */
GEN_KERNEL_1D_IN1(CEILF_KERNEL, float *, float *, ceilf);
GEN_KERNEL_1D_IN1(CEILD_KERNEL, double *, double *, ceil);
GEN_KERNEL_1D_IN1(CEILI_KERNEL, int *, int *, ceil);

/* floor(int) simply copies the input to the output */
__device__ inline float floor(int data1) {
  return data1;
}
/* floor_kernel */
GEN_KERNEL_1D_IN1(FLOORF_KERNEL, float *, float *, floorf);
GEN_KERNEL_1D_IN1(FLOORD_KERNEL, double *, double *, floor);
GEN_KERNEL_1D_IN1(FLOORI_KERNEL, int  *, int *, floor);


/* conjugate_kernel */
__device__ inline float conjugate(float data1) {
  return data1;
}
__device__ inline Complex conjugate(Complex data1) {
  return make_float2(data1.x, -1.0*data1.y);
}
__device__ inline double conjugate(double data1) {
  return data1;
}
__device__ inline DoubleComplex conjugate(DoubleComplex data1) {
  return  make_double2(data1.x,-1.0*data1.y);
}
__device__ inline int conjugate(int data1) {
  return data1;
}
GEN_KERNEL_1D_IN1(CONJUGATEF_KERNEL, float *, float *, conjugate);
GEN_KERNEL_1D_IN1(CONJUGATEC_KERNEL, Complex *, Complex *, conjugate);
GEN_KERNEL_1D_IN1(CONJUGATED_KERNEL, double *, double *, conjugate);
GEN_KERNEL_1D_IN1(CONJUGATECD_KERNEL, DoubleComplex *, DoubleComplex *, conjugate);
GEN_KERNEL_1D_IN1(CONJUGATEI_KERNEL, int *, int *, conjugate);


/* zeros_kernel */
// We use a standard 1D_IN1 kernel, but the input is not used
extern "C" {
__global__ void ZEROSF_KERNEL(int n,  int offset, float *idata1, float *odata)
{  
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset; 
    if ((xIndex - offset) < n) 
        odata[xIndex] = 0.0; 
} 
__global__ void ZEROSC_KERNEL(int n,  int offset, float *idata1, Complex *odata)
{  
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset; 
    if ((xIndex - offset) < n) {
	Complex tmp = make_float2(0.0,0.0);
        odata[xIndex] = tmp; 
    }
}
__global__ void ZEROSD_KERNEL(int n,  int offset, double *idata1, double *odata)
{  
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset; 
    if ((xIndex - offset) < n) 
        odata[xIndex] = 0.0; 
}
__global__ void ZEROSCD_KERNEL(int n,  int offset, float *idata1, DoubleComplex *odata)
{  
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset; 
    if ((xIndex - offset) < n) {
	DoubleComplex tmp = make_double2(0.0,0.0);
        odata[xIndex] = tmp; 
    }
}
__global__ void ZEROSI_KERNEL(int n,  int offset, int *idata1, int *odata)
{  
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset; 
    if ((xIndex - offset) < n) 
        odata[xIndex] = 0; 
}

/* ones_kernel */
__global__ void ONESF_KERNEL(int n,  int offset, float *idata1, float *odata)
{  
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset; 
    if ((xIndex - offset) < n) 
        odata[xIndex] = 1.0; 
} 
__global__ void ONESC_KERNEL(int n,  int offset, float *idata1, Complex *odata)
{  
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset; 
    if ((xIndex - offset) < n) {
	Complex tmp = make_float2(1.0,0.0);
        odata[xIndex] = tmp; 
    }
}  
__global__ void ONESD_KERNEL(int n,  int offset, double *idata1, double *odata)
{  
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset; 
    if ((xIndex - offset) < n) 
        odata[xIndex] = 1.0; 
}
__global__ void ONESCD_KERNEL(int n,  int offset, float *idata1, DoubleComplex *odata)
{  
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset; 
    if ((xIndex - offset) < n) {
	DoubleComplex tmp = make_double2(1.0,0.0);
        odata[xIndex] = tmp; 
    }
}  
__global__ void ONESI_KERNEL(int n,  int offset, int *idata1, int *odata)
{  
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset; 
    if ((xIndex - offset) < n) 
        odata[xIndex] = 1; 
}
}

/* uminus_kernel */
__device__ inline float uminus(float data) {
  return -1.0*data;
}
__device__ inline Complex uminus(Complex data1) {
  return make_float2(-1.0*data1.x,-1.0*data1.y);
}
__device__ inline double uminus(double data) {
  return -1.0*data;
}
__device__ inline DoubleComplex uminus(DoubleComplex data1) {
  return make_double2(-1.0*data1.x, -1.0*data1.y);
}
__device__ inline int uminus(int data) {
  return -1*data;
}
GEN_KERNEL_1D_IN1(UMINUSF_KERNEL, float *, float *, uminus);
GEN_KERNEL_1D_IN1(UMINUSC_KERNEL, Complex *, Complex *, uminus);
GEN_KERNEL_1D_IN1(UMINUSD_KERNEL, double *, double *, uminus);
GEN_KERNEL_1D_IN1(UMINUSCD_KERNEL, DoubleComplex *, DoubleComplex *, uminus);
GEN_KERNEL_1D_IN1(UMINUI_KERNEL, int *, int *, uminus);

/* not_kernel */
__device__ inline float cnot(float data) {
  return (!data);
}
__device__ inline float cnot(double data) {
  return (!data);
}
__device__ inline int cnot(int data) {
  return (!data);
}
GEN_KERNEL_1D_IN1(NOTF_KERNEL, float *, float *, cnot);
GEN_KERNEL_1D_IN1(NOTD_KERNEL, double *, float *, cnot);
GEN_KERNEL_1D_IN1(NOTI_KERNEL, int *, int *, cnot);

/* times_kernel */
__device__ inline float times(float data1, float data2) {
  return data1*data2;
}
__device__ inline Complex times(Complex data1, Complex data2) {
  return make_float2(data1.x * data2.x - data1.y * data2.y, data1.x * data2.y + data1.y * data2.x);
}
__device__ inline double times(double data1, double data2) {
  return data1*data2;
}
__device__ inline DoubleComplex times(DoubleComplex data1, DoubleComplex data2) {
  return make_double2(data1.x * data2.x - data1.y * data2.y, data1.x * data2.y + data1.y * data2.x);
}
GEN1_KERNEL_1D_IN1_IN2_ALL(TIMES, times)

//GEN1_KERNEL_1D_IN1_IN2_ALL(TIMES2, times)
/*extern "C" {__global__ void TIMES2_F_F_KERNEL(int n , int offset, int i1, int i2, float *odata, int right)
{
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    int xIndex1 = (i1<0)?xIndex:i1; 
    int xIndex2 = (i2<0)?xIndex:i2; 
    if ((xIndex - offset) < n) {
        float d1 = tex1Dfetch_float(texref_f1_a, xIndex1);
        float d2 = tex1Dfetch_float(texref_f1_b, xIndex2);
        odata[xIndex] = (right==1)?times(d1,d2):times(d2,d1);
    }
}
__global__ void TIMES2_C_C_KERNEL(int n , int offset, int i1, int i2, Complex *odata, int right)
{
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    int xIndex1 = (i1<0)?xIndex:i1; 
    int xIndex2 = (i2<0)?xIndex:i2; 
    if ((xIndex - offset) < n) {
        Complex d1 = tex1Dfetch_Complex(texref_c1_a, xIndex1);
        Complex d2 = tex1Dfetch_Complex(texref_c1_b, xIndex2);
        odata[xIndex] = (right==1)?times(d1,d2):times(d2,d1);
    }
}
__global__ void TIMES2_D_D_KERNEL(int n , int offset, int i1, int i2, double *odata, int right)
{
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    int xIndex1 = (i1<0)?xIndex:i1; 
    int xIndex2 = (i2<0)?xIndex:i2; 
    if ((xIndex - offset) < n) {
        double d1 = tex1Dfetch_double(texref_d1_a, xIndex1);
        double d2 = tex1Dfetch_double(texref_d1_b, xIndex2);
        odata[xIndex] = (right==1)?times(d1,d2):times(d2,d1);
    }
}
__global__ void TIMES2_CD_CD_KERNEL(int n , int offset, int i1, int i2, DoubleComplex *odata, int right)
{
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    int xIndex1 = (i1<0)?xIndex:i1; 
    int xIndex2 = (i2<0)?xIndex:i2; 
    if ((xIndex - offset) < n) {
        DoubleComplex d1 = tex1Dfetch_DoubleComplex(texref_cd1_a, xIndex1);
        DoubleComplex d2 = tex1Dfetch_DoubleComplex(texref_cd1_b, xIndex2);
	//DoubleComplex a = {1.0,1.0};
        //odata[xIndex] = a;
        odata[xIndex] = (right==1)?times(d1,d2):times(d2,d1);
    }
}
}*/

/* rdivide_kernel */
__device__ inline float rdivide(float data1, float data2) {
  return data1/data2;
}
__device__ inline Complex rdivide(Complex data1, Complex data2) {
  Complex c = make_float2(0.0,0.0);
  float den = data2.x*data2.x + data2.y*data2.y;
  c.x = (data1.x * data2.x + data1.y * data2.y)/den;
  c.y = (-data1.x * data2.y + data1.y * data2.x)/den;
  //c.y = (data1.x * data2.y - data1.y * data2.x)/den;
  return c;
}
__device__ inline double rdivide(double data1, double data2) {
  return data1/data2;
}
__device__ inline DoubleComplex rdivide(DoubleComplex data1, DoubleComplex data2) {
  DoubleComplex c = make_double2(0.0,0.0);
  double den = data2.x*data2.x + data2.y*data2.y;
  c.x = (data1.x * data2.x + data1.y * data2.y)/den;
  c.y = (-data1.x * data2.y + data1.y * data2.x)/den;
  //c.y = (data1.x * data2.y - data1.y * data2.x)/den;
  return c;
}
GEN1_KERNEL_1D_IN1_IN2_ALL(RDIVIDE, rdivide)

/* ldivide_kernel */
__device__ inline float ldivide(float data2, float data1) {
  return data1/data2;
}
__device__ inline Complex ldivide(Complex data2, Complex data1) {
  Complex c = make_float2(0.0,0.0);
  float den = data2.x*data2.x + data2.y*data2.y;
  c.x = (data1.x * data2.x + data1.y * data2.y)/den;
  c.y = (-data1.x * data2.y + data1.y * data2.x)/den;
  //c.y = (data1.x * data2.y - data1.y * data2.x)/den;
  return c;
}
__device__ inline double ldivide(double data2, double data1) {
  return data1/data2;
}
__device__ inline DoubleComplex ldivide(DoubleComplex data2, DoubleComplex data1) {
  DoubleComplex c = make_double2(0.0,0.0);
  double den = data2.x*data2.x + data2.y*data2.y;
  c.x = (data1.x * data2.x + data1.y * data2.y)/den;
  c.y = (-data1.x * data2.y + data1.y * data2.x)/den;
  //c.y = (data1.x * data2.y - data1.y * data2.x)/den;
  return c;
}
GEN1_KERNEL_1D_IN1_IN2_ALL(LDIVIDE, ldivide)

/* plus_kernel */
__device__ inline float plus(float data1, float data2) {
  return data1+data2;
}
__device__ inline Complex plus(Complex data1, Complex data2) {
  Complex c = make_float2(0.0,0.0);
  c.x = (data1.x + data2.x);
  c.y = (data1.y + data2.y);
  return c;
}
__device__ inline double plus(double data1, double data2) {
  return data1+data2;
}
__device__ inline DoubleComplex plus(DoubleComplex data1, DoubleComplex data2) {
  DoubleComplex c = make_double2(0.0,0.0);
  c.x = (data1.x + data2.x);
  c.y = (data1.y + data2.y);
  return c;
}

//GEN_KERNEL_1D_IN1_IN2_ALL(PLUS, plus)
GEN1_KERNEL_1D_IN1_IN2(PLUS_F_F_KERNEL, float   , float   , float   , plus, float);
GEN1_KERNEL_1D_IN1_IN2(PLUS_F_C_KERNEL, float  , Complex , Complex , plus, Complex);
GEN1_KERNEL_1D_IN1_IN2(PLUS_F_D_KERNEL, float  , double  , float   , plus, double);
GEN1_KERNEL_1D_IN1_IN2(PLUS_F_CD_KERNEL, float          , DoubleComplex  , Complex  , plus, DoubleComplex);
GEN1_KERNEL_1D_IN1_IN2(PLUS_C_F_KERNEL, Complex, float   , Complex , plus, Complex);
GEN1_KERNEL_1D_IN1_IN2(PLUS_C_C_KERNEL, Complex , Complex , Complex , plus, Complex);
GEN1_KERNEL_1D_IN1_IN2(PLUS_C_D_KERNEL, Complex  , double  , Complex   , plus, DoubleComplex);
GEN1_KERNEL_1D_IN1_IN2(PLUS_C_CD_KERNEL, Complex       , DoubleComplex  , Complex   , plus, DoubleComplex);
GEN1_KERNEL_1D_IN1_IN2(PLUS_D_F_KERNEL, double , float   , float   , plus, double);
GEN1_KERNEL_1D_IN1_IN2(PLUS_D_C_KERNEL, double   , Complex , Complex   , plus, DoubleComplex);
GEN1_KERNEL_1D_IN1_IN2(PLUS_D_D_KERNEL, double  , double  , double  , plus, double);
GEN1_KERNEL_1D_IN1_IN2(PLUS_D_CD_KERNEL,  double        , DoubleComplex , DoubleComplex , plus, DoubleComplex);
GEN1_KERNEL_1D_IN1_IN2(PLUS_CD_F_KERNEL, DoubleComplex  , float          , Complex  , plus, DoubleComplex);
GEN1_KERNEL_1D_IN1_IN2(PLUS_CD_C_KERNEL, DoubleComplex , Complex        , Complex   , plus, DoubleComplex);
GEN1_KERNEL_1D_IN1_IN2(PLUS_CD_D_KERNEL,  DoubleComplex , double        , DoubleComplex , plus, DoubleComplex);
GEN1_KERNEL_1D_IN1_IN2(PLUS_CD_CD_KERNEL, DoubleComplex , DoubleComplex , DoubleComplex , plus, DoubleComplex);








/* power_kernel */
__device__ inline float mypower(float data1, float data2) {
  return powf(data1,data2);
}
__device__ inline Complex mypower(Complex data1, Complex data2)
{
    Complex c = make_float2(0.0,0.0);
    float ang = atan2f(data1.y,data1.x); 
    float mag = (data1.x==0)?data1.y:sqrtf(data1.y*data1.y + data1.x*data1.x);
    float pre = powf(mag,data2.x)*expf(-data2.y*ang);
    c.x = pre*cosf(data2.x*ang + data2.y*logf(mag));
    c.y = pre*sinf(data2.x*ang + data2.y*logf(mag));
    return c;
}
__device__ inline double mypower(double data1, double data2) {
  return pow((data1),(data2));
}
__device__ inline DoubleComplex mypower(DoubleComplex data1, DoubleComplex data2)
{
    DoubleComplex c = make_double2(0.0,0.0);
    double ang = atan2(data1.y,data1.x); 
    double mag = (data1.x==0)?data1.y:sqrt(data1.y*data1.y + data1.x*data1.x);
    double pre = pow(mag,data2.x)*exp(-data2.y*ang);
    c.x = pre*cos(data2.x*ang + data2.y*log(mag));
    c.y = pre*sin(data2.x*ang + data2.y*log(mag));
    return c;
}

GEN1_KERNEL_1D_IN1_IN2(POWER_F_F_KERNEL, float   , float   , float   , mypower, float);
GEN1_KERNEL_1D_IN1_IN2(POWER_F_C_KERNEL, float  , Complex , Complex , mypower, Complex);
GEN1_KERNEL_1D_IN1_IN2(POWER_F_D_KERNEL, float  , double  , float   , mypower, double);
GEN1_KERNEL_1D_IN1_IN2(POWER_F_CD_KERNEL, float          , DoubleComplex  , Complex  , mypower, DoubleComplex);
GEN1_KERNEL_1D_IN1_IN2(POWER_C_F_KERNEL, Complex, float   , Complex , mypower, Complex);
GEN1_KERNEL_1D_IN1_IN2(POWER_C_C_KERNEL, Complex , Complex , Complex , mypower, Complex);
GEN1_KERNEL_1D_IN1_IN2(POWER_C_D_KERNEL, Complex  , double  , Complex   , mypower, DoubleComplex);
GEN1_KERNEL_1D_IN1_IN2(POWER_C_CD_KERNEL, Complex       , DoubleComplex  , Complex   , mypower, DoubleComplex);
GEN1_KERNEL_1D_IN1_IN2(POWER_D_F_KERNEL, double , float   , float   , mypower, float); //<---wrong, should be mypower, double. Cannot compile
GEN1_KERNEL_1D_IN1_IN2(POWER_D_C_KERNEL, double   , Complex , Complex   , mypower, Complex); //<---wrong, should be mypower, DoubleComplex. Cannot compile
GEN1_KERNEL_1D_IN1_IN2(POWER_D_D_KERNEL, double  , double  , double  , mypower, double);
GEN1_KERNEL_1D_IN1_IN2(POWER_D_CD_KERNEL,  double        , DoubleComplex , DoubleComplex , mypower, DoubleComplex);
GEN1_KERNEL_1D_IN1_IN2(POWER_CD_F_KERNEL, DoubleComplex  , float          , Complex  , mypower, Complex); //<---wrong, should be mypower, DoubleComplex. Cannot compile
GEN1_KERNEL_1D_IN1_IN2(POWER_CD_C_KERNEL, DoubleComplex , Complex        , Complex   , mypower, Complex);//<---wrong, should be mypower, DoubleComplex. Cannot compile
GEN1_KERNEL_1D_IN1_IN2(POWER_CD_D_KERNEL,  DoubleComplex , double        , DoubleComplex , mypower, DoubleComplex);
GEN1_KERNEL_1D_IN1_IN2(POWER_CD_CD_KERNEL, DoubleComplex , DoubleComplex , DoubleComplex , mypower, DoubleComplex);
//GEN1_KERNEL_1D_IN1_IN2_ALL(POWER, mypower)

/* minus_kernel */
__device__ inline float minus(float data1, float data2) {
  return (data1-data2);
}
__device__ inline Complex minus(Complex data1, Complex data2) {
  Complex c = make_float2(0.0,0.0);
  c.x = (data1.x - data2.x);
  c.y = (data1.y - data2.y);
  return c;
}
__device__ inline double minus(double data1, double data2) {
  return (data1-data2);
}
__device__ inline DoubleComplex minus(DoubleComplex data1, DoubleComplex data2) {
  DoubleComplex c = make_double2(0.0,0.0);
  c.x = (data1.x - data2.x);
  c.y = (data1.y - data2.y);
  return c;
}
GEN1_KERNEL_1D_IN1_IN2_ALL(MINUS, minus)


/* lt_kernel */
// imaginary part ignored
__device__ inline float lt(float data1, float data2) {
  return ((data1<data2)?1.0:0.0);
}
__device__ inline float lt(Complex data1, Complex data2) {
  return ((data1.x<data2.x)?1.0:0.0);
}
__device__ inline double lt(double data1, double data2) {
  return ((data1<data2)?1.0:0.0);
}
__device__ inline double lt(DoubleComplex data1, DoubleComplex data2) {
  return ((data1.x<data2.x)?1.0:0.0);
}
GEN1_KERNEL_1D_IN1_IN2_OUTFLOAT_ALL(LT, lt)

//GEN_KERNEL_1D_IN1_IN2(LTF_KERNEL, float , float , float , ltf);
//GEN_KERNEL_1D_IN1_IN2(LTC_KERNEL, Complex , Complex , float , ltc);
//GEN_KERNEL_1D_IN1_IN2SCALAR(LTF_SCALAR_KERNEL, float , float , float , ltf);
//GEN_KERNEL_1D_IN1_IN2SCALAR(LTC_SCALAR_KERNEL, Complex , float , float , ltc);
//GEN_KERNEL_1D_IN1_IN2(LTD_KERNEL, double , double , double , ltd);
//GEN_KERNEL_1D_IN1_IN2(LTCD_KERNEL, DoubleComplex , DoubleComplex , double , ltcd);
//GEN_KERNEL_1D_IN1_IN2SCALAR(LTD_SCALAR_KERNEL, double , double , double , ltd);
//GEN_KERNEL_1D_IN1_IN2SCALAR(LTCD_SCALAR_KERNEL, DoubleComplex , double , double , ltcd);

/* gt_kernel */
// imaginary part ignored
__device__ inline float gt(float data1, float data2) {
  return ((data1>data2)?1.0:0.0);
}
__device__ inline float gt(Complex data1, Complex data2) {
  return ((data1.x>data2.x)?1.0:0.0);
}
__device__ inline double gt(double data1, double data2) {
  return ((data1>data2)?1.0:0.0);
}
__device__ inline double gt(DoubleComplex data1, DoubleComplex data2) {
  return ((data1.x>data2.x)?1.0:0.0);
}
GEN1_KERNEL_1D_IN1_IN2_OUTFLOAT_ALL(GT, gt)
//GEN_KERNEL_1D_IN1_IN2(GTF_KERNEL, float , float , float , gtf);
//GEN_KERNEL_1D_IN1_IN2(GTC_KERNEL, Complex , Complex , float , gtc);
//GEN_KERNEL_1D_IN1_IN2SCALAR(GTF_SCALAR_KERNEL, float , float , float , gtf);
//GEN_KERNEL_1D_IN1_IN2SCALAR(GTC_SCALAR_KERNEL, Complex , float , float , gtc);
//GEN_KERNEL_1D_IN1_IN2(GTD_KERNEL, double , double , double , gtd);
//GEN_KERNEL_1D_IN1_IN2(GTCD_KERNEL, DoubleComplex , DoubleComplex , double , gtcd);
//GEN_KERNEL_1D_IN1_IN2SCALAR(GTD_SCALAR_KERNEL, double , double , double , gtd);
//GEN_KERNEL_1D_IN1_IN2SCALAR(GTCD_SCALAR_KERNEL, DoubleComplex , double , double , gtcd);

/* le_kernel */
// imaginary part ignored
__device__ inline float le(float data1, float data2) {
  return ((data1<=data2)?1.0:0.0);
}
__device__ inline float le(Complex data1, Complex data2) {
  return ((data1.x<=data2.x)?1.0:0.0);
}
__device__ inline double le(double data1, double data2) {
  return ((data1<=data2)?1.0:0.0);
}
__device__ inline double le(DoubleComplex data1, DoubleComplex data2) {
  return ((data1.x<=data2.x)?1.0:0.0);
}
GEN1_KERNEL_1D_IN1_IN2_OUTFLOAT_ALL(LE, le)
//GEN_KERNEL_1D_IN1_IN2(LEF_KERNEL, float , float , float , lef);
//GEN_KERNEL_1D_IN1_IN2(LEC_KERNEL, Complex , Complex , float , lec);
//GEN_KERNEL_1D_IN1_IN2SCALAR(LEF_SCALAR_KERNEL, float , float , float , lef);
//GEN_KERNEL_1D_IN1_IN2SCALAR(LEC_SCALAR_KERNEL, Complex , float , float , lec);
//GEN_KERNEL_1D_IN1_IN2(LED_KERNEL, double , double , double , led);
//GEN_KERNEL_1D_IN1_IN2(LECD_KERNEL, DoubleComplex , DoubleComplex , double , lecd);
//GEN_KERNEL_1D_IN1_IN2SCALAR(LED_SCALAR_KERNEL, double , double , double , led);
//GEN_KERNEL_1D_IN1_IN2SCALAR(LECD_SCALAR_KERNEL, DoubleComplex , double , double , lecd);

/* ge_kernel */
// imaginary part ignored
__device__ inline float ge(float data1, float data2) {
  return ((data1 >= data2)?1.0:0.0);
}
// imaginary part ignored
__device__ inline float ge(Complex data1, Complex data2) {
  return ((data1.x >= data2.x)?1.0:0.0);
}
__device__ inline double ge(double data1, double data2) {
  return ((data1 >= data2)?1.0:0.0);
}
// imaginary part ignored
__device__ inline double ge(DoubleComplex data1, DoubleComplex data2) {
  return ((data1.x >= data2.x)?1.0:0.0);
}
GEN1_KERNEL_1D_IN1_IN2_OUTFLOAT_ALL(GE, ge)
//GEN_KERNEL_1D_IN1_IN2(GEF_KERNEL, float , float , float , gef);
//GEN_KERNEL_1D_IN1_IN2(GEC_KERNEL, Complex , Complex , float , gec);
//GEN_KERNEL_1D_IN1_IN2SCALAR(GEF_SCALAR_KERNEL, float , float , float , gef);
//GEN_KERNEL_1D_IN1_IN2SCALAR(GEC_SCALAR_KERNEL, Complex , float , float , gec);
//GEN_KERNEL_1D_IN1_IN2(GED_KERNEL, double , double , double , ged);
//GEN_KERNEL_1D_IN1_IN2(GECD_KERNEL, DoubleComplex , DoubleComplex , double , gecd);
//GEN_KERNEL_1D_IN1_IN2SCALAR(GED_SCALAR_KERNEL, double , double , double , ged);
//GEN_KERNEL_1D_IN1_IN2SCALAR(GECD_SCALAR_KERNEL, DoubleComplex , double , double , gecd);

/* ne_kernel */
__device__ inline float ne(float data1, float data2) {
  return ((data1!=data2)?1.0:0.0);
}
__device__ inline float ne(Complex data1, Complex data2) {
  return (data1.x!=data2.x)||(data1.y!=data2.y);
}
__device__ inline double ne(double data1, double data2) {
  return ((data1!=data2)?1.0:0.0);
}
__device__ inline double ne(DoubleComplex data1, DoubleComplex data2) {
  return (data1.x!=data2.x)||(data1.y!=data2.y);
}
GEN1_KERNEL_1D_IN1_IN2_OUTFLOAT_ALL(NE, ne)
//GEN_KERNEL_1D_IN1_IN2(NEF_KERNEL, float , float , float , nef);
//GEN_KERNEL_1D_IN1_IN2(NEC_KERNEL, Complex , Complex , float , nec);
//GEN_KERNEL_1D_IN1_IN2SCALAR(NEF_SCALAR_KERNEL, float , float , float , nef);
//GEN_KERNEL_1D_IN1_IN2SCALAR(NEC_SCALAR_KERNEL, Complex , float , float , nec);
//GEN_KERNEL_1D_IN1_IN2(NED_KERNEL, double , double , double , ned);
//GEN_KERNEL_1D_IN1_IN2(NECD_KERNEL, DoubleComplex , DoubleComplex , double , necd);
//GEN_KERNEL_1D_IN1_IN2SCALAR(NED_SCALAR_KERNEL, double , double , double , ned);
//GEN_KERNEL_1D_IN1_IN2SCALAR(NECD_SCALAR_KERNEL, DoubleComplex , double , double , necd);

/* eq_kernel */
__device__ inline float eq(float data1, float data2) {
  return ((data1==data2)?1.0:0.0);
}
__device__ inline float eq(Complex data1, Complex data2) {
  return (data1.x==data2.x)&&(data1.y==data2.y);
}
__device__ inline double eq(double data1, double data2) {
  return ((data1==data2)?1.0:0.0);
}
__device__ inline double eq(DoubleComplex data1, DoubleComplex data2) {
  return (data1.x==data2.x)&&(data1.y==data2.y);
}
GEN1_KERNEL_1D_IN1_IN2_OUTFLOAT_ALL(EQ, eq)
//GEN_KERNEL_1D_IN1_IN2(EQF_KERNEL, float , float , float , eqf);
//GEN_KERNEL_1D_IN1_IN2(EQC_KERNEL, Complex , Complex , float , eqc);
//GEN_KERNEL_1D_IN1_IN2SCALAR(EQF_SCALAR_KERNEL, float , float , float , eqf);
//GEN_KERNEL_1D_IN1_IN2SCALAR(EQC_SCALAR_KERNEL, Complex , float , float , eqc);
//GEN_KERNEL_1D_IN1_IN2(EQD_KERNEL, double , double , double , eqd);
//GEN_KERNEL_1D_IN1_IN2(EQCD_KERNEL, DoubleComplex , DoubleComplex , double , eqcd);
//GEN_KERNEL_1D_IN1_IN2SCALAR(EQD_SCALAR_KERNEL, double , double , double , eqd);
//GEN_KERNEL_1D_IN1_IN2SCALAR(EQCD_SCALAR_KERNEL, DoubleComplex , double , double , eqcd);

/* and_kernel */
__device__ inline float cand(float data1, float data2) {
  return data1 && data2;
}
__device__ inline float cand(double data1, double data2) {
  return data1 && data2;
}
GEN1_KERNEL_1D_IN1_IN2_OUTFLOAT_REAL(AND, cand)

/* or_kernel */
__device__ inline float cor(float data1, float data2) {
  return data1 || data2;
}
__device__ inline float cor(double data1, double data2) {
  return data1 || data2;
}
GEN1_KERNEL_1D_IN1_IN2_OUTFLOAT_REAL(OR, cor)

/* fmax_kernel */
__device__ inline double myfmax(float data1, float data2) {
  return fmaxf(data1,data2);
}
GEN1_KERNEL_1D_IN1_IN2_REAL(FMAX, myfmax)
//GEN_KERNEL_1D_IN1_IN2(FMAXF_KERNEL, float , float , float , fmaxf);
//GEN_KERNEL_1D_IN1_IN2SCALAR(FMAXF_SCALAR_KERNEL, float , float , float , fmaxf);
//GEN_KERNEL_1D_IN1_IN2(FMAXD_KERNEL, double , double , double , fmax);
//GEN_KERNEL_1D_IN1_IN2SCALAR(FMAXD_SCALAR_KERNEL, double , double , double , fmax);


/* fmin_kernel */
__device__ inline double myfmin(float data1, float data2) {
  return fminf(data1,data2);
}
GEN1_KERNEL_1D_IN1_IN2_REAL(FMIN, myfmin)
//GEN_KERNEL_1D_IN1_IN2(FMINF_KERNEL, float , float , float , fminf);
//GEN_KERNEL_1D_IN1_IN2SCALAR(FMINF_SCALAR_KERNEL, float , float , float , fminf);
//GEN_KERNEL_1D_IN1_IN2(FMIND_KERNEL, double , double , double , fmin);
//GEN_KERNEL_1D_IN1_IN2SCALAR(FMIND_SCALAR_KERNEL, double , double , double , fmin);


#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define cache(i) CUT_BANK_CHECKER(((float*)&cache[0]), (i))
#else
#define cache(i) cache[i]
#endif


/* packfC2C */
#define WARP2 16
#define WARP4 8

extern "C" {__global__ void PACKFC2C_KERNEL(int n, int onlyreal,  float* re_idata, float* im_idata, float* odata)
{
  __shared__ float cache[BLOCK_DIM1D*2+BLOCK_DIM1D/WARP4];
  unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x;
  unsigned int offset = 0;

  offset = threadIdx.x >> 3;


  /* read real part */
  if (xIndex < n) {
    //printf("r t %d, %d (%g) -> %d\n", threadIdx.x, xIndex, re_idata[xIndex], 2*threadIdx.x+offset);
    cache(2*threadIdx.x+offset) = re_idata[xIndex];
  }
  __syncthreads();
  /* read imaginary part */
  if (xIndex < n) {
    cache(2*threadIdx.x+offset+1) = (onlyreal==0)?im_idata[xIndex]:0.0;
    //printf("r t %d, %d (%g) -> %d\n", threadIdx.x, xIndex, im_idata[xIndex], 2*threadIdx.x+offset+1 );
  }
  __syncthreads();
  
  //printf("Write\n");
  offset = floorf(threadIdx.x/WARP2);
  //printf("offs %d\n",offset);
  xIndex = blockIdx.x * 2 * BLOCK_DIM1D + threadIdx.x;
  if (xIndex < 2*n) {
    odata[xIndex]   = (cache(threadIdx.x+offset));
    //printf("w t %d, %d (%g) -> %d\n", threadIdx.x, threadIdx.x+offset, odata[xIndex], xIndex);
  }
  __syncthreads();

  offset = (threadIdx.x + BLOCK_DIM1D) >> 4;

  xIndex += BLOCK_DIM1D;
  if (xIndex < 2*n) {
    odata[xIndex] = (cache(threadIdx.x+offset+BLOCK_DIM1D));
    //printf("w t %d, %d (%g) -> %d\n", threadIdx.x, threadIdx.x+offset+BLOCK_DIM1D+offset2, odata[xIndex], xIndex);
  }

}

__global__ void PACKDC2C_KERNEL(int n, int onlyreal,  double* re_idata, double* im_idata, double* odata)
{
  __shared__ double cache[BLOCK_DIM1D*2+BLOCK_DIM1D/WARP4];
  unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x;
  unsigned int offset = 0;

  offset = threadIdx.x >> 3;


  /* read real part */
  if (xIndex < n) {
    cache(2*threadIdx.x+offset) = re_idata[xIndex];
  }
  __syncthreads();
  /* read imaginary part */
  if (xIndex < n) {
    cache(2*threadIdx.x+offset+1) = (onlyreal==0)?im_idata[xIndex]:0.0;
  }
  __syncthreads();
  
  offset = floorf(threadIdx.x/WARP2);
  xIndex = blockIdx.x * 2 * BLOCK_DIM1D + threadIdx.x;
  if (xIndex < 2*n) {
    odata[xIndex]   = (cache(threadIdx.x+offset));
  }
  __syncthreads();

  offset = (threadIdx.x + BLOCK_DIM1D) >> 4;

  xIndex += BLOCK_DIM1D;
  if (xIndex < 2*n) {
    odata[xIndex] = (cache(threadIdx.x+offset+BLOCK_DIM1D));
  }

}

/*
 * Read imaginary and/or real part of idata into re_odata and im_odata depending on
 * mode
 * 0 - REAL, IMAG
 * 1 - REAL
 * 2 - IMAG
 */
__global__ void UNPACKFC2C_KERNEL(int n, int mode,  float* idata, float* re_odata, float* im_odata)
{
  __shared__ float cache[BLOCK_DIM1D*2+BLOCK_DIM1D/WARP4];
  unsigned int offset = 0;

  

  offset = threadIdx.x >> 4;
  int xIndex = blockIdx.x * 2 * BLOCK_DIM1D + threadIdx.x;
  
  if (xIndex < 2*n) {
    //printf("idata %g\n",idata[xIndex]);
    cache(threadIdx.x+offset) = idata[xIndex];
  }
  __syncthreads();

  offset = (threadIdx.x + BLOCK_DIM1D) >> 4;
  xIndex += BLOCK_DIM1D;

  if (xIndex < 2*n) {
    //printf("idata %g\n",idata[xIndex]);
    cache(threadIdx.x+offset+BLOCK_DIM1D) = idata[xIndex];
  }
  __syncthreads();
  
  offset = threadIdx.x >> 3;

  xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x;

  /* read real part */
  if (xIndex < n) {
    if (mode != 2) {
      //printf("re(%d)=%g\n",xIndex,cache(2*threadIdx.x+offset));
      re_odata[xIndex] = cache(2*threadIdx.x+offset);
    }
  }
  
  /* read imaginary part */
  if (xIndex < n) {
    if (mode != 1) {
      //printf("im(%d)=%g\n",xIndex,cache(2*threadIdx.x+offset+1));
      im_odata[xIndex] = cache(2*threadIdx.x+offset+1); 
    }
  }
  
  

}

__global__ void UNPACKDC2C_KERNEL(int n, int mode,  double* idata, double* re_odata, double* im_odata)
{
  __shared__ double cache[BLOCK_DIM1D*2+BLOCK_DIM1D/WARP4];
  unsigned int offset = 0;

  

  offset = threadIdx.x >> 4;
  int xIndex = blockIdx.x * 2 * BLOCK_DIM1D + threadIdx.x;
  
  if (xIndex < 2*n) {
    cache(threadIdx.x+offset) = idata[xIndex];
  }
  __syncthreads();

  offset = (threadIdx.x + BLOCK_DIM1D) >> 4;
  xIndex += BLOCK_DIM1D;

  if (xIndex < 2*n) {
    cache(threadIdx.x+offset+BLOCK_DIM1D) = idata[xIndex];
  }
  __syncthreads();
  
  offset = threadIdx.x >> 3;

  xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x;

  /* read real part */
  if (xIndex < n) {
    if (mode != 2) {
      re_odata[xIndex] = cache(2*threadIdx.x+offset);
    }
  }
  
  /* read imaginary part */
  if (xIndex < n) {
    if (mode != 1) {
      im_odata[xIndex] = cache(2*threadIdx.x+offset+1); 
    }
  }
}

#undef WARP2
#undef WARP4

/* copymemory kernel */
__global__ void COPYMEMORY_KERNEL(int n, float *idata, float *odata)
{
    __shared__ float cache[BLOCK_DIM1D];
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x;
    if (xIndex < n)
        cache[threadIdx.x] = idata[xIndex];
    __syncthreads();
    if (xIndex < n)
        odata[xIndex] = cache[threadIdx.x];
}                        

/*
 * subsindexf
 */ 
/*__global__ void SUBSINDEXF_KERNEL(int n, const float idxshift,  const int texwidth, 
		                  float *ix, float *odata)
{
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x;
    //unsigned int iIndex = ix[xIndex]+idxshift;
    //unsigned int texy   = (int) floorf(iIndex/texwidth);
    //unsigned int texx   = iIndex - texy*texwidth;
    if (xIndex < n) 
       odata[xIndex] = tex1Dfetch(texref1, ix[xIndex]+idxshift);
       
    //odata[xIndex] = tex2D(texref2, (float) texx, (float) texy);
}
*/

/*
 * subsindexc
 */ 
/*__global__ void SUBSINDEXC_KERNEL(int n, const float idxshift,  const int texwidth, 
		                  float *ix, Complex *odata)
{
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x;
    //unsigned int iIndex = ix[xIndex]+idxshift;
    //unsigned int texy   = (int) floorf(iIndex/texwidth);
    //unsigned int texx   = iIndex - texy*texwidth;
    if (xIndex < n)  {
       odata[xIndex] = (Complex) tex1Dfetch(texref4, ix[xIndex]+idxshift);
    }
       //odata[xIndex] = tex2D(texref3, (float) texx, (float) texy);
        
}*/
/*
 * fillVector1
 * Fill odata starting from offs and with increment incr. 
 * n    - number of elements
 * offs - offset to be used when filling
 * incr - increment
 * m - used to calculate effective index (xIndex % m)
 * type 
 *   0 - Only real
 *   1 - Only complex
 *   2 - Real/Complex
 *    
 */
__global__ void FILLVECTOR1F_KERNEL(int n,  int offset,  float *odata, int m, int p, int offsetp, int type)
{
    int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    float incr = tex1Dfetch_float(texref_f1_a,0);
    float offs = tex1Dfetch_float(texref_f1_b,0);
    if ((xIndex -offset)  < n) {
	if (((xIndex+offsetp) % p)==0)
          odata[xIndex] = incr*(xIndex % m) + offs;
    }
}

__global__ void FILLVECTOR1C_KERNEL(int n,  int offset,  Complex *odata, int m, int p, int offsetp,  int type)
{
    int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    float incr = tex1Dfetch_float(texref_f1_a,0);
    float offs = tex1Dfetch_float(texref_f1_b,0);
    if ((xIndex -offset)  < n) {
	if (((xIndex+offsetp) % p)==0) {
          Complex c = make_float2(0.0,0.0);
	  c.x = incr*(xIndex % m) + offs;
	  c.y = c.x;
          if (type==0)
            odata[xIndex].x = c.x;
          if (type==1)
            odata[xIndex].y = c.y;
          if (type==2)
            odata[xIndex] = c;
        }
    }
}
__global__ void FILLVECTOR1D_KERNEL(int n,  int offset,  double *odata, int m, int p, int offsetp, int type) {
    int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    double incr = tex1Dfetch_double(texref_d1_a,0);
    double offs = tex1Dfetch_double(texref_d1_b,0);
    if ((xIndex -offset)  < n) {
	if (((xIndex+offsetp) % p)==0)
          odata[xIndex] = incr*(xIndex % m) + offs;
    }
}

__global__ void FILLVECTOR1CD_KERNEL(int n,  int offset,  DoubleComplex *odata, int m, int p, int offsetp, int type) {
    int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    double incr = tex1Dfetch_double(texref_d1_a,0);
    double offs = tex1Dfetch_double(texref_d1_b,0);
    if ((xIndex -offset)  < n) {
	if (((xIndex+offsetp) % p)==0) {
          DoubleComplex c = make_double2(0.0,0.0);
	  c.x = incr*(xIndex % m) + offs;
	  c.y = c.x;
          if (type==0)
            odata[xIndex].x = c.x;
          if (type==1)
            odata[xIndex].y = c.y;
          if (type==2)
            odata[xIndex] = c;
        }
    }
}

/*
 * fillVector
 * Fill odata starting from offs and with increment incr
 * n    - number of elements
 * offs - offset to be used when filling
 * incr - increment
 */
__global__ void FILLVECTORF_KERNEL(int n,  int offset,  float *odata)
{
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    float incr = tex1Dfetch_float(texref_f1_a,0);
    float offs = tex1Dfetch_float(texref_f1_b,0);
    if ((xIndex -offset)  < n)
        odata[xIndex] = incr*xIndex + offs;
}
__global__ void FILLVECTORC_KERNEL(int n,  int offset, Complex *odata)
{
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    float incr = tex1Dfetch_float(texref_f1_a,0);
    float offs = tex1Dfetch_float(texref_f1_b,0);
    if ((xIndex -offset)  < n) {
        Complex c = make_float2(0.0,0.0);
        c.x = incr*xIndex + offs;
        odata[xIndex] = c;
    }
}
__global__ void FILLVECTORD_KERNEL(int n,  int offset, double *odata)
{
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    double incr = tex1Dfetch_double(texref_d1_a,0);
    double offs = tex1Dfetch_double(texref_d1_b,0);
    if ((xIndex -offset)  < n)
        odata[xIndex] = incr*xIndex + offs;
}
__global__ void FILLVECTORCD_KERNEL(int n,  int offset, DoubleComplex *odata)
{
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    double incr = tex1Dfetch_double(texref_d1_a,0);
    double offs = tex1Dfetch_double(texref_d1_b,0);
    if ((xIndex -offset)  < n) {
        DoubleComplex c = make_double2(0.0,0.0);
        c.x = incr*xIndex + offs;
        odata[xIndex] = c;
    }
}

/*
 * sum
 * If I sum along a direction the result will be vectors with all the dimensions except the
 * one I did the sum.
 * If J = rand(2,2,2,2)
 * then sum(J,3) is the same as J(:,:,1,:)+J(:,:,2,:)
 * then sum(J,1) is the same as J(1,:,:,:)+J(2,:,:,:)
 * then sum(J,2) is the same as J(:,1,:,:)+J(:,2,:,:)
 *
 * Nthreads is the number of elements the resuling vector. 
 * If J=(a,b,c,d) and sum over c NThreads and the size of the result is
 * a*b*d
 *
 * The threads are grouped depending on the dimension I sum. If I sum over c,
 * then I will have a*b*d threads, grouped into a*b and d. 
 *
 * Example
 * (a,b,c,d) = (2,3,4,5) sum over c
 * Nthreads 2*3*5
 * M = iterations = 4
 * There will be 5 groups of 2*3 threads. 2*3 will be contiguos and the separation
 * between groups is 4*3*2. With a group I consider some threads that work on memory
 * areas that are contiguos
 * So, to do the calculation I need Nthread, M, GroupSize = 2*3, GroupOffset = 4*3*2. I need
 * also the step at each iteration given by incr=2*3. This means that every GroupSize threads 
 * I have to jump to another memory region with GroupOffset
 *
 * Note: column-major format
 *
 * Example
 * (a,b,c,d) = (2,3,4,5) sum over a
 * Nthreads 3*4*5
 * M = 2
 * GroupSize = 1
 * GroupOffset = 2
 * incr = 1
 * 
 * Example
 * (a,b,c,d) = (2,3,4,5) sum over b
 * Nthreads 2*4*5
 * M = 3
 * GroupSize = 2
 * GroupOffset = 2*3
 * incr = 2
 *
 * For a given vector J and sum over D, the rule is
 * 1. Prepend a "1" to the size of vector
 * 2. Nthread: multiply all dim. except D
 * 3. M: D
 * 4. GroupSize: prod(dim before D)
 * 5. GroupOffset: GroupSize *D
 * 6. incr = GroupSize
 *
 * NThreads - number of threads (also size of odata)
 * M - iterations
 * GroupSize - see above
 * GroupOffset - see above 
 * texwidth - width of the 2D texture
 * odata - output vector
 *
 * Kernel
 * 1D kernel, only first threads < NThreads are workers. 
 *
 * Note: don't use 1D texture because the number of elements is limited to 65536
 *
 */

__global__ void SUMF_TEX_KERNEL(int n,  int offset, int Nthreads, int M, int GroupSize, int GroupOffset, float *odata)
{
  unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x; // index used for output
  unsigned int iGroup = (int) floorf(xIndex/GroupSize);
  unsigned int iIndex = xIndex - iGroup*(GroupSize  - GroupOffset);
  // Domanda, il for è meglio farlo dentro if o fuori?
  float sum = 0.0;
  if (xIndex < Nthreads) { 

    for (int i=0;i<M;i++) {
      //if (xIndex < Nthreads) 
      sum += tex1Dfetch(texref_f1_a, iIndex);
      iIndex += GroupSize;
    }

    odata[xIndex] = sum;
  }
}

__global__ void SUMC_TEX_KERNEL(int n,  int offset, int Nthreads, int M, int GroupSize, int GroupOffset, Complex *odata)
{
  unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x; // index used for output
  unsigned int iGroup = (int) floorf(xIndex/GroupSize);
  unsigned int iIndex = xIndex - iGroup*(GroupSize  - GroupOffset);
  // Domanda, il for è meglio farlo dentro if o fuori?
  Complex sum = make_float2(0.0,0.0);
  if (xIndex < Nthreads) { 

    for (int i=0;i<M;i++) {
      //if (xIndex < Nthreads) 
      sum = plus(sum,tex1Dfetch(texref_c1_a, iIndex));
      iIndex += GroupSize;
    }

    odata[xIndex] = sum;
  }
}

__global__ void SUMD_TEX_KERNEL(int n,  int offset, int Nthreads, int M, int GroupSize, int GroupOffset, double *odata)
{
  unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x; // index used for output
  unsigned int iGroup = (int) floorf(xIndex/GroupSize);
  unsigned int iIndex = xIndex - iGroup*(GroupSize  - GroupOffset);
  // Domanda, il for è meglio farlo dentro if o fuori?
  double sum = 0.0;
  if (xIndex < Nthreads) { 

    for (int i=0;i<M;i++) {
      //if (xIndex < Nthreads) 
      sum += tex1Dfetch_double(texref_d1_a, iIndex);
      iIndex += GroupSize;
    }

    odata[xIndex] = sum;
  }
}

__global__ void SUMCD_TEX_KERNEL(int n,  int offset, int Nthreads, int M, int GroupSize, int GroupOffset, DoubleComplex *odata)
{
  unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x; // index used for output
  unsigned int iGroup = (int) floorf(xIndex/GroupSize);
  unsigned int iIndex = xIndex - iGroup*(GroupSize  - GroupOffset);
  // Domanda, il for è meglio farlo dentro if o fuori?
  DoubleComplex sum = make_double2(0.0,0.0);
  if (xIndex < Nthreads) { 

    for (int i=0;i<M;i++) {
      //if (xIndex < Nthreads) 
      sum = plus(sum,tex1Dfetch_DoubleComplex(texref_cd1_a, iIndex));
      iIndex += GroupSize;
    }

    odata[xIndex] = sum;
  }
}



/* The following kernel uses 2D texture */
__global__ void SUMF_KERNEL(int Nthreads, int M, int GroupSize, int GroupOffset, const int texwidth, float *odata)
{
  unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x; // index used for output
  unsigned int iGroup = (int) floorf(xIndex/GroupSize);
  unsigned int iIndex = xIndex - iGroup*(GroupSize  - GroupOffset);
  unsigned int texy   = 0;
  unsigned int texx   = 0;
  // Domanda, il for è meglio farlo dentro if o fuori?
  float sum = 0.0;
  if (xIndex < Nthreads) { 

    for (int i=0;i<M;i++) {
      texy   = (int) floorf(iIndex/texwidth);
      texx   = iIndex - texy*texwidth;

      //if (xIndex < Nthreads) 
      sum += tex2D(texref_f2_a, (float) texx, (float) texy);
      iIndex += GroupSize;

    }

    odata[xIndex] = sum;
  }
}

#define MYBLOCK_DIM1D (BLOCK_DIM2D*BLOCK_DIM2D*2) 
__global__ void SUM1F_TEX_KERNEL(int Nthreads, int M, int GroupSize, int GroupOffset, float *odata)
{
 
 unsigned int threadIdxEff = threadIdx.x + blockDim.x*threadIdx.y;
 unsigned int effThread = (threadIdxEff % (Nthreads - blockIdx.x * MYBLOCK_DIM1D));
 unsigned int xIndex = blockIdx.x * MYBLOCK_DIM1D + effThread; 
 unsigned int iGroup = (int) floorf(xIndex/GroupSize);
 unsigned int iIndex = xIndex - iGroup*(GroupSize  - GroupOffset);

 __shared__ float sum[MYBLOCK_DIM1D];
   
 int blockThreads = (int) fminf((Nthreads - blockIdx.x * MYBLOCK_DIM1D), MYBLOCK_DIM1D);

 sum[threadIdxEff] = 0.0;

 int Ntot =  (int) floorf(MYBLOCK_DIM1D/blockThreads);
 int Nresidual = MYBLOCK_DIM1D - Ntot*blockThreads;
 if (effThread < Nresidual) 
   Ntot ++;

 int Noffs = (int) floorf(threadIdxEff/blockThreads); 
 
 int Meff = (int) floorf(M/Ntot);
 Meff = Meff==0?1:Meff;  
 Meff = (M-Meff*Noffs)<=0?0:Meff;  
 Meff = (Noffs<(M-Meff*Ntot))?Meff+1:Meff;
 
 
 for (int i=0;i<Meff;i++) {
    sum[threadIdxEff] += tex1Dfetch(texref_f1_a, iIndex+GroupSize*Noffs+GroupSize*i*Ntot);
 }

 syncthreads();
 
 // only the first threads must clean the cache and init the index
 float finalsum = 0.0;
 if (threadIdxEff < blockThreads) {
   for  (int i=0;i<Ntot;i++) {
     finalsum += sum[threadIdxEff + blockThreads*i]; 
   }
   odata[xIndex] = finalsum;

 }
 /*syncthreads();

 xIndex = blockIdx.x * MYBLOCK_DIM1D + threadIdx.x; 
 // sum
 if (xIndex < Nthreads) {
   odata[xIndex] = finalsum;
 }*/

}


/*__global__ void SUM1F_TEX_KERNEL(int n , int offset, int Nthreads, int M, int GroupSize, int GroupOffset, float *odata)
{
  unsigned int xIndex = blockIdx.x; // index used for output
  //unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x; // index used for output
  unsigned int iGroup = (int) floorf(xIndex/GroupSize);
  unsigned int iIndex = xIndex - iGroup*(GroupSize  - GroupOffset);
  // Domanda, il for è meglio farlo dentro if o fuori?
  float sum = 0.0;
 
  if (threadIdx.x < Nthreads) { 
    for (int i=0;i<M;i++) {
      //if (xIndex < Nthreads) 
      sum += tex1Dfetch(texref_f1_a, iIndex +);
      iIndex += GroupSize;
    }

    odata[xIndex] = sum;
  }
}*/
/*
 * 
 *
 * n : number of threads
 * odata : output vector of indexes
 *
 * Explanation:
 *
 * 1) each thread generates a number based on xIndex
 * 2) each thread calculates his position as a N-dimensional value
 *    and translates to a 1D index values
 * 3) Have to manage this type of indexing A([1 2 3],[4 5 6])
 *    or A(1:2,:)
 * 4) Need to have idx0, idx1, idx2 for each dimension storing indexing
 *    of type A([1 2 3],[4 5 6])
 * 5) Matlab passes the calculated indexes to function subsref as array of values.
 *    I have to copy these values to GPU memory and use here. 
 * 6) configuration parameters and idxn vector all stored into texture. 
 *
 *    The texture has the following format
 *    nd0: number dimension 0 
 *    nd1: " 
 *    nd2: " 
 *    nd3: " 
 *    nd4: " 
 *    nidx0: size of idx0
 *    nidx1: size of idx1
 *    nidx2: size of idx2
 *    nidx3: size of idx3
 *    nidx4: size of idx4
 *    idx0:
 *    idx1
 *    idx2
 *    idx3
 *    idx4
 *
 *  7) Index in vectors idx0 idx1 etc are considered to be in C format. If not
 *     use the idxshift to shift the value
 *
 *
 *
 * General formulas
 * Given the ND position (p0,p1,p2,p3,p4) and
 * dimensions (nd0, nd1, nd2, nd3, nd4)
 * 
 * The translation on the position to 1D is
 * mypos = p0 + p1*nd0 + p2*nd0*nd1 + p3*nd0*nd1*nd2 + p4*nd0*nd1*nd2*nd3;
 * or
 * mypos = p0 + nd0*(p1 + nd1*(p2 + nd2*(p3 + p4*nd3)))
 *
 * A(2:3,:,2)
 */
/*__global__ void  SUBSINDEXF_KERNEL(unsigned int n, 
			    float *odata, int idxshift, unsigned int offset)
{
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    int nd0 = tex1Dfetch(texref_i1_a, 0); 
    int nd1 = tex1Dfetch(texref_i1_a, 1); 
    int nd2 = tex1Dfetch(texref_i1_a, 2); 
    int nd3 = tex1Dfetch(texref_i1_a, 3); 
    int nd4 = tex1Dfetch(texref_i1_a, 4); 
    
    int nidx0 = tex1Dfetch(texref_i1_a, 5); 
    int nidx1 = tex1Dfetch(texref_i1_a, 6); 
    int nidx2 = tex1Dfetch(texref_i1_a, 7); 
    int nidx3 = tex1Dfetch(texref_i1_a, 8); 
    int nidx4 = tex1Dfetch(texref_i1_a, 9); 

// define idx positions
#define IDX0 (15) 
#define IDX1 (IDX0+nidx0) 
#define IDX2 (IDX1+nidx1) 
#define IDX3 (IDX2+nidx2) 
#define IDX4 (IDX3+nidx3) 

// with p0i, etc we calculate the 5D index given the 1D index xIndex
#define p0i  (fmodf(xIndex,nidx0))
#define p1i  (fmodf(floorf(xIndex/nidx0),nidx1))
#define p2i  (fmodf(floorf(xIndex/(nidx0*nidx1)),nidx2))
#define p3i  (fmodf(floorf(xIndex/(nidx0*nidx1*nidx2)),nidx3))
#define p4i  (fmodf(floorf(xIndex/(nidx0*nidx1*nidx2*nidx3)),nidx4))

    int p0 = tex1Dfetch(texref_i1_a,IDX0+p0i)+idxshift;
    int p1 = tex1Dfetch(texref_i1_a,IDX1+p1i)+idxshift;
    int p2 = tex1Dfetch(texref_i1_a,IDX2+p2i)+idxshift;
    int p3 = tex1Dfetch(texref_i1_a,IDX3+p3i)+idxshift;
    int p4 = tex1Dfetch(texref_i1_a,IDX4+p4i)+idxshift;
    
    unsigned int mypos;

    if ((xIndex - offset) < n) {
       //mypos = p0 + p1*nd0 + p2*nd0*nd1 + p3*nd0*nd1*nd2 + p4*nd0*nd1*nd2*nd3;
       mypos = p0 + nd0*(p1 + nd1*(p2 + nd2*(p3 + p4*nd3)));
       //odata[xIndex] = p0i;
       odata[xIndex] = tex1Dfetch(texref1,mypos);
    }
       
    //odata[xIndex] = tex2D(texref2, (float) texx, (float) texy);
}*/
/***********************************************************************************/
/* SUBSREF */
__global__ void  PERMSUBSINDEX1F_KERNEL(unsigned int n, int offset,  
			    float *odata, float *idata, int nsiz, int nidx, int nnd, int nncons, int dir, int scalar) {

#define SIZ(x)   tex1Dfetch(texref_i1_a, (x))
#define PERM(x)  tex1Dfetch(texref_i1_a, (x)+nsiz)
#define IDX(x)   tex1Dfetch(texref_i1_a, (x)+nsiz+nsiz)
#define ND(x)    tex1Dfetch(texref_i1_a, (x)+nsiz+nsiz+nidx) 
#define NCONS(x) tex1Dfetch(texref_i1_a, (x)+nsiz+nsiz+nidx+nnd)

// nd is dimensions of idata, starting from 1
// ndims(idata) = 2 3 2
// nd = 1 2 3 2
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    int k = 1; // the same a cumulative product
    //int offs = 0;
    for (int i=0;i<nsiz;i++) {
      k = k*SIZ(i);
      //if ( NCONS(i) )
      //  offs += 1; 
      //else
      //  offs += SIZ(i); 
    }

    if ((xIndex - offset) < n) {
      /* If scalar==1 means that RHS is a scalar and LHS is not. For example
       * A(1:100) = 1
       * In this case the RHS index is not advanced with xIndex
       */
      int ndx = xIndex + 1;
      if ((dir==0)&&(scalar==1))
        ndx = 1;

      int linidx = 0;
      for (int i=nsiz-1;i>=0;i--) {
	k = k/SIZ(PERM(i));

        int offs = 0;
	int cumprod = 1;
        for (int ii=0;ii<PERM(i);ii++) {
	  cumprod = cumprod*SIZ(ii);
	  if (NCONS(ii))
	    offs = offs + 1;
	  else
	    offs = offs + SIZ(ii);
	}
	
        int vi = ((ndx - 1) % k) + 1;         
        int vj = (ndx - vi)/k + 1;
	
	// to go to a linear we use the formula from sub2ind in Matlab
	int myidx = 0;
	if (NCONS(PERM(i)))
          myidx = IDX(offs) + NCONS(PERM(i))*(vj-1);	
	else
          myidx = IDX(offs + (vj-1));	
	
	linidx += myidx*cumprod;

	// nd-1*(p0+nd0*(p1+nd1*(p2+nd2*(p3+p4*nd3)))) 
        ndx = vi;
        	
      }
      if (dir==0) {
        odata[xIndex] = tex1Dfetch(texref_f1_a, linidx); //idata[linidx];
      } else {
	if (scalar==0)
          odata[linidx] = tex1Dfetch(texref_f1_a, xIndex); //idata[xIndex];
	else
          odata[linidx] = tex1Dfetch(texref_f1_a, 0); //idata[xIndex];
      }

    }
}

#undef SIZ
#undef PERM
#undef IDX
#undef ND
#undef NCONS

__global__ void  PERMSUBSINDEX1C_KERNEL(unsigned int n, int offset,  
			    Complex *odata, Complex *idata, int nsiz, int nidx, int nnd, int nncons, int dir, int scalar) {

#define SIZ(x)   tex1Dfetch(texref_i1_a, (x))
#define PERM(x)  tex1Dfetch(texref_i1_a, (x)+nsiz)
#define IDX(x)   tex1Dfetch(texref_i1_a, (x)+nsiz+nsiz)
#define ND(x)    tex1Dfetch(texref_i1_a, (x)+nsiz+nsiz+nidx) 
#define NCONS(x) tex1Dfetch(texref_i1_a, (x)+nsiz+nsiz+nidx+nnd)


// nd is dimensions of idata, starting from 1
// ndims(idata) = 2 3 2
// nd = 1 2 3 2
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    int k = 1; // the same a cumulative product
    //int offs = 0;
    for (int i=0;i<nsiz;i++) {
      k = k*SIZ(i);
      //if ( NCONS(i) )
      //  offs += 1; 
      //else
      //  offs += SIZ(i); 
    }

    if ((xIndex - offset) < n) {
      /* If scalar==1 means that RHS is a scalar and LHS is not. For example
       * A(1:100) = 1
       * In this case the RHS index is not advanced with xIndex
       */
      int ndx = xIndex + 1;
      if ((dir==0)&&(scalar==1))
        ndx = 1;

      int linidx = 0;
      for (int i=nsiz-1;i>=0;i--) {
	k = k/SIZ(PERM(i));

        int offs = 0;
	int cumprod = 1;
        for (int ii=0;ii<PERM(i);ii++) {
	  cumprod = cumprod*SIZ(ii);
	  if (NCONS(ii))
	    offs = offs + 1;
	  else
	    offs = offs + SIZ(ii);
	}
	
        int vi = ((ndx - 1) % k) + 1;         
        int vj = (ndx - vi)/k + 1;
	
	// to go to a linear we use the formula from sub2ind in Matlab
	int myidx = 0;
	if (NCONS(PERM(i)))
          myidx = IDX(offs) + NCONS(PERM(i))*(vj-1);	
	else
          myidx = IDX(offs + (vj-1));	
	
	linidx += myidx*cumprod;

	// nd-1*(p0+nd0*(p1+nd1*(p2+nd2*(p3+p4*nd3)))) 
        ndx = vi;
        	
      }
      if (dir==0) {
        odata[xIndex] = tex1Dfetch_Complex(texref_c1_a, linidx); //idata[linidx];
      } else {
	if (scalar==0)
          odata[linidx] = tex1Dfetch_Complex(texref_c1_a, xIndex); //idata[xIndex];
	else
          odata[linidx] = tex1Dfetch_Complex(texref_c1_a, 0); //idata[xIndex];
      }

    }
}
#undef SIZ
#undef PERM
#undef IDX
#undef ND
#undef NCONS

__global__ void  PERMSUBSINDEX1D_KERNEL(unsigned int n, int offset,  
			    double *odata, double *idata, int nsiz, int nidx, int nnd, int nncons, int dir, int scalar) {

#define SIZ(x)   tex1Dfetch(texref_i1_a, (x))
#define PERM(x)  tex1Dfetch(texref_i1_a, (x)+nsiz)
#define IDX(x)   tex1Dfetch(texref_i1_a, (x)+nsiz+nsiz)
#define ND(x)    tex1Dfetch(texref_i1_a, (x)+nsiz+nsiz+nidx) 
#define NCONS(x) tex1Dfetch(texref_i1_a, (x)+nsiz+nsiz+nidx+nnd)


// nd is dimensions of idata, starting from 1
// ndims(idata) = 2 3 2
// nd = 1 2 3 2
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    int k = 1; // the same a cumulative product
    //int offs = 0;
    for (int i=0;i<nsiz;i++) {
      k = k*SIZ(i);
      //if ( NCONS(i) )
      //  offs += 1; 
      //else
      //  offs += SIZ(i); 
    }

    if ((xIndex - offset) < n) {
      /* If scalar==1 means that RHS is a scalar and LHS is not. For example
       * A(1:100) = 1
       * In this case the RHS index is not advanced with xIndex
       */
      int ndx = xIndex + 1;
      if ((dir==0)&&(scalar==1))
        ndx = 1;

      int linidx = 0;
      for (int i=nsiz-1;i>=0;i--) {
	k = k/SIZ(PERM(i));

        int offs = 0;
	int cumprod = 1;
        for (int ii=0;ii<PERM(i);ii++) {
	  cumprod = cumprod*SIZ(ii);
	  if (NCONS(ii))
	    offs = offs + 1;
	  else
	    offs = offs + SIZ(ii);
	}
	
        int vi = ((ndx - 1) % k) + 1;         
        int vj = (ndx - vi)/k + 1;
	
	// to go to a linear we use the formula from sub2ind in Matlab
	int myidx = 0;
	if (NCONS(PERM(i)))
          myidx = IDX(offs) + NCONS(PERM(i))*(vj-1);	
	else
          myidx = IDX(offs + (vj-1));	
	
	linidx += myidx*cumprod;

	// nd-1*(p0+nd0*(p1+nd1*(p2+nd2*(p3+p4*nd3)))) 
        ndx = vi;
        	
      }
      if (dir==0) {
        odata[xIndex] = tex1Dfetch_double(texref_d1_a, linidx); //idata[linidx];
      } else {
	if (scalar==0)
          odata[linidx] = tex1Dfetch_double(texref_d1_a, xIndex); //idata[xIndex];
	else
          odata[linidx] = tex1Dfetch_double(texref_d1_a, 0); //idata[xIndex];
      }

    }
}
#undef SIZ
#undef PERM
#undef IDX
#undef ND
#undef NCONS

__global__ void  PERMSUBSINDEX1CD_KERNEL(unsigned int n, int offset,  
			    DoubleComplex *odata, DoubleComplex *idata, int nsiz, int nidx, int nnd, int nncons, int dir, int scalar) {

#define SIZ(x)   tex1Dfetch(texref_i1_a, (x))
#define PERM(x)  tex1Dfetch(texref_i1_a, (x)+nsiz)
#define IDX(x)   tex1Dfetch(texref_i1_a, (x)+nsiz+nsiz)
#define ND(x)    tex1Dfetch(texref_i1_a, (x)+nsiz+nsiz+nidx) 
#define NCONS(x) tex1Dfetch(texref_i1_a, (x)+nsiz+nsiz+nidx+nnd)


// nd is dimensions of idata, starting from 1
// ndims(idata) = 2 3 2
// nd = 1 2 3 2
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    int k = 1; // the same a cumulative product
    //int offs = 0;
    for (int i=0;i<nsiz;i++) {
      k = k*SIZ(i);
      //if ( NCONS(i) )
      //  offs += 1; 
      //else
      //  offs += SIZ(i); 
    }

    if ((xIndex - offset) < n) {
      /* If scalar==1 means that RHS is a scalar and LHS is not. For example
       * A(1:100) = 1
       * In this case the RHS index is not advanced with xIndex
       */
      int ndx = xIndex + 1;
      if ((dir==0)&&(scalar==1))
        ndx = 1;

      int linidx = 0;
      for (int i=nsiz-1;i>=0;i--) {
	k = k/SIZ(PERM(i));

        int offs = 0;
	int cumprod = 1;
        for (int ii=0;ii<PERM(i);ii++) {
	  cumprod = cumprod*SIZ(ii);
	  if (NCONS(ii))
	    offs = offs + 1;
	  else
	    offs = offs + SIZ(ii);
	}
	
        int vi = ((ndx - 1) % k) + 1;         
        int vj = (ndx - vi)/k + 1;
	
	// to go to a linear we use the formula from sub2ind in Matlab
	int myidx = 0;
	if (NCONS(PERM(i)))
          myidx = IDX(offs) + NCONS(PERM(i))*(vj-1);	
	else
          myidx = IDX(offs + (vj-1));	
	
	linidx += myidx*cumprod;

	// nd-1*(p0+nd0*(p1+nd1*(p2+nd2*(p3+p4*nd3)))) 
        ndx = vi;
        	
      }
      if (dir==0) {
        odata[xIndex] = tex1Dfetch_DoubleComplex(texref_cd1_a, linidx); //idata[linidx];
      } else {
	if (scalar==0)
          odata[linidx] = tex1Dfetch_DoubleComplex(texref_cd1_a, xIndex); //idata[xIndex];
	else
          odata[linidx] = tex1Dfetch_DoubleComplex(texref_cd1_a, 0); //idata[xIndex];
      }

    }
}
#undef SIZ
#undef PERM
#undef IDX
#undef ND
#undef NCONS


/***********************************************************************************/

/* SUBSREF */
__global__ void  SUBSINDEX1F_KERNEL(unsigned int n, int offset,  
			    float *odata, float *idata, int nsiz, int nidx, int nnd, int nncons, int dir, int scalar) {
/*__global__ void SUBSREF(int n, 
                      int offset, 
		      int nsiz,
		      //float   * siz,
		      //float   * idx,
		      //float   * nd,
		      //float   * ncons,
                      int   * siz,
		      int   * idx,
		      int   * nd,
		      int   * ncons,
                      float * odata,
                      float * idata,
		      int dir
		      )*/
#define SIZ(x)   tex1Dfetch(texref_i1_a, (x))
#define IDX(x)   tex1Dfetch(texref_i1_a, (x)+nsiz)
#define ND(x)    tex1Dfetch(texref_i1_a, (x)+nsiz+nidx) 
#define NCONS(x) tex1Dfetch(texref_i1_a, (x)+nsiz+nidx+nnd)

// nd is dimensions of idata, starting from 1
// ndims(idata) = 2 3 2
// nd = 1 2 3 2
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    int k = 1;
    int offs = 0;
    for (int i=0;i<nsiz;i++) {
      k = k*SIZ(i);
      if ( NCONS(i) )
        offs += 1; 
      else
        offs += SIZ(i); 
    }

    if ((xIndex - offset) < n) {
      /* If scalar==1 means that RHS is a scalar and LHS is not. For example
       * A(1:100) = 1
       * In this case the RHS index is not advanced with xIndex
       */
      int ndx = xIndex + 1;
      if ((dir==0)&&(scalar==1))
        ndx = 1;

      int linidx = 0;
      /* The following loop is in part the implementation of the ind2sub
       * Matlab function
       */
      for (int i=nsiz-1;i>=0;i--) {
	k = k/SIZ(i);

	if (NCONS(i))
	  offs = offs - 1;
	else
	  offs = offs - SIZ(i);
	
        int vi = ((ndx - 1) % k) + 1;         
        int vj = (ndx - vi)/k + 1;
	
	if (NCONS(i))
          linidx = ND(i)*( IDX(offs) + NCONS(i)*(vj-1) + linidx);	
	else
          linidx = ND(i)*( IDX(offs + (vj-1)) + linidx);	

	// nd-1*(p0+nd0*(p1+nd1*(p2+nd2*(p3+p4*nd3)))) 
        ndx = vi;
        	
      }
      if (dir==0) {
        odata[xIndex] = tex1Dfetch(texref_f1_a, linidx); //idata[linidx];
      } else {
	if (scalar==0)
          odata[linidx] = tex1Dfetch(texref_f1_a, xIndex); //idata[xIndex];
	else
          odata[linidx] = tex1Dfetch(texref_f1_a, 0); //idata[xIndex];
      }

    }
}
#undef SIZ
#undef IDX
#undef ND
#undef NCONS

__global__ void  SUBSINDEX1C_KERNEL(unsigned int n, int offset,  
			    Complex *odata, Complex *idata, int nsiz, int nidx, int nnd, int nncons, int dir, int scalar) {
/*__global__ void SUBSREF(int n, 
                      int offset, 
		      int nsiz,
		      //float   * siz,
		      //float   * idx,
		      //float   * nd,
		      //float   * ncons,
                      int   * siz,
		      int   * idx,
		      int   * nd,
		      int   * ncons,
                      float * odata,
                      float * idata,
		      int dir
		      )*/
#define SIZ(x)   tex1Dfetch(texref_i1_a, (x))
#define IDX(x)   tex1Dfetch(texref_i1_a, (x)+nsiz)
#define ND(x)    tex1Dfetch(texref_i1_a, (x)+nsiz+nidx) 
#define NCONS(x) tex1Dfetch(texref_i1_a, (x)+nsiz+nidx+nnd)

// nd is dimensions of idata, starting from 1
// ndims(idata) = 2 3 2
// nd = 1 2 3 2
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    int k = 1;
    int offs = 0;
    for (int i=0;i<nsiz;i++) {
      k = k*SIZ(i);
      if ( NCONS(i) )
        offs += 1; 
      else
        offs += SIZ(i); 
    }

    if ((xIndex - offset) < n) {
      /* If scalar==1 means that RHS is a scalar and LHS is not. For example
       * A(1:100) = 1
       * In this case the RHS index is not advanced with xIndex
       */
      int ndx = xIndex + 1;
      if ((dir==0)&&(scalar==1))
        ndx = 1;

      int linidx = 0;
      for (int i=nsiz-1;i>=0;i--) {
	k = k/SIZ(i);

	if (NCONS(i))
	  offs = offs - 1;
	else
	  offs = offs - SIZ(i);
	
        int vi = ((ndx - 1) % k) + 1;         
        int vj = (ndx - vi)/k + 1;
	
	if (NCONS(i))
          linidx = ND(i)*( IDX(offs) + NCONS(i)*(vj-1) + linidx);	
	else
          linidx = ND(i)*( IDX(offs + (vj-1)) + linidx);	

	// nd-1*(p0+nd0*(p1+nd1*(p2+nd2*(p3+p4*nd3)))) 
        ndx = vi;
        	
      }
      if (dir==0) {
        odata[xIndex] = tex1Dfetch_Complex(texref_c1_a, linidx); //idata[linidx];
      } else {
	if (scalar==0)
          odata[linidx] = tex1Dfetch_Complex(texref_c1_a, xIndex); //idata[xIndex];
	else
          odata[linidx] = tex1Dfetch_Complex(texref_c1_a, 0); //idata[xIndex];
      }

    }
}
#undef SIZ
#undef IDX
#undef ND
#undef NCONS

__global__ void  SUBSINDEX1D_KERNEL(unsigned int n, int offset,  
			    double *odata, double *idata, int nsiz, int nidx, int nnd, int nncons, int dir, int scalar) {
/*__global__ void SUBSREF(int n, 
                      int offset, 
		      int nsiz,
		      //float   * siz,
		      //float   * idx,
		      //float   * nd,
		      //float   * ncons,
                      int   * siz,
		      int   * idx,
		      int   * nd,
		      int   * ncons,
                      float * odata,
                      float * idata,
		      int dir
		      )*/
#define SIZ(x)   tex1Dfetch(texref_i1_a, (x))
#define IDX(x)   tex1Dfetch(texref_i1_a, (x)+nsiz)
#define ND(x)    tex1Dfetch(texref_i1_a, (x)+nsiz+nidx) 
#define NCONS(x) tex1Dfetch(texref_i1_a, (x)+nsiz+nidx+nnd)

// nd is dimensions of idata, starting from 1
// ndims(idata) = 2 3 2
// nd = 1 2 3 2
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    int k = 1;
    int offs = 0;
    for (int i=0;i<nsiz;i++) {
      k = k*SIZ(i);
      if ( NCONS(i) )
        offs += 1; 
      else
        offs += SIZ(i); 
    }

    if ((xIndex - offset) < n) {
       /* If scalar==1 means that RHS is a scalar and LHS is not. For example
       * A(1:100) = 1
       * In this case the RHS index is not advanced with xIndex
       */
      int ndx = xIndex + 1;
      if ((dir==0)&&(scalar==1))
        ndx = 1;

      int linidx = 0;
      for (int i=nsiz-1;i>=0;i--) {
	k = k/SIZ(i);

	if (NCONS(i))
	  offs = offs - 1;
	else
	  offs = offs - SIZ(i);
	
        int vi = ((ndx - 1) % k) + 1;         
        int vj = (ndx - vi)/k + 1;
	
	if (NCONS(i))
          linidx = ND(i)*( IDX(offs) + NCONS(i)*(vj-1) + linidx);	
	else
          linidx = ND(i)*( IDX(offs + (vj-1)) + linidx);	

	// nd-1*(p0+nd0*(p1+nd1*(p2+nd2*(p3+p4*nd3)))) 
        ndx = vi;
        	
      }
      if (dir==0) {
        odata[xIndex] = tex1Dfetch_double(texref_d1_a, linidx); //idata[linidx];
      } else {
	if (scalar==0)
          odata[linidx] = tex1Dfetch_double(texref_d1_a, xIndex); //idata[xIndex];
	else
          odata[linidx] = tex1Dfetch_double(texref_d1_a, 0); //idata[xIndex];
      }

    }
}
#undef SIZ
#undef IDX
#undef ND
#undef NCONS

__global__ void  SUBSINDEX1CD_KERNEL(unsigned int n, int offset,  
			    DoubleComplex *odata, DoubleComplex *idata, int nsiz, int nidx, int nnd, int nncons, int dir, int scalar) {
/*__global__ void SUBSREF(int n, 
                      int offset, 
		      int nsiz,
		      //float   * siz,
		      //float   * idx,
		      //float   * nd,
		      //float   * ncons,
                      int   * siz,
		      int   * idx,
		      int   * nd,
		      int   * ncons,
                      float * odata,
                      float * idata,
		      int dir
		      )*/
#define SIZ(x)   tex1Dfetch(texref_i1_a, (x))
#define IDX(x)   tex1Dfetch(texref_i1_a, (x)+nsiz)
#define ND(x)    tex1Dfetch(texref_i1_a, (x)+nsiz+nidx) 
#define NCONS(x) tex1Dfetch(texref_i1_a, (x)+nsiz+nidx+nnd)

// nd is dimensions of idata, starting from 1
// ndims(idata) = 2 3 2
// nd = 1 2 3 2
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    int k = 1;
    int offs = 0;
    for (int i=0;i<nsiz;i++) {
      k = k*SIZ(i);
      if ( NCONS(i) )
        offs += 1; 
      else
        offs += SIZ(i); 
    }

    if ((xIndex - offset) < n) {
       /* If scalar==1 means that RHS is a scalar and LHS is not. For example
       * A(1:100) = 1
       * In this case the RHS index is not advanced with xIndex
       */
      int ndx = xIndex + 1;
      if ((dir==0)&&(scalar==1))
        ndx = 1;

      int linidx = 0;
      for (int i=nsiz-1;i>=0;i--) {
	k = k/SIZ(i);

	if (NCONS(i))
	  offs = offs - 1;
	else
	  offs = offs - SIZ(i);
	
        int vi = ((ndx - 1) % k) + 1;         
        int vj = (ndx - vi)/k + 1;
	
	if (NCONS(i))
          linidx = ND(i)*( IDX(offs) + NCONS(i)*(vj-1) + linidx);	
	else
          linidx = ND(i)*( IDX(offs + (vj-1)) + linidx);	

	// nd-1*(p0+nd0*(p1+nd1*(p2+nd2*(p3+p4*nd3)))) 
        ndx = vi;
        	
      }
      if (dir==0) {
        odata[xIndex] = tex1Dfetch_DoubleComplex(texref_cd1_a, linidx); //idata[linidx];
      } else {
	if (scalar==0)
          odata[linidx] = tex1Dfetch_DoubleComplex(texref_cd1_a, xIndex); //idata[xIndex];
	else
          odata[linidx] = tex1Dfetch_DoubleComplex(texref_cd1_a, 0); //idata[xIndex];
      }

    }
}
#undef SIZ
#undef IDX
#undef ND
#undef NCONS


__global__ void  SUBSINDEXF_KERNEL(unsigned int n, int offset,  
			    float *odata, int idxshift)
{
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    int nd0 = tex1Dfetch(texref_i1_a, 0); 
    int nd1 = tex1Dfetch(texref_i1_a, 1); 
    int nd2 = tex1Dfetch(texref_i1_a, 2); 
    int nd3 = tex1Dfetch(texref_i1_a, 3); 
    int nd4 = tex1Dfetch(texref_i1_a, 4); 
    
    int nidx0 = tex1Dfetch(texref_i1_a, 5); 
    int nidx1 = tex1Dfetch(texref_i1_a, 6); 
    int nidx2 = tex1Dfetch(texref_i1_a, 7); 
    int nidx3 = tex1Dfetch(texref_i1_a, 8); 
    int nidx4 = tex1Dfetch(texref_i1_a, 9);

    int ncons0 = tex1Dfetch(texref_i1_a, 10); 
    int ncons1 = tex1Dfetch(texref_i1_a, 11); 
    int ncons2 = tex1Dfetch(texref_i1_a, 12); 
    int ncons3 = tex1Dfetch(texref_i1_a, 13); 
    int ncons4 = tex1Dfetch(texref_i1_a, 14); 

    int neff0 = (ncons0==0)?nidx0:1;
    int neff1 = (ncons1==0)?nidx1:1;
    int neff2 = (ncons2==0)?nidx2:1;
    int neff3 = (ncons3==0)?nidx3:1;
    //int neff4 = (ncons4==0)?nidx4:1;


// define idx positions

#define IDX0 (15) 
#define IDX1 (IDX0+neff0) 
#define IDX2 (IDX1+neff1) 
#define IDX3 (IDX2+neff2) 
#define IDX4 (IDX3+neff3) 

    int p0 = (ncons0==0)?tex1Dfetch(texref_i1_a,IDX0+(xIndex % nidx0))+idxshift:
	    tex1Dfetch(texref_i1_a,IDX0)+idxshift+ ncons0*(xIndex % nidx0);
    int p1 = (ncons1==0)?tex1Dfetch(texref_i1_a,IDX1+((xIndex/nidx0) % nidx1))+idxshift:
	    tex1Dfetch(texref_i1_a,IDX1)+idxshift+ ncons1*((xIndex/nidx0) % nidx1);
    int p2 = (ncons2==0)?tex1Dfetch(texref_i1_a,IDX2+((xIndex/(nidx0*nidx1)) % nidx2))+idxshift:
	    tex1Dfetch(texref_i1_a,IDX2)+idxshift+ ncons2*((xIndex/(nidx0*nidx1)) % nidx2);
    int p3 = (ncons3==0)?tex1Dfetch(texref_i1_a,IDX3+((xIndex/(nidx0*nidx1*nidx2)) % nidx3))+idxshift:
	    tex1Dfetch(texref_i1_a,IDX3)+idxshift+ ncons3*((xIndex/(nidx0*nidx1*nidx2)) % nidx3);
    int p4 = (ncons4==0)?tex1Dfetch(texref_i1_a,IDX4+((xIndex/(nidx0*nidx1*nidx2*nidx3)) % nidx4))+idxshift:
	    tex1Dfetch(texref_i1_a,IDX4)+idxshift+ ncons4*((xIndex/(nidx0*nidx1*nidx2*nidx3)) % nidx4);

    unsigned int mypos;

    if ((xIndex - offset) < n) {
       //mypos = p0 + p1*nd0 + p2*nd0*nd1 + p3*nd0*nd1*nd2 + p4*nd0*nd1*nd2*nd3;
       mypos = p0 + nd0*(p1 + nd1*(p2 + nd2*(p3 + p4*nd3)));
       odata[xIndex] = tex1Dfetch(texref_f1_a,mypos);
    }
       
    //odata[xIndex] = tex2D(texref2, (float) texx, (float) texy);
}


__global__ void  SUBSINDEXC_KERNEL(int n, int offset, 
			    Complex *odata, int idxshift)
{
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    int nd0 = tex1Dfetch(texref_i1_a, 0); 
    int nd1 = tex1Dfetch(texref_i1_a, 1); 
    int nd2 = tex1Dfetch(texref_i1_a, 2); 
    int nd3 = tex1Dfetch(texref_i1_a, 3); 
    int nd4 = tex1Dfetch(texref_i1_a, 4); 
    
    int nidx0 = tex1Dfetch(texref_i1_a, 5); 
    int nidx1 = tex1Dfetch(texref_i1_a, 6); 
    int nidx2 = tex1Dfetch(texref_i1_a, 7); 
    int nidx3 = tex1Dfetch(texref_i1_a, 8); 
    int nidx4 = tex1Dfetch(texref_i1_a, 9);

    int ncons0 = tex1Dfetch(texref_i1_a, 10); 
    int ncons1 = tex1Dfetch(texref_i1_a, 11); 
    int ncons2 = tex1Dfetch(texref_i1_a, 12); 
    int ncons3 = tex1Dfetch(texref_i1_a, 13); 
    int ncons4 = tex1Dfetch(texref_i1_a, 14); 

    int neff0 = (ncons0==0)?nidx0:1;
    int neff1 = (ncons1==0)?nidx1:1;
    int neff2 = (ncons2==0)?nidx2:1;
    int neff3 = (ncons3==0)?nidx3:1;
    //int neff4 = (ncons4==0)?nidx4:1; 

// define idx positions
   int p0 = (ncons0==0)?tex1Dfetch(texref_i1_a,IDX0+(xIndex % nidx0))+idxshift:
	    tex1Dfetch(texref_i1_a,IDX0)+idxshift+ ncons0*(xIndex % nidx0);
    int p1 = (ncons1==0)?tex1Dfetch(texref_i1_a,IDX1+((xIndex/nidx0) % nidx1))+idxshift:
	    tex1Dfetch(texref_i1_a,IDX1)+idxshift+ ncons1*((xIndex/nidx0) % nidx1);
    int p2 = (ncons2==0)?tex1Dfetch(texref_i1_a,IDX2+((xIndex/(nidx0*nidx1)) % nidx2))+idxshift:
	    tex1Dfetch(texref_i1_a,IDX2)+idxshift+ ncons2*((xIndex/(nidx0*nidx1)) % nidx2);
    int p3 = (ncons3==0)?tex1Dfetch(texref_i1_a,IDX3+((xIndex/(nidx0*nidx1*nidx2)) % nidx3))+idxshift:
	    tex1Dfetch(texref_i1_a,IDX3)+idxshift+ ncons3*((xIndex/(nidx0*nidx1*nidx2)) % nidx3);
    int p4 = (ncons4==0)?tex1Dfetch(texref_i1_a,IDX4+((xIndex/(nidx0*nidx1*nidx2*nidx3)) % nidx4))+idxshift:
	    tex1Dfetch(texref_i1_a,IDX4)+idxshift+ ncons4*((xIndex/(nidx0*nidx1*nidx2*nidx3)) % nidx4);

    unsigned int mypos;

    if ((xIndex-offset) < n) {
       //mypos = p0 + p1*nd0 + p2*nd0*nd1 + p3*nd0*nd1*nd2 + p4*nd0*nd1*nd2*nd3;
       mypos = p0 + nd0*(p1 + nd1*(p2 + nd2*(p3 + p4*nd3)));
       odata[xIndex] = tex1Dfetch(texref_c1_a,mypos);
    }
       
    //odata[xIndex] = tex2D(texref2, (float) texx, (float) texy);
}

__global__ void  SUBSINDEXD_KERNEL(unsigned int n, int offset,  
			    double *odata, int idxshift)
{
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    int nd0 = tex1Dfetch(texref_i1_a, 0); 
    int nd1 = tex1Dfetch(texref_i1_a, 1); 
    int nd2 = tex1Dfetch(texref_i1_a, 2); 
    int nd3 = tex1Dfetch(texref_i1_a, 3); 
    int nd4 = tex1Dfetch(texref_i1_a, 4); 
    
    int nidx0 = tex1Dfetch(texref_i1_a, 5); 
    int nidx1 = tex1Dfetch(texref_i1_a, 6); 
    int nidx2 = tex1Dfetch(texref_i1_a, 7); 
    int nidx3 = tex1Dfetch(texref_i1_a, 8); 
    int nidx4 = tex1Dfetch(texref_i1_a, 9);

    int ncons0 = tex1Dfetch(texref_i1_a, 10); 
    int ncons1 = tex1Dfetch(texref_i1_a, 11); 
    int ncons2 = tex1Dfetch(texref_i1_a, 12); 
    int ncons3 = tex1Dfetch(texref_i1_a, 13); 
    int ncons4 = tex1Dfetch(texref_i1_a, 14); 

    int neff0 = (ncons0==0)?nidx0:1;
    int neff1 = (ncons1==0)?nidx1:1;
    int neff2 = (ncons2==0)?nidx2:1;
    int neff3 = (ncons3==0)?nidx3:1;
    //int neff4 = (ncons4==0)?nidx4:1;


// define idx positions
#define IDX0 (15) 
#define IDX1 (IDX0+neff0) 
#define IDX2 (IDX1+neff1) 
#define IDX3 (IDX2+neff2) 
#define IDX4 (IDX3+neff3) 

    int p0 = (ncons0==0)?tex1Dfetch(texref_i1_a,IDX0+(xIndex % nidx0))+idxshift:
	    tex1Dfetch(texref_i1_a,IDX0)+idxshift+ ncons0*(xIndex % nidx0);
    int p1 = (ncons1==0)?tex1Dfetch(texref_i1_a,IDX1+((xIndex/nidx0) % nidx1))+idxshift:
	    tex1Dfetch(texref_i1_a,IDX1)+idxshift+ ncons1*((xIndex/nidx0) % nidx1);
    int p2 = (ncons2==0)?tex1Dfetch(texref_i1_a,IDX2+((xIndex/(nidx0*nidx1)) % nidx2))+idxshift:
	    tex1Dfetch(texref_i1_a,IDX2)+idxshift+ ncons2*((xIndex/(nidx0*nidx1)) % nidx2);
    int p3 = (ncons3==0)?tex1Dfetch(texref_i1_a,IDX3+((xIndex/(nidx0*nidx1*nidx2)) % nidx3))+idxshift:
	    tex1Dfetch(texref_i1_a,IDX3)+idxshift+ ncons3*((xIndex/(nidx0*nidx1*nidx2)) % nidx3);
    int p4 = (ncons4==0)?tex1Dfetch(texref_i1_a,IDX4+((xIndex/(nidx0*nidx1*nidx2*nidx3)) % nidx4))+idxshift:
	    tex1Dfetch(texref_i1_a,IDX4)+idxshift+ ncons4*((xIndex/(nidx0*nidx1*nidx2*nidx3)) % nidx4);

    unsigned int mypos;

    if ((xIndex - offset) < n) {
       mypos = p0 + nd0*(p1 + nd1*(p2 + nd2*(p3 + p4*nd3)));
       odata[xIndex] = tex1Dfetch_double(texref_d1_a,mypos);
    }
}


__global__ void  SUBSINDEXCD_KERNEL(int n, int offset,
			    DoubleComplex *odata, int idxshift)
{
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    int nd0 = tex1Dfetch(texref_i1_a, 0); 
    int nd1 = tex1Dfetch(texref_i1_a, 1); 
    int nd2 = tex1Dfetch(texref_i1_a, 2); 
    int nd3 = tex1Dfetch(texref_i1_a, 3); 
    int nd4 = tex1Dfetch(texref_i1_a, 4); 
    
    int nidx0 = tex1Dfetch(texref_i1_a, 5); 
    int nidx1 = tex1Dfetch(texref_i1_a, 6); 
    int nidx2 = tex1Dfetch(texref_i1_a, 7); 
    int nidx3 = tex1Dfetch(texref_i1_a, 8); 
    int nidx4 = tex1Dfetch(texref_i1_a, 9);

    int ncons0 = tex1Dfetch(texref_i1_a, 10); 
    int ncons1 = tex1Dfetch(texref_i1_a, 11); 
    int ncons2 = tex1Dfetch(texref_i1_a, 12); 
    int ncons3 = tex1Dfetch(texref_i1_a, 13); 
    int ncons4 = tex1Dfetch(texref_i1_a, 14); 

    int neff0 = (ncons0==0)?nidx0:1;
    int neff1 = (ncons1==0)?nidx1:1;
    int neff2 = (ncons2==0)?nidx2:1;
    int neff3 = (ncons3==0)?nidx3:1;
    //int neff4 = (ncons4==0)?nidx4:1; 

// define idx positions
   int p0 = (ncons0==0)?tex1Dfetch(texref_i1_a,IDX0+(xIndex % nidx0))+idxshift:
	    tex1Dfetch(texref_i1_a,IDX0)+idxshift+ ncons0*(xIndex % nidx0);
    int p1 = (ncons1==0)?tex1Dfetch(texref_i1_a,IDX1+((xIndex/nidx0) % nidx1))+idxshift:
	    tex1Dfetch(texref_i1_a,IDX1)+idxshift+ ncons1*((xIndex/nidx0) % nidx1);
    int p2 = (ncons2==0)?tex1Dfetch(texref_i1_a,IDX2+((xIndex/(nidx0*nidx1)) % nidx2))+idxshift:
	    tex1Dfetch(texref_i1_a,IDX2)+idxshift+ ncons2*((xIndex/(nidx0*nidx1)) % nidx2);
    int p3 = (ncons3==0)?tex1Dfetch(texref_i1_a,IDX3+((xIndex/(nidx0*nidx1*nidx2)) % nidx3))+idxshift:
	    tex1Dfetch(texref_i1_a,IDX3)+idxshift+ ncons3*((xIndex/(nidx0*nidx1*nidx2)) % nidx3);
    int p4 = (ncons4==0)?tex1Dfetch(texref_i1_a,IDX4+((xIndex/(nidx0*nidx1*nidx2*nidx3)) % nidx4))+idxshift:
	    tex1Dfetch(texref_i1_a,IDX4)+idxshift+ ncons4*((xIndex/(nidx0*nidx1*nidx2*nidx3)) % nidx4);

    unsigned int mypos;

    if ((xIndex-offset) < n) {
       mypos = p0 + nd0*(p1 + nd1*(p2 + nd2*(p3 + p4*nd3)));
       odata[xIndex] = tex1Dfetch_DoubleComplex(texref_cd1_a,mypos);
    }
}

#undef IDX0
#undef IDX1
#undef IDX2
#undef IDX3
#undef IDX4
/*#undef p0i
#undef p1i
#undef p2i
#undef p3i
#undef p4i*/

/*
 * odata is the output of R2C FFT, with symmetric coefficients. This
 * kernel complets the matrix odata according to symmetry. 
 * M - odata rows
 * N - odata columns
 * Q - odata 3rd dim 
 * offset - used when there are more streams
 *
 * The part to be filled is given starting from M/2+1 on first dimension
 * For a given index, the symmetric coefficient is in
 * c = A(mod(M-i,M),mod(N-j,N),mod(Q-k,Q));
 *
 * This kernel calculates the 3D index (pi,pj,pk) from a 1D index xIndex. Then applies the
 * above formula
 *
 */
__global__ void FFTSYMMC_KERNEL(int Nthreads, int M, int N, int Q, Complex *odata, int offset, int batch)
{
  unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset; // index used for output

//#define pi  (fmodf(xIndex,M))
//#define pj  (fmodf(floorf(xIndex/M),N))
//#define pk  (fmodf(floorf(xIndex/(M*N)),Q))
  int pi  = xIndex % M;
  int pj  = (xIndex/M) % N ;
  int pk  = (xIndex/(M*N)) % Q;

// follow symmetric positions
//#define pis  (fmodf(M-pi,M))
//#define pjs  (fmodf(N-pj,N))
//#define pks  (fmodf(Q-pk,Q))
  int pis  = (M-pi) % M;
  int pjs  = (N-pj) % N;
  int pks  = (Q-pk) % Q;

  // batch mode is used for the 1D FFT on arrays. In this case the symmetry is given
  // for each column, it means that I do not have to consider symmetry along 2nd an 3rd dimension
  if (batch==1) {
    pjs = pj;
    pks = pk;
  }
  
  // is batch is 2 it means that we did fft2 on a 3d array
  if (batch==2) {
    pks = pk;
  }


   int half = ((int) M/2) + 1; // first part of the matrix

  // 3D index is given by (pi, pj, pk)
  // threads with pi< M/2+1 copy the data from input vector, otherwise
  // calculate the position of symmetric element and copy

  // symmetric position 
  // c = A(mod(M-i,M)+1,mod(N-j,N)+1,mod(Q-k,Q)+1);

  if ((xIndex - offset) < Nthreads) {
   int symmpos = pis + M*(pjs + N*pks);
   int pos =0;
   int sign = 1;
   (pi < half) ? pos = xIndex: pos = symmpos; 
   (pi < half) ? sign = 1: sign  = -1; 
   Complex d = tex1Dfetch(texref_c1_a,pos);
   d.y = sign*d.y; // conj
   odata[xIndex] = d;
  }
}

__global__ void FFTSYMMCD_KERNEL(int Nthreads, int M, int N, int Q, DoubleComplex *odata, int offset, int batch)
{
  unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset; // index used for output

//#define pi  (fmodf(xIndex,M))
//#define pj  (fmodf(floorf(xIndex/M),N))
//#define pk  (fmodf(floorf(xIndex/(M*N)),Q))
  int pi  = xIndex % M;
  int pj  = (xIndex/M) % N ;
  int pk  = (xIndex/(M*N)) % Q;

// follow symmetric positions
//#define pis  (fmodf(M-pi,M))
//#define pjs  (fmodf(N-pj,N))
//#define pks  (fmodf(Q-pk,Q))
  int pis  = (M-pi) % M;
  int pjs  = (N-pj) % N;
  int pks  = (Q-pk) % Q;

  // batch mode is used for the 1D FFT on arrays. In this case the symmetry is given
  // for each column, it means that I do not have to consider symmetry along 2nd an 3rd dimension
  if (batch==1) {
    pjs = pj;
    pks = pk;
  }
  
  // is batch is 2 it means that we did fft2 on a 3d array
  if (batch==2) {
    pks = pk;
  }


   int half = ((int) M/2) + 1; // first part of the matrix

  // 3D index is given by (pi, pj, pk)
  // threads with pi< M/2+1 copy the data from input vector, otherwise
  // calculate the position of symmetric element and copy

  // symmetric position 
  // c = A(mod(M-i,M)+1,mod(N-j,N)+1,mod(Q-k,Q)+1);

  if ((xIndex - offset) < Nthreads) {
   int symmpos = pis + M*(pjs + N*pks);
   int pos =0;
   int sign = 1;
   (pi < half) ? pos = xIndex: pos = symmpos; 
   (pi < half) ? sign = 1: sign  = -1; 
   DoubleComplex d = tex1Dfetch_DoubleComplex(texref_cd1_a,pos);
   d.y = sign*d.y; // conj
   odata[xIndex] = d;
  }
}

__global__ void  CHECKTEXTURE_KERNEL(int n,
                                  float *odata, int offset)
{
 unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x;

 if (xIndex < n) {
   odata[xIndex] = tex1Dfetch(texref_f1_a,xIndex+offset);
 }
}


/*
 * REALIMAG
 * Depending on mode and direction the kernel perform the following operations
 *
 * dir
 * 0 - REAL to COMPLEX
 * 1 - COMPLEX to REAL
 * mode
 * 0 - REAL, IMAG
 * 1 - REAL
 * 2 - IMAG
 */
__global__ void  REALIMAGF_KERNEL(int n, Complex *data, float *re, float *im, int dir, int mode, 
								unsigned int offset)
{
 unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;

 if ((xIndex -offset) < n) {
   if (dir == 0) {
	   
      Complex tmp = data[xIndex];
      if (mode!=2) 
        tmp.x = re[xIndex];
      if (mode!=1)      
        tmp.y = im[xIndex];      
      
      data[xIndex] = tmp;
   
   } else {
      Complex tmp = data[xIndex];
      if (mode!=2) 
        re[xIndex] = tmp.x;
      if (mode!=1)      
        im[xIndex] = tmp.y;      
   }	   
 }
}
__global__ void  REALIMAGD_KERNEL(int n, DoubleComplex *data, double *re, double *im, int dir, int mode, 
								unsigned int offset)
{
 unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;

 if ((xIndex -offset) < n) {
   if (dir == 0) {
	   
      DoubleComplex tmp = data[xIndex];
      if (mode!=2) 
        tmp.x = re[xIndex];
      if (mode!=1)      
        tmp.y = im[xIndex];      
      
      data[xIndex] = tmp;
   
   } else {
      DoubleComplex tmp = data[xIndex];
      if (mode!=2) 
        re[xIndex] = tmp.x;
      if (mode!=1)      
        im[xIndex] = tmp.y;      
   }	   
 }
}
}

/* Casting kernels */

/* FLOAT to DOUBLE */
GEN_KERNEL_1D_IN1(FLOAT_TO_DOUBLE_KERNEL, float *, double *, (double));
/* FLOAT to INTEGER */
GEN_KERNEL_1D_IN1(FLOAT_TO_INTEGER_KERNEL, float *, int *, (int));

/* DOUBLE to FLOAT */
GEN_KERNEL_1D_IN1(DOUBLE_TO_FLOAT_KERNEL, double *, float *, (float));
/* DOUBLE to INTEGER */
GEN_KERNEL_1D_IN1(DOUBLE_TO_INTEGER_KERNEL, double *, int *, (int));

/* INTEGER TO DOUBLE */
GEN_KERNEL_1D_IN1(INTEGER_TO_DOUBLE_KERNEL, int *, double *, (double));
/* INTEGER TO FLOAT */
GEN_KERNEL_1D_IN1(INTEGER_TO_FLOAT_KERNEL, int *, float *, (float));




//}

#endif
