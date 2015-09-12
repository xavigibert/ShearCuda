/*
     Copyright (C) 2013  University of Maryland
 
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

#include "shearcudafunctions.h"

// CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "cutil_inline_runtime.h"
#include "cuda_common.h"

#include <iostream>

ShearCudaFunctions g_ShearFunc;

/*************************************************
 * HOST DRIVERS
 *************************************************/
typedef struct hostdrv_pars {
  hostdrv_pars() {
    par = NULL;
    psize = 0;
    align = __alignof(int);
  }
  hostdrv_pars(void *p, int s) {
    par = p;
    psize = s;
    align = __alignof(int);
  }
  hostdrv_pars(void *p, int s, size_t t) {
    par = p;
    psize = s;
    align = t;
  }
  void * par;
  unsigned int psize;
  size_t align;
} hostdrv_pars_t;

#define ALIGN_UP(offset, alignment) \
      (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

/*************************************************
 * UTILS
 *************************************************/

//Round a / b to nearest higher integer value
int iDivUp(int a, int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

/*************************************************
 * HOST DRIVERS
 *************************************************/
void hostGPUDRV(CUfunction drvfun, int N, int nrhs, hostdrv_pars_t *prhs)
{
    unsigned int maxthreads = MAXTHREADS_STREAM;
    int nstreams = iDivUp(N, maxthreads*BLOCK_DIM1D);
    for (int str = 0; str < nstreams; str++) {
        int offset = str * maxthreads * BLOCK_DIM1D;
        int size = 0;
        if (str == (nstreams - 1))
            size = N - str * maxthreads * BLOCK_DIM1D;
        else
            size = maxthreads * BLOCK_DIM1D;


        int gridx = iDivUp(size, BLOCK_DIM1D); // number of x blocks

        // setup execution parameters
        cutilSafeCall( cuFuncSetBlockShape(drvfun, BLOCK_DIM1D, 1, 1));
        cutilSafeCall( cuFuncSetSharedSize(drvfun, 0) );


        // add parameters
        int poffset = 0;

        // CUDA kernels interface
        // N: number of elements
        // offset: used for streams
        ALIGN_UP(poffset, __alignof(size));
        cutilSafeCall( cuParamSeti(drvfun, poffset, size) );
        poffset += sizeof(size);

        ALIGN_UP(poffset, __alignof(offset));
        cutilSafeCall( cuParamSeti(drvfun, poffset, offset) );
        poffset += sizeof(offset);

        for (int p=0;p<nrhs;p++) {
            ALIGN_UP(poffset, prhs[p].align);
            cutilSafeCall( cuParamSetv(drvfun, poffset, prhs[p].par, prhs[p].psize) );
            poffset += prhs[p].psize;
        }

        cutilSafeCall( cuParamSetSize(drvfun, poffset) );
        cutilSafeCall( cuLaunchGridAsync(drvfun, gridx, 1, 0) );
    }
}

void hostGPUDRV_Large(CUfunction drvfun, int N, int nrhs, hostdrv_pars_t *prhs)
{
    unsigned int maxthreads = MAXTHREADS_STREAM;
    int nstreams = iDivUp(N, maxthreads*BLOCK_DIM1D_LARGE);
    for (int str = 0; str < nstreams; str++) {
        int offset = str * maxthreads * BLOCK_DIM1D_LARGE;
        int size = 0;
        if (str == (nstreams - 1))
            size = N - str * maxthreads * BLOCK_DIM1D_LARGE;
        else
            size = maxthreads * BLOCK_DIM1D_LARGE;


        int gridx = iDivUp(size, BLOCK_DIM1D_LARGE); // number of x blocks

        // setup execution parameters
        cutilSafeCall( cuFuncSetBlockShape(drvfun, BLOCK_DIM1D_LARGE, 1, 1));
        cutilSafeCall( cuFuncSetSharedSize(drvfun, 0) );


        // add parameters
        int poffset = 0;

        // CUDA kernels interface
        // N: number of elements
        // offset: used for streams
        ALIGN_UP(poffset, __alignof(size));
        cutilSafeCall( cuParamSeti(drvfun, poffset, size) );
        poffset += sizeof(size);

        ALIGN_UP(poffset, __alignof(offset));
        cutilSafeCall( cuParamSeti(drvfun, poffset, offset) );
        poffset += sizeof(offset);

        for (int p=0;p<nrhs;p++) {
            ALIGN_UP(poffset, prhs[p].align);
            cutilSafeCall( cuParamSetv(drvfun, poffset, prhs[p].par, prhs[p].psize) );
            poffset += prhs[p].psize;
        }

        cutilSafeCall( cuParamSetSize(drvfun, poffset) );
        cutilSafeCall( cuLaunchGridAsync(drvfun, gridx, 1, 0) );
    }
}

ShearCudaFunctions::ShearCudaFunctions() :
    supportsFloat(false),
    supportsDouble(false),
    modulateConjAndNormalizeFC(NULL),
    modulateConjAndNormalizeDC(NULL),
    modulateAndNormalizeManyFC(NULL),
    modulateAndNormalizeManyDC(NULL),
    addVectorF(NULL),
    addVectorD(NULL),
    sumVectorsF(NULL),
    sumVectorsD(NULL),
    mulMatrixByScalarF(NULL),
    mulMatrixByScalarD(NULL),
    atrousSubsampleF(NULL),
    atrousSubsampleD(NULL),
    atrousUpsampleF(NULL),
    atrousUpsampleD(NULL),
    atrousConvolutionF(NULL),
    atrousConvolutionD(NULL),
    hardThresholdF(NULL),
    hardThresholdD(NULL),
    hardThresholdFC(NULL),
    hardThresholdDC(NULL),
    softThresholdF(NULL),
    softThresholdD(NULL),
    softThresholdFC(NULL),
    softThresholdDC(NULL),
    symExtF(NULL),
    symExtD(NULL),
    scalarVectorMulF(NULL),
    scalarVectorMulD(NULL),
    mrdwtRowF(NULL),
    mrdwtRowD(NULL),
    mrdwtColF(NULL),
    mrdwtColD(NULL),
    mirdwtRowF(NULL),
    mirdwtRowD(NULL),
    mirdwtColF(NULL),
    mirdwtColD(NULL),    
    complexToRealF(NULL),
    complexToRealD(NULL),
    realToComplexF(NULL),
    realToComplexD(NULL),
    reduceMaxAbsVal256F(NULL),
    reduceMaxAbsVal256D(NULL),
    reduceMaxAbsVal256FC(NULL),
    reduceMaxAbsVal256DC(NULL),
    reduceNorm256F(NULL),
    reduceNorm256D(NULL),
    reduceNorm256FC(NULL),
    reduceNorm256DC(NULL),
    reduceNormErr256F(NULL),
    reduceNormErr256D(NULL),
    reduceNormErr256FC(NULL),
    reduceNormErr256DC(NULL),
    reduceSum256F(NULL),
    reduceSum256D(NULL),
    zeroPadF2FC(NULL),
    zeroPadF2DC(NULL),
    zeroPadD2FC(NULL),
    zeroPadD2DC(NULL),
    prepareMyerFiltersFC(NULL),
    prepareMyerFiltersDC(NULL),
    convert8UToF(NULL),
    convert8UToD(NULL),
    convertFTo8U(NULL),
    convertDTo8U(NULL),
    m_timer(NULL)
{
   
}
    
bool ShearCudaFunctions::LoadGpuFunctions( const CUmodule *drvmod )
{
    bool bSuccessF = true, bSuccessD = true;
    
    // load GPU single precision functions
    bSuccessF = bSuccessF && (cuModuleGetFunction(&modulateConjAndNormalizeFC, *drvmod, "modulateConjAndNormalizeFC") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&modulateAndNormalizeManyFC, *drvmod, "modulateAndNormalizeManyFC") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&addVectorF, *drvmod, "addVectorF") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&sumVectorsF, *drvmod, "sumVectorsF") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&mulMatrixByScalarF, *drvmod, "mulMatrixByScalarF") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&atrousSubsampleF, *drvmod, "atrousSubsampleF") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&atrousUpsampleF, *drvmod, "atrousUpsampleF") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&atrousConvolutionF, *drvmod, "atrousConvolutionF") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&hardThresholdF, *drvmod, "hardThresholdF") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&hardThresholdFC, *drvmod, "hardThresholdFC") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&softThresholdF, *drvmod, "softThresholdF") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&softThresholdFC, *drvmod, "softThresholdFC") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&symExtF, *drvmod, "symExtF") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&scalarVectorMulF, *drvmod, "scalarVectorMulF") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&mrdwtRowF, *drvmod, "mrdwtRowF") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&mrdwtColF, *drvmod, "mrdwtColF") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&mirdwtRowF, *drvmod, "mirdwtRowF") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&mirdwtColF, *drvmod, "mirdwtColF") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&complexToRealF, *drvmod, "complexToRealF") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&realToComplexF, *drvmod, "realToComplexF") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&reduceMaxAbsVal256F, *drvmod, "reduceMaxAbsVal256F") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&reduceMaxAbsVal256FC, *drvmod, "reduceMaxAbsVal256FC") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&reduceNorm256F, *drvmod, "reduceNorm256F") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&reduceNorm256FC, *drvmod, "reduceNorm256FC") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&reduceNormErr256F, *drvmod, "reduceNormErr256F") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&reduceNormErr256FC, *drvmod, "reduceNormErr256FC") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&reduceSum256F, *drvmod, "reduceSum256F") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&zeroPadF2FC, *drvmod, "zeroPadF2FC") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&prepareMyerFiltersFC, *drvmod, "prepareMyerFiltersFC") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&convert8UToF, *drvmod, "convert8UToF") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&convertFTo8U, *drvmod, "convertFTo8U") == CUDA_SUCCESS);

    // load GPU double precision functions
    bSuccessD = bSuccessD && (cuModuleGetFunction(&modulateConjAndNormalizeDC, *drvmod, "modulateConjAndNormalizeDC") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&modulateAndNormalizeManyDC, *drvmod, "modulateAndNormalizeManyDC") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&addVectorD, *drvmod, "addVectorD") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&sumVectorsD, *drvmod, "sumVectorsD") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&mulMatrixByScalarD, *drvmod, "mulMatrixByScalarD") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&atrousSubsampleD, *drvmod, "atrousSubsampleD") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&atrousUpsampleD, *drvmod, "atrousUpsampleD") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&atrousConvolutionD, *drvmod, "atrousConvolutionD") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&hardThresholdD, *drvmod, "hardThresholdD") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&hardThresholdDC, *drvmod, "hardThresholdDC") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&softThresholdD, *drvmod, "softThresholdD") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&softThresholdDC, *drvmod, "softThresholdDC") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&symExtD, *drvmod, "symExtD") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&scalarVectorMulD, *drvmod, "scalarVectorMulD") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&mrdwtRowD, *drvmod, "mrdwtRowD") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&mrdwtColD, *drvmod, "mrdwtColD") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&mirdwtRowD, *drvmod, "mirdwtRowD") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&mirdwtColD, *drvmod, "mirdwtColD") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&reduceMaxAbsVal256D, *drvmod, "reduceMaxAbsVal256D") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&reduceMaxAbsVal256DC, *drvmod, "reduceMaxAbsVal256DC") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&complexToRealD, *drvmod, "complexToRealD") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&realToComplexD, *drvmod, "realToComplexD") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&reduceNorm256D, *drvmod, "reduceNorm256D") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&reduceNorm256DC, *drvmod, "reduceNorm256DC") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&reduceNormErr256D, *drvmod, "reduceNormErr256D") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&reduceNormErr256DC, *drvmod, "reduceNormErr256D") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&reduceSum256D, *drvmod, "reduceSum256D") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&zeroPadF2DC, *drvmod, "zeroPadF2DC") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&zeroPadD2DC, *drvmod, "zeroPadD2DC") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&prepareMyerFiltersDC, *drvmod, "prepareMyerFiltersDC") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&zeroPadD2FC, *drvmod, "zeroPadD2FC") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&convert8UToD, *drvmod, "convert8UToD") == CUDA_SUCCESS);
    bSuccessD = bSuccessD && (cuModuleGetFunction(&convertDTo8U, *drvmod, "convertDTo8U") == CUDA_SUCCESS);

    supportsFloat = bSuccessF;
    supportsDouble = bSuccessD;
    
    return bSuccessF;
}

void ShearCudaFunctions::symExt( void* d_Output, int outputRows, int outputCols,
             void* d_Input, int inputRows, int inputCols,
             int topOffset, int leftOffset, gpuTYPE_t type_signal ) const
{
    hostdrv_pars_t gpuprhs[8];
    int gpunrhs = 8;
    gpuprhs[0] = hostdrv_pars(&d_Output,sizeof(d_Output),__alignof(d_Output));
    gpuprhs[1] = hostdrv_pars(&outputRows,sizeof(outputRows),__alignof(outputRows));
    gpuprhs[2] = hostdrv_pars(&outputCols,sizeof(outputCols),__alignof(outputCols));
    gpuprhs[3] = hostdrv_pars(&d_Input,sizeof(d_Input),__alignof(d_Input));
    gpuprhs[4] = hostdrv_pars(&inputRows,sizeof(inputRows),__alignof(inputRows));
    gpuprhs[5] = hostdrv_pars(&inputCols,sizeof(inputCols),__alignof(inputCols));
    gpuprhs[6] = hostdrv_pars(&topOffset,sizeof(topOffset),__alignof(topOffset));
    gpuprhs[7] = hostdrv_pars(&leftOffset,sizeof(leftOffset),__alignof(leftOffset));

    int N = inputRows * inputCols;

    m_timer->startTimer( GpuTimes::symExt );
    if( type_signal == gpuFLOAT )
        hostGPUDRV(symExtF, N, gpunrhs, gpuprhs);
    else
        hostGPUDRV(symExtD, N, gpunrhs, gpuprhs);
    m_timer->stopTimer( GpuTimes::symExt );
}

void ShearCudaFunctions::atrousConvolution( void* d_Out, const void* d_Subsampled,
        void *d_Filter, int outW, int subW_padded,
        int filterLen, int level, gpuTYPE_t type_signal ) const
{
    hostdrv_pars_t gpuprhs[7];
    int gpunrhs = 7;
    gpuprhs[0] = hostdrv_pars(&d_Out,sizeof(d_Out),__alignof(d_Out));
    gpuprhs[1] = hostdrv_pars(&d_Subsampled,sizeof(d_Subsampled),__alignof(d_Subsampled));
    gpuprhs[2] = hostdrv_pars(&d_Filter,sizeof(d_Filter),__alignof(d_Filter));
    gpuprhs[3] = hostdrv_pars(&outW,sizeof(outW),__alignof(outW));
    gpuprhs[4] = hostdrv_pars(&subW_padded,sizeof(subW_padded),__alignof(subW_padded));
    gpuprhs[5] = hostdrv_pars(&filterLen,sizeof(filterLen),__alignof(filterLen));
    gpuprhs[6] = hostdrv_pars(&level,sizeof(level),__alignof(level));

    int N = outW * outW;

    m_timer->startTimer( GpuTimes::atrousConvolution );
    if( type_signal == gpuFLOAT )
        hostGPUDRV_Large(atrousConvolutionF, N, gpunrhs, gpuprhs);
    else
        hostGPUDRV_Large(atrousConvolutionD, N, gpunrhs, gpuprhs);
    m_timer->stopTimer( GpuTimes::atrousConvolution );
}

void ShearCudaFunctions::atrousSubsample( void* d_Subsampled, const void* d_Signal,
        int subW, int signalW, int level, gpuTYPE_t type_signal ) const
{
    hostdrv_pars_t gpuprhs[5];
    int gpunrhs = 5;
    gpuprhs[0] = hostdrv_pars(&d_Subsampled,sizeof(d_Subsampled),__alignof(d_Subsampled));
    gpuprhs[1] = hostdrv_pars(&d_Signal,sizeof(d_Signal),__alignof(d_Signal));
    gpuprhs[2] = hostdrv_pars(&subW,sizeof(subW),__alignof(subW));
    gpuprhs[3] = hostdrv_pars(&signalW,sizeof(signalW),__alignof(signalW));
    gpuprhs[4] = hostdrv_pars(&level,sizeof(level),__alignof(level));

    int N = subW * subW * level * level;

    m_timer->startTimer( GpuTimes::atrousSubsample );
    if( type_signal == gpuFLOAT )
        hostGPUDRV(atrousSubsampleF, N, gpunrhs, gpuprhs);
    else
        hostGPUDRV(atrousSubsampleD, N, gpunrhs, gpuprhs);
    m_timer->stopTimer( GpuTimes::atrousSubsample );
}

void ShearCudaFunctions::modulateConjAndNormalize(
        void* d_Dst, const void* d_Src, int fftW, int fftH, int numElem,
        int padding, gpuTYPE_t type_signal ) const
{
    hostdrv_pars_t gpuprhs[3];
    int gpunrhs = 3;
    gpuprhs[0] = hostdrv_pars(&d_Dst,sizeof(d_Dst),__alignof(d_Dst));
    gpuprhs[1] = hostdrv_pars(&d_Src,sizeof(d_Src),__alignof(d_Src));

    int N = fftW * fftH * numElem;

    m_timer->startTimer( GpuTimes::modulateAndNormalize );
    if( type_signal == gpuCFLOAT )
    {
        float factor = 1.0f / (float)(fftW * fftH);
        gpuprhs[2] = hostdrv_pars(&factor,sizeof(factor),__alignof(factor));
        hostGPUDRV(modulateConjAndNormalizeFC, N, gpunrhs, gpuprhs);
    }
    else
    {
        double factor = 1.0 / (double)(fftW * fftH);
        gpuprhs[2] = hostdrv_pars(&factor,sizeof(factor),__alignof(factor));
        hostGPUDRV(modulateConjAndNormalizeDC, N, gpunrhs, gpuprhs);
    }
    m_timer->stopTimer( GpuTimes::modulateAndNormalize );
}

// Modulate and normalize a single data element with a filter bank
void ShearCudaFunctions::modulateAndNormalizeMany(
        void* d_Dst, const void* d_SrcData, const void* d_SrcKernel,
        int fftW, int fftH, int numElem, gpuTYPE_t type_signal ) const
{
    hostdrv_pars_t gpuprhs[7];
    int gpunrhs = 7;
    gpuprhs[0] = hostdrv_pars(&d_Dst,sizeof(d_Dst),__alignof(d_Dst));
    gpuprhs[1] = hostdrv_pars(&d_SrcData,sizeof(d_SrcData),__alignof(d_SrcData));
    gpuprhs[2] = hostdrv_pars(&d_SrcKernel,sizeof(d_SrcKernel),__alignof(d_SrcKernel));
    gpuprhs[3] = hostdrv_pars(&fftW,sizeof(fftW),__alignof(fftW));
    gpuprhs[4] = hostdrv_pars(&fftH,sizeof(fftH),__alignof(fftH));
    gpuprhs[5] = hostdrv_pars(&numElem,sizeof(numElem),__alignof(numElem));

    int N = fftW * fftH;

    m_timer->startTimer( GpuTimes::modulateAndNormalize );
    if( type_signal == gpuCFLOAT )
    {
        float factor = 1.0f / (float)(fftW * fftH);
        gpuprhs[6] = hostdrv_pars(&factor,sizeof(factor),__alignof(factor));
        hostGPUDRV(modulateAndNormalizeManyFC, N, gpunrhs, gpuprhs);
    }
    else
    {
        double factor = 1.0 / (double)(fftW * fftH);
        gpuprhs[6] = hostdrv_pars(&factor,sizeof(factor),__alignof(factor));
        hostGPUDRV(modulateAndNormalizeManyDC, N, gpunrhs, gpuprhs);
    }
    m_timer->stopTimer( GpuTimes::modulateAndNormalize );
}

// Add 2 vectors and save result on first one
void ShearCudaFunctions::addVector(
        void* d_Dst, const void* d_Src,
        int numElem, gpuTYPE_t type_signal ) const
{
    hostdrv_pars_t gpuprhs[2];
    int gpunrhs = 2;
    gpuprhs[0] = hostdrv_pars(&d_Dst,sizeof(d_Dst),__alignof(d_Dst));
    gpuprhs[1] = hostdrv_pars(&d_Src,sizeof(d_Src),__alignof(d_Src));

    int N = numElem;

    m_timer->startTimer( GpuTimes::addVector );
    if( type_signal == gpuFLOAT )
        hostGPUDRV(addVectorF, N, gpunrhs, gpuprhs);
    else
        hostGPUDRV(addVectorD, N, gpunrhs, gpuprhs);
    m_timer->stopTimer( GpuTimes::addVector );
}


// Add all components together (all components of d_Src are added toghether and result is saved into d_Dst)
void ShearCudaFunctions::sumVectors(
        void* d_Dst, const void* d_Src, int numElem,
        int numComponents, gpuTYPE_t type_signal ) const
{
    hostdrv_pars_t gpuprhs[3];
    int gpunrhs = 3;
    gpuprhs[0] = hostdrv_pars(&d_Dst,sizeof(d_Dst),__alignof(d_Dst));
    gpuprhs[1] = hostdrv_pars(&d_Src,sizeof(d_Src),__alignof(d_Src));
    gpuprhs[2] = hostdrv_pars(&numComponents,sizeof(numComponents),__alignof(numComponents));

    int N = numElem;

    m_timer->startTimer( GpuTimes::sumVectors );
    if( type_signal == gpuFLOAT )
        hostGPUDRV(sumVectorsF, N, gpunrhs, gpuprhs);
    else
        hostGPUDRV(sumVectorsD, N, gpunrhs, gpuprhs);
    m_timer->stopTimer( GpuTimes::sumVectors );
}

// Multiply a vector by a scalar
void ShearCudaFunctions::mulMatrixByScalar(
        void* d_Dst, const void* d_Src, double scalar,
        int numElem, gpuTYPE_t type_signal ) const
{
    hostdrv_pars_t gpuprhs[3];
    int gpunrhs = 3;
    gpuprhs[0] = hostdrv_pars(&d_Dst,sizeof(d_Dst),__alignof(d_Dst));
    gpuprhs[1] = hostdrv_pars(&d_Src,sizeof(d_Src),__alignof(d_Src));

    int N = numElem;

    m_timer->startTimer( GpuTimes::mulMatrixByScalar );
    if( type_signal == gpuFLOAT )
    {
        float fScalar = (float)scalar;
        gpuprhs[2] = hostdrv_pars(&fScalar,sizeof(fScalar),__alignof(fScalar));
        hostGPUDRV(mulMatrixByScalarF, N, gpunrhs, gpuprhs);
    }
    else
    {
        gpuprhs[2] = hostdrv_pars(&scalar,sizeof(scalar),__alignof(scalar));
        hostGPUDRV(mulMatrixByScalarD, N, gpunrhs, gpuprhs);
    }
    m_timer->stopTimer( GpuTimes::mulMatrixByScalar );
}

void ShearCudaFunctions::applyHardThreshold(
        void* d_Dst, const void* d_Src, int numElem,
        double th, gpuTYPE_t type_signal ) const
{
    hostdrv_pars_t gpuprhs[3];
    int gpunrhs = 3;
    gpuprhs[0] = hostdrv_pars(&d_Dst,sizeof(d_Dst),__alignof(d_Dst));
    gpuprhs[1] = hostdrv_pars(&d_Src,sizeof(d_Src),__alignof(d_Src));

    int N = numElem;

    m_timer->startTimer( GpuTimes::hardThreshold );
    switch( type_signal )
    {
    case gpuFLOAT:
    {
        float fTh = (float)th;
        gpuprhs[2] = hostdrv_pars(&fTh,sizeof(fTh),__alignof(fTh));
        hostGPUDRV(hardThresholdF, N, gpunrhs, gpuprhs);
        break;
    }
    case gpuDOUBLE:
    {
        gpuprhs[2] = hostdrv_pars(&th,sizeof(th),__alignof(th));
        hostGPUDRV(hardThresholdD, N, gpunrhs, gpuprhs);
        break;
    }
    case gpuCFLOAT:
    {
        float fTh_sq = (float)(th*th);
        gpuprhs[2] = hostdrv_pars(&fTh_sq,sizeof(fTh_sq),__alignof(fTh_sq));
        hostGPUDRV(hardThresholdFC, N, gpunrhs, gpuprhs);
        break;
    }
    case gpuCDOUBLE:
    {
        double th_sq = th*th;
        gpuprhs[2] = hostdrv_pars(&th_sq,sizeof(th_sq),__alignof(th_sq));
        hostGPUDRV(hardThresholdDC, N, gpunrhs, gpuprhs);
        break;
    }
    }

    m_timer->stopTimer( GpuTimes::hardThreshold );
}

void ShearCudaFunctions::applySoftThreshold(
    void* d_Dst, const void* d_Src, int numElem,
    double th, gpuTYPE_t type_signal ) const
{
    hostdrv_pars_t gpuprhs[3];
    int gpunrhs = 3;
    gpuprhs[0] = hostdrv_pars(&d_Dst,sizeof(d_Dst),__alignof(d_Dst));
    gpuprhs[1] = hostdrv_pars(&d_Src,sizeof(d_Src),__alignof(d_Src));

    int N = numElem;

    m_timer->startTimer( GpuTimes::softThreshold );
    switch( type_signal )
    {
    case gpuFLOAT:
    {
        float fTh = (float)th;
        gpuprhs[2] = hostdrv_pars(&fTh,sizeof(fTh),__alignof(fTh));
        hostGPUDRV(softThresholdF, N, gpunrhs, gpuprhs);
        break;
    }
    case gpuDOUBLE:
    {
        gpuprhs[2] = hostdrv_pars(&th,sizeof(th),__alignof(th));
        hostGPUDRV(softThresholdD, N, gpunrhs, gpuprhs);
        break;
    }
    case gpuCFLOAT:
    {
        float fTh = (float)th;
        gpuprhs[2] = hostdrv_pars(&fTh,sizeof(fTh),__alignof(fTh));
        hostGPUDRV(softThresholdFC, N, gpunrhs, gpuprhs);
        break;
    }
    case gpuCDOUBLE:
    {
        gpuprhs[2] = hostdrv_pars(&th,sizeof(th),__alignof(th));
        hostGPUDRV(softThresholdDC, N, gpunrhs, gpuprhs);
        break;
    }
    }
    m_timer->stopTimer( GpuTimes::softThreshold );
}

void ShearCudaFunctions::mrdwtRow(
        const void* d_Signal, int numRows, int numCols,
        const void* d_Filter, int filterLen, int level,
        void* d_yl, void* d_yh, gpuTYPE_t type_signal ) const
{
    hostdrv_pars_t gpuprhs[8];
    int gpunrhs = 8;
    gpuprhs[0] = hostdrv_pars(&d_Signal,sizeof(d_Signal),__alignof(d_Signal));
    gpuprhs[1] = hostdrv_pars(&numRows,sizeof(numRows),__alignof(numRows));
    gpuprhs[2] = hostdrv_pars(&numCols,sizeof(numCols),__alignof(numCols));
    gpuprhs[3] = hostdrv_pars(&d_Filter,sizeof(d_Filter),__alignof(d_Filter));
    gpuprhs[4] = hostdrv_pars(&filterLen,sizeof(filterLen),__alignof(filterLen));
    gpuprhs[5] = hostdrv_pars(&level,sizeof(level),__alignof(level));
    gpuprhs[6] = hostdrv_pars(&d_yl,sizeof(d_yl),__alignof(d_yl));
    gpuprhs[7] = hostdrv_pars(&d_yh,sizeof(d_yh),__alignof(d_yh));

    int N = numRows * numCols;

    m_timer->startTimer( GpuTimes::mrdwtRow );
    if( type_signal == gpuFLOAT )
        hostGPUDRV(mrdwtRowF, N, gpunrhs, gpuprhs);
    else
        hostGPUDRV(mrdwtRowD, N, gpunrhs, gpuprhs);
    m_timer->stopTimer( GpuTimes::mrdwtRow );
}

void ShearCudaFunctions::mrdwtCol(
        const void* d_Signal, int numRows, int numCols,
        const void* d_Filter, int filterLen, int level,
        void* d_yl, void* d_yh, gpuTYPE_t type_signal ) const
{
    hostdrv_pars_t gpuprhs[8];
    int gpunrhs = 8;
    gpuprhs[0] = hostdrv_pars(&d_Signal,sizeof(d_Signal),__alignof(d_Signal));
    gpuprhs[1] = hostdrv_pars(&numRows,sizeof(numRows),__alignof(numRows));
    gpuprhs[2] = hostdrv_pars(&numCols,sizeof(numCols),__alignof(numCols));
    gpuprhs[3] = hostdrv_pars(&d_Filter,sizeof(d_Filter),__alignof(d_Filter));
    gpuprhs[4] = hostdrv_pars(&filterLen,sizeof(filterLen),__alignof(filterLen));
    gpuprhs[5] = hostdrv_pars(&level,sizeof(level),__alignof(level));
    gpuprhs[6] = hostdrv_pars(&d_yl,sizeof(d_yl),__alignof(d_yl));
    gpuprhs[7] = hostdrv_pars(&d_yh,sizeof(d_yh),__alignof(d_yh));

    int N = numRows * numCols;

    m_timer->startTimer( GpuTimes::mrdwtCol );
    if( type_signal == gpuFLOAT )
        hostGPUDRV(mrdwtColF, N, gpunrhs, gpuprhs);
    else
        hostGPUDRV(mrdwtColD, N, gpunrhs, gpuprhs);
    m_timer->stopTimer( GpuTimes::mrdwtCol );
}

// Inverse redundant wavelet functions
void ShearCudaFunctions::mirdwtRow(
        const void* d_xinl, const void* d_xinh, int numRows, int numCols,
        const void* d_Filter, int filterLen, int level,
        void* d_xout, gpuTYPE_t type_signal ) const
{
    hostdrv_pars_t gpuprhs[8];
    int gpunrhs = 8;
    gpuprhs[0] = hostdrv_pars(&d_xinl,sizeof(d_xinl),__alignof(d_xinl));
    gpuprhs[1] = hostdrv_pars(&d_xinh,sizeof(d_xinl),__alignof(d_xinh));
    gpuprhs[2] = hostdrv_pars(&numRows,sizeof(numRows),__alignof(numRows));
    gpuprhs[3] = hostdrv_pars(&numCols,sizeof(numCols),__alignof(numCols));
    gpuprhs[4] = hostdrv_pars(&d_Filter,sizeof(d_Filter),__alignof(d_Filter));
    gpuprhs[5] = hostdrv_pars(&filterLen,sizeof(filterLen),__alignof(filterLen));
    gpuprhs[6] = hostdrv_pars(&level,sizeof(level),__alignof(level));
    gpuprhs[7] = hostdrv_pars(&d_xout,sizeof(d_xout),__alignof(d_xout));

    int N = numRows * numCols;

    m_timer->startTimer( GpuTimes::mirdwtRow );
    if( type_signal == gpuFLOAT )
        hostGPUDRV(mirdwtRowF, N, gpunrhs, gpuprhs);
    else
        hostGPUDRV(mirdwtRowD, N, gpunrhs, gpuprhs);
    m_timer->stopTimer( GpuTimes::mirdwtRow );
}

void ShearCudaFunctions::mirdwtCol(
        const void* d_xinl, const void* d_xinh, int numRows, int numCols,
        const void* d_Filter, int filterLen, int level,
        void* d_xout, gpuTYPE_t type_signal ) const
{
    hostdrv_pars_t gpuprhs[8];
    int gpunrhs = 8;
    gpuprhs[0] = hostdrv_pars(&d_xinl,sizeof(d_xinl),__alignof(d_xinl));
    gpuprhs[1] = hostdrv_pars(&d_xinh,sizeof(d_xinl),__alignof(d_xinh));
    gpuprhs[2] = hostdrv_pars(&numRows,sizeof(numRows),__alignof(numRows));
    gpuprhs[3] = hostdrv_pars(&numCols,sizeof(numCols),__alignof(numCols));
    gpuprhs[4] = hostdrv_pars(&d_Filter,sizeof(d_Filter),__alignof(d_Filter));
    gpuprhs[5] = hostdrv_pars(&filterLen,sizeof(filterLen),__alignof(filterLen));
    gpuprhs[6] = hostdrv_pars(&level,sizeof(level),__alignof(level));
    gpuprhs[7] = hostdrv_pars(&d_xout,sizeof(d_xout),__alignof(d_xout));

    int N = numRows * numCols;

    m_timer->startTimer( GpuTimes::mirdwtCol );
    if( type_signal == gpuFLOAT )
        hostGPUDRV(mirdwtColF, N, gpunrhs, gpuprhs);
    else
        hostGPUDRV(mirdwtColD, N, gpunrhs, gpuprhs);
    m_timer->stopTimer( GpuTimes::mirdwtCol );
}

// Calculate maximum absolute value within 256 samples (source can be either real
// or complex, but result is always complex)
void ShearCudaFunctions::reduceMaxAbsVal256(
        void* d_Dst, const void* d_Src, int numElem, gpuTYPE_t type_signal ) const
{
    hostdrv_pars_t gpuprhs[2];
    int gpunrhs = 2;
    gpuprhs[0] = hostdrv_pars(&d_Dst,sizeof(d_Dst),__alignof(d_Dst));
    gpuprhs[1] = hostdrv_pars(&d_Src,sizeof(d_Src),__alignof(d_Src));

    int N = numElem;

    m_timer->startTimer( GpuTimes::reduceMaxAbsVal256 );
    switch( type_signal )
    {
    case gpuFLOAT:
        hostGPUDRV(reduceMaxAbsVal256F, N, gpunrhs, gpuprhs);
        break;
    case gpuDOUBLE:
        hostGPUDRV(reduceMaxAbsVal256D, N, gpunrhs, gpuprhs);
        break;
    case gpuCFLOAT:
        hostGPUDRV(reduceMaxAbsVal256FC, N, gpunrhs, gpuprhs);
        break;
    case gpuCDOUBLE:
        hostGPUDRV(reduceMaxAbsVal256DC, N, gpunrhs, gpuprhs);
        break;
    }
    m_timer->stopTimer( GpuTimes::reduceMaxAbsVal256 );    
}

// Reduce vector Lp norm
void ShearCudaFunctions::reduceNorm256(
        void* d_Dst, const void* d_Src, double p, int numElem, gpuTYPE_t type_signal ) const
{
    hostdrv_pars_t gpuprhs[3];
    int gpunrhs = 3;
    gpuprhs[0] = hostdrv_pars(&d_Dst,sizeof(d_Dst),__alignof(d_Dst));
    gpuprhs[1] = hostdrv_pars(&d_Src,sizeof(d_Src),__alignof(d_Src));

    int N = numElem;

    m_timer->startTimer( GpuTimes::reduceNorm256 );
    switch( type_signal )
    {
    case gpuFLOAT:
    {
        float fP = (float)p;
        gpuprhs[2] = hostdrv_pars(&fP,sizeof(fP),__alignof(fP));
        hostGPUDRV(reduceNorm256F, N, gpunrhs, gpuprhs);
        break;
    }
    case gpuDOUBLE:
        gpuprhs[2] = hostdrv_pars(&p,sizeof(p),__alignof(p));
        hostGPUDRV(reduceNorm256D, N, gpunrhs, gpuprhs);
        break;
    case gpuCFLOAT:
    {
        float fP = (float)p;
        gpuprhs[2] = hostdrv_pars(&fP,sizeof(fP),__alignof(fP));
        hostGPUDRV(reduceNorm256FC, N, gpunrhs, gpuprhs);
        break;
    }
    case gpuCDOUBLE:
        gpuprhs[2] = hostdrv_pars(&p,sizeof(p),__alignof(p));
        hostGPUDRV(reduceNorm256DC, N, gpunrhs, gpuprhs);
        break;
    default:
        std::cerr << "ERROR: ShearCudaFunctions::reduceNorm256() type not supported" << std::endl;
        break;
    }
    m_timer->stopTimer( GpuTimes::reduceNorm256 );
}

// Reduce vector Lp norm of the difference between 2 vectors
void ShearCudaFunctions::reduceNormErr256(
        void* d_Dst, const void* d_SrcA, const void* d_SrcB, double p, int numElem, gpuTYPE_t type_signal ) const
{
    hostdrv_pars_t gpuprhs[4];
    int gpunrhs = 4;
    gpuprhs[0] = hostdrv_pars(&d_Dst,sizeof(d_Dst),__alignof(d_Dst));
    gpuprhs[1] = hostdrv_pars(&d_SrcA,sizeof(d_SrcA),__alignof(d_SrcA));
    gpuprhs[2] = hostdrv_pars(&d_SrcB,sizeof(d_SrcB),__alignof(d_SrcB));

    int N = numElem;

    m_timer->startTimer( GpuTimes::reduceNormErr256 );
    switch( type_signal )
    {
    case gpuFLOAT:
    {
        float fP = (float)p;
        gpuprhs[3] = hostdrv_pars(&fP,sizeof(fP),__alignof(fP));
        hostGPUDRV(reduceNormErr256F, N, gpunrhs, gpuprhs);
        break;
    }
    case gpuDOUBLE:
        gpuprhs[3] = hostdrv_pars(&p,sizeof(p),__alignof(p));
        hostGPUDRV(reduceNormErr256D, N, gpunrhs, gpuprhs);
        break;
    case gpuCFLOAT:
    {
        float fP = (float)p;
        gpuprhs[3] = hostdrv_pars(&fP,sizeof(fP),__alignof(fP));
        hostGPUDRV(reduceNormErr256FC, N, gpunrhs, gpuprhs);
        break;
    }
    case gpuCDOUBLE:
        gpuprhs[3] = hostdrv_pars(&p,sizeof(p),__alignof(p));
        hostGPUDRV(reduceNormErr256DC, N, gpunrhs, gpuprhs);
        break;
    default:
        std::cerr << "ERROR: ShearCudaFunctions::reduceNormErr256() type not supported" << std::endl;
        break;
    }
    m_timer->stopTimer( GpuTimes::reduceNormErr256 );
}

// Reduce vector sum
void ShearCudaFunctions::reduceSum256(
        void* d_Dst, const void* d_Src, int numElem, gpuTYPE_t type_signal ) const
{
    hostdrv_pars_t gpuprhs[2];
    int gpunrhs = 2;
    gpuprhs[0] = hostdrv_pars(&d_Dst,sizeof(d_Dst),__alignof(d_Dst));
    gpuprhs[1] = hostdrv_pars(&d_Src,sizeof(d_Src),__alignof(d_Src));

    int N = numElem;

    m_timer->startTimer( GpuTimes::reduceSum256 );
    if( type_signal == gpuFLOAT )
        hostGPUDRV(reduceSum256F, N, gpunrhs, gpuprhs);
    else
        hostGPUDRV(reduceSum256D, N, gpunrhs, gpuprhs);
    m_timer->stopTimer( GpuTimes::reduceSum256 );
}

// Convert real vector to complex
void ShearCudaFunctions::realToComplex(
        void* d_DstComplex, const void* d_SrcReal, int numElem, gpuTYPE_t type_signal ) const
{
    hostdrv_pars_t gpuprhs[2];
    int gpunrhs = 2;
    gpuprhs[0] = hostdrv_pars(&d_DstComplex,sizeof(d_DstComplex),__alignof(d_DstComplex));
    gpuprhs[1] = hostdrv_pars(&d_SrcReal,sizeof(d_SrcReal),__alignof(d_SrcReal));

    int N = numElem;

    m_timer->startTimer( GpuTimes::realToComplex );
    if( type_signal == gpuFLOAT || type_signal == gpuCFLOAT )
        hostGPUDRV(realToComplexF, N, gpunrhs, gpuprhs);
    else
        hostGPUDRV(realToComplexD, N, gpunrhs, gpuprhs);
    m_timer->stopTimer( GpuTimes::realToComplex );
}

// Convert complex vector to real
void ShearCudaFunctions::complexToReal(
        void* d_DstReal, const void* d_SrcComplex, int numElem, gpuTYPE_t type_signal ) const
{
    hostdrv_pars_t gpuprhs[2];
    int gpunrhs = 2;
    gpuprhs[0] = hostdrv_pars(&d_DstReal,sizeof(d_DstReal),__alignof(d_DstReal));
    gpuprhs[1] = hostdrv_pars(&d_SrcComplex,sizeof(d_SrcComplex),__alignof(d_SrcComplex));

    int N = numElem;

    m_timer->startTimer( GpuTimes::complexToReal );
    if( type_signal == gpuFLOAT || type_signal == gpuCFLOAT )
        hostGPUDRV(complexToRealF, N, gpunrhs, gpuprhs);
    else
        hostGPUDRV(complexToRealD, N, gpunrhs, gpuprhs);
    m_timer->stopTimer( GpuTimes::complexToReal );
}

// Zero padding
void ShearCudaFunctions::zeroPad(
        void* d_Dst, int dstDimX, int dstDimY, int dstDimZ, gpuTYPE_t type_dst,
        const void* d_Src, int srcDimX, int srcDimY, int srcDimZ, gpuTYPE_t type_src ) const
{
    hostdrv_pars_t gpuprhs[8];
    int gpunrhs = 8;
    gpuprhs[0] = hostdrv_pars(&d_Dst,sizeof(d_Dst),__alignof(d_Dst));
    gpuprhs[1] = hostdrv_pars(&dstDimX,sizeof(dstDimX),__alignof(dstDimX));
    gpuprhs[2] = hostdrv_pars(&dstDimY,sizeof(dstDimY),__alignof(dstDimY));
    gpuprhs[3] = hostdrv_pars(&dstDimZ,sizeof(dstDimZ),__alignof(dstDimZ));
    gpuprhs[4] = hostdrv_pars(&d_Src,sizeof(d_Src),__alignof(d_Src));
    gpuprhs[5] = hostdrv_pars(&srcDimX,sizeof(srcDimX),__alignof(srcDimX));
    gpuprhs[6] = hostdrv_pars(&srcDimY,sizeof(srcDimY),__alignof(srcDimY));
    gpuprhs[7] = hostdrv_pars(&srcDimZ,sizeof(srcDimZ),__alignof(srcDimZ));

    int N = dstDimX * dstDimY;

    m_timer->startTimer( GpuTimes::zeroPad );
    if( type_src == gpuFLOAT && type_dst == gpuCFLOAT )
        hostGPUDRV(zeroPadF2FC, N, gpunrhs, gpuprhs);
    else if( type_src == gpuFLOAT && type_dst == gpuCDOUBLE )
        hostGPUDRV(zeroPadF2DC, N, gpunrhs, gpuprhs);
    else if ( type_src == gpuDOUBLE && type_dst == gpuCFLOAT )
        hostGPUDRV(zeroPadD2FC, N, gpunrhs, gpuprhs);
    else if( type_src == gpuDOUBLE && type_dst == gpuCDOUBLE )
        hostGPUDRV(zeroPadD2DC, N, gpunrhs, gpuprhs);
    m_timer->stopTimer( GpuTimes::zeroPad );
}

void ShearCudaFunctions::prepareMyerFilters(
    void* d_Filter, int size, int numDir, gpuTYPE_t type_signal ) const
{
    hostdrv_pars_t gpuprhs[3];
    int gpunrhs = 3;
    gpuprhs[0] = hostdrv_pars(&d_Filter,sizeof(d_Filter),__alignof(d_Filter));
    gpuprhs[1] = hostdrv_pars(&size,sizeof(size),__alignof(size));
    gpuprhs[2] = hostdrv_pars(&numDir,sizeof(numDir),__alignof(numDir));

    int N = size * size;

    m_timer->startTimer( GpuTimes::prepareMyerFilters );
    if( type_signal == gpuCFLOAT )
        hostGPUDRV(prepareMyerFiltersFC, N, gpunrhs, gpuprhs);
    else
        hostGPUDRV(prepareMyerFiltersDC, N, gpunrhs, gpuprhs);
    m_timer->stopTimer( GpuTimes::prepareMyerFilters );
}

// Image conversion
bool ShearCudaFunctions::convert(
        void* d_Dst, gpuTYPE_t type_dst, const void* d_Src, gpuTYPE_t type_src, int numElem) const
{
    hostdrv_pars_t gpuprhs[2];
    int gpunrhs = 2;
    gpuprhs[0] = hostdrv_pars(&d_Dst,sizeof(d_Dst),__alignof(d_Dst));
    gpuprhs[1] = hostdrv_pars(&d_Src,sizeof(d_Src),__alignof(d_Src));

    int N = numElem;
    bool ret = true;

    m_timer->startTimer( GpuTimes::convert );
    if( type_src == gpuFLOAT && type_dst == gpuUINT8 )
        hostGPUDRV(convertFTo8U, N, gpunrhs, gpuprhs);
    else if( type_src == gpuDOUBLE && type_dst == gpuUINT8 )
        hostGPUDRV(convertDTo8U, N, gpunrhs, gpuprhs);
    else if ( type_src == gpuUINT8 && type_dst == gpuFLOAT )
        hostGPUDRV(convert8UToF, N, gpunrhs, gpuprhs);
    else if( type_src == gpuUINT8 && type_dst == gpuDOUBLE )
        hostGPUDRV(convert8UToD, N, gpunrhs, gpuprhs);
    else
        ret = false;
    m_timer->stopTimer( GpuTimes::convert );

    return ret;
}

/////////////////////////////////////////////////////////////////////////////////
//                             HELPER FUNCTIONS                                //
/////////////////////////////////////////////////////////////////////////////////

bool ShearCudaFunctions::atrousConvolutionDevice(
        void* d_TempSubsampled, void* d_outArray, int O_SRowLength, int /* O_SColLength */,
        const void *d_SArray, int SRowLength, int SColLength,
        int nM, void* d_Filter, int filterLen, gpuTYPE_t type_signal ) const
{
    int level;
    switch (nM) {
        case 1: level = -1; break;
        case 2: level = 0; break;
        case 4: level = 1; break;
        case 8: level = 2; break;
        default: return false;
    }

    if (SRowLength!=SColLength)
        return false;

    // Retrieve filter
    const int fftW = O_SRowLength / nM + filterLen - 1;

    // Subsample signal
    atrousSubsample( d_TempSubsampled, d_SArray, fftW, SRowLength, nM, type_signal );

    // Upsample and do convolution
    atrousConvolution( d_outArray, d_TempSubsampled, d_Filter, O_SRowLength, fftW, filterLen, nM, type_signal);

    return true;
}

// Return sizes of different data types
int ShearCudaFunctions::elemSize( gpuTYPE_t type_signal )
{
    switch(type_signal)
    {
    case gpuDOUBLE: return sizeof(double);
    case gpuFLOAT: return sizeof(float);
    case gpuCDOUBLE: return 2 * sizeof(double);
    case gpuCFLOAT: return 2 * sizeof(float);
    case gpuINT32: return 4;
    case gpuUINT8: return 1;
    default:
        std::cerr << "Invalid data type " << type_signal << std::endl;
        exit(1);
        return 0;
    }
}
