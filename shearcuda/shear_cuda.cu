/*
 * Author: Xavier Gibert-Serra (gibert@umiacs.umd.edu)
 *
 * Copyright 2012-2013 University of Maryland. All rights reserved.
 *
 * This software contains source code provided by NVIDIA Corporation.
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 */


//#include <assert.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>
#include "cuda_common.h"
#include "GPUkernel.hh"
#include "complex_helper.h"

#define BLOCK_DIM1D_LARGE 512

#define LINPOS(row,col,numcols) (((row)*(numcols))+(col))

////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size (do a = a .* conj(b) )
// This function contains source code provided by NVIDIA Corporation.
////////////////////////////////////////////////////////////////////////////////
inline __device__ void mulAndScale(fComplex& a, const fComplex& b, const float& c){
    fComplex t = {c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y)};
    a = t;
}

inline __device__ void mulAndScale(dComplex& a, const dComplex& b, const double& c){
    dComplex t = {c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y)};
    a = t;
}

extern "C" __global__ void modulateConjAndNormalizeFC(
    int n,
    int offset,
    fComplex *d_Dst,
    fComplex *d_Src,
    float c
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        fComplex a = d_Src[idx];
        fComplex b = d_Dst[idx];
        a.y = -a.y;

        mulAndScale(a, b, c);

        d_Dst[idx] = a;
    }
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void modulateConjAndNormalizeDC(
    int n,
    int offset,
    dComplex *d_Dst,
    dComplex *d_Src,
    double c
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        dComplex a = d_Src[idx];
        dComplex b = d_Dst[idx];
        a.y = -a.y;

        mulAndScale(a, b, c);

        d_Dst[idx] = a;
    }
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Modulate and normalize with the same filter on multiple data
////////////////////////////////////////////////////////////////////////////////
template <class dataType, class realType> inline __device__ void modulateAndNormalizeMany_kernel(
    int n,
    int offset,
    dataType* d_Dst,
    const dataType* d_SrcData,
    const dataType* d_SrcKernel,
    int kernelSize,
    int numElem,
    realType c
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        int idxLinear = idx;

        // Use same data for all directions
        dataType data = d_SrcData[idxLinear];

        // Process all directions
        for( int idxDir = 0; idxDir < numElem; idxDir++, idxLinear += kernelSize )
        {
            dataType a = d_SrcKernel[idxLinear];
            mulAndScale(a, data, c);
            d_Dst[idxLinear] = a;
        }
    }
}

extern "C" __global__ void modulateAndNormalizeManyFC(
    int n,
    int offset,
    fComplex* d_Dst,
    const fComplex* d_SrcData,
    const fComplex* d_SrcKernel,
    int kernelSize,
    int numElem,
    float c
){
    modulateAndNormalizeMany_kernel<fComplex,float>(n, offset, d_Dst, d_SrcData, d_SrcKernel, kernelSize, numElem, c);
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void modulateAndNormalizeManyDC(
    int n,
    int offset,
    dComplex* d_Dst,
    const dComplex* d_SrcData,
    const dComplex* d_SrcKernel,
    int kernelSize,
    int numElem,
    double c
){
    modulateAndNormalizeMany_kernel<dComplex,double>(n, offset, d_Dst, d_SrcData, d_SrcKernel, kernelSize, numElem, c);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Modulate and normalize
////////////////////////////////////////////////////////////////////////////////
template <class dataType, class realType> inline __device__ void modulateAndNormalize3D_kernel(
    int n,
    int offset,
    dataType* d_Dst,
    const dataType* d_SrcData,
    const dataType* d_SrcKernel,
    int dataWH,
    int dataD,
    realType c
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        // Output coordinates
        int idxLinear = idx;

        // Process all depths
        for( int z = 0; z < dataD; z++, idxLinear += dataWH )
        {
            dataType a = d_SrcKernel[idxLinear];
            dataType b = d_SrcData[idxLinear];
            mulAndScale(a, b, c);
            d_Dst[idxLinear] = a;
        }
    }
}

extern "C" __global__ void modulateAndNormalize3DF(
    int n,
    int offset,
    fComplex* d_Dst,
    const fComplex* d_SrcData,
    const fComplex* d_SrcKernel,
    int dataWH,
    int dataD,
    float c
){
    modulateAndNormalize3D_kernel<fComplex,float>(n, offset, d_Dst, d_SrcData, d_SrcKernel, dataWH, dataD, c);
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void modulateAndNormalize3DD(
    int n,
    int offset,
    dComplex* d_Dst,
    const dComplex* d_SrcData,
    const dComplex* d_SrcKernel,
    int dataWH,
    int dataD,
    double c
){
    modulateAndNormalize3D_kernel<dComplex,double>(n, offset, d_Dst, d_SrcData, d_SrcKernel, dataWH, dataD, c);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Add 2 vectors and save result on first one
////////////////////////////////////////////////////////////////////////////////
inline __device__ void addVector(float& a, const float& b){
    a += b;
}

extern "C" __global__ void addVectorF(
    int n,
    int offset,
    float *d_Dst,
    float *d_Src
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        float a = d_Src[idx];
        float b = d_Dst[idx];

        addVector(a, b);

        d_Dst[idx] = a;
    }
}

#if __CUDA_ARCH__ >= 130
inline __device__ void addVector(double& a, const double& b){
    a += b;
}

extern "C" __global__ void addVectorD(
    int n,
    int offset,
    double *d_Dst,
    double *d_Src
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        double a = d_Src[idx];
        double b = d_Dst[idx];

        addVector(a, b);

        d_Dst[idx] = a;
    }
}
#endif

inline __device__ int iDivUp_dev(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

////////////////////////////////////////////////////////////////////////////////
// Add k vectors and save result on d_dst
////////////////////////////////////////////////////////////////////////////////

extern "C" __global__ void sumVectorsF(
    int n,
    int offset,
    float *d_Dst,
    float *d_Src,
    int numComponents
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        float sum = 0.0f;
        for( int j = 0; j < numComponents; j++, d_Src += n )
            sum += d_Src[idx];

        d_Dst[idx] = sum;
    }
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void sumVectorsD(
    int n,
    int offset,
    double *d_Dst,
    double *d_Src,
    int numComponents
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        double sum = 0.0;
        for( int j = 0; j < numComponents; j++, d_Src += n )
            sum += d_Src[idx];

        d_Dst[idx] = sum;
    }
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Multiply a vector by a scalar
////////////////////////////////////////////////////////////////////////////////

extern "C" __global__ void mulMatrixByScalarF(
    int n,
    int offset,
    float *d_Dst,
    float *d_Src,
    float scalar
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        d_Dst[idx] = scalar * d_Src[idx];
    }
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void mulMatrixByScalarD(
    int n,
    int offset,
    double *d_Dst,
    double *d_Src,
    double scalar
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        d_Dst[idx] = scalar * d_Src[idx];
    }
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Subsample 2D image (atrousSubsampleF and atrousSubsampleD)
////////////////////////////////////////////////////////////////////////////////
template <class dataType> inline __device__ void atrousSubsample_kernel(
    int n,
    int offset,
    dataType* d_Subsampled,
    const dataType* d_Signal,
    int subW,
    int signalW,
    int level
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        const int subSize = subW * subW;
    
        // Determine which subimage to generate (there are level x level subimages)
        const int subImgIdx = idx / subSize;      // Linear index
        const int subImgIdxCol = subImgIdx % level;
        const int subImgIdxRow = subImgIdx / level;

        // Determine which pixel in subimage to generate (each image is of size subW x subW)
        const int subIdx = idx % subSize;         // Linear index
        const int subCol = subIdx % subW;
        const int subRow = subIdx / subW;

        // Calculate corresponding pixel in original image
        const int row = level - 1 + subImgIdxRow + level * subRow;
        const int col = level - 1 + subImgIdxCol + level * subCol;

        // Move data
        d_Subsampled[idx] = d_Signal[LINPOS(row,col,signalW)];
    }
}

extern "C" __global__ void atrousSubsampleF(
    int n,
    int offset,
    float* d_Subsampled,
    const float* d_Signal,
    int subW,
    int signalW,
    int level
){
    atrousSubsample_kernel<float>(n, offset, d_Subsampled, d_Signal, subW, signalW, level);
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void atrousSubsampleD(
    int n,
    int offset,
    double* d_Subsampled,
    const double* d_Signal,
    int subW,
    int signalW,
    int level
){
    atrousSubsample_kernel<double>(n, offset, d_Subsampled, d_Signal, subW, signalW, level);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Upsubsample 2D image (atrousUpsampleF and atrousUpsampleD)
////////////////////////////////////////////////////////////////////////////////
template <class dataType> inline __device__ void atrousUpsample_kernel(
    int n,
    int offset,
    dataType* d_Out,
    const dataType* d_Subsampled,
    int outW,
    int subW_padded,
    int padding,
    int level
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        const int subSize_padded = subW_padded * subW_padded;
        const int subW = outW / level;
        const int subSize = subW * subW;

        // Determine which subimage to operate on (there are level x level subimages)
        const int subImgIdx = idx / subSize;      // Linear index
        const int subImgIdxCol = subImgIdx % level;
        const int subImgIdxRow = subImgIdx / level;

        // Find pointer to subimage
        const dataType* d_SubImage = d_Subsampled + subImgIdx * subSize_padded;

        // Determine which pixel in subimage to operate on (each image is of size subW x subW)
        const int subIdx = idx % subSize;         // Linear index
        const int subCol = subIdx % subW;
        const int subRow = subIdx / subW;

        // Calculate corresponding pixel in original image
        const int row = subImgIdxRow + level * subRow;
        const int col = subImgIdxCol + level * subCol;

        // Move data
        d_Out[LINPOS(row,col,outW)] = d_SubImage[LINPOS(subRow+padding,subCol+padding,subW_padded)];
    }
}

extern "C" __global__ void atrousUpsampleF(
    int n,
    int offset,
    float* d_Out,
    const float* d_Subsampled,
    int outW,
    int subW_padded,
    int padding,
       int level
){
    atrousUpsample_kernel<float>(n, offset, d_Out, d_Subsampled, outW, subW_padded, padding, level);
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void atrousUpsampleD(
    int n,
    int offset,
    double* d_Out,
    const double* d_Subsampled,
    int outW,
    int subW_padded,
    int padding,
    int level
){
    atrousUpsample_kernel<double>(n, offset, d_Out, d_Subsampled, outW, subW_padded, padding, level);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// A trous convolution kernel (atrousConvolutionF and atrousConvolutionD)
////////////////////////////////////////////////////////////////////////////////
template <class dataType> inline __device__ void atrousConvolution_kernel(
    int n,
    int offset,
    dataType* d_Out,
    const dataType* d_Subsampled,
    dataType *d_Filter,
    int outW,
    int subW_padded,
    int filterLen,
    int level
){
    __shared__ dataType shared_filter[512];
    __shared__ dataType data_cache[3072];

    const int padding = filterLen - 1;
    const int filterSize = filterLen * filterLen;

    const unsigned int idx = BLOCK_DIM1D_LARGE * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        // Load filters into shared memory
        int i = threadIdx.x;
        while (i<filterSize)
        {
            shared_filter[i] = d_Filter[i];
            i += BLOCK_DIM1D_LARGE;
        }

        const int subSize_padded = subW_padded * subW_padded;
        const int subW = outW / level;
        const int subSize = subW * subW;

        // Determine which subimage to operate on (there are level x level subimages)
        i = BLOCK_DIM1D_LARGE * blockIdx.x + threadIdx.x;
        const int subImgIdx = i / subSize;      // Linear index
        const int subImgIdxCol = subImgIdx % level;
        const int subImgIdxRow = subImgIdx / level;

        // Find pointer to subimage
        const dataType* d_SubImage = d_Subsampled + subImgIdx * subSize_padded;

        // Determine which pixel in subimage to operate on (each image is of size subW x subW)
        const int subIdx = i % subSize;         // Linear index
        const int subCol = subIdx % subW + padding;
        const int subRow = subIdx / subW + padding;

        // Each tread in the block will calculate a row range of pixels in the image
        // The data will be loaded once in shared memory, and then each thread will
        // multiply it with the appropriate coefficients in the filter
        // We need to determine what part of the images the other threads in the block
        // will process
        // NOTE: We implicitly assume that subW * subW is a multiple of the block size (256)
        const int first_subIdx = (BLOCK_DIM1D_LARGE * blockIdx.x) % subSize;
        int last_subIdx = first_subIdx + BLOCK_DIM1D_LARGE;
        //if (last_subIdx > subSize) last_subIdx = last_subIdx % subSize;

        const int first_subRow = first_subIdx / subW;
        const int last_subRow = last_subIdx / subW + 2 * padding;

        // The number of rows that fit in shared memory is
        int nNumRowsPerBlock = 3072 / subW_padded;
        // The region of interest on which to operate is
        const int roiSize = (last_subRow - first_subRow) * subW_padded;
        int nBlockSize = nNumRowsPerBlock * subW_padded;

        // The number of blocks to process is
        const int nNumBlocks = iDivUp_dev( roiSize, nBlockSize );

        // Initialize sum
        dataType sum = 0;

        for (int blockIdx=0; blockIdx<nNumBlocks; blockIdx++)
        {
            // Image linear offset to beginning of block
            const int offsetRows = blockIdx * nNumRowsPerBlock;
            const int offset = blockIdx * nBlockSize;

            __syncthreads();

            // Adjust for shorter last block
            if (blockIdx == nNumBlocks-1) {
                nBlockSize = roiSize - blockIdx * nBlockSize;
                nNumRowsPerBlock = nBlockSize / subW_padded;
            }

            // Read data to shared memory
            i = threadIdx.x;
            while (i<nBlockSize)
            {
                data_cache[i] = d_SubImage[first_subRow * subW_padded + offset + i];
                i += BLOCK_DIM1D_LARGE;
            }

            __syncthreads();

            // Range of rows and columns to perform operations
            int startRow = max(offsetRows + first_subRow, subRow - padding);
            int endRow = min(offsetRows + last_subRow, subRow + 1);
            endRow = min( endRow, nNumRowsPerBlock + offsetRows + first_subRow );
            int startCol = max(0, subCol - padding);
            int endCol = min(subW_padded, subCol + 1);

            // Operate on shared memory
            for( int srcRow = startRow; srcRow < endRow; srcRow++ )
            {
                int filterRow = srcRow - subRow + padding;
                dataType *pDataRow = data_cache + (srcRow - offsetRows - first_subRow) * subW_padded;
                dataType *pFilterRow = shared_filter + filterRow * filterLen;
                for( int srcCol = startCol; srcCol < endCol; srcCol++ )
                {
                    int filterCol = srcCol - subCol + padding;
                    sum += pDataRow[srcCol] * pFilterRow[filterCol];
                }
            }
        }

        // Calculate corresponding pixel in original image
        const int row = subImgIdxRow + level * (subRow - padding);
        const int col = subImgIdxCol + level * (subCol - padding);

        // Collect result
        d_Out[LINPOS(row,col,outW)] = sum;
    }
}

extern "C" __global__ void atrousConvolutionF(
    int n,
    int offset,
    float* d_Out,
    const float* d_Subsampled,
    float *d_Filter,
    int outW,
    int subW_padded,
    int filterLen,
    int level
){
    atrousConvolution_kernel<float>(n, offset, d_Out, d_Subsampled, d_Filter, outW, subW_padded, filterLen, level);
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void atrousConvolutionD(
    int n,
    int offset,
    double* d_Out,
    const double* d_Subsampled,
    double *d_Filter,
    int outW,
    int subW_padded,
    int filterLen,
    int level
){
    atrousConvolution_kernel<double>(n, offset, d_Out, d_Subsampled, d_Filter, outW, subW_padded, filterLen, level);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Apply hard thresholding
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void hardThresholdF(
    int n,
    int offset,
    float* d_Dst,
    const float* d_Src,
    float th
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        float val = d_Src[idx];

        float res;
        if( fabs(val) > th )
            res = val;
        else
            res = 0;

        d_Dst[idx] = res;
    }
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void hardThresholdD(
    int n,
    int offset,
    double* d_Dst,
    const double* d_Src,
    double th
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        double val = d_Src[idx];

        double res;
        if( fabs(val) > th )
            res = val;
        else
            res = 0;

        d_Dst[idx] = res;
    }
}
#endif

extern "C" __global__ void hardThresholdFC(
    int n,
    int offset,
    fComplex* d_Dst,
    const fComplex* d_Src,
    float th_sq
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        fComplex val = d_Src[idx];

        fComplex res;
        if( val.x * val.x + val.y * val.y > th_sq )
            res = val;
        else
            res.x = res.y = 0;

        d_Dst[idx] = res;
    }
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void hardThresholdDC(
    int n,
    int offset,
    dComplex* d_Dst,
    const dComplex* d_Src,
    double th_sq
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        dComplex val = d_Src[idx];

        dComplex res;
        if( val.x * val.x + val.y * val.y > th_sq )
            res = val;
        else
            res.x = res.y = 0;

        d_Dst[idx] = res;
    }
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Apply soft thresholding
////////////////////////////////////////////////////////////////////////////////
inline __device__ double fabs(const dComplex& a){
    return sqrt(a.x * a.x + a.y * a.y);
}

inline __device__ float fabs(const fComplex& a){
    return sqrt(a.x * a.x + a.y * a.y);
}

extern "C" __global__ void softThresholdF(
    int n,
    int offset,
    float* d_Dst,
    const float* d_Src,
    float th
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        float val = d_Src[idx];

        float res;
        if( fabs(val) > th )
            res = (val > 0 ? val - th : val + th);
        else
            res = 0;

        d_Dst[idx] = res;
    }
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void softThresholdD(
    int n,
    int offset,
    double* d_Dst,
    const double* d_Src,
    double th
){  
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        double val = d_Src[idx];

        double res;
        if( fabs(val) > th )
            res = (val > 0 ? val - th : val + th);
        else
            res = 0;

        d_Dst[idx] = res;
    }
}
#endif

extern "C" __global__ void softThresholdFC(
    int n,
    int offset,
    fComplex* d_Dst,
    const fComplex* d_Src,
    float th
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        fComplex val = d_Src[idx];
        float absVal = fabs(val);

        fComplex res;
        if( absVal > th )
        {
            float factor = (absVal - th) / absVal;
            res.x = val.x * factor;
            res.y = val.y * factor;
        }
        else
            res.x = res.y = 0;

        d_Dst[idx] = res;
    }
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void softThresholdDC(
    int n,
    int offset,
    dComplex* d_Dst,
    const dComplex* d_Src,
    float th
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        dComplex val = d_Src[idx];
        double absVal = fabs(val);

        dComplex res;
        if( absVal > th )
        {
            float factor = (absVal - th) / absVal;
            res.x = val.x * factor;
            res.y = val.y * factor;
        }
        else
            res.x = res.y = 0;

        d_Dst[idx] = res;
    }
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Symmetric extension
////////////////////////////////////////////////////////////////////////////////
template <class dataType> inline __device__ void symExt_kernel(
    int n,
    int offset,
    dataType* d_Output,
    int outputRows,
    int outputCols,
    const dataType* d_Input,
    int inputRows,
    int inputCols,
    int topOffset,
    int leftOffset
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        // Determine source row and column
        int row = idx / inputRows;
        int col = idx % inputCols;

        // Copy data from global memory
        float val = d_Input[LINPOS(row,col,inputCols)];

        // Calculate candidate target locations in output image
        int o_row_Top = topOffset - 1 - row;
        int o_row_Cent = row + topOffset;
        int o_row_Bot = topOffset + 2 * inputRows - 1 - row; 
        int o_col_Left = leftOffset - 1 - col;
        int o_col_Cent = col + leftOffset;
        int o_col_Right = leftOffset + 2 * inputCols - 1 - col;

        // Save shifted image
        d_Output[LINPOS(o_row_Cent,o_col_Cent,outputCols)] = val;

        // Save extension
        if( o_row_Top >= 0 )
        {
            d_Output[LINPOS(o_row_Top,o_col_Cent,outputCols)] = val;
            if( o_col_Left >= 0 )
                d_Output[LINPOS(o_row_Top,o_col_Left,outputCols)] = val;
            else if( o_col_Right < outputCols )
                d_Output[LINPOS(o_row_Top,o_col_Right,outputCols)] = val;
        }
        else if( o_row_Bot < outputRows )
        {
            d_Output[LINPOS(o_row_Bot,o_col_Cent,outputCols)] = val;
            if( o_col_Left >= 0 )
                d_Output[LINPOS(o_row_Bot,o_col_Left,outputCols)] = val;
            else if( o_col_Right < outputCols )
                d_Output[LINPOS(o_row_Bot,o_col_Right,outputCols)] = val;
        }
        if( o_col_Left >= 0 )
            d_Output[LINPOS(o_row_Cent,o_col_Left,outputCols)] = val;
        else if( o_col_Right < outputCols )
            d_Output[LINPOS(o_row_Cent,o_col_Right,outputCols)] = val;
    }
}

extern "C" __global__ void symExtF(
    int n,
    int offset,
    float* d_Output,
    int outputRows,
    int outputCols,
    const float* d_Input,
    int inputRows,
    int inputCols,
    int topOffset,
    int leftOffset
){
    symExt_kernel<float>(n, offset, d_Output, outputRows, outputCols, d_Input, inputRows, inputCols, topOffset, leftOffset);
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void symExtD(
    int n,
    int offset,
    double* d_Output,
    int outputRows,
    int outputCols,
    const double* d_Input,
    int inputRows,
    int inputCols,
    int topOffset,
    int leftOffset
){
    symExt_kernel<double>(n, offset, d_Output, outputRows, outputCols, d_Input, inputRows, inputCols, topOffset, leftOffset);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Multiply scalar by a vector
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void scalarVectorMulF(
    int n,
    int offset,
    float* d_Data,
    float scalar
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
        d_Data[idx] = d_Data[idx] * scalar;
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void scalarVectorMulD(
    int n,
    int offset,
    double* d_Data,
    double scalar
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
        d_Data[idx] = d_Data[idx] * scalar;
}
#endif

////////////////////////////////////////////////////////////////////////////////
// 1D Undecimated Wavelet transform along rows (columns in MATLAB)
////////////////////////////////////////////////////////////////////////////////
template <class dataType> inline __device__ void mrdwtRow_kernel(
    int n,
    int offset,
    const dataType* d_X,
    int numRows,   // m
    int numCols,   // n
    const dataType* d_Filter,   // h
    int filterLen,              // lh
    int level,                  // actual_L
    dataType* d_yl,     // Low pass
    dataType* d_yh      // High pass
){
    __shared__ dataType h0_filter[32];
    __shared__ dataType h1_filter[32];
    __shared__ dataType shared_cache[320];       // Data cache

    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    const unsigned int blockStart_idx = idx - threadIdx.x;
    if ((idx - offset) < n)
    {
        // Load filters into shared memory
        int j = threadIdx.x;
        if( j < filterLen )
        {
            dataType val = d_Filter[j];
            // Mirror h0
            h0_filter[j] = val;
            // Modulate h1
            if( j%2 == 0)
                h1_filter[filterLen-1-j] = -val;
            else
                h1_filter[filterLen-1-j] = val;
        }

        // Output coordinates
        int idxRow = idx / numCols;
        int idxCol = idx % numCols;
        int blockStart_idxRow = blockStart_idx / numCols;   // First row processed by this thread block

        // Calculate coordinates to operate on
        int actual_n = numCols >> (level - 1);    // Row index to read from
        int sample_f = 1 << (level-1);      // Number of blocks (sample_f == n_cb )
        int temp_block_len = actual_n + filterLen;  // Block lenght corresponding to one decimated row
        int blockStart_offset = temp_block_len * sample_f * (idxRow - blockStart_idxRow);   // Offset to first element in shared memory
        dataType* buffer = shared_cache + blockStart_offset;
        int n_c = idxCol / actual_n;
        int i = idxCol % actual_n;
        int ic = n_c + sample_f * i;
        int c_o_a = (numRows==1 ? numCols*(level-1) : 0);

        // Load data into cache
        buffer[temp_block_len * n_c + i] = d_X[LINPOS(idxRow,ic,numCols)];
        // Load periodic extension
        if( i < filterLen - 1 )
        {
            buffer[temp_block_len * n_c + actual_n + i] = d_X[LINPOS(idxRow,(ic+numCols) % numCols,numCols)];
        }
        __syncthreads();

        // Perform convolution on rows
        dataType x0 = 0, x1 = 0;
        int idxOffset = temp_block_len * n_c + i;
        for( int k = 0; k < filterLen; k++ )
        {
            x0 += buffer[idxOffset + k] * h0_filter[k];
            x1 += buffer[idxOffset + k] * h1_filter[k];
        }

        // Save results
        d_yl[LINPOS(idxRow,ic,numCols)] = x0;
        d_yh[LINPOS(idxRow,c_o_a+ic,numCols)] = x1;
    }
}

extern "C" __global__ void mrdwtRowF(
    int n,
    int offset,
    const float* d_X,
    int numRows,   // m
    int numCols,   // n
    const float* d_Filter,   // h
    int filterLen,              // lh
    int level,                  // actual_L
    float* d_yl,     // Low pass
    float* d_yh      // High pass
){
    mrdwtRow_kernel<float>(n, offset, d_X, numRows, numCols, d_Filter, filterLen, level, d_yl, d_yh);
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void mrdwtRowD(
    int n,
    int offset,
    const double* d_X,
    int numRows,   // m
    int numCols,   // n
    const double* d_Filter,   // h
    int filterLen,              // lh
    int level,                  // actual_L
    double* d_yl,     // Low pass
    double* d_yh      // High pass
){
    mrdwtRow_kernel<double>(n, offset, d_X, numRows, numCols, d_Filter, filterLen, level, d_yl, d_yh);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// 1D Undecimated Wavelet transform along columns (rows in MATLAB)
////////////////////////////////////////////////////////////////////////////////
template <class dataType> inline __device__ void mrdwtCol_kernel(
    int n,
    int offset,
    const dataType* d_X,
    int numRows,   // m
    int numCols,   // n
    const dataType* d_Filter,   // h
    int filterLen,              // lh
    int level,                  // actual_L
    dataType* d_yl,     // Low pass
    dataType* d_yh      // High pass
){
    __shared__ dataType h0_filter[32];
    __shared__ dataType h1_filter[32];
    __shared__ dataType shared_cache[320];       // Data cache

    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    const unsigned int blockStart_idx = idx - threadIdx.x;
    if ((idx - offset) < n)
    {
        // Load filters into shared memory
        int j = threadIdx.x;
        if( j < filterLen )
        {
            dataType val = d_Filter[j];
            // Mirror h0
            h0_filter[j] = val;
            // Modulate h1
            if( j%2 == 0)
                h1_filter[filterLen-1-j] = -val;
            else
                h1_filter[filterLen-1-j] = val;
        }

        // Output coordinates
        int idxCol = idx / numRows;
        int idxRow = idx % numRows;
        int blockStart_idxCol = blockStart_idx / numCols;   // First column processed by this thread block

        // Calculate coordinates to operate on
        int actual_m = numRows >> (level - 1);    // Column index to read from
        int sample_f = 1 << (level-1);      // Number of row blocks (sample_f == n_rb )
        int temp_block_len = actual_m + filterLen;
        int blockStart_offset = temp_block_len * sample_f * (idxCol - blockStart_idxCol);   // Offset to first element in shared memory
        dataType* buffer = shared_cache + blockStart_offset;
        int n_r = idxRow / actual_m;
        int i = idxRow % actual_m;
        int ir = n_r + sample_f * i;

        // Load data into cache
        buffer[temp_block_len * n_r + i] = d_X[LINPOS(ir,idxCol,numCols)];
        // Load periodic extension
        if( i < filterLen - 1 )
        {
            buffer[temp_block_len * n_r + actual_m + i] = d_X[LINPOS((ir+numRows) % numRows,idxCol,numCols)];
        }
        __syncthreads();

        // Perform convolution on columns
        dataType x0 = 0, x1 = 0;
        int idxOffset = temp_block_len * n_r + i;
        for( int k = 0; k < filterLen; k++ )
        {
            x0 += buffer[idxOffset + k] * h0_filter[k];
            x1 += buffer[idxOffset + k] * h1_filter[k];
        }

        // Save results
        d_yl[LINPOS(ir,idxCol,numCols)] = x0;
        d_yh[LINPOS(ir,idxCol,numCols)] = x1;
    }
}

extern "C" __global__ void mrdwtColF(
    int n,
    int offset,
    const float* d_X,
    int numRows,   // m
    int numCols,   // n
    const float* d_Filter,   // h
    int filterLen,              // lh
    int level,                  // actual_L
    float* d_yl,     // Low pass
    float* d_yh      // High pass
){
    mrdwtCol_kernel<float>(n, offset, d_X, numRows, numCols, d_Filter, filterLen, level, d_yl, d_yh);
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void mrdwtColD(
    int n,
    int offset,
    const double* d_X,
    int numRows,   // m
    int numCols,   // n
    const double* d_Filter,   // h
    int filterLen,              // lh
    int level,                  // actual_L
    double* d_yl,     // Low pass
    double* d_yh      // High pass
){
    mrdwtCol_kernel<double>(n, offset, d_X, numRows, numCols, d_Filter, filterLen, level, d_yl, d_yh);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// 1D Undecimated Wavelet transform along rows (columns in MATLAB)
// Large (512) block size version
////////////////////////////////////////////////////////////////////////////////
template <class dataType> inline __device__ void mrdwtRow512_kernel(
    int n,
    int offset,
    const dataType* d_X,
    int numRows,   // m
    int numCols,   // n
    const dataType* d_Filter,   // h
    int filterLen,              // lh
    int level,                  // actual_L
    dataType* d_yl,     // Low pass
    dataType* d_yh      // High pass
){
    __shared__ dataType h0_filter[32];
    __shared__ dataType h1_filter[32];
    __shared__ dataType shared_cache[640];       // Data cache

    const unsigned int idx = BLOCK_DIM1D_LARGE * blockIdx.x + threadIdx.x + offset;
    const unsigned int blockStart_idx = idx - threadIdx.x;
    if ((idx - offset) < n)
    {
        // Load filters into shared memory
        int j = threadIdx.x;
        if( j < filterLen )
        {
            dataType val = d_Filter[j];
            // Mirror h0
            h0_filter[j] = val;
            // Modulate h1
            if( j%2 == 0)
                h1_filter[filterLen-1-j] = -val;
            else
                h1_filter[filterLen-1-j] = val;
        }

        // Output coordinates
        int idxRow = idx / numCols;
        int idxCol = idx % numCols;
        int blockStart_idxRow = blockStart_idx / numCols;   // First row processed by this thread block

        // Calculate coordinates to operate on
        int actual_n = numCols >> (level - 1);    // Row index to read from
        int sample_f = 1 << (level-1);      // Number of blocks (sample_f == n_cb )
        int temp_block_len = actual_n + filterLen;  // Block lenght corresponding to one decimated row
        int blockStart_offset = temp_block_len * sample_f * (idxRow - blockStart_idxRow);   // Offset to first element in shared memory
        dataType* buffer = shared_cache + blockStart_offset;
        int n_c = idxCol / actual_n;
        int i = idxCol % actual_n;
        int ic = n_c + sample_f * i;
        int c_o_a = (numRows==1 ? numCols*(level-1) : 0);

        // Load data into cache
        buffer[temp_block_len * n_c + i] = d_X[LINPOS(idxRow,ic,numCols)];
        // Load periodic extension
        if( i < filterLen - 1 )
        {
            buffer[temp_block_len * n_c + actual_n + i] = d_X[LINPOS(idxRow,(ic+numCols) % numCols,numCols)];
        }
        __syncthreads();

        // Perform convolution on rows
        dataType x0 = 0, x1 = 0;
        int idxOffset = temp_block_len * n_c + i;
        for( int k = 0; k < filterLen; k++ )
        {
            x0 += buffer[idxOffset + k] * h0_filter[k];
            x1 += buffer[idxOffset + k] * h1_filter[k];
        }

        // Save results
        d_yl[LINPOS(idxRow,ic,numCols)] = x0;
        d_yh[LINPOS(idxRow,c_o_a+ic,numCols)] = x1;
    }
}

extern "C" __global__ void mrdwtRow512F(
    int n,
    int offset,
    const float* d_X,
    int numRows,   // m
    int numCols,   // n
    const float* d_Filter,   // h
    int filterLen,              // lh
    int level,                  // actual_L
    float* d_yl,     // Low pass
    float* d_yh      // High pass
){
    mrdwtRow512_kernel<float>(n, offset, d_X, numRows, numCols, d_Filter, filterLen, level, d_yl, d_yh);
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void mrdwtRow512D(
    int n,
    int offset,
    const double* d_X,
    int numRows,   // m
    int numCols,   // n
    const double* d_Filter,   // h
    int filterLen,              // lh
    int level,                  // actual_L
    double* d_yl,     // Low pass
    double* d_yh      // High pass
){
    mrdwtRow512_kernel<double>(n, offset, d_X, numRows, numCols, d_Filter, filterLen, level, d_yl, d_yh);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// 1D Undecimated Wavelet transform along columns (rows in MATLAB)
// Large (512) block size version
////////////////////////////////////////////////////////////////////////////////
template <class dataType> inline __device__ void mrdwtCol512_kernel(
    int n,
    int offset,
    const dataType* d_X,
    int numRows,   // m
    int numCols,   // n
    const dataType* d_Filter,   // h
    int filterLen,              // lh
    int level,                  // actual_L
    dataType* d_yl,     // Low pass
    dataType* d_yh      // High pass
){
    __shared__ dataType h0_filter[32];
    __shared__ dataType h1_filter[32];
    __shared__ dataType shared_cache[640];       // Data cache

    const unsigned int idx = BLOCK_DIM1D_LARGE * blockIdx.x + threadIdx.x + offset;
    const unsigned int blockStart_idx = idx - threadIdx.x;
    if ((idx - offset) < n)
    {
        // Load filters into shared memory
        int j = threadIdx.x;
        if( j < filterLen )
        {
            dataType val = d_Filter[j];
            // Mirror h0
            h0_filter[j] = val;
            // Modulate h1
            if( j%2 == 0)
                h1_filter[filterLen-1-j] = -val;
            else
                h1_filter[filterLen-1-j] = val;
        }

        // Output coordinates
        int idxCol = idx / numRows;
        int idxRow = idx % numRows;
        int blockStart_idxCol = blockStart_idx / numCols;   // First column processed by this thread block

        // Calculate coordinates to operate on
        int actual_m = numRows >> (level - 1);    // Column index to read from
        int sample_f = 1 << (level-1);      // Number of row blocks (sample_f == n_rb )
        int temp_block_len = actual_m + filterLen;
        int blockStart_offset = temp_block_len * sample_f * (idxCol - blockStart_idxCol);   // Offset to first element in shared memory
        dataType* buffer = shared_cache + blockStart_offset;
        int n_r = idxRow / actual_m;
        int i = idxRow % actual_m;
        int ir = n_r + sample_f * i;

        // Load data into cache
        buffer[temp_block_len * n_r + i] = d_X[LINPOS(ir,idxCol,numCols)];
        // Load periodic extension
        if( i < filterLen - 1 )
        {
            buffer[temp_block_len * n_r + actual_m + i] = d_X[LINPOS((ir+numRows) % numRows,idxCol,numCols)];
        }
        __syncthreads();

        // Perform convolution on columns
        dataType x0 = 0, x1 = 0;
        int idxOffset = temp_block_len * n_r + i;
        for( int k = 0; k < filterLen; k++ )
        {
            x0 += buffer[idxOffset + k] * h0_filter[k];
            x1 += buffer[idxOffset + k] * h1_filter[k];
        }

        // Save results
        d_yl[LINPOS(ir,idxCol,numCols)] = x0;
        d_yh[LINPOS(ir,idxCol,numCols)] = x1;
    }
}

extern "C" __global__ void mrdwtCol512F(
    int n,
    int offset,
    const float* d_X,
    int numRows,   // m
    int numCols,   // n
    const float* d_Filter,   // h
    int filterLen,              // lh
    int level,                  // actual_L
    float* d_yl,     // Low pass
    float* d_yh      // High pass
){
    mrdwtCol512_kernel<float>(n, offset, d_X, numRows, numCols, d_Filter, filterLen, level, d_yl, d_yh);
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void mrdwtCol512D(
    int n,
    int offset,
    const double* d_X,
    int numRows,   // m
    int numCols,   // n
    const double* d_Filter,   // h
    int filterLen,              // lh
    int level,                  // actual_L
    double* d_yl,     // Low pass
    double* d_yh      // High pass
){
    mrdwtCol512_kernel<double>(n, offset, d_X, numRows, numCols, d_Filter, filterLen, level, d_yl, d_yh);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// 1D Undecimated Inverse Wavelet transform along rows (columns in MATLAB)
////////////////////////////////////////////////////////////////////////////////
template <class dataType> inline __device__ void mirdwtRow_kernel(
    int n,
    int offset,
    const dataType* d_xinl,     // Low pass
    const dataType* d_xinh,      // High pass
    int numRows,   // m
    int numCols,   // n
    const dataType* d_Filter,   // g
    int filterLen,              // lg
    int level,                  // actual_L
    dataType* d_xout       // Merged output
){
    __shared__ dataType g0_filter[32];
    __shared__ dataType g1_filter[32];
    __shared__ dataType shared_cache_xl[320];       // Data cache for d_xinl
    __shared__ dataType shared_cache_xh[320];       // Data cache for d_xinh

    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    const unsigned int blockStart_idx = idx - threadIdx.x;
    if ((idx - offset) < n)
    {
        // Load filters into shared memory
        int j = threadIdx.x;
        if( j < filterLen )
        {
            dataType val = d_Filter[j];
            // Mirror g0
            g0_filter[filterLen-1-j] = val/2;
            // Modulate g1
            if( j%2 == 0)
                g1_filter[j] = -val/2;
            else
                g1_filter[j] = val/2;
        }

        // Output coordinates
        int idxRow = idx / numCols;
        int idxCol = idx % numCols;
        int blockStart_idxRow = blockStart_idx / numCols;   // First row processed by this thread block

        // Calculate coordinates to operate on
        int actual_n = numCols >> (level - 1);    // Row index to read from
        int sample_f = 1 << (level-1);      // Number of blocks (sample_f == n_cb )
        int temp_block_len = actual_n + filterLen;  // Block lenght corresponding to one decimated row
        int blockStart_offset = temp_block_len * sample_f * (idxRow - blockStart_idxRow);   // Offset to first element in shared memory
        dataType* buffer_xl = shared_cache_xl + blockStart_offset;
        dataType* buffer_xh = shared_cache_xh + blockStart_offset;
        int n_c = idxCol / actual_n;
        int i = idxCol % actual_n;
        int ic = n_c + sample_f * i;
        int c_o_a = (numRows==1 ? numCols*(level-1) : 0);

        // Load data into cache
        buffer_xl[temp_block_len * n_c + i + filterLen - 1] = d_xinl[LINPOS(idxRow,ic,numCols)];
        buffer_xh[temp_block_len * n_c + i + filterLen - 1] = d_xinh[LINPOS(idxRow,c_o_a+ic,numCols)];
        __syncthreads();
        // Load periodic extension
        if( i < filterLen - 1 )
        {
            buffer_xl[temp_block_len * n_c + i] = buffer_xl[temp_block_len * n_c + i + actual_n];
//d_xinl[LINPOS(idxRow,(ic+numCols-filterLen+1) % numCols,numCols)];
            buffer_xh[temp_block_len * n_c + i] = buffer_xh[temp_block_len * n_c + i + actual_n];
//d_xinh[LINPOS(idxRow,(c_o_a+ic+numCols-filterLen+1) % numCols,numCols)];
        }
        __syncthreads();

        // Perform convolution on rows
        dataType x0 = 0;
        int idxOffset = temp_block_len * n_c + i;
        for( int k = 0; k < filterLen; k++ )
        {
            x0 += buffer_xl[idxOffset + k] * g0_filter[k] +
                  buffer_xh[idxOffset + k] * g1_filter[k];
        }

        // Save result
        d_xout[LINPOS(idxRow,ic,numCols)] = x0;
    }
}

extern "C" __global__ void mirdwtRowF(
    int n,
    int offset,
    const float* d_xinl,     // Low pass
    const float* d_xinh,     // High pass
    int numRows,   // m
    int numCols,   // n
    const float* d_Filter,   // g
    int filterLen,              // lh
    int level,                  // actual_L
    float* d_xout       // Merged output
){
    mirdwtRow_kernel<float>(n, offset, d_xinl, d_xinh, numRows, numCols, d_Filter, filterLen, level, d_xout);
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void mirdwtRowD(
    int n,
    int offset,
    const double* d_xinl,     // Low pass
    const double* d_xinh,     // High pass
    int numRows,   // m
    int numCols,   // n
    const double* d_Filter,   // g
    int filterLen,              // lh
    int level,                  // actual_L
    double* d_xout        // Merged output
){
    mirdwtRow_kernel<double>(n, offset, d_xinl, d_xinh, numRows, numCols, d_Filter, filterLen, level, d_xout);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// 1D Undecimated Inverse Wavelet transform along columns (rows in MATLAB)
////////////////////////////////////////////////////////////////////////////////
template <class dataType> inline __device__ void mirdwtCol_kernel(
    int n,
    int offset,
    const dataType* d_xinl,     // Low pass
    const dataType* d_xinh,     // High pass
    int numRows,   // m
    int numCols,   // n
    const dataType* d_Filter,   // g
    int filterLen,              // lg
    int level,                  // actual_L
    dataType* d_xout        // Merged output
){
    __shared__ dataType g0_filter[32];
    __shared__ dataType g1_filter[32];
    __shared__ dataType shared_cache_xl[320];       // Data cache for d_xinl
    __shared__ dataType shared_cache_xh[320];       // Data cache for d_xinh

    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    const unsigned int blockStart_idx = idx - threadIdx.x;
    if ((idx - offset) < n)
    {
        // Load filters into shared memory
        int j = threadIdx.x;
        if( j < filterLen )
        {
            dataType val = d_Filter[j];
            // Mirror g0
            g0_filter[filterLen-1-j] = val/2;
            // Modulate g1
            if( j%2 == 0)
                g1_filter[j] = -val/2;
            else
                g1_filter[j] = val/2;
        }

        // Output coordinates
        int idxCol = idx / numRows;
        int idxRow = idx % numRows;
        int blockStart_idxCol = blockStart_idx / numCols;   // First column processed by this thread block

        // Calculate coordinates to operate on
        int actual_m = numRows >> (level - 1);    // Column index to read from
        int sample_f = 1 << (level-1);      // Number of row blocks (sample_f == n_rb )
        int temp_block_len = actual_m + filterLen;
        int blockStart_offset = temp_block_len * sample_f * (idxCol - blockStart_idxCol);   // Offset to first element in shared memory
        dataType* buffer_xl = shared_cache_xl + blockStart_offset;
        dataType* buffer_xh = shared_cache_xh + blockStart_offset;
        int n_r = idxRow / actual_m;
        int i = idxRow % actual_m;
        int ir = n_r + sample_f * i;

        // Load data into cache
        buffer_xl[temp_block_len * n_r + i + filterLen - 1] = d_xinl[LINPOS(ir,idxCol,numCols)];
        buffer_xh[temp_block_len * n_r + i + filterLen - 1] = d_xinh[LINPOS(ir,idxCol,numCols)];
        __syncthreads();
        // Load periodic extension
        if( i < filterLen - 1 )
        {
            buffer_xl[temp_block_len * n_r + i] = buffer_xl[temp_block_len * n_r + i + actual_m];
            buffer_xh[temp_block_len * n_r + i] = buffer_xh[temp_block_len * n_r + i + actual_m];
        }
        __syncthreads();

        // Perform convolution on columns
        dataType x0 = 0;
        int idxOffset = temp_block_len * n_r + i;
        for( int k = 0; k < filterLen; k++ )
        {
            x0 += buffer_xl[idxOffset + k] * g0_filter[k] +
                  buffer_xh[idxOffset + k] * g1_filter[k];
        }

        // Save results
        d_xout[LINPOS(ir,idxCol,numCols)] = x0;
    }
}

extern "C" __global__ void mirdwtColF(
    int n,
    int offset,
    const float* d_xinl,     // Low pass
    const float* d_xinh,     // High pass
    int numRows,   // m
    int numCols,   // n
    const float* d_Filter,   // h
    int filterLen,              // lh
    int level,                  // actual_L
    float* d_xout        // Merged output
){
    mirdwtCol_kernel<float>(n, offset, d_xinl, d_xinh, numRows, numCols, d_Filter, filterLen, level, d_xout);
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void mirdwtColD(
    int n,
    int offset,
    const double* d_xinl,     // Low pass
    const double* d_xinh,     // High pass
    int numRows,   // m
    int numCols,   // n
    const double* d_Filter,   // h
    int filterLen,              // lh
    int level,                  // actual_L
    double* d_xout        // Merged output
){
    mirdwtCol_kernel<double>(n, offset, d_xinl, d_xinh, numRows, numCols, d_Filter, filterLen, level, d_xout);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// 1D Undecimated Inverse Wavelet transform along rows (columns in MATLAB)
// Large (512) block size version
////////////////////////////////////////////////////////////////////////////////
template <class dataType> inline __device__ void mirdwtRow512_kernel(
    int n,
    int offset,
    const dataType* d_xinl,     // Low pass
    const dataType* d_xinh,      // High pass
    int numRows,   // m
    int numCols,   // n
    const dataType* d_Filter,   // g
    int filterLen,              // lg
    int level,                  // actual_L
    dataType* d_xout       // Merged output
){
    __shared__ dataType g0_filter[32];
    __shared__ dataType g1_filter[32];
    __shared__ dataType shared_cache_xl[640];       // Data cache for d_xinl
    __shared__ dataType shared_cache_xh[640];       // Data cache for d_xinh

    const unsigned int idx = BLOCK_DIM1D_LARGE * blockIdx.x + threadIdx.x + offset;
    const unsigned int blockStart_idx = idx - threadIdx.x;
    if ((idx - offset) < n)
    {
        // Load filters into shared memory
        int j = threadIdx.x;
        if( j < filterLen )
        {
            dataType val = d_Filter[j];
            // Mirror g0
            g0_filter[filterLen-1-j] = val/2;
            // Modulate g1
            if( j%2 == 0)
                g1_filter[j] = -val/2;
            else
                g1_filter[j] = val/2;
        }

        // Output coordinates
        int idxRow = idx / numCols;
        int idxCol = idx % numCols;
        int blockStart_idxRow = blockStart_idx / numCols;   // First row processed by this thread block

        // Calculate coordinates to operate on
        int actual_n = numCols >> (level - 1);    // Row index to read from
        int sample_f = 1 << (level-1);      // Number of blocks (sample_f == n_cb )
        int temp_block_len = actual_n + filterLen;  // Block lenght corresponding to one decimated row
        int blockStart_offset = temp_block_len * sample_f * (idxRow - blockStart_idxRow);   // Offset to first element in shared memory
        dataType* buffer_xl = shared_cache_xl + blockStart_offset;
        dataType* buffer_xh = shared_cache_xh + blockStart_offset;
        int n_c = idxCol / actual_n;
        int i = idxCol % actual_n;
        int ic = n_c + sample_f * i;
        int c_o_a = (numRows==1 ? numCols*(level-1) : 0);

        // Load data into cache
        buffer_xl[temp_block_len * n_c + i + filterLen - 1] = d_xinl[LINPOS(idxRow,ic,numCols)];
        buffer_xh[temp_block_len * n_c + i + filterLen - 1] = d_xinh[LINPOS(idxRow,c_o_a+ic,numCols)];
        __syncthreads();
        // Load periodic extension
        if( i < filterLen - 1 )
        {
            buffer_xl[temp_block_len * n_c + i] = buffer_xl[temp_block_len * n_c + i + actual_n];
//d_xinl[LINPOS(idxRow,(ic+numCols-filterLen+1) % numCols,numCols)];
            buffer_xh[temp_block_len * n_c + i] = buffer_xh[temp_block_len * n_c + i + actual_n];
//d_xinh[LINPOS(idxRow,(c_o_a+ic+numCols-filterLen+1) % numCols,numCols)];
        }
        __syncthreads();

        // Perform convolution on rows
        dataType x0 = 0;
        int idxOffset = temp_block_len * n_c + i;
        for( int k = 0; k < filterLen; k++ )
        {
            x0 += buffer_xl[idxOffset + k] * g0_filter[k] +
                  buffer_xh[idxOffset + k] * g1_filter[k];
        }

        // Save result
        d_xout[LINPOS(idxRow,ic,numCols)] = x0;
    }
}

extern "C" __global__ void mirdwtRow512F(
    int n,
    int offset,
    const float* d_xinl,     // Low pass
    const float* d_xinh,     // High pass
    int numRows,   // m
    int numCols,   // n
    const float* d_Filter,   // g
    int filterLen,              // lh
    int level,                  // actual_L
    float* d_xout       // Merged output
){
    mirdwtRow512_kernel<float>(n, offset, d_xinl, d_xinh, numRows, numCols, d_Filter, filterLen, level, d_xout);
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void mirdwtRow512D(
    int n,
    int offset,
    const double* d_xinl,     // Low pass
    const double* d_xinh,     // High pass
    int numRows,   // m
    int numCols,   // n
    const double* d_Filter,   // g
    int filterLen,              // lh
    int level,                  // actual_L
    double* d_xout        // Merged output
){
    mirdwtRow512_kernel<double>(n, offset, d_xinl, d_xinh, numRows, numCols, d_Filter, filterLen, level, d_xout);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// 1D Undecimated Inverse Wavelet transform along columns (rows in MATLAB)
// Large (512) block size version
////////////////////////////////////////////////////////////////////////////////
template <class dataType> inline __device__ void mirdwtCol512_kernel(
    int n,
    int offset,
    const dataType* d_xinl,     // Low pass
    const dataType* d_xinh,     // High pass
    int numRows,   // m
    int numCols,   // n
    const dataType* d_Filter,   // g
    int filterLen,              // lg
    int level,                  // actual_L
    dataType* d_xout        // Merged output
){
    __shared__ dataType g0_filter[32];
    __shared__ dataType g1_filter[32];
    __shared__ dataType shared_cache_xl[640];       // Data cache for d_xinl
    __shared__ dataType shared_cache_xh[640];       // Data cache for d_xinh

    const unsigned int idx = BLOCK_DIM1D_LARGE * blockIdx.x + threadIdx.x + offset;
    const unsigned int blockStart_idx = idx - threadIdx.x;
    if ((idx - offset) < n)
    {
        // Load filters into shared memory
        int j = threadIdx.x;
        if( j < filterLen )
        {
            dataType val = d_Filter[j];
            // Mirror g0
            g0_filter[filterLen-1-j] = val/2;
            // Modulate g1
            if( j%2 == 0)
                g1_filter[j] = -val/2;
            else
                g1_filter[j] = val/2;
        }

        // Output coordinates
        int idxCol = idx / numRows;
        int idxRow = idx % numRows;
        int blockStart_idxCol = blockStart_idx / numCols;   // First column processed by this thread block

        // Calculate coordinates to operate on
        int actual_m = numRows >> (level - 1);    // Column index to read from
        int sample_f = 1 << (level-1);      // Number of row blocks (sample_f == n_rb )
        int temp_block_len = actual_m + filterLen;
        int blockStart_offset = temp_block_len * sample_f * (idxCol - blockStart_idxCol);   // Offset to first element in shared memory
        dataType* buffer_xl = shared_cache_xl + blockStart_offset;
        dataType* buffer_xh = shared_cache_xh + blockStart_offset;
        int n_r = idxRow / actual_m;
        int i = idxRow % actual_m;
        int ir = n_r + sample_f * i;

        // Load data into cache
        buffer_xl[temp_block_len * n_r + i + filterLen - 1] = d_xinl[LINPOS(ir,idxCol,numCols)];
        buffer_xh[temp_block_len * n_r + i + filterLen - 1] = d_xinh[LINPOS(ir,idxCol,numCols)];
        __syncthreads();
        // Load periodic extension
        if( i < filterLen - 1 )
        {
            buffer_xl[temp_block_len * n_r + i] = buffer_xl[temp_block_len * n_r + i + actual_m];
            buffer_xh[temp_block_len * n_r + i] = buffer_xh[temp_block_len * n_r + i + actual_m];
        }
        __syncthreads();

        // Perform convolution on columns
        dataType x0 = 0;
        int idxOffset = temp_block_len * n_r + i;
        for( int k = 0; k < filterLen; k++ )
        {
            x0 += buffer_xl[idxOffset + k] * g0_filter[k] +
                  buffer_xh[idxOffset + k] * g1_filter[k];
        }

        // Save results
        d_xout[LINPOS(ir,idxCol,numCols)] = x0;
    }
}

extern "C" __global__ void mirdwtCol512F(
    int n,
    int offset,
    const float* d_xinl,     // Low pass
    const float* d_xinh,     // High pass
    int numRows,   // m
    int numCols,   // n
    const float* d_Filter,   // h
    int filterLen,              // lh
    int level,                  // actual_L
    float* d_xout        // Merged output
){
    mirdwtCol512_kernel<float>(n, offset, d_xinl, d_xinh, numRows, numCols, d_Filter, filterLen, level, d_xout);
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void mirdwtCol512D(
    int n,
    int offset,
    const double* d_xinl,     // Low pass
    const double* d_xinh,     // High pass
    int numRows,   // m
    int numCols,   // n
    const double* d_Filter,   // h
    int filterLen,              // lh
    int level,                  // actual_L
    double* d_xout        // Merged output
){
    mirdwtCol512_kernel<double>(n, offset, d_xinl, d_xinh, numRows, numCols, d_Filter, filterLen, level, d_xout);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Calculate maximum absolute value within 256 samples (source can be either real
// or complex, but result is always real)
////////////////////////////////////////////////////////////////////////////////
template <class dataType, class realType> inline __device__ void reduceMaxAbsVal256_kernel(
    int n,
    int offset,
    realType* d_Dst,
    const dataType* d_Src
){
    __shared__ realType max_buffer[256];
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    const int i = threadIdx.x;
    max_buffer[i] = 0;  // Initialize shared memory
    bool inRange = (idx - offset) < n;

    // Load absolute value from memory to buffer
    if( inRange ) max_buffer[i] = fabs(d_Src[idx]);
    __syncthreads();
                
    if( i < 128 && inRange ) max_buffer[i] = max( max_buffer[i], max_buffer[i+128] );
    __syncthreads();

    if( i < 64 && inRange ) max_buffer[i] = max( max_buffer[i], max_buffer[i+64] );
    __syncthreads();

    if( i < 32 && inRange ) max_buffer[i] = max( max_buffer[i], max_buffer[i+32] );
    __syncthreads();

    if( i < 16 && inRange ) max_buffer[i] = max( max_buffer[i], max_buffer[i+16] );
    __syncthreads();

    if( i < 8 && inRange ) max_buffer[i] = max( max_buffer[i], max_buffer[i+8] );
    __syncthreads();

    if( i < 4 && inRange) max_buffer[i] = max( max_buffer[i], max_buffer[i+4] );
    __syncthreads();

    if( i < 2 && inRange ) max_buffer[i] = max( max_buffer[i], max_buffer[i+2] );
    __syncthreads();

    if( i < 1 )
    {
        d_Dst[idx/256] = max( max_buffer[i], max_buffer[i+1] );
    }
}

extern "C" __global__ void reduceMaxAbsVal256F(
    int n,
    int offset,
    float* d_Dst,
    const float* d_Src
){
    reduceMaxAbsVal256_kernel<float,float>(n, offset, d_Dst, d_Src );
}

extern "C" __global__ void reduceMaxAbsVal256FC(
    int n,
    int offset,
    float* d_Dst,
    const fComplex* d_Src
){
    reduceMaxAbsVal256_kernel<fComplex,float>(n, offset, d_Dst, d_Src );
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void reduceMaxAbsVal256D(
    int n,
    int offset,
    double* d_Dst,
    const double* d_Src
){
    reduceMaxAbsVal256_kernel<double,double>(n, offset, d_Dst, d_Src );
}

extern "C" __global__ void reduceMaxAbsVal256DC(
    int n,
    int offset,
    double* d_Dst,
    const dComplex* d_Src
){
    reduceMaxAbsVal256_kernel<dComplex,double>(n, offset, d_Dst, d_Src );
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Calculate vector norm within 256 samples (source can be either real
// or complex, but result is always real)
////////////////////////////////////////////////////////////////////////////////
template <class dataType, class realType> inline __device__ void reduceNorm256_kernel(
    int n,
    int offset,
    realType* d_Dst,
    const dataType* d_Src,
    realType p
){
    __shared__ realType sum_buffer[256];
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    const int i = threadIdx.x;
    sum_buffer[i] = 0;  // Initialize shared memory
    bool inRange = (idx - offset) < n;

    // Load absolute value from memory to buffer and take pth power
    if( inRange )
        sum_buffer[i] = pow(fabs(d_Src[idx]), p);
    __syncthreads();

    if( i < 128 && inRange ) sum_buffer[i] += sum_buffer[i+128];
    __syncthreads();

    if( i < 64 && inRange ) sum_buffer[i] += sum_buffer[i+64];
    __syncthreads();

    if( i < 32 && inRange ) sum_buffer[i] += sum_buffer[i+32];
    __syncthreads();

    if( i < 16 && inRange ) sum_buffer[i] += sum_buffer[i+16];
    __syncthreads();

    if( i < 8 && inRange ) sum_buffer[i] += sum_buffer[i+8];
    __syncthreads();

    if( i < 4 && inRange) sum_buffer[i] += sum_buffer[i+4];
    __syncthreads();

    if( i < 2 && inRange ) sum_buffer[i] += sum_buffer[i+2];
    __syncthreads();

    if( i < 1 )
    {
        sum_buffer[i] += sum_buffer[i+1];
        d_Dst[idx/256] = sum_buffer[i];
    }
}

extern "C" __global__ void reduceNorm256F(
    int n,
    int offset,
    float* d_Dst,
    const float* d_Src,
    float p
){
    reduceNorm256_kernel<float,float>(n, offset, d_Dst, d_Src, p );
}

extern "C" __global__ void reduceNorm256FC(
    int n,
    int offset,
    float* d_Dst,
    const fComplex* d_Src,
    float p
){
    reduceNorm256_kernel<fComplex,float>(n, offset, d_Dst, d_Src, p );
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void reduceNorm256D(
    int n,
    int offset,
    double* d_Dst,
    const double* d_Src,
    double p
){
    reduceNorm256_kernel<double,double>(n, offset, d_Dst, d_Src, p );
}

extern "C" __global__ void reduceNorm256DC(
    int n,
    int offset,
    double* d_Dst,
    const dComplex* d_Src,
    double p
){
    reduceNorm256_kernel<dComplex,double>(n, offset, d_Dst, d_Src, p );
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Calculate vector sum within 256 samples
////////////////////////////////////////////////////////////////////////////////
template <class dataType> inline __device__ void reduceSum256_kernel(
    int n,
    int offset,
    dataType* d_Dst,
    const dataType* d_Src
){
    __shared__ dataType sum_buffer[256];
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    const int i = threadIdx.x;
    sum_buffer[i] = 0;  // Initialize shared memory
    bool inRange = (idx - offset) < n;

    // Load value from memory to buffer
    if( inRange ) sum_buffer[i] = d_Src[idx];
    __syncthreads();

    if( i < 128 && inRange ) sum_buffer[i] += sum_buffer[i+128];
    __syncthreads();

    if( i < 64 && inRange ) sum_buffer[i] += sum_buffer[i+64];
    __syncthreads();

    if( i < 32 && inRange ) sum_buffer[i] += sum_buffer[i+32];
    __syncthreads();

    if( i < 16 && inRange ) sum_buffer[i] += sum_buffer[i+16];
    __syncthreads();

    if( i < 8 && inRange ) sum_buffer[i] += sum_buffer[i+8];
    __syncthreads();

    if( i < 4 && inRange) sum_buffer[i] += sum_buffer[i+4];
    __syncthreads();

    if( i < 2 && inRange ) sum_buffer[i] += sum_buffer[i+2];
    __syncthreads();

    if( i < 1 )
    {
        sum_buffer[i] += sum_buffer[i+1];
        d_Dst[idx/256] = sum_buffer[i];
    }
}

extern "C" __global__ void reduceSum256F(
    int n,
    int offset,
    float* d_Dst,
    const float* d_Src
){
    reduceSum256_kernel<float>(n, offset, d_Dst, d_Src );
}

//extern "C" __global__ void reduceSum256FC(
//    int n,
//    int offset,
//    fComplex* d_Dst,
//    const fComplex* d_Src
//){
//    reduceSum256_kernel<fComplex>(n, offset, d_Dst, d_Src );
//}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void reduceSum256D(
    int n,
    int offset,
    double* d_Dst,
    const double* d_Src
){
    reduceSum256_kernel<double>(n, offset, d_Dst, d_Src );
}

//extern "C" __global__ void reduceSum256DC(
//    int n,
//    int offset,
//    dComplex* d_Dst,
//    const dComplex* d_Src
//){
//    reduceSum256_kernel<dComplex>(n, offset, d_Dst, d_Src );
//}
#endif

////////////////////////////////////////////////////////////////////////////////
// Data conversion kernels
////////////////////////////////////////////////////////////////////////////////
template <class dataType, class realType> inline __device__ void complexToReal_kernel(
    int n,
    int offset,
    realType* d_DstReal,
    const dataType* d_SrcComplex
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        // Load data to local variable to ensure coalesced memory access
        dataType val = d_SrcComplex[idx];
        // Write data to real vector
        d_DstReal[idx] = val.x;
    }
}


template <class dataType, class realType> inline __device__ void realToComplex_kernel(
    int n,
    int offset,
    dataType* d_DstComplex,
    const realType* d_SrcReal
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        // Load data to local variable
        dataType val;
        val.x = d_SrcReal[idx];
        val.y = 0;
        // Write data to complex vector
        d_DstComplex[idx] = val;
    }
}

extern "C" __global__ void complexToRealF(
    int n,
    int offset,
    float* d_DstReal,
    const fComplex* d_SrcComplex
){
    complexToReal_kernel<fComplex,float>(n, offset, d_DstReal, d_SrcComplex );
}

extern "C" __global__ void realToComplexF(
    int n,
    int offset,
    fComplex* d_DstComplex,
    const float* d_SrcReal
){
    realToComplex_kernel<fComplex,float>(n, offset, d_DstComplex, d_SrcReal );
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void complexToRealD(
    int n,
    int offset,
    double* d_DstReal,
    const dComplex* d_SrcComplex
){
    complexToReal_kernel<dComplex,double>(n, offset, d_DstReal, d_SrcComplex );
}

extern "C" __global__ void realToComplexD(
    int n,
    int offset,
    dComplex* d_DstComplex,
    const double* d_SrcReal
){
    realToComplex_kernel<dComplex,double>(n, offset, d_DstComplex, d_SrcReal );
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Zero padding kernels for complex FFTs
////////////////////////////////////////////////////////////////////////////////
template <class dstType, class srcType> inline __device__ void zeroPadR2C_kernel(
    int n,
    int offset,
    dstType* d_DstComplex,
    int dstDimX,
    int dstDimY,
    int dstDimZ,
    const srcType* d_SrcReal,
    int srcDimX,
    int srcDimY,
    int srcDimZ,
    int shiftX,
    int shiftY
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        int x = idx % dstDimX;
        int y = idx / dstDimX;
        // circularly shift x and y dimensions
        int src_x = (x + shiftX + dstDimX) % dstDimX;
        int src_y = (y + shiftY + dstDimY) % dstDimY;
        for( int z = 0; z < dstDimZ; z++ )
        {
            dstType val = {0};
            if( src_x < srcDimX && src_y < srcDimY && z < srcDimZ )
            {
                val.x = d_SrcReal[z*srcDimX*srcDimY + src_y*srcDimX + src_x];
            }
            d_DstComplex[z*dstDimX*dstDimY + y*dstDimX + x] = val;
        }
    }
}

extern "C" __global__ void zeroPadF2FC(
    int n,
    int offset,
    fComplex* d_DstComplex,
    int dstDimX,
    int dstDimY,
    int dstDimZ,
    const float* d_SrcReal,
    int srcDimX,
    int srcDimY,
    int srcDimZ,
    int shiftX,
    int shiftY
){
    zeroPadR2C_kernel<fComplex,float>(n, offset, d_DstComplex, dstDimX, dstDimY, dstDimZ, d_SrcReal, srcDimX, srcDimY, srcDimZ, shiftX, shiftY );
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void zeroPadF2DC(
    int n,
    int offset,
    dComplex* d_DstComplex,
    int dstDimX,
    int dstDimY,
    int dstDimZ,
    const float* d_SrcReal,
    int srcDimX,
    int srcDimY,
    int srcDimZ,
    int shiftX,
    int shiftY
){
    zeroPadR2C_kernel<dComplex,float>(n, offset, d_DstComplex, dstDimX, dstDimY, dstDimZ, d_SrcReal, srcDimX, srcDimY, srcDimZ, shiftX, shiftY );
}

extern "C" __global__ void zeroPadD2DC(
    int n,
    int offset,
    dComplex* d_DstComplex,
    int dstDimX,
    int dstDimY,
    int dstDimZ,
    const double* d_SrcReal,
    int srcDimX,
    int srcDimY,
    int srcDimZ,
    int shiftX,
    int shiftY
){
    zeroPadR2C_kernel<dComplex,double>(n, offset, d_DstComplex, dstDimX, dstDimY, dstDimZ, d_SrcReal, srcDimX, srcDimY, srcDimZ, shiftX, shiftY );
}

extern "C" __global__ void zeroPadD2FC(
    int n,
    int offset,
    fComplex* d_DstComplex,
    int dstDimX,
    int dstDimY,
    int dstDimZ,
    const double* d_SrcReal,
    int srcDimX,
    int srcDimY,
    int srcDimZ,
    int shiftX,
    int shiftY
){
    zeroPadR2C_kernel<fComplex,double>(n, offset, d_DstComplex, dstDimX, dstDimY, dstDimZ, d_SrcReal, srcDimX, srcDimY, srcDimZ, shiftX, shiftY );
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Zero padding kernels for real FFTs
////////////////////////////////////////////////////////////////////////////////
template <class dstType, class srcType> inline __device__ void zeroPadR2R_kernel(
    int n,
    int offset,
    dstType* d_DstReal,
    int dstDimX,
    int dstDimY,
    int dstDimZ,
    const srcType* d_SrcReal,
    int srcDimX,
    int srcDimY,
    int srcDimZ,
    int shiftX,
    int shiftY
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        int x = idx % dstDimX;
        int y = idx / dstDimX;
        // circularly shift x and y dimensions
        int src_x = (x + shiftX + dstDimX) % dstDimX;
        int src_y = (y + shiftY + dstDimY) % dstDimY;
        for( int z = 0; z < dstDimZ; z++ )
        {
            dstType val = 0;
            if( src_x < srcDimX && src_y < srcDimY && z < srcDimZ )
            {
                val = d_SrcReal[z*srcDimX*srcDimY + src_y*srcDimX + src_x];
            }
            d_DstReal[z*dstDimX*dstDimY + y*dstDimX + x] = val;
        }
    }
}

extern "C" __global__ void zeroPadF2F(
    int n,
    int offset,
    float* d_DstReal,
    int dstDimX,
    int dstDimY,
    int dstDimZ,
    const float* d_SrcReal,
    int srcDimX,
    int srcDimY,
    int srcDimZ,
    int shiftX,
    int shiftY
){
    zeroPadR2R_kernel<float,float>(n, offset, d_DstReal, dstDimX, dstDimY, dstDimZ, d_SrcReal, srcDimX, srcDimY, srcDimZ, shiftX, shiftY );
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void zeroPadF2D(
    int n,
    int offset,
    double* d_DstReal,
    int dstDimX,
    int dstDimY,
    int dstDimZ,
    const float* d_SrcReal,
    int srcDimX,
    int srcDimY,
    int srcDimZ,
    int shiftX,
    int shiftY
){
    zeroPadR2R_kernel<double,float>(n, offset, d_DstReal, dstDimX, dstDimY, dstDimZ, d_SrcReal, srcDimX, srcDimY, srcDimZ, shiftX, shiftY );
}

extern "C" __global__ void zeroPadD2D(
    int n,
    int offset,
    double* d_DstReal,
    int dstDimX,
    int dstDimY,
    int dstDimZ,
    const double* d_SrcReal,
    int srcDimX,
    int srcDimY,
    int srcDimZ,
    int shiftX,
    int shiftY
){
    zeroPadR2R_kernel<double,double>(n, offset, d_DstReal, dstDimX, dstDimY, dstDimZ, d_SrcReal, srcDimX, srcDimY, srcDimZ, shiftX, shiftY );
}

extern "C" __global__ void zeroPadD2F(
    int n,
    int offset,
    float* d_DstReal,
    int dstDimX,
    int dstDimY,
    int dstDimZ,
    const double* d_SrcReal,
    int srcDimX,
    int srcDimY,
    int srcDimZ,
    int shiftX,
    int shiftY
){
    zeroPadR2R_kernel<float,double>(n, offset, d_DstReal, dstDimX, dstDimY, dstDimZ, d_SrcReal, srcDimX, srcDimY, srcDimZ, shiftX, shiftY );
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Filter Myer preparation kernel
////////////////////////////////////////////////////////////////////////////////
//  % Reference MATLAB code (input shear, output dshear)
//  for j = 1:length(num)
//      d=sqrt(sum((shear_f{i}).*conj(shear_f{i}),3));
//      for k = 1:num(j)
//          dshear_f{j}(:,:,k)=shear_f{j}(:,:,k)./d;
//      end
//  end
////////////////////////////////////////////////////////////////////////////////
template <class dataType, class realType> inline __device__ void prepareMyerFilters_kernel(
    int n,
    int offset,
    dataType* d_Filter,
    int size,
    int numDir
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        realType d = 0;
        dataType dshear[32];
        for( int idxDir = 0; idxDir < numDir; idxDir++ )
        {
            dataType shear = d_Filter[idxDir * size + idx];
            dshear[idxDir] = shear;
            d = d + shear.x * shear.x + shear.y * shear.y;
        }
        d = sqrt(d);
        for( int idxDir = 0; idxDir < numDir; idxDir++ )
        {
            d_Filter[idxDir * size + idx] = dshear[idxDir] / d;
        }
    }
}

extern "C" __global__ void prepareMyerFiltersFC(
    int n,
    int offset,
    fComplex* d_Filter,
    int size,
    int numDir
){
    prepareMyerFilters_kernel<fComplex,float>(n, offset, d_Filter, size, numDir );
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void prepareMyerFiltersDC(
    int n,
    int offset,
    dComplex* d_Filter,
    int size,
    int numDir
){
    prepareMyerFilters_kernel<dComplex,double>(n, offset, d_Filter, size, numDir );
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Image conversion kernels
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void convertFTo8U(
    int n,
    int offset,
    unsigned char* d_Dst,
    const float* d_Src
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        float val = d_Src[idx];
        d_Dst[idx] = (val > 255 ? 255 : (val < 0 ? 0 : (unsigned char)val));
    }
}

extern "C" __global__ void convert8UToF(
    int n,
    int offset,
    float* d_Dst,
    const unsigned char* d_Src
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        d_Dst[idx] = d_Src[idx];
    }
}

#if __CUDA_ARCH__ >= 130
extern "C" __global__ void convertDTo8U(
    int n,
    int offset,
    unsigned char* d_Dst,
    const double* d_Src
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        double val = d_Src[idx];
        d_Dst[idx] = (val > 255 ? 255 : (val < 0 ? 0 : (unsigned char)val));
    }
}

extern "C" __global__ void convert8UToD(
    int n,
    int offset,
    double* d_Dst,
    const unsigned char* d_Src
){
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if ((idx - offset) < n)
    {
        d_Dst[idx] = d_Src[idx];
    }
}
#endif
