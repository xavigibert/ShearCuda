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

#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#ifdef UNIX
#include <stdint.h>
#endif

#include "mex.h"

#include <vector>

// CUDA
#include "cuda.h"
#include "cuda_runtime.h"

#include "GPUmat.hh"
#include "ShearCudaFunctions.h"
#include "MexUtil.h"

// static paramaters
static ShearCudaFunctions func;
static GpuTimes* gt;

static int init = 0;

static GPUmat *gm;

// This function decomposes one scale
void decomposeAndReconstructLevel(mxArray** mx_output, mxArray* L_cell, mxArray* F_cell, gpuTYPE_t dataClass, float* bandEst, double thr);

// Same as MATLAB's nextpow2
int nextpow2(int val);
int nextpow3(int val);

inline int min(int a, int b) {
    return a < b ? a : b;
}

/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function L = ShDecThRec_cuda(F,L,level,dataClass,nsstScalars,sigma,thr)
% Computes shearlet coefficients, thresholds them, and performs reconstruction
% in a single function call (this is done so we don't need to keep the whole)
% transformation in memory
%Input:
%        F          : Windowing filter cell array
%        L          : BandPass data/Other preproccessed data
%        level      : level of decomposition
%        dataClass  : 'GPUsingle' or 'GPUdouble'
%        dstScalars : relative thresholds for each subband
%        sigma      : global threshold scale factor
%        thr        : thresholds scale factor for each level
%Output:
%        L          : Reconstructed BandPass Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // One-time initialization
    if (init == 0)
    {
        // load GPUmat
        gm = gmGetGPUmat();

        // load module
        CUmodule *drvmod = gmGetModule("shear_cuda");

        // Get timers
        gt = GpuTimes::getGpuTimesObject();
        func.setTimer(gt);

        // load GPU functions
        if( !func.LoadGpuFunctions( drvmod ) )
            mexErrMsgTxt("Unable to load GPU functions.");

        init = 1;
    }

    if( nrhs != 7)
        mexErrMsgTxt( "Incorrect number of input arguments" );
    if( nlhs != 1 )
        mexErrMsgTxt( "Incorrect number of output arguments" );

    // Check that input data is a cell
    const mxArray* mx_F =  prhs[0];
    if( !mxIsCell( mx_F ) )
        mexErrMsgTxt( "First input argument is not cell" );

    const mxArray* mx_L =  prhs[1];
    if( !mxIsCell( mx_L ) )
        mexErrMsgTxt( "Second input argument is not cell" );

    int level = (int)mxGetScalar(prhs[2]);

    const mxArray* mx_dataClass = prhs[3];
    if( !mxIsChar(mx_dataClass) )
        mexErrMsgTxt( "Third argument is not a string" );
    char* strDataClass = mxArrayToString(mx_dataClass);
    gpuTYPE_t dataClass = gpuNOTDEF;
    if( strcmp(strDataClass, "GPUsingle")==0 )
        dataClass = gpuFLOAT;
    else if( strcmp( strDataClass, "GPUdouble")==0 )
        dataClass = gpuDOUBLE;
    else {
        mxFree(strDataClass);
        mexErrMsgTxt( "Forth argument should be 'GPUsingle' or 'GPUdouble'" );
    }
    mxFree(strDataClass);

    const mxArray* mx_dstScalars =  prhs[4];
    if( !mxIsStruct(mx_dstScalars) )
        mexErrMsgTxt( "Fifth argument should be a struct" );
    mxArray* mx_bandEst = mxGetField(mx_dstScalars,0,"bandEst");
    if( !mxIsCell(mx_bandEst) )
        mexErrMsgTxt( "dstScalars.bandEst should be cell" );

    double sigma =  mxGetScalar(prhs[5]);

    const mxArray* mx_thr = prhs[6];
    if( !mxIsDouble(mx_thr) || mxGetNumberOfElements(mx_thr) < level )
        mexErrMsgTxt( "Seventh argument should be double precission vector with at least 'level' elements" );
    const double* thr = mxGetPr(mx_thr);

    // Create cell to hold the result
    mwSize cellDims[2];
    cellDims[0] = 1;
    cellDims[1] = level;
    mxArray* result = mxCreateCellArray(2, cellDims);

    // Process each level
    for( int l = 0; l < level; l++ )
    {
        // Declare result (it will be allocated during reconstruction)
        mxArray* bp_cell = NULL;
        // Get image input
        mxArray* L_cell = mxGetCell(mx_L, l);
        // Process all 3 pyramidal cones
        for( int pyrConeIdx = 0; pyrConeIdx < 3; pyrConeIdx++ )
        {
            int subs[2] = {pyrConeIdx,l};

            mwIndex F_idx = mxCalcSingleSubscript(mx_F, 2, subs);
            mxArray* F_cell = mxGetCell(mx_F, F_idx);

            mwIndex bandEst_idx = mxCalcSingleSubscript(mx_bandEst, 2, subs);
            mxArray* bandEst_cell = mxGetCell(mx_bandEst, bandEst_idx);

            if( !mxIsSingle(bandEst_cell) )
                mexErrMsgTxt("Cells in 'dstScalars.bandEst'' should be 'single'");

            decomposeAndReconstructLevel(&bp_cell, L_cell, F_cell, dataClass, (float*)mxGetData(bandEst_cell), sigma * thr[l]);
        }
        mxSetCell(result, l, bp_cell);
    }

    plhs[0] = result;
}

// This function performs shearlet decomposition, thresholding and reconstruction.
// The output is accumulated in matrix mx_output (this matrix is allocated if a NULL pointer is passed)
void decomposeAndReconstructLevel(mxArray** mx_output, mxArray* L_cell, mxArray* F_cell, gpuTYPE_t dataClass, float* bandEst, double thr)
{
    if( !mxIsCell(F_cell) )
        mexErrMsgTxt("F must be a cell of cells");
    if( mxGetNumberOfDimensions(F_cell) != 2 )
        mexErrMsgTxt("Invalid number of dimensions in F cell");

    const int* F_dims = mxGetDimensions(F_cell);
    mxArray* shCoeff = mxCreateCellArray(2,F_dims);

    // Get dimensions and data types for L and F
    GPUtype L_gpu = gm->gputype.getGPUtype(L_cell);
    gpuTYPE_t L_type = gm->gputype.getType(L_gpu);
    if( L_type != gpuFLOAT && L_type != gpuDOUBLE )
        mexErrMsgTxt("Elements of L should be 'GPUsingle' or 'GPUdouble'");
    if( gm->gputype.getNdims(L_gpu) !=3 )
        mexErrMsgTxt("Windows should be 3D");
    const int* L_dims = gm->gputype.getSize(L_gpu);
    const void* L_ptr = gm->gputype.getGPUptr(L_gpu);

    mxArray* F_subcell = mxGetCell(F_cell,0);
    GPUtype F_gpu = gm->gputype.getGPUtype(F_subcell);
    gpuTYPE_t F0_type = gm->gputype.getType(F_gpu);
    if( F0_type != gpuFLOAT && F0_type != gpuDOUBLE )
        mexErrMsgTxt("Elements of F should be 'GPUsingle' or 'GPUdouble'");
    if( gm->gputype.getNdims(F_gpu) !=3 )
        mexErrMsgTxt("Data should be 3D");
    const int* F_subdims = gm->gputype.getSize(F_gpu);
    int filterLen = F_subdims[0];

    if( L_dims[0] == 0 || L_dims[0] != L_dims[1] || L_dims[0] != L_dims[2]
            || F_subdims[0] == 0 || F_subdims[0] != F_subdims[1] || F_subdims[0] != F_subdims[2] )
        mexErrMsgTxt("Invalid cell dimensions");
    int elem_size = (dataClass == gpuFLOAT ? sizeof(float) : sizeof(double));

    // Allocate (or check) results matrix
    void* d_result;
    if( *mx_output == NULL )
    {
        GPUtype output_gpu = gm->gputype.create(dataClass, 3, L_dims, NULL);
        d_result = const_cast<void*>( gm->gputype.getGPUptr(output_gpu) );
        cmexSafeCall(cudaMemset(d_result, 0, L_dims[0]*L_dims[1]*L_dims[2]*elem_size));
        *mx_output = gm->gputype.createMxArray(output_gpu);
    }
    else
    {
        GPUtype output_gpu = gm->gputype.getGPUtype(*mx_output);
        d_result = const_cast<void*>( gm->gputype.getGPUptr(output_gpu) );
    }

    // Calculate dimensions for 3D FFT assuming that  every element of F has same dimensions
    //int fft_len = min( nextpow2(L_dims[0] + filterLen - 1), nextpow3(L_dims[0] + filterLen - 1) );
    int fft_len = nextpow2(L_dims[0] + filterLen - 1);

    // Create FFT plans
    cufftHandle planR2C, planC2R;
    if( dataClass == gpuFLOAT )
    {
        cufftmexSafeCall(cufftPlan3d(&planR2C, fft_len, fft_len, fft_len, CUFFT_R2C));
        cufftmexSafeCall(cufftPlan3d(&planC2R, fft_len, fft_len, fft_len, CUFFT_C2R));
    }
    else
    {
        cufftmexSafeCall(cufftPlan3d(&planR2C, fft_len, fft_len, fft_len, CUFFT_D2Z));
        cufftmexSafeCall(cufftPlan3d(&planC2R, fft_len, fft_len, fft_len, CUFFT_Z2D));
    }
    //cufftSetCompatibilityMode(planR2C, CUFFT_COMPATIBILITY_NATIVE);
    //cufftSetCompatibilityMode(planC2R, CUFFT_COMPATIBILITY_NATIVE);

    // Allocate device memory for the padded inputs and their FFTs
    void* D_padded_L;
    void* D_padded_F;
    void* D_fft_L;
    void* D_fft_F;
    cmexSafeCall(cudaMalloc(&D_padded_L, fft_len * fft_len * fft_len * elem_size));
    cmexSafeCall(cudaMalloc(&D_fft_L, fft_len * fft_len * (fft_len/2+1) * elem_size * 2));

    // Zero-pad matrix L and perform forward FFT
    func.zeroPad(D_padded_L, fft_len, fft_len, fft_len, dataClass, L_ptr, L_dims[0], L_dims[1], L_dims[2], L_type);
    gt->startTimer( GpuTimes::fftFwd );
    if( dataClass == gpuFLOAT )
        cufftmexSafeCall(cufftExecR2C(planR2C, (cufftReal*)D_padded_L, (cufftComplex*)D_fft_L));
    else
        cufftmexSafeCall(cufftExecD2Z(planR2C, (cufftDoubleReal*)D_padded_L, (cufftDoubleComplex*)D_fft_L));
    gt->stopTimer( GpuTimes::fftFwd );
    cmexSafeCall(cudaFree(D_padded_L));

    cmexSafeCall(cudaMalloc(&D_padded_F, fft_len * fft_len * fft_len * elem_size));
    cmexSafeCall(cudaMalloc(&D_fft_F, fft_len * fft_len * (fft_len/2+1) * elem_size * 2));

    // Process all directions
    int numElem = mxGetNumberOfElements(F_cell);

    // Allocate matrix for shearlet coefficients
    GPUtype sh_gpu = gm->gputype.create(dataClass, 3, L_dims, NULL);
    void* sh_ptr = const_cast<void*>( gm->gputype.getGPUptr(sh_gpu) );
    void* ptr_valid;    // Calculate pointer to first valid element in result (starting at filterLen/2,filterLen/2,filterLen/2)
    int offset = filterLen/2;
    int linOffset = offset * fft_len * fft_len + offset * fft_len + offset;
    if( dataClass == gpuFLOAT )
        ptr_valid = (float*)D_padded_F + linOffset;
    else
        ptr_valid = (double*)D_padded_F + linOffset;

    for( int i = 0; i < numElem; i++ )
    {
        F_subcell = mxGetCell(F_cell,i);
        F_gpu = gm->gputype.getGPUtype(F_subcell);
        gpuTYPE_t F_type = gm->gputype.getType(F_gpu);
        if( F_type != gpuFLOAT && F_type != gpuDOUBLE )
            mexErrMsgTxt("Elements of F should be 'GPUsingle' or 'GPUdouble'");
        if( gm->gputype.getNdims(F_gpu) !=3 )
            mexErrMsgTxt("Data should be 3D");
        F_subdims = gm->gputype.getSize(F_gpu);
        if( F_subdims[0] != filterLen || F_subdims[1] != filterLen || F_subdims[2] != filterLen )
            mexErrMsgTxt("Inconsistent F dimensions");
        if( F_type != F0_type )
            mexErrMsgTxt("Inconsistent F type");
        const void* F_ptr = gm->gputype.getGPUptr(F_gpu);

        // Zero-pad matrix F and perform forward FFT
        func.zeroPad(D_padded_F, fft_len, fft_len, fft_len, dataClass, F_ptr, F_subdims[0], F_subdims[1], F_subdims[2], F_type);
        gt->startTimer( GpuTimes::fftFwd );
        if( dataClass == gpuFLOAT )
            cufftmexSafeCall(cufftExecR2C(planR2C, (cufftReal*)D_padded_F, (cufftComplex*)D_fft_F));
        else
            cufftmexSafeCall(cufftExecD2Z(planR2C, (cufftDoubleReal*)D_padded_F, (cufftDoubleComplex*)D_fft_F));
        gt->stopTimer( GpuTimes::fftFwd );
        // Modulate
        func.modulateAndNormalize3D(D_fft_F, D_fft_F, D_fft_L, fft_len, fft_len, fft_len, 1, dataClass);
        // Perform inverse FFT
        gt->startTimer( GpuTimes::fftInv );
        if( dataClass == gpuFLOAT )
            cufftmexSafeCall(cufftExecC2R(planC2R, (cufftComplex*)D_fft_F, (cufftReal*)D_padded_F));
        else
            cufftmexSafeCall(cufftExecZ2D(planC2R, (cufftDoubleComplex*)D_fft_F, (cufftDoubleReal*)D_padded_F));
        gt->stopTimer( GpuTimes::fftInv );

        // Remove zero padding and save results
        func.zeroPad(sh_ptr, L_dims[0], L_dims[1], L_dims[2], dataClass, ptr_valid, fft_len, fft_len, fft_len, dataClass);
        //mxSetCell(shCoeff, i, gm->gputype.createMxArray(sh_gpu));

        // Apply threshold to coefficients
        func.applyHardThreshold(sh_ptr, sh_ptr, L_dims[0] * L_dims[1] * L_dims[2], thr * bandEst[i], dataClass);

        // Accumulate results
        func.addVector(d_result, sh_ptr, L_dims[0] * L_dims[1] * L_dims[2], dataClass);
    }

    // Release temporary memory
    cmexSafeCall(cudaFree(D_fft_L));
    cmexSafeCall(cudaFree(D_fft_F));
    cmexSafeCall(cudaFree(D_padded_F));

    // Release FFT plans
    cufftmexSafeCall(cufftDestroy(planR2C));
    cufftmexSafeCall(cufftDestroy(planC2R));
}

int nextpow2(int val)
{
    val = abs(val);
    int res = 1;
    while( res < val )
        res *= 2;
    return res;
}

int nextpow3(int val)
{
    val = abs(val);
    int res = 1;
    while( res < val )
        res *= 3;
    return res;
}
