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


// CUDA
#include "cuda.h"
#include "cuda_runtime.h"

#include "GPUmat.hh"
#include "ShearCudaFunctions.h"
#include "ShearDictionary.h"

// static paramaters
static ShearCudaFunctions func;
static GpuTimes* gt;

static int init = 0;

static GPUmat *gm;

// Forward declarations

// A trous decomposition
// INPUTS: x, g0, g1, level
// OUTPUT: y (must be an array of length level+1)
void atrousdec(const GPUmat *gm, const ShearCudaFunctions& func, const GPUtype& x, const GPUtype& h0, const GPUtype& h1, int numLevels, GPUtype* y,
               bool alloc_result, GPUtype* tempBuffer);

// Perform convolution
void convolution(const GPUtype* y, const ShearDictionary& shear, GPUtype* d, bool alloc_result = true );

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // One-time initialization
    if (init == 0)
    {
        // Initialize function
        //mexLock();

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

    if( nrhs < 3 && nrhs > 4 )
        mexErrMsgTxt( "Incorrect number of input arguments" );
    if( nlhs > 1 )
        mexErrMsgTxt( "Incorrect number of output arguments" );

    // Grab parameters
    const mxArray* mx_signal = prhs[0];          // Input signal
    const mxArray* mx_pfilt  = prhs[1];          // Filter coefficients
    const mxArray* mx_shear  = prhs[2];          // Struct containing shearlet dictionary and FFT plans
    const mxArray* mx_coeff  = NULL;             // In-place result
    if( nrhs > 3 ) mx_coeff  = prhs[3];
    
    // Fallback to CPU implementation if data type is single or double
    if( mxIsSingle(mx_signal) || mxIsDouble(mx_signal) )
    {
        mexCallMATLAB(nlhs, plhs, nrhs, const_cast<mxArray **>(prhs), "shear_trans");
        return;
    }

    // Check parameter  types
    GPUtype x = gm->gputype.getGPUtype(mx_signal);
    gpuTYPE_t type_signal = gm->gputype.getType(x);
    if( type_signal == gpuDOUBLE && !func.supportsDouble )
        mexErrMsgTxt("GPUdouble requires compute capability >= 2.0");
    if( type_signal != gpuDOUBLE && type_signal != gpuFLOAT )
        mexErrMsgTxt("Signal should be GPUsingle or GPUdouble");

    // Get input signal dimensions
    if( gm->gputype.getNdims(x) != 2 )
        mexErrMsgTxt("Input data should be 2-dimensional");

    const int * signalSize = gm->gputype.getSize(x);

    if( signalSize[0] != 256 && signalSize[0] != 512 && signalSize[0] != 1024 )
        mexErrMsgTxt("Input data size not supported (supported sizes are 256x256, 512x512 and 1024x1024)");
    if( signalSize[0] != signalSize[1] )
        mexErrMsgTxt("Input data should be a square");

    // Check that sherlet dictionary has the same dimensions
    ShearDictionary shear;
    if( !shear.loadFromMx( mx_shear, gm ) )
        mexErrMsgTxt( "Invalid shearlet dictionary" );
    if( shear.filterLen() != signalSize[0] )
        mexErrMsgTxt( "Input data size does not match dictionary size" );
    int numScales = shear.numScales();

    // Get filters for subsampling
    GPUtype h0, h1;
    char str_atrousfilter[32] = {0};
    char str_pfilt[32] = {0};
    mxArray* mx_atrousfilter = mxGetField( mx_shear, 0, "atrousfilter" );
    mxGetString(mx_atrousfilter, str_atrousfilter, 32);
    mxGetString(mx_pfilt, str_pfilt, 32);
    if( strcmp( str_atrousfilter, str_pfilt ) == 0 )
    {
        // Use preloaded filters to reduce memory transfers
        mxArray* mx_h0 = mxGetField( mx_shear, 0, "h0" );
        mxArray* mx_h1 = mxGetField( mx_shear, 0, "h1" );
        h0 = gm->gputype.getGPUtype( mx_h0 );
        h1 = gm->gputype.getGPUtype( mx_h1 );
    }
    else
    {
        // Query filters
        mxArray* atrousfilters[4];
        mxArray* filterParams[2];
        filterParams[0] = const_cast<mxArray*>( mx_pfilt );
        if( type_signal == gpuFLOAT )
            filterParams[1] = mxCreateString("GPUsingle");
        else
            filterParams[1] = mxCreateString("GPUdouble");
        mexCallMATLAB(4, atrousfilters, 2, filterParams, "atrousfilters");
        h0 = gm->gputype.getGPUtype( atrousfilters[0] );
        h1 = gm->gputype.getGPUtype( atrousfilters[1] );
    }

    // Call atrousdec
    std::vector<GPUtype> y( numScales + 1 );

    // Retrieve temporary buffers
    GPUtype tempBuffer[3];
    mxArray* mx_tempBuffer = mxGetField( mx_shear, 0, "tempBuffer");
    for( int j = 0; j < 3; j++ )
    {
        mxArray* mx_elem = mxGetCell( mx_tempBuffer, j );
        tempBuffer[j] = gm->gputype.getGPUtype( mx_elem );
    }

    // Retrieve decomposition buffers
    mxArray* mx_tempDec = mxGetField( mx_shear, 0, "tempDec");
    for( int j = 0; j < numScales + 1; j++)
    {
        mxArray* mx_elem = mxGetCell( mx_tempDec, j );
        y[j] = gm->gputype.getGPUtype( mx_elem );
    }

    atrousdec(gm, func, x, h0, h1, shear.numScales(), &y[0], false, tempBuffer);

    // Return y as a cell (DEBUG code)
    //    plhs[0] = mxCreateCellMatrix( 1, numScales );
    //    for (int j = 0; j < numScales + 1; j++)
    //    {
    //        mxArray* pCellElement = gm->gputype.createMxArray( y[j] );
    //        mxSetCell(plhs[0], j, pCellElement);
    //    }
    //    return;

    // Allocate result
    std::vector<GPUtype> d( numScales + 1 );

    // Grab elements of mx_coeff
    if( mx_coeff != NULL )
    {
        if( mxGetNumberOfElements( mx_coeff ) != numScales + 1 )
            mexErrMsgTxt( "Incorrect number of cell elements");

        for( int j = 0; j < numScales + 1; j++ )
        {
            mxArray* mx_elem = mxGetCell(mx_coeff, j);
            d[j] = gm->gputype.getGPUtype(mx_elem);
        }
    }

    // Perform convolution
    convolution( &y[0], shear, &d[0], mx_coeff == NULL );

    if( nlhs > 0 )
    {
        // Return d as a cell
        plhs[0] = mxCreateCellMatrix( numScales + 1, 1 );
        for (int j = 0; j < numScales + 1; j++)
        {
            mxArray* pCellElement = gm->gputype.createMxArray( d[j] );
            mxSetCell(plhs[0], j, pCellElement);
        }
    }
}

// Perform convolution
void convolution(const GPUtype* y, const ShearDictionary& shear, GPUtype* d, bool alloc_result)
{
    // Calculate dimensions and allocate temporary buffer
    gpuTYPE_t type_signal = shear.dataType();
    gpuTYPE_t type_real = (type_signal == gpuCFLOAT ? gpuFLOAT : gpuDOUBLE );
    int filterLen = shear.filterLen();
    int elem_size = (type_signal == gpuCFLOAT ? sizeof(float) : sizeof(double));
    int temp_size = filterLen * filterLen * elem_size * 2;
    // Fourier transform of the data
    void* d_DataSpectrum;
    cmexSafeCall( cudaMalloc( &d_DataSpectrum, temp_size ));
    // Fourier transform of the product
    void* d_TempSpectrum;
    cmexSafeCall( cudaMalloc( &d_TempSpectrum, temp_size * shear.maxDirections()) );

    // Dimensions for output matrices
    int dims[3];
    dims[0] = filterLen;
    dims[1] = filterLen;

    // Process scales
    for( int scale = 0; scale < shear.numScales(); scale++ )
    {
        int numDir = shear.numDirections(scale);

        // Get pointer to input
        const void* y_ptr = gm->gputype.getGPUptr( y[scale+1] );

        // Convert input to complex (we will use d_TempSpectrum as temporary storage)
        //func.realToComplex( d_TempSpectrum, y_ptr, filterLen * filterLen, type_signal );

        // Allocate output and get pointer
        if( alloc_result )
        {
            dims[2] = numDir;
            d[scale+1] = gm->gputype.create( type_real, 3, dims, NULL );
        }
        void* d_ptr = const_cast<void*>( gm->gputype.getGPUptr( d[scale+1] ) );

        gt->startTimer( GpuTimes::fftFwd );
        if( type_signal == gpuCFLOAT )
            cufftmexSafeCall( cufftExecR2C(shear.fftPlanOne(), (cufftReal *)y_ptr, (cufftComplex *)d_DataSpectrum ));
        else
            cufftmexSafeCall( cufftExecD2Z(shear.fftPlanOne(), (cufftDoubleReal *)y_ptr, (cufftDoubleComplex *)d_DataSpectrum ));
        gt->stopTimer( GpuTimes::fftFwd );

        func.modulateAndNormalizeMany( d_TempSpectrum, d_DataSpectrum, shear.getBuffer(scale), filterLen, filterLen, numDir, type_real );

        gt->startTimer( GpuTimes::fftInv );
        if( type_signal == gpuCFLOAT )
            cufftmexSafeCall( cufftExecC2R(shear.ifftPlanMany(scale), (cufftComplex *)d_TempSpectrum, (cufftReal *)d_ptr ));
        else
            cufftmexSafeCall( cufftExecZ2D(shear.ifftPlanMany(scale), (cufftDoubleComplex *)d_TempSpectrum, (cufftDoubleReal *)d_ptr ));
        gt->stopTimer( GpuTimes::fftInv );

    }

    // Allocate base scale and transfer output
    if( alloc_result )
        d[0] = gm->gputype.create( type_real, 2, dims, NULL );
    const void* y0_ptr = gm->gputype.getGPUptr( y[0] );
    void* d0_ptr = const_cast<void*>( gm->gputype.getGPUptr( d[0] ) );
    cmexSafeCall( cudaMemcpy( d0_ptr, y0_ptr, filterLen * filterLen * elem_size, cudaMemcpyDeviceToDevice ));

    // Free temporary buffers
    cmexSafeCall( cudaFree( d_DataSpectrum ) );
    cmexSafeCall( cudaFree( d_TempSpectrum ) );
}
