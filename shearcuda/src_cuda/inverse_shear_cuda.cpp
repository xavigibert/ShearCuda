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

// A trous reconstruction
// INPUTS: y, g0, g1, level
// OUTPUT: y (must be an array of length level+1)
void atrousrec(const GPUmat *gm, const ShearCudaFunctions& func, const GPUtype* y, const GPUtype& g0, const GPUtype& g1, int numLevels,
               GPUtype& outputImage, bool alloc_result, GPUtype* tempBuffer);

// apply directional shearlet filters to decomposed images for each scale
void applyfilters(const GPUtype* d, const ShearDictionary& shear, GPUtype* y );

// Inputs: (d,pfilt,shear)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 3)
        mexErrMsgTxt("Wrong number of arguments");

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

    if( nrhs != 3 )
        mexErrMsgTxt( "Incorrect number of input arguments" );
    if( nlhs != 1 )
        mexErrMsgTxt( "Incorrect number of output arguments" );

    // Grab parameters
    const mxArray* mx_d      = prhs[0];          // Decomposed input signal
    const mxArray* mx_pfilt  = prhs[1];          // Filter coefficients
    const mxArray* mx_shear  = prhs[2];          // Struct containing shearlet dictionary and FFT plans
    
    // Check that data is a cell
    if( !mxIsCell( mx_d ) )
        mexErrMsgTxt( "Input data must be a cell" );
    int numElements_d = mxGetNumberOfElements( mx_d );
    if( numElements_d < 1 )
        mexErrMsgTxt( "Input cell is empty" );

    // Check type of first element
    const mxArray* mx_d_elem = mxGetCell(mx_d, 0);

    // Fallback to CPU implementation if data type is single or double
    if( mxIsSingle(mx_d_elem) || mxIsDouble(mx_d_elem) )
    {
        mexCallMATLAB(nlhs, plhs, nrhs, const_cast<mxArray **>(prhs), "inverse_shear");
        return;
    }

    // Grab elements of d
    std::vector<GPUtype> d( numElements_d );
    for( int idx = 0; idx < numElements_d; idx++ )
    {
        mx_d_elem = mxGetCell(mx_d, idx);
        d[idx] = gm->gputype.getGPUtype(mx_d_elem);
    }

    // Check parameter types
    gpuTYPE_t type_signal = gm->gputype.getType(d[0]);
    if( type_signal == gpuDOUBLE && !func.supportsDouble )
        mexErrMsgTxt("GPUdouble requires compute capability >= 2.0");
    if( type_signal != gpuDOUBLE && type_signal != gpuFLOAT )
        mexErrMsgTxt("Signal should be GPUsingle or GPUdouble");

    // Get input signal dimensions
    if( gm->gputype.getNdims(d[0]) != 2 )
        mexErrMsgTxt("Input data should be 2-dimensional");

    const int * signalSize = gm->gputype.getSize(d[0]);

    if( signalSize[0] != 256 && signalSize[0] != 512 && signalSize[0] != 1024 )
        mexErrMsgTxt("Input data size not supported (supported sizes are 256x256, 512x512 and 1024x1024)");
    if( signalSize[0] != signalSize[1] )
        mexErrMsgTxt("Input data should be a square");

    // Check that shearlet dictionary has the same dimensions
    ShearDictionary shear;
    if( !shear.loadFromMx( mx_shear, gm ) )
        mexErrMsgTxt( "Invalid shearlet dictionary" );
    if( shear.filterLen() != signalSize[0] )
        mexErrMsgTxt( "Input data size does not match dictionary size" );
    int numScales = shear.numScales();
    if( (int)mxGetNumberOfElements( mx_d ) != numScales + 1 )
	mexErrMsgTxt("Input data dimension do not match dictionary");

    // Get filters for subsampling
    GPUtype g0, g1;
    char str_atrousfilter[32] = {0};
    char str_pfilt[32] = {0};
    mxArray* mx_atrousfilter = mxGetField( mx_shear, 0, "atrousfilter" );
    mxGetString(mx_atrousfilter, str_atrousfilter, 32);
    mxGetString(mx_pfilt, str_pfilt, 32);
    if( strcmp( str_atrousfilter, str_pfilt ) == 0 )
    {
        // Use preloaded filters to reduce memory transfers
        mxArray* mx_g0 = mxGetField( mx_shear, 0, "g0" );
        mxArray* mx_g1 = mxGetField( mx_shear, 0, "g1" );
        g0 = gm->gputype.getGPUtype( mx_g0 );
        g1 = gm->gputype.getGPUtype( mx_g1 );
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
        g0 = gm->gputype.getGPUtype( atrousfilters[2] );
        g1 = gm->gputype.getGPUtype( atrousfilters[3] );
    }

    // apply directional shearlet filters to decomposed images
    // for each scale j
    std::vector<GPUtype> y( numScales + 1 );

    // Retrieve decomposition buffers
    mxArray* mx_tempDec = mxGetField( mx_shear, 0, "tempDec");
    for( int j = 0; j < numScales + 1; j++)
    {
        mxArray* mx_elem = mxGetCell( mx_tempDec, j );
        y[j] = gm->gputype.getGPUtype( mx_elem );
    }

    applyfilters( &d[0], shear, &y[0] );

    // Return y as a cell (DEBUG code)
//    plhs[0] = mxCreateCellMatrix( 1, numScales + 1 );
//    for (int j = 0; j < numScales + 1; j++)
//    {
//        mxArray* pCellElement = gm->gputype.createMxArray( y[j] );
//        mxSetCell(plhs[0], j, pCellElement);
//    }
//    return;


    // Prepare output
    GPUtype x;

    // Retrieve temporary buffers
    GPUtype tempBuffer[3];
    mxArray* mx_tempBuffer = mxGetField( mx_shear, 0, "tempBuffer");
    for( int j = 0; j < 3; j++ )
    {
        mxArray* mx_elem = mxGetCell( mx_tempBuffer, j );
        tempBuffer[j] = gm->gputype.getGPUtype( mx_elem );
    }

    // Call atrousrec
    atrousrec(gm, func, &y[0], g0, g1, shear.numScales(), x, true, tempBuffer);

    // Return matrix x
    plhs[0] = gm->gputype.createMxArray( x );
}

// apply directional shearlet filters to decomposed images for each scale
void applyfilters(const GPUtype* d, const ShearDictionary& shear, GPUtype* y )
{
    // Calculate dimensions and allocate temporary buffer
    gpuTYPE_t type_signal = shear.dataType();
    gpuTYPE_t type_real = (type_signal == gpuCFLOAT ? gpuFLOAT : gpuDOUBLE );
    int filterLen = shear.filterLen();
    int elem_size = (type_signal == gpuCFLOAT ? sizeof(float) : sizeof(double));
    int temp_size = filterLen * filterLen * elem_size * 2;
    void* d_DataSpectrum;
    cmexSafeCall( cudaMalloc( &d_DataSpectrum, temp_size * shear.maxDirections() ));
    void* d_Temp;
    cmexSafeCall( cudaMalloc( &d_Temp, temp_size * shear.maxDirections() ));

    // Allocate output image components
    int dims[2];
    dims[0] = filterLen;
    dims[1] = filterLen;

    for( int scale = 0; scale < shear.numScales(); scale++ )
    {
        int numDir = shear.numDirections(scale);

        // Get pointer to input
        const void* d_ptr = gm->gputype.getGPUptr( d[scale+1] );

        // Get pointer to image component
        void* y_ptr = const_cast<void*>( gm->gputype.getGPUptr( y[scale+1] ));

        // Filter all directions
        if( type_signal == gpuCFLOAT )
        {
            gt->startTimer( GpuTimes::fftFwd );
            cufftmexSafeCall( cufftExecR2C( shear.fftPlanMany(scale), (cufftReal *)d_ptr, (cufftComplex *)d_DataSpectrum ));
            gt->stopTimer( GpuTimes::fftFwd );
            func.modulateConjAndNormalize( d_DataSpectrum, shear.getBuffer(scale), filterLen, filterLen, numDir, type_signal);
            gt->startTimer( GpuTimes::fftInv );
            cufftmexSafeCall( cufftExecC2R( shear.ifftPlanMany(scale), (cufftComplex *)d_DataSpectrum, (cufftReal *)d_Temp ));
            gt->stopTimer( GpuTimes::fftInv );
        }
        else
        {
            gt->startTimer( GpuTimes::fftFwd );
            cufftmexSafeCall( cufftExecD2Z( shear.fftPlanMany(scale), (cufftDoubleReal *)d_ptr, (cufftDoubleComplex *)d_DataSpectrum ));
            gt->stopTimer( GpuTimes::fftFwd );
            func.modulateConjAndNormalize( d_DataSpectrum, shear.getBuffer(scale), filterLen, filterLen, numDir, type_signal);
            gt->startTimer( GpuTimes::fftInv );
            cufftmexSafeCall( cufftExecZ2D( shear.ifftPlanMany(scale), (cufftDoubleComplex *)d_DataSpectrum, (cufftDoubleReal *)d_Temp ));
            gt->stopTimer( GpuTimes::fftInv );
        }
        // Convert data from complex to real (result goes to d_DataSpectrum)
        //func.complexToReal( d_DataSpectrum, d_Temp, filterLen * filterLen * numDir, type_signal );

        // Add all components together
        func.sumVectors( y_ptr, d_Temp, filterLen * filterLen, shear.numDirections(scale), type_real);
    }

    // Allocate base scale and transfer output
    const void* d0_ptr = gm->gputype.getGPUptr( d[0] );
    void* y0_ptr = const_cast<void*>( gm->gputype.getGPUptr( y[0] ) );
    cmexSafeCall( cudaMemcpy( y0_ptr, d0_ptr, filterLen * filterLen * elem_size, cudaMemcpyDeviceToDevice ));

    // Free temporary buffer
    cmexSafeCall( cudaFree( d_Temp ));
    cmexSafeCall( cudaFree( d_DataSpectrum ));
}
