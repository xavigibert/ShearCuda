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

// A trous reconstruction
// INPUTS: y, g0, g1, level
// OUTPUT: y (must be an array of length level+1)
void atrousrec(const GPUmat *gm, const ShearCudaFunctions& func, const GPUtype* y, const GPUtype& g0, const GPUtype& g1, int numLevels,
               GPUtype& outputImage, bool alloc_result, GPUtype* tempBuffer);

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

    if( nrhs < 2 || nrhs > 3 )
        mexErrMsgTxt( "Incorrect number of input arguments" );
    if( nlhs != 1 )
        mexErrMsgTxt( "Incorrect number of output arguments" );

    // Grab parameters
    const mxArray* mx_signal = prhs[0];          // Input signal
    const mxArray* mx_coeff  = prhs[1];          // weights across frequency bands
    const mxArray* mx_shear  = NULL;             // Shearlet data structure containing temp buffers
    if( nrhs > 2 ) mx_shear = prhs[2];
    
    // Fallback to CPU implementation if data type is single or double
    if( mxIsSingle(mx_signal) || mxIsDouble(mx_signal) )
    {
        mexCallMATLAB(nlhs, plhs, nrhs, const_cast<mxArray **>(prhs), "re_weight_band");
        return;
    }

    // Check parameter types
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

    // Get pointer to coeff list
    if( !mxIsDouble(mx_coeff) )
        mexErrMsgTxt( "Weights should be double" );
    int numBands = mxGetNumberOfElements( mx_coeff );
    if( numBands <= 1 )
        mexErrMsgTxt( "There should be at least one band" );
    double* pBands = mxGetPr( mx_coeff );

    // Get filters for subsampling
    mxArray* atrousfilters[4];
    mxArray* filterParams[2];
    filterParams[0] = mxCreateString("maxflat");
    if( type_signal == gpuFLOAT )
        filterParams[1] = mxCreateString("GPUsingle");
    else
        filterParams[1] = mxCreateString("GPUdouble");
    mexCallMATLAB(4, atrousfilters, 2, filterParams, "atrousfilters");
    GPUtype h0 = gm->gputype.getGPUtype( atrousfilters[0] );
    GPUtype h1 = gm->gputype.getGPUtype( atrousfilters[1] );
    GPUtype g0 = gm->gputype.getGPUtype( atrousfilters[2] );
    GPUtype g1 = gm->gputype.getGPUtype( atrousfilters[3] );

    // Call atrousdec
    std::vector<GPUtype> y( numBands );
    if( mx_shear != NULL )
    {
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
        for( int j = 0; j < numBands; j++)
        {
            mxArray* mx_elem = mxGetCell( mx_tempDec, j );
            y[j] = gm->gputype.getGPUtype( mx_elem );
        }
        atrousdec(gm, func, x, h0, h1, numBands - 1, &y[0], false, tempBuffer);
    }
    else
        atrousdec(gm, func, x, h0, h1, numBands - 1, &y[0], true, NULL);

    // Apply weights
    for( int band = 0; band < numBands; band++ )
    {
        void* y_ptr = const_cast<void*>( gm->gputype.getGPUptr( y[band] ) );
        func.mulMatrixByScalar( y_ptr, y_ptr, pBands[band], signalSize[0] * signalSize[1], type_signal );
    }

    // Prepare output
    GPUtype result;

    // Call atrousrec
    atrousrec(gm, func, &y[0], g0, g1, numBands - 1, result, true, NULL);

    // Return matrix x
    plhs[0] = gm->gputype.createMxArray( result );
}
