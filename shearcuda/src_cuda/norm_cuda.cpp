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
#include "npps.h"

#include "GPUmat.hh"
#include "ShearCudaFunctions.h"
#include "MexUtil.h"

#include <vector>

// static paramaters
static ShearCudaFunctions func;
static GpuTimes* gt;

static int init = 0;

static GPUmat *gm;

// Forward declarations

// Get L^K norm of a vector
double getNorm( const void* image_ptr, void* scratch_buf, double p, int numElem, gpuTYPE_t type_signal );
// Allocate scratch memory for getNorm() function
void* getNorm_alloc( int numElem, gpuTYPE_t type_signal );
// Free scratch memory used by getNorm() function
void getNorm_free( void* pBuffer );

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

    if( nrhs < 1 || nrhs > 3 )
        mexErrMsgTxt( "Incorrect number of input arguments" );
    if( nlhs > 1 )
        mexErrMsgTxt( "Incorrect number of output arguments" );

    // Grab parameters
    const mxArray* mx_input  = prhs[0];     // Input signal

    // Fallback to CPU implementation if data type is single or double
    if( mxIsSingle(mx_input) || mxIsDouble(mx_input) )
    {
        mexCallMATLAB(nlhs, plhs, nrhs, const_cast<mxArray **>(prhs), "norm");
        return;
    }

    double p = 2.0;    // Default to L2 norm
    int scale_idx = 0;   // Select scale index

    // Check input arguments and get argument values
    if( nrhs > 1)
    {
        const mxArray* mx_p      = prhs[1];     // L_p Norm power
        if( !mxIsDouble(mx_p) )
            mexErrMsgTxt("Second argument should be double");
        p = mxGetScalar( mx_p );
    }

    if( nrhs > 2 )
    {
        const mxArray* mx_scale_idx = prhs[2];
        if( !mxIsDouble(mx_scale_idx) )
            mexErrMsgTxt("Second argument should be double");
        scale_idx = (int)mxGetScalar( mx_scale_idx ) - 1;
    }

    GPUtype gpuInput = gm->gputype.getGPUtype( mx_input );
    const int* dims = gm->gputype.getSize( gpuInput );

    //int numElem = gm->gputype.getNumel( gpuInput );
    int numElem = dims[0] * dims[1];
    gpuTYPE_t type_signal = gm->gputype.getType( gpuInput );
    const void* input_ptr = gm->gputype.getGPUptr( gpuInput );
    if( gm->gputype.getNdims( gpuInput ) > 2 )
    {
        // Select scale to operate on
        input_ptr = (char*)input_ptr + scale_idx * numElem * func.elemSize( type_signal );
    }

    // Calculate vector norm
    void* scratch_buf = getNorm_alloc( numElem, type_signal );
    double norm = getNorm( input_ptr, scratch_buf, p, numElem, type_signal );
    getNorm_free( scratch_buf );

    // Return norm
    plhs[0] = mxCreateDoubleScalar( norm );
}

//int iDivUp(int a, int b){
//    return (a % b != 0) ? (a / b + 1) : (a / b);
//}

// Allocate scratch memory for getNorm() function
// The norm reduction kernel works by reducing the search space
// by 256 at each kernel invocation
void* getNorm_alloc( int numElem, gpuTYPE_t type_signal )
{
    int elem_size_real = (type_signal == gpuDOUBLE || type_signal == gpuCDOUBLE  ?
                     sizeof(double) : sizeof(float));
    int buffer_len = 0;

    // Calculate elements needed for each iteration
    while( numElem > 1 )
    {
        numElem = iDivUp(numElem, 256);
        buffer_len += numElem;
    }

    // Allocate memory
    void* buffer_ptr;
    cmexSafeCall( cudaMalloc( &buffer_ptr, buffer_len * elem_size_real));
    return buffer_ptr;
}

// Free scratch memory used by getNorm() function
void getNorm_free( void* pBuffer )
{
    cmexSafeCall( cudaFree( pBuffer ) );
}

// Get Lp norm of a vector (Note: this function assumes that numElem > 1)
double getNorm( const void* image_ptr, void* scratch_buf, double p, int numElem, gpuTYPE_t type_signal )
{
    int elem_size_real = (type_signal == gpuDOUBLE || type_signal == gpuCDOUBLE  ?
                          sizeof(double) : sizeof(float));

    const void* buffer_src = image_ptr;
    void* buffer_dst = scratch_buf;

    // The first step is calculate the norm of each element and reduce sum
    func.reduceNorm256( buffer_dst, buffer_src, p, numElem, type_signal );

    buffer_src = buffer_dst;
    numElem = iDivUp(numElem, 256);
    buffer_dst = (char*)buffer_dst + numElem * elem_size_real;

    // We know that reduceNorm256 returns always real values
    if( type_signal == gpuCDOUBLE )
        type_signal = gpuDOUBLE;
    if( type_signal == gpuCFLOAT )
        type_signal = gpuFLOAT;

    // Subsequent reduction steps only involve summations
    while( numElem > 1 )
    {
        func.reduceSum256( buffer_dst, buffer_src, numElem, type_signal );
        buffer_src = buffer_dst;
        numElem = iDivUp(numElem, 256);
        buffer_dst = (char*)buffer_dst + numElem * elem_size_real;
    }

    // Transfer result
    if( type_signal == gpuDOUBLE )
    {
        double h_Norm;
        cmexSafeCall( cudaMemcpy( &h_Norm, buffer_src, sizeof(h_Norm), cudaMemcpyDeviceToHost ));
        return pow( h_Norm, 1.0 / p );
    }
    else
    {
        float h_Norm;
        cmexSafeCall( cudaMemcpy( &h_Norm, buffer_src, sizeof(h_Norm), cudaMemcpyDeviceToHost ));
        return pow( h_Norm, 1.0 / p );
    }
}
