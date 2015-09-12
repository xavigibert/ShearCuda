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

// Get maximum absolute value of an image
double getMaxAbsVal( const void* image_ptr, void* scratch_buf, int numElem, gpuTYPE_t type_signal );
// Allocate scratch memory for getMaxAbsVal() function
void* getMaxAbsVal_alloc( int numElem, gpuTYPE_t type_signal );
// Free scratch memory used by getMaxAbsVal() function
void getMaxAbsVal_free( void* pBuffer );

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

    if( nrhs != 6 )
        mexErrMsgTxt( "Incorrect number of input arguments" );
    if( nlhs > 1 )
        mexErrMsgTxt( "Incorrect number of output arguments" );

    // Grab parameters
    const mxArray* mx_coeff  = prhs[0];     // transform coefficients (wavelets or shearlets)
    const mxArray* mx_lambda = prhs[1];     // threshold parameter (any positive number)
    const mxArray* mx_option = prhs[2];     // thresholding for wavelets, option = 2 : thesholding for shearlets
    const mxArray* mx_E      = prhs[3];     // l^2 norm of shearlets (see com_norm.m)
    const mxArray* mx_sc     = prhs[4];     // row vector [s(0),s(1),...,s(L)] where each entry s(j) is a
                                            // thresholding parameter for each scale j.
    const mxArray* mx_opt    = prhs[5];     // opt == 1 : compute max magnitude of coeff otherwise return 0 for output maxi.

    // Fallback to CPU implementation if data type is single or double
    if( mxIsSingle(mx_coeff) || mxIsDouble(mx_coeff) )
    {
        mexCallMATLAB(nlhs, plhs, nrhs, const_cast<mxArray **>(prhs), "thresh");
        return;
    }
    else if( mxIsCell(mx_coeff) && (mxIsSingle(mxGetCell(mx_coeff, 0)) && mxIsDouble(mxGetCell(mx_coeff, 0))))
    {
        mexCallMATLAB(nlhs, plhs, nrhs, const_cast<mxArray **>(prhs), "thresh");
        return;
    }

    // Check input arguments
    if( !mxIsDouble(mx_E) || !mxIsDouble(mx_sc) )
        mexErrMsgTxt("Forth and fifth arguments should be double");

    // Resulting maximum value                                                                                                               
    double global_max_val = 0.0;

    // Get argument values
    double lambda = mxGetScalar( mx_lambda );
    int option = (int)mxGetScalar( mx_option );
    int opt = (int)mxGetScalar( mx_opt );

    int numScales = mxGetM( mx_E );
    int maxDirections = mxGetN( mx_E );
    double* E = mxGetPr( mx_E );
    double* sc = mxGetPr( mx_sc );

    // Grab elements of mx_coeff
    std::vector<GPUtype> Ct;
    if( option == 1)
    {
        if( mxGetNumberOfElements( mx_coeff ) < 2 )
            mexErrMsgTxt( "Number of cell elements of 'Ct' should be at least 2");
        
        // For wavelents apply thresholding to second element only
        Ct.resize( 1 );
        mxArray* mx_elem = mxGetCell(mx_coeff, 1);
        Ct[0] = gm->gputype.getGPUtype(mx_elem);
    }
    else
    {
        if( option == 2 && mxGetNumberOfElements( mx_sc ) != numScales )
            mexErrMsgTxt( "Number of elmenents of 'sc'' does not match number of rows of 'E'" );

        // Get number of elements
        int numElements_Ct = mxGetNumberOfElements( mx_coeff );
        if( numElements_Ct != numScales )
            mexErrMsgTxt( "Number of cell elements of 'Ct' does not match number of rows of 'E'");

        Ct.resize( numElements_Ct );
        for( int idx = 0; idx < numElements_Ct; idx++ )
        {
            mxArray* mx_elem = mxGetCell(mx_coeff, idx);
            Ct[idx] = gm->gputype.getGPUtype(mx_elem);
        }
    }

    // Check parameter types
    gpuTYPE_t type_signal = gm->gputype.getType( Ct[0] );
    if( type_signal == gpuDOUBLE && !func.supportsDouble )
        mexErrMsgTxt("GPUdouble requires compute capability >= 2.0");
    if( type_signal != gpuDOUBLE && type_signal != gpuFLOAT )
        mexErrMsgTxt("Signal should be GPUsingle or GPUdouble");
    
    void* temp_buffer;
    if( opt == 1 )
    {
        // Allocate temporary buffers
        const int* dims = gm->gputype.getSize( Ct[0] );
        int imageSize = dims[0] * dims[1];
        temp_buffer = getMaxAbsVal_alloc( imageSize, type_signal );
    }

    // Apply thresholding
    for( int idxScale = 0; idxScale < Ct.size(); idxScale++ )
    {
        int numDirections = 1;
        const int* dims = gm->gputype.getSize( Ct[idxScale] );
        int imageSize = dims[0] * dims[1];
        if( gm->gputype.getNdims( Ct[idxScale] ) > 2 )
            numDirections = dims[2];
        void* scale_ptr = const_cast<void*>( gm->gputype.getGPUptr( Ct[idxScale] ));
        type_signal = gm->gputype.getType( Ct[idxScale] );
        int elem_size = func.elemSize( type_signal );

        for( int idxDirection = 0; idxDirection < numDirections; idxDirection++ )
        {
            // Apply hard threshold in-place
            double th, valE;
            if( option == 1 )
            {   // Wavelets
                valE = 1.0;
                th = lambda;
            }
            else
            {   // Shearlets
                valE = E[ idxScale + idxDirection * numScales ];
                th = sc[idxScale] * lambda * valE;
            }
            void* image_ptr = (char*)scale_ptr + idxDirection * imageSize * elem_size;
            func.applyHardThreshold( image_ptr, image_ptr, imageSize, th, type_signal );
            if( opt == 1 )
            {
                // Find maximum value
                double maxVal = getMaxAbsVal( image_ptr, temp_buffer, imageSize, type_signal ) / valE;
                if( maxVal > global_max_val )
                    global_max_val = maxVal;
            }
        }
    }
    
    if( opt == 1 )
    {
        // Free temporary buffers
        getMaxAbsVal_free( temp_buffer );
    }

    // Return maximum value (if computed)
    plhs[0] = mxCreateDoubleScalar( global_max_val );
}

//int iDivUp(int a, int b){
//    return (a % b != 0) ? (a / b + 1) : (a / b);
//}

// Allocate scratch memory for getMaxAbsVal() function
// The maxAbsVal reduction kernel works by reducing the search space
// by 256 at each kernel invocation
void* getMaxAbsVal_alloc( int numElem, gpuTYPE_t type_signal )
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

// Free scratch memory used by getMaxAbsVal() function
void getMaxAbsVal_free( void* pBuffer )
{
    cmexSafeCall( cudaFree( pBuffer ) );
}

// Get maximum absolute value of an image (Note: this function assumes that numElem > 1)
double getMaxAbsVal( const void* image_ptr, void* scratch_buf, int numElem, gpuTYPE_t type_signal )
{
    int elem_size_real = (type_signal == gpuDOUBLE || type_signal == gpuCDOUBLE  ?
                          sizeof(double) : sizeof(float));

    const void* buffer_src = image_ptr;
    void* buffer_dst = scratch_buf;

    while( numElem > 1 )
    {
        func.reduceMaxAbsVal256( buffer_dst, buffer_src, numElem, type_signal );
        buffer_src = buffer_dst;
        numElem = iDivUp(numElem, 256);
        buffer_dst = (char*)buffer_dst + numElem * elem_size_real;

        // We know that reduceMaxAbsVal512 returns always real values
        if( type_signal == gpuCDOUBLE )
            type_signal = gpuDOUBLE;
        if( type_signal == gpuCFLOAT )
            type_signal = gpuFLOAT;
    }

    // Transfer result
    if( type_signal == gpuDOUBLE )
    {
        double h_MaxAbs;
        cmexSafeCall( cudaMemcpy( &h_MaxAbs, buffer_src, sizeof(h_MaxAbs), cudaMemcpyDeviceToHost ));
        return h_MaxAbs;
    }
    else
    {
        float h_MaxAbs;
        cmexSafeCall( cudaMemcpy( &h_MaxAbs, buffer_src, sizeof(h_MaxAbs), cudaMemcpyDeviceToHost ));
        return h_MaxAbs;
    }
}
