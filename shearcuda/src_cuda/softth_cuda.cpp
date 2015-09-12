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

    if( nrhs != 2 )
        mexErrMsgTxt( "Incorrect number of input arguments" );
    if( nlhs > 1 )
        mexErrMsgTxt( "Incorrect number of output arguments" );

    // Grab parameters
    const mxArray* mx_signal  = prhs[0];     // 2D image data
    const mxArray* mx_th      = prhs[1];     // Threshold

    // Fallback to CPU implementation if data type is single or double
    if( mxIsSingle(mx_signal) || mxIsDouble(mx_signal) )
    {
        mexCallMATLAB(nlhs, plhs, nrhs, const_cast<mxArray **>(prhs), "SoftThresh");
        return;
    }

    // Grab data pointer
    GPUtype gpu_signal = gm->gputype.getGPUtype(mx_signal);

    // Check parameter type
    gpuTYPE_t type_signal = gm->gputype.getType( gpu_signal );
    if( type_signal == gpuDOUBLE && !func.supportsDouble )
        mexErrMsgTxt("GPUdouble requires compute capability >= 2.0");
    if( type_signal != gpuDOUBLE && type_signal != gpuFLOAT )
        mexErrMsgTxt("Signal should be GPUsingle or GPUdouble");
    
    int elem_size = ( type_signal == gpuDOUBLE ? sizeof(double) : sizeof(float) );

    // Grab threshold value
    double th = mxGetScalar(mx_th);

    // Apply thresholding
    const int* dims = gm->gputype.getSize( gpu_signal );
    int imageSize = dims[0] * dims[1];
    void* data_ptr = const_cast<void*>( gm->gputype.getGPUptr( gpu_signal ));

    // Apply soft threshold in-place
    func.applySoftThreshold( data_ptr, data_ptr, imageSize, th, type_signal );
}
