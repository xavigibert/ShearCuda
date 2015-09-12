/*
 * Author: Xavier Gibert-Serra (gibert@umiacs.umd.edu)
 *
 * Copyright (C) 2013 University of Maryland. All rights reserved.
 *
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
#include "CracksCudaFunctions.h"
#include "MexUtil.h"

// static paramaters
static CracksCudaFunctions func;
static CracksGpuTimes* gt;

static int init = 0;

static GPUmat *gm;


// This function takes a set of Shearlet coefficients and generates low-level
// features for crack segmentation
//
// INPUTS: [0] shearCt -- (cell array with one element per scale, real-valued
//               single shearlet coefficients)
//         [1] shearAngles -- (cell array with one element per scale, containing the
//               angles (in radians, single precision) corresponding to each
//               directional filter
//         [2] w -- vector of weights (one element per scale)
// OUTPUT: [0] features (width x height x 4N) feature vector
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
        CUmodule *drvmod = gmGetModule("cracks_cuda");

        // Get timers
        gt = CracksGpuTimes::getGpuTimesObject();
        func.setTimer(gt);

        // load GPU functions
        if( !func.LoadGpuFunctions( drvmod ) )
            mexErrMsgTxt("Unable to load GPU functions.");

        init = 1;
    }

    if( nrhs != 3)
        mexErrMsgTxt( "Incorrect number of input arguments" );
    if( nlhs != 1 )
        mexErrMsgTxt( "Incorrect number of output arguments" );

    // Check input arguments
    const mxArray* mx_shearCt =  prhs[0];
    if( !mxIsCell( mx_shearCt ) )
        mexErrMsgTxt( "First input argument is not a cell" );
    int numScales = (int)mxGetNumberOfElements( mx_shearCt ) - 1;
    if( numScales < 1 )
        mexErrMsgTxt( "'shearCt' cell is empty" );

    const mxArray* mx_shearAngles = prhs[1];
    if( !mxIsCell( mx_shearAngles ) )
        mexErrMsgTxt( "Second input argument is not a cell");
    if( (int)mxGetNumberOfElements( mx_shearAngles ) != numScales )
        mexErrMsgTxt( "'shearAngles' does not have the correct number of elements" );

    // Determine input data type and dimensions
    mxArray* mx_shearCt_elem = mxGetCell( mx_shearCt, 0 );
    GPUtype gpu_shearCt = gm->gputype.getGPUtype( mx_shearCt_elem );
    gpuTYPE_t real_type;
    real_type = gm->gputype.getType( gpu_shearCt );
    if( real_type != gpuFLOAT )
    {
        mexErrMsgTxt( "Input data should be real-valued of class 'GPUsingle'" );
        return;
    }
    const int* dims = gm->gputype.getSize( gpu_shearCt );
    int imageSize = dims[0] * dims[1];

    // Get weights
    const mxArray* mx_w = prhs[2];
    if( !mxIsDouble(mx_w) || (int)mxGetNumberOfElements(mx_w)!=numScales )
        mexErrMsgTxt( "Invalid vector of weights" );
    const double* w = (const double*)mxGetData(mx_w);

    //  Allocate output array
    int odims[3];
    odims[0] = dims[0];
    odims[1] = dims[1];
    odims[2] = numScales * 4;
    GPUtype gpu_features = gm->gputype.create( real_type, 3, odims, NULL );
    float* d_features_ptr = const_cast<float*>( (const float*)gm->gputype.getGPUptr( gpu_features ));

    // Extract features for each scale
    for( int idxScale = 0; idxScale < numScales; idxScale++ )
    {
        mx_shearCt_elem = mxGetCell( mx_shearCt, idxScale + 1 );
        gpu_shearCt = gm->gputype.getGPUtype( mx_shearCt_elem );
        int ndims = gm->gputype.getNdims( gpu_shearCt );
        const int* dims = gm->gputype.getSize( gpu_shearCt );
        int numDirections = 1;
        if( ndims > 2 )
            numDirections = dims[2];
        if( dims[0] != dims[1] || ndims > 3 )
            mexErrMsgTxt( "Invalid filter dimensions" );
        const float* d_shearCt_ptr = static_cast<const float*>( gm->gputype.getGPUptr( gpu_shearCt ));

        mxArray* mx_shearAngle_elem = mxGetCell( mx_shearAngles, idxScale );
        GPUtype gpu_shearAngle = gm->gputype.getGPUtype( mx_shearAngle_elem );
        const float* d_angles_ptr = static_cast<const float*>( gm->gputype.getGPUptr( gpu_shearAngle ));

        func.llCrackFeatures( d_shearCt_ptr, d_angles_ptr, (float)w[idxScale], numDirections, imageSize, d_features_ptr );

        // Update output pointer to next scale
        d_features_ptr += 4 * imageSize;
    }

    // Assign output
    plhs[0] = gm->gputype.createMxArray( gpu_features );
}
