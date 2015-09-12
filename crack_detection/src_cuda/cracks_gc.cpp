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
#include "ShearCudaFunctions.h"
#include "MexUtil.h"
#include <nppi.h>

// static paramaters
static CracksCudaFunctions func;
static CracksGpuTimes* gt;

static int init = 0;

static GPUmat *gm;

// Forward declarations

// Run graphcuts with an 8 neighborhood. Inputs are in GPU memory, results to be returned in CPU memory
bool graphCut8(int * d_terminals, int * d_left_transposed, int * d_right_transposed, int * d_top,
               int * d_topleft, int * d_topright, int * d_bottom, int * d_bottomleft, int * d_bottomright,
               int width, int height, unsigned char * pLabels);

// This function takes a graph structure created by gc_crack_affinities and an optional
// bias and runs the graph cut algorithm
//
// INPUTS: [0] gc -- struct containing parameters for graph-cut function
//         [1] lambda -- scale factor on the terminal capacities (optional)
//         [2] bias -- bias on the terminal capacities (optional)
// OUTPUT: [0] labels -- UINT8 matrix containing the estimated labels
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

    if( nrhs != 1 && nrhs != 3)
        mexErrMsgTxt( "Incorrect number of input arguments" );
    if( nlhs != 1 )
        mexErrMsgTxt( "Incorrect number of output arguments" );

    // Check input arguments
    const mxArray* mx_gc =  prhs[0];
    if( !mxIsStruct(mx_gc) )
        mexErrMsgTxt( "First argument must be a structure" );

    // Get all fields
    GPUtype gpu_graph_terminals = gm->gputype.getGPUtype( mxGetField( mx_gc, 0, "graph_terminals") );
    GPUtype gpu_graph_left = gm->gputype.getGPUtype( mxGetField( mx_gc, 0, "graph_left") );
    GPUtype gpu_graph_right = gm->gputype.getGPUtype( mxGetField( mx_gc, 0, "graph_right") );
    GPUtype gpu_graph_top = gm->gputype.getGPUtype( mxGetField( mx_gc, 0, "graph_top") );
    GPUtype gpu_graph_topleft = gm->gputype.getGPUtype( mxGetField( mx_gc, 0, "graph_topleft") );
    GPUtype gpu_graph_topright = gm->gputype.getGPUtype( mxGetField( mx_gc, 0, "graph_topright") );
    GPUtype gpu_graph_bottom = gm->gputype.getGPUtype( mxGetField( mx_gc, 0, "graph_bottom") );
    GPUtype gpu_graph_bottomleft = gm->gputype.getGPUtype( mxGetField( mx_gc, 0, "graph_bottomleft") );
    GPUtype gpu_graph_bottomright = gm->gputype.getGPUtype( mxGetField( mx_gc, 0, "graph_bottomright") );
    int32_t* ptr_graph_terminals = (int32_t*)const_cast<void*>( gm->gputype.getGPUptr(gpu_graph_terminals) );
    int32_t* ptr_graph_left = (int32_t*)const_cast<void*>( gm->gputype.getGPUptr(gpu_graph_left) );
    int32_t* ptr_graph_right = (int32_t*)const_cast<void*>( gm->gputype.getGPUptr(gpu_graph_right) );
    int32_t* ptr_graph_top = (int32_t*)const_cast<void*>( gm->gputype.getGPUptr(gpu_graph_top) );
    int32_t* ptr_graph_topleft = (int32_t*)const_cast<void*>( gm->gputype.getGPUptr(gpu_graph_topleft) );
    int32_t* ptr_graph_topright = (int32_t*)const_cast<void*>( gm->gputype.getGPUptr(gpu_graph_topright) );
    int32_t* ptr_graph_bottom = (int32_t*)const_cast<void*>( gm->gputype.getGPUptr(gpu_graph_bottom) );
    int32_t* ptr_graph_bottomleft = (int32_t*)const_cast<void*>( gm->gputype.getGPUptr(gpu_graph_bottomleft) );
    int32_t* ptr_graph_bottomright = (int32_t*)const_cast<void*>( gm->gputype.getGPUptr(gpu_graph_bottomright) );

    const int* dims = gm->gputype.getSize( gpu_graph_terminals );
    int width = dims[0];
    int height = dims[1];

    // Recalculate terminals if necessary
    if( nrhs >= 3 )
    {
        float lambda = (float)mxGetScalar(prhs[1]);
        float bias = (float)mxGetScalar(prhs[2]);
        GPUtype gpu_sum_fv = gm->gputype.getGPUtype( mxGetField( mx_gc, 0, "sum_fv") );
        float* ptr_sum_fv = (float*)const_cast<void*>( gm->gputype.getGPUptr(gpu_sum_fv) );
        func.graphTerminals( ptr_sum_fv, ptr_graph_terminals, width, height, lambda, bias );
    }

    // Allocate result
    plhs[0] = mxCreateNumericArray(2, dims, mxUINT8_CLASS, mxREAL);
    uint8_t* pLabels = (uint8_t*)mxGetData(plhs[0]);

    // Call graph-cuts
    graphCut8( ptr_graph_terminals, ptr_graph_left, ptr_graph_right, ptr_graph_top, ptr_graph_topleft,
               ptr_graph_topright, ptr_graph_bottom, ptr_graph_bottomleft, ptr_graph_bottomright,
               width, height, pLabels);
}

// Run graphcuts with an 8 neighborhood. Inputs are in GPU memory, results to be returned in CPU memory
bool graphCut8(int * d_terminals, int * d_left_transposed, int * d_right_transposed, int * d_top,
               int * d_topleft, int * d_topright, int * d_bottom, int * d_bottomleft, int * d_bottomright,
               int width, int height, unsigned char * pLabels)
{
    bool bRet = true;       // Return value

    Npp8u* d_labels;
    size_t pitch, transposed_pitch, labels_pitch;
    NppiSize size;
    size.width = width;
    size.height = height;

    // Allocate Graph
    cmexSafeCall( cudaMallocPitch(&d_labels, &labels_pitch, width*sizeof(Npp8u), height));
    pitch = width*sizeof(Npp32s);
    transposed_pitch = height*sizeof(Npp32s);

    int scratch_gc_size;
    nppiGraphcut8GetSize(size, &scratch_gc_size);

    Npp8u* d_scratch_mem;
    NppiGraphcutState* pState;
    cmexSafeCall( cudaMalloc(&d_scratch_mem, scratch_gc_size) );

    // Initialize graph cut
    if( nppiGraphcut8InitAlloc(size, &pState, d_scratch_mem) != NPP_SUCCESS )
        bRet = false;

    if( nppiGraphcut8_32s8u(d_terminals, d_left_transposed, d_right_transposed, d_top, d_topleft, d_topright, d_bottom, d_bottomleft, d_bottomright,
        (int)pitch, (int)transposed_pitch, size, d_labels, (int)labels_pitch, pState) != NPP_SUCCESS )
        bRet = false;

    // Transfer results
    cmexSafeCall( cudaMemcpy2D(pLabels, width * sizeof(Npp8u), d_labels, labels_pitch, width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost) );

    cmexSafeCall( cudaFree(d_labels) );
    cmexSafeCall( cudaFree(d_scratch_mem) );
    nppiGraphcutFree( pState );

    return bRet;
}

