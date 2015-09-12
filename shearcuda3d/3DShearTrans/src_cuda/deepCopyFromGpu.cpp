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
#include "MexUtil.h"

static int init = 0;

static GPUmat *gm;

// Deep copy functions
mxArray* deepCopyCellFromGpu(const mxArray* src);
mxArray* copyMatrixFromGpu(const mxArray* src);

// This function makes a deep copy of a cell object (possible containing subcells)
// from CPU memory to GPU, while maintaining the same structure and data types
//
// INPUTS: [0] x -- arbitrary cell containing single or double matrices
// OUTPUT: [0] y -- cell with the same structure as x containing GPUsingle or
//                  GPUdouble matrices
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // One-time initialization
    if (init == 0)
    {
        // load GPUmat
        gm = gmGetGPUmat();

        init = 1;
    }

    if( nrhs != 1)
        mexErrMsgTxt( "Incorrect number of input arguments" );
    if( nlhs != 1 )
        mexErrMsgTxt( "Incorrect number of output arguments" );

    // Check that input data is a cell
    const mxArray* mx_input =  prhs[0];
    if( !mxIsCell( mx_input ) )
        mexErrMsgTxt( "Input argument is not cell" );

    plhs[0] = deepCopyCellFromGpu(prhs[0]);
}

// Deep copy functions
mxArray* deepCopyCellFromGpu(const mxArray* src)
{
    if( mxGetNumberOfDimensions(src) != 2 )
        mexErrMsgTxt( "Unsupported number of cell dimensions" );

    const mwSize* cellDims = mxGetDimensions(src);
    mxArray* result = mxCreateCellArray(mxGetNumberOfDimensions(src), cellDims);
    int numElem = (int)mxGetNumberOfElements( src );
    for( int i = 0; i < numElem; i++ ) {
        mxArray* srcElem = mxGetCell(src, i);
        if( srcElem != NULL )
        {
            mxArray* dstElem;
            if( mxIsCell(srcElem) )
                dstElem = deepCopyCellFromGpu(srcElem);
            else
                dstElem = copyMatrixFromGpu(srcElem);
            mxSetCell(result,i,dstElem);
        }
    }
    return result;
}

mxArray* copyMatrixFromGpu(const mxArray* mx_src)
{
    GPUtype src = gm->gputype.getGPUtype(mx_src);
    gpuTYPE_t gpu_type = gm->gputype.getType(src);
    if( gpu_type != gpuDOUBLE && gpu_type != gpuFLOAT )
        mexErrMsgTxt("Input should be GPUsingle or GPUdouble");

    int numDims = gm->gputype.getNdims(src);
    const int* idims = gm->gputype.getSize(src);
    std::vector<mwSize> dims(numDims);
    for( int i = 0; i < numDims; i++)
        dims[i] = idims[i];
    mxClassID mx_type = (gpu_type == gpuDOUBLE ? mxDOUBLE_CLASS : mxSINGLE_CLASS);
    mxArray* dst = mxCreateNumericArray(numDims, &dims[0], mx_type, mxREAL);
    int data_size = (gpu_type == gpuDOUBLE ? sizeof(double) : sizeof(float));
    const void* src_ptr = gm->gputype.getGPUptr( src );
    int dst_size = (int)mxGetNumberOfElements(dst) * data_size;

    // Transfer data
    cmexSafeCall( cudaMemcpy(mxGetPr(dst), src_ptr, dst_size, cudaMemcpyDeviceToHost));

    return dst;
}
