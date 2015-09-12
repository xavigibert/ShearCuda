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
mxArray* deepCopyCellToGpu(const mxArray* src);
mxArray* copyMatrixToGpu(const mxArray* src);

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

    if( nrhs != 1 && nrhs != 2 )
        mexErrMsgTxt( "Incorrect number of input arguments" );
    if( nlhs != 1 )
        mexErrMsgTxt( "Incorrect number of output arguments" );

    // Check that input data is a cell
    const mxArray* mx_input =  prhs[0];
    if( !mxIsCell( mx_input ) )
        mexErrMsgTxt( "Input argument is not cell" );

    plhs[0] = deepCopyCellToGpu(prhs[0]);
}

// Deep copy functions
mxArray* deepCopyCellToGpu(const mxArray* src)
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
                dstElem = deepCopyCellToGpu(srcElem);
            else
                dstElem = copyMatrixToGpu(srcElem);
            mxSetCell(result,i,dstElem);
        }
    }
    return result;
}

mxArray* copyMatrixToGpu(const mxArray* src)
{
    if( !mxIsDouble(src) && !mxIsSingle(src))
        mexErrMsgTxt( "Matrix is neither real nor double" );
    if( mxIsComplex(src) )
        mexErrMsgTxt( "Matrix is complex" );

    int numDims = (int)mxGetNumberOfDimensions(src);
    const mwSize* dims = mxGetDimensions(src);
    std::vector<int> idims(numDims);
    for( int i = 0; i < numDims; i++)
        idims[i] = dims[i];
    gpuTYPE_t gpu_type = mxIsDouble(src) ? gpuDOUBLE : gpuFLOAT;
    int data_size = mxIsDouble(src) ? sizeof(double) : sizeof(float);
    GPUtype dst = gm->gputype.create( gpu_type, numDims, &idims[0], NULL );
    void* dst_ptr = const_cast<void*>( gm->gputype.getGPUptr( dst ) );
    int dst_size = (int)mxGetNumberOfElements(src) * data_size;

    // Transfer data
    cmexSafeCall( cudaMemcpy(dst_ptr, mxGetPr(src), dst_size, cudaMemcpyHostToDevice));

    return gm->gputype.createMxArray( dst );
}
