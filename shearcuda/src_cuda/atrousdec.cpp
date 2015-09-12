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
#include "MexUtil.h"
#include <vector>

// A trous decomposition
// INPUTS: x, h0, h1, level
// OUTPUT: y (must be an array of length level+1)
void atrousdec(const GPUmat *gm, const ShearCudaFunctions& func, const GPUtype& x, const GPUtype& h0, const GPUtype& h1, int numLevels, GPUtype* y,
               bool alloc_result, GPUtype* tempBuffer)
{
    // Get data size and dimensions
    const int* dims = gm->gputype.getSize( x );
    int inputRows = dims[0];
    int inputCols = dims[1];
    gpuTYPE_t type_signal = gm->gputype.getType( x );
    void* d_InputImage = const_cast<void*>( gm->gputype.getGPUptr( x ) );

    // Calculate dimensions and allocate temporary buffer
    int temp_size = (2*inputRows) * (2*inputCols)
            * (type_signal == gpuFLOAT ? sizeof(float) : sizeof(double));
    void* d_Temp;
    void* d_TempSubsampled;
    if( tempBuffer == NULL )
    {
        cmexSafeCall( cudaMalloc( &d_Temp, temp_size ));
        cmexSafeCall( cudaMalloc( &d_TempSubsampled, temp_size ));
    }
    else
    {
        d_Temp = const_cast<void*>( gm->gputype.getGPUptr( tempBuffer[0] ));
        d_TempSubsampled = const_cast<void*>( gm->gputype.getGPUptr( tempBuffer[1] ));
    }

    // Allocate results pointers
    std::vector<void*> imgComp( numLevels + 1);
    for( int idxScale = 0; idxScale < numLevels + 1; idxScale++ )
    {
        if( alloc_result )
            y[idxScale] = gm->gputype.create( type_signal, 2, dims, NULL);
        imgComp[idxScale] = const_cast<void*>( gm->gputype.getGPUptr( y[idxScale] ) );
    }

    // Get length of analysis filters
    const int len_h0 = gm->gputype.getSize( h0 )[0];
    const int len_h1 = gm->gputype.getSize( h1 )[0];
    void* d_h0 = const_cast<void*>( gm->gputype.getGPUptr( h0 ) );
    void* d_h1 = const_cast<void*>( gm->gputype.getGPUptr( h1 ) );

    //// % First level
    //// shift = [1, 1]; % delay compensation
    //// y1 = conv2(symext(x,h1,shift),h1,'valid');
    int paddedRows = inputRows + len_h1 - 1;
    int paddedCols = inputCols + len_h1 - 1;
    void* d_y1 = imgComp[ numLevels ];
    func.symExt( d_Temp, paddedRows, paddedCols, d_InputImage, inputRows, inputCols, len_h1/2, len_h1/2, type_signal );
    func.atrousConvolutionDevice( d_TempSubsampled, d_y1, inputRows, inputCols, d_Temp, paddedRows, paddedCols, 1, d_h1, len_h1, type_signal );

    //// y0 = conv2(symext(x,h0,shift),h0,'valid');
    paddedRows = inputRows + len_h0 - 1;
    paddedCols = inputCols + len_h0 - 1;
    func.symExt( d_Temp, paddedRows, paddedCols, d_InputImage, inputRows, inputCols, len_h0/2, len_h0/2, type_signal );
    void* d_y0 = imgComp[ 0 ];
    func.atrousConvolutionDevice( d_TempSubsampled, d_y0, inputRows, inputCols, d_Temp, paddedRows, paddedCols, 1, d_h0, len_h0, type_signal );

    for( int i = 0; i < numLevels-1; i++ )
    {
        int shift = 1 - (1<<i);
        int m = 2<<i;

        void* d_x = d_y0;

        //// y1 = cuda_atrousc(symext(x,upsample2df(h1,i),shift),h1,I2 * L,'h1');
        //// y{Nlevels-i+1} = y1;
        paddedRows = inputRows + m * len_h1 - 1;
        paddedCols = inputCols + m * len_h1 - 1;
        int offsetRows = (m * len_h1)/2 - shift;
        int offsetCols = (m * len_h1)/2 - shift;
        d_y1 = imgComp[ numLevels - 1 - i ];
        func.symExt( d_Temp, paddedRows, paddedCols, d_x, inputRows, inputCols, offsetRows, offsetCols, type_signal );
        func.atrousConvolutionDevice( d_TempSubsampled, d_y1, inputRows, inputCols, d_Temp, paddedRows, paddedCols, m, d_h1, len_h1, type_signal );

        //// y0 = cuda_atrousc(symext(x,upsample2df(h0,i),shift),h0,I2 * L,'h0');
        //// x=y0;
        paddedRows = inputRows + m * len_h0 - 1;
        paddedCols = inputCols + m * len_h0 - 1;
        offsetRows = (m * len_h0)/2 - shift;
        offsetCols = (m * len_h0)/2 - shift;
        func.symExt( d_Temp, paddedRows, paddedCols, d_x, inputRows, inputCols, offsetRows, offsetCols, type_signal );
        func.atrousConvolutionDevice( d_TempSubsampled, d_y0, inputRows, inputCols, d_Temp, paddedRows, paddedCols, m, d_h0, len_h0, type_signal );
    }

    if( tempBuffer == NULL )
    {
        // Free temporary buffer
        cmexSafeCall( cudaFree( d_Temp ) );
        cmexSafeCall( cudaFree( d_TempSubsampled ) );
    }
}
