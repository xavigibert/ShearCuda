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


// A trous reconstruction
// INPUTS: y, g0, g1, level
// OUTPUT: outputImage
void atrousrec(const GPUmat *gm, const ShearCudaFunctions& func, const GPUtype* y, const GPUtype& g0, const GPUtype& g1, int numLevels,
               GPUtype& outputImage, bool alloc_result, GPUtype* tempBuffer)
{
    // Get data size and dimensions
    const int* dims = gm->gputype.getSize( y[0] );
    int inputRows = dims[0];
    int inputCols = dims[1];
    gpuTYPE_t type_signal = gm->gputype.getType( y[0] );
    std::vector<void*> y_ptr( numLevels + 1 );
    for( int i = 0; i < numLevels + 1; i++ )
        y_ptr[i] = const_cast<void*>( gm->gputype.getGPUptr( y[i] ));

    // Calculate dimensions and allocate temporary buffer
    int temp_size = (2*inputRows) * (2*inputCols)
            * (type_signal == gpuFLOAT ? sizeof(float) : sizeof(double));
    void* d_AtrousTemp;
    void* d_Temp;
    void* d_TempSubsampled;
    if( tempBuffer == NULL )
    {
        cmexSafeCall( cudaMalloc( &d_AtrousTemp, temp_size ));
        cmexSafeCall( cudaMalloc( &d_Temp, temp_size ));
        cmexSafeCall( cudaMalloc( &d_TempSubsampled, temp_size ));
    }
    else
    {
        d_AtrousTemp = const_cast<void*>( gm->gputype.getGPUptr( tempBuffer[0] ));
        d_Temp = const_cast<void*>( gm->gputype.getGPUptr( tempBuffer[1] ));
        d_TempSubsampled = const_cast<void*>( gm->gputype.getGPUptr( tempBuffer[2] ));
    }

    // Allocate results pointer
    if( alloc_result )
        outputImage = gm->gputype.create( type_signal, 2, dims, NULL );
    void* outputImage_ptr = const_cast<void*>( gm->gputype.getGPUptr( outputImage ));

    // Get length of synthesis filters
    const int len_g0 = gm->gputype.getSize( g0 )[0];
    const int len_g1 = gm->gputype.getSize( g1 )[0];
    void* d_g0 = const_cast<void*>( gm->gputype.getGPUptr( g0 ) );
    void* d_g1 = const_cast<void*>( gm->gputype.getGPUptr( g1 ) );

    //// % First Nlevels - 1 levels
    void* d_x = y_ptr[0];
    for( int i = numLevels - 2; i >= 0; i-- )
    {
        int shift = 1 - (1<<i);
        int m = 2<<i;

        void* d_y1 = y_ptr[numLevels - 1 - i];

        //// x = cuda_atrousc(symext(x,upsample2df(g0,i),shift),g0,L*I2,'g0') + cuda_atrousc(symext(y1,upsample2df(g1,i),shift),g1,L*I2,'g1');
        int paddedRows = inputRows + m * len_g1 - 1;
        int paddedCols = inputCols + m * len_g1 - 1;
        int offsetRows = (m * len_g1)/2 - shift;
        int offsetCols = (m * len_g1)/2 - shift;
        func.symExt( d_Temp, paddedRows, paddedCols, d_y1, inputRows, inputCols, offsetRows, offsetCols, type_signal );
        func.atrousConvolutionDevice( d_TempSubsampled, d_AtrousTemp, inputRows, inputCols, d_Temp, paddedRows, paddedCols, m, d_g1, len_g1, type_signal );

        paddedRows = inputRows + m * len_g0 - 1;
        paddedCols = inputCols + m * len_g0 - 1;
        offsetRows = (m * len_g0)/2 - shift;
        offsetCols = (m * len_g0)/2 - shift;
        func.symExt( d_Temp, paddedRows, paddedCols, d_x, inputRows, inputCols, offsetRows, offsetCols, type_signal );
        d_x = outputImage_ptr;
        func.atrousConvolutionDevice( d_TempSubsampled, d_x, inputRows, inputCols, d_Temp, paddedRows, paddedCols, m, d_g0, len_g0, type_signal );
        func.addVector( d_x, d_AtrousTemp, inputRows * inputCols, type_signal );
    }

    //// % Reconstruct first level
    //// x = conv2(symext(x,g0,shift),g0,'valid')+ conv2(symext(y{Nlevels+1},g1,shift),g1,'valid');
    int paddedRows = inputRows + len_g1 - 1;
    int paddedCols = inputCols + len_g1 - 1;
    func.symExt( d_Temp, paddedRows, paddedCols, y_ptr[numLevels], inputRows, inputCols, len_g1/2, len_g1/2, type_signal );
    func.atrousConvolutionDevice( d_TempSubsampled, d_AtrousTemp, inputRows, inputCols, d_Temp, paddedRows, paddedCols, 1, d_g1, len_g1, type_signal );

    paddedRows = inputRows + len_g0 - 1;
    paddedCols = inputCols + len_g0 - 1;
    func.symExt( d_Temp, paddedRows, paddedCols, d_x, inputRows, inputCols, len_g0/2, len_g0/2, type_signal );
    func.atrousConvolutionDevice( d_TempSubsampled, d_x, inputRows, inputCols, d_Temp, paddedRows, paddedCols, 1, d_g0, len_g0, type_signal );

    func.addVector( d_x, d_AtrousTemp, inputRows * inputCols, type_signal );

    // Free temporary buffer
    if( tempBuffer == NULL )
    {
        cmexSafeCall( cudaFree( d_TempSubsampled ) );
        cmexSafeCall( cudaFree( d_Temp ) );
        cmexSafeCall( cudaFree( d_AtrousTemp ) );
    }
}
