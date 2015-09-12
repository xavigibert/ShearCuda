/*
     Copyright (C) 2012  GP-you Group (http://gp-you.org)
 
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



#include "GPUkernel.hh"

// textures
texture<float, 1>          texref_f1_a;
texture<int2, 1>           texref_d1_a;

typedef float2  Complex;
typedef double2 DoubleComplex;

// double precision texture fetch
static __inline__ __device__ double tex1Dfetch_double(texture<int2, 1> t, double i)
{
    int2 v = tex1Dfetch(t,i);
#if !defined(CUDA_NO_SM_13_DOUBLE_INTRINSICS)    
    return __hiloint2double(v.y, v.x);
#else
    return 0.0;
#endif
}

extern "C" {

/* FLOAT */
__global__ void LININTERF(int n, 
                      int offset, 
                      float * index, 
                      float * odata)
{  
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    if ((xIndex - offset) < n) 
        odata[xIndex] = tex1Dfetch(texref_f1_a,index[xIndex]);
}

/* DOUBLE */
__global__ void LININTERD(int n, 
                      int offset, 
                      double * index, 
                      double * odata)
{  
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;
    if ((xIndex - offset) < n) 
        odata[xIndex] = tex1Dfetch_double(texref_d1_a, index[xIndex]);
}

}
