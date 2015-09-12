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

typedef float2  Complex;
typedef double2 DoubleComplex;

/* EYE
 * Matlab function eye. 
 *EYE(N) is the N-by-N identity matrix
 */
__device__ inline float eye_float(int index, int maxindex, int step) {
    return ((index <= maxindex)&&((index % step)==0)) ? 1.0:0.0;
}
__device__ inline double eye_double(int index, int maxindex, int step) {
  return ((index <= maxindex)&&((index % step)==0)) ? 1.0:0.0;
}
__device__ inline Complex eye_Complex(int index, int maxindex, int step) {
  return ((index <= maxindex)&&((index % step)==0)) ? make_float2(1.0,0.0):make_float2(0.0,0.0);
}
__device__ inline DoubleComplex eye_DoubleComplex(int index,int maxindex, int step) {
  return ((index <= maxindex)&&((index % step)==0)) ? make_double2(1.0,0.0):make_double2(0.0,0.0);
}

/* EYE TEMPLATE */
#define EYE_TEMPLATE(KERNAME, OUTARG1, FUNNAME) \
extern "C" { __global__ void KERNAME (int n, int offset, OUTARG1 * odata, int maxindex, int step)\
{  \
    unsigned int xIndex = blockIdx.x * BLOCK_DIM1D + threadIdx.x + offset;\
    if ((xIndex - offset) < n) \
        odata[xIndex] = FUNNAME (xIndex, maxindex, step);\
}\
}\

EYE_TEMPLATE(EYEF,  float,   eye_float)
EYE_TEMPLATE(EYEC,  Complex, eye_Complex)
EYE_TEMPLATE(EYED,  double,  eye_double)
EYE_TEMPLATE(EYEDC, DoubleComplex, eye_DoubleComplex)
