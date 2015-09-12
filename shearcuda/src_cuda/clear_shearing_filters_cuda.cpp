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
#include "ShearDictionary.h"

// static paramaters
static int init = 0;
static GPUmat *gm;

// This function takes a cell array of single or double precission
// shearlet coefficients and generates a data structure that can be
// used by shear_trans_cuda and inverse_shear_cuda
//
// INPUTS: [0] shearCuda (data structure containing the following elements
//             * fftPlanOne (plan for FFT of a one real image to complex spectrum)
//             * fftPlanMany (array of FFT plans (R2C), one multiplan per scale)
//             * ifftPlanMany (array of IFFT plans (C22), one multiplan per scale)
//             * filter (cell array of filters, one cell element per scale)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // One-time initialization
    if (init == 0)
    {
        // Initialize function
        //mexLock();

        // load GPUmat
        gm = gmGetGPUmat();

        init = 1;
    }

    if( nrhs != 1)
        mexErrMsgTxt( "Incorrect number of input arguments" );

    ShearDictionary shear;
    if( !shear.loadFromMx( prhs[0], gm ) )
        mexErrMsgTxt( "Invalid shearlet dictionary" );

    // Check each scale
    for( int idxScale = 0; idxScale < shear.numScales(); idxScale++ )
    {
        // Check if previous plan has been reused
        bool bFound = false;
        for( int idxOther = 0; idxOther < idxScale; idxOther++ )
        {
            if( shear.fftPlanMany(idxScale) == shear.fftPlanMany(idxOther) )
            {
                bFound = true;
                break;
            }
        }

        // Release FFT plan
        if( !bFound )
        {
            cufftmexSafeCall( cufftDestroy( shear.fftPlanMany(idxScale) ));
            cufftmexSafeCall( cufftDestroy( shear.ifftPlanMany(idxScale) ));
        }
    }

    // Destroy single image plan
    cufftmexSafeCall( cufftDestroy( shear.fftPlanOne() ));
}
