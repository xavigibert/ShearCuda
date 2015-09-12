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
#include "ShearCudaFunctions.h"
#include "MexUtil.h"

// static paramaters
static ShearCudaFunctions func;
static GpuTimes* gt;

static int init = 0;

static GPUmat *gm;

// Zero pad and prepare shearing filters
void prepareFilters( void* dst, void* temp_ptr, int dstSize, int numDir, gpuTYPE_t data_type, const void* src, int srcSize, cufftHandle fftPlan );

// This function takes a cell array of single or double precission
// shearlet coefficients and generates a data structure that can be
// used by shear_trans_cuda and inverse_shear_cuda
//
// INPUTS: [0] shearFilters (cell array with one element per scale, real-valued
//               single of double in space domain)
//         [1] dataLen (scalar indicating the length of the data)
// OUTPUT: [0] shearCuda (data structure containing the following elements
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

    if( nrhs != 2)
        mexErrMsgTxt( "Incorrect number of input arguments" );
    if( nlhs != 1 )
        mexErrMsgTxt( "Incorrect number of output arguments" );

    // Check that input data is a cell
    const mxArray* mx_input =  prhs[0];
    if( !mxIsCell( mx_input ) )
        mexErrMsgTxt( "Input argument is not cell" );
    int numScales = (int)mxGetNumberOfElements( mx_input );
    if( numScales < 1 )
        mexErrMsgTxt( "Input cell is empty" );
    int filterLen = (int)mxGetScalar(prhs[1]);

    mxArray* mx_cell_elem = mxGetCell( mx_input, 0 );

    // Determine input data type
    GPUtype gpu_cell_elem = gm->gputype.getGPUtype( mx_cell_elem );
    gpuTYPE_t real_type;
    real_type = gm->gputype.getType( gpu_cell_elem );
    if( real_type != gpuFLOAT && real_type != gpuDOUBLE )
    {
        mexErrMsgTxt( "Input data should be real-valued of class 'GPUsingle' or 'GPUdouble'" );
        return;
    }
    gpuTYPE_t data_type = (real_type == gpuFLOAT ? gpuCFLOAT : gpuCDOUBLE );

    // Preallocate temporary buffers for subsampling
    mxArray* mx_temp_buffer = mxCreateCellMatrix( 1, 3 );
    int buff_dims[2] = {2*filterLen, 2*filterLen};
    GPUtype gpuTempBuffer[3];
    for( int j = 0; j < 3; j++ )
    {
        gpuTempBuffer[j] = gm->gputype.create( real_type, 2, buff_dims, NULL );
        mxArray* mx_elem = gm->gputype.createMxArray( gpuTempBuffer[j] );
        mxSetCell( mx_temp_buffer, j, mx_elem );
    }
    // Get pointer to temporary buffer
    void* d_temp_ptr = const_cast<void*>( gm->gputype.getGPUptr( gpuTempBuffer[0] ) );

    // Allocate temporary storage
    std::vector<int> numDirections( numScales );

    // Prepare elements of output data structure
    mwSize plan_dims[2] = { numScales, 1 };
    mxArray* mx_fftPlanMany =  mxCreateNumericArray( 2, plan_dims, mxUINT32_CLASS, mxREAL );
    mxArray* mx_ifftPlanMany =  mxCreateNumericArray( 2, plan_dims, mxUINT32_CLASS, mxREAL );
    mxArray* mx_filter = mxCreateCellMatrix( 1, numScales );
    mwSize single_dims[2] = { 1, 1 };
    mxArray* mx_fftPlanOne = mxCreateNumericArray( 2, single_dims, mxUINT32_CLASS, mxREAL );

    // Populate output results
    cufftHandle* fftPlanMany = (cufftHandle*)mxGetData( mx_fftPlanMany );
    cufftHandle* ifftPlanMany = (cufftHandle*)mxGetData( mx_ifftPlanMany );
    cufftHandle* fftPlanOne = (cufftHandle*)mxGetData( mx_fftPlanOne );

    // Check each scale
    for( int idxScale = 0; idxScale < numScales; idxScale++ )
    {
        mx_cell_elem = mxGetCell( mx_input, idxScale );
        gpu_cell_elem = gm->gputype.getGPUtype( mx_cell_elem );
        int ndims = gm->gputype.getNdims( gpu_cell_elem );
        const int* dims = gm->gputype.getSize( gpu_cell_elem );
        if( ndims > 2 )
            numDirections[idxScale] = dims[2];
        else
            numDirections[idxScale] = 1;
        if( dims[0] != dims[1] || ndims > 3 )
            mexErrMsgTxt( "Invalid filter dimensions" );

        // Zero-padded dimensions
        int idims[3];
        idims[0] = filterLen;
        idims[1] = filterLen;
        idims[2] = dims[2];

        // Check if previous plan can be reused
        bool bFound = false;
        for( int idxOther = 0; idxOther < idxScale; idxOther++ )
        {
            if( numDirections[idxScale] == numDirections[idxOther] )
            {
                fftPlanMany[idxScale] = fftPlanMany[idxOther];
                ifftPlanMany[idxScale] = ifftPlanMany[idxOther];

                bFound = true;
                break;
            }
        }

        // Create FFT plans
        if( !bFound )
        {
            if( real_type == gpuFLOAT )
            {
                cufftmexSafeCall( cufftPlanMany( &fftPlanMany[idxScale], 2, idims, NULL, 0, 0, NULL, 0, 0, CUFFT_R2C, numDirections[idxScale] ) );
                cufftmexSafeCall( cufftPlanMany( &ifftPlanMany[idxScale], 2, idims, NULL, 0, 0, NULL, 0, 0, CUFFT_C2R, numDirections[idxScale] ) );
            }
            else
            {
                cufftmexSafeCall( cufftPlanMany( &fftPlanMany[idxScale], 2, idims, NULL, 0, 0, NULL, 0, 0, CUFFT_D2Z, numDirections[idxScale] ) );
                cufftmexSafeCall( cufftPlanMany( &ifftPlanMany[idxScale], 2, idims, NULL, 0, 0, NULL, 0, 0, CUFFT_Z2D, numDirections[idxScale] ) );
            }
        }

        // Prepare zero-padded filters
        idims[1] = (filterLen/2+1);     // R2C FFT only generates non-redundant elements
        GPUtype dshear = gm->gputype.create(data_type, 3, idims, NULL );
        void* dshear_ptr = const_cast<void*>( gm->gputype.getGPUptr( dshear ));
        const void* w_s_ptr = gm->gputype.getGPUptr( gpu_cell_elem );
        prepareFilters( dshear_ptr, d_temp_ptr, filterLen, numDirections[idxScale], real_type, w_s_ptr, dims[0], fftPlanMany[idxScale] );

        // Transfer filters to mx_filter
        mxArray* mx_filter_elem = gm->gputype.createMxArray( dshear );
        mxSetCell( mx_filter, idxScale, mx_filter_elem );
    }
    // Create single image plan
    if( data_type == gpuCFLOAT )
        cufftmexSafeCall( cufftPlan2d( fftPlanOne, filterLen, filterLen, CUFFT_R2C ) );
    else
        cufftmexSafeCall( cufftPlan2d( fftPlanOne, filterLen, filterLen, CUFFT_D2Z ) );

    // Preload standar 'maxflat' atrous filter
    mxArray* mx_atrousfilter = mxCreateString("maxflat");

    // Get filters for subsampling
    mxArray* atrousfilters[4];
    mxArray* filterParams[2];
    filterParams[0] = mx_atrousfilter;
    if( data_type == gpuCFLOAT )
        filterParams[1] = mxCreateString("GPUsingle");
    else
        filterParams[1] = mxCreateString("GPUdouble");

    mexCallMATLAB(4, atrousfilters, 2, filterParams, "atrousfilters");

    // Preallocate temporary buffers for decomposition
    mxArray* mx_temp_dec = mxCreateCellMatrix( 1, numScales + 1 );
    int dbuff_dims[2] = {filterLen, filterLen};
    std::vector<GPUtype> gpuDecBuffer(numScales + 1);
    for( int j = 0; j < numScales + 1; j++ )
    {
        gpuDecBuffer[j] = gm->gputype.create( real_type, 2, dbuff_dims, NULL );
        mxArray* mx_elem = gm->gputype.createMxArray( gpuDecBuffer[j] );
        mxSetCell( mx_temp_dec, j, mx_elem );
    }

    // Return results
    const char *field_names[] = { "filter", "fftPlanMany", "ifftPlanMany", "fftPlanOne", "atrousfilter",
                                  "h0", "h1", "g0", "g1", "tempBuffer", "tempDec"};
    const mwSize field_dims[2] = { 1, 1 };
    plhs[0] = mxCreateStructArray( 2, field_dims, 11, field_names );
    mxSetField( plhs[0], 0, "filter", mx_filter );
    mxSetField( plhs[0], 0, "fftPlanMany", mx_fftPlanMany );
    mxSetField( plhs[0], 0, "ifftPlanMany", mx_ifftPlanMany );
    mxSetField( plhs[0], 0, "fftPlanOne", mx_fftPlanOne );
    mxSetField( plhs[0], 0, "atrousfilter", mx_atrousfilter );
    mxSetField( plhs[0], 0, "h0", atrousfilters[0] );
    mxSetField( plhs[0], 0, "h1", atrousfilters[1] );
    mxSetField( plhs[0], 0, "g0", atrousfilters[2] );
    mxSetField( plhs[0], 0, "g1", atrousfilters[3] );
    mxSetField( plhs[0], 0, "tempBuffer", mx_temp_buffer );
    mxSetField( plhs[0], 0, "tempDec", mx_temp_dec );
}

// Zero pad and prepare shearing filters
void prepareFilters( void* dst, void* temp_ptr, int dstSize, int numDir, gpuTYPE_t data_type, const void* src, int srcSize, cufftHandle fftPlan )
{
    func.zeroPad(temp_ptr, dstSize, dstSize, numDir, data_type, src, srcSize, srcSize, numDir, data_type, srcSize/2, srcSize/2);

    gt->startTimer( GpuTimes::fftFwd );
    if( data_type == gpuFLOAT )
        cufftmexSafeCall( cufftExecR2C(fftPlan, (cufftReal *)temp_ptr, (cufftComplex *)dst ));
    else
        cufftmexSafeCall( cufftExecD2Z(fftPlan, (cufftDoubleReal *)temp_ptr, (cufftDoubleComplex *)dst ));
    gt->stopTimer( GpuTimes::fftFwd );
    
    func.prepareMyerFilters(dst, dstSize * (dstSize/2 + 1), numDir, data_type);
}
