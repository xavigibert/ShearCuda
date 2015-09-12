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
#include "npps.h"


#include "GPUmat.hh"
#include "ShearCudaFunctions.h"
#include "MexUtil.h"

#include <math.h>

// static paramaters
static ShearCudaFunctions func;
static GpuTimes* gt;

static int init = 0;

static GPUmat *gm;

#define even(x)  ((x & 1) ? 0 : 1)
#define isint(x) ((x - floor(x)) > 0.0 ? 0 : 1)
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

// Forward declarations
void mrdwt(const void* x, int m, int n, const void* h, int lh, int L,
           void* yl, void* yh, gpuTYPE_t type_signal);

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

    /* check for correct # of input variables */
    if (nrhs>4){
        mexErrMsgTxt("There are at most 4 input parameters allowed!");
        return;
    }
    if (nrhs<2){
        mexErrMsgTxt("There are at least 2 input parameters required!");
        return;
    }

    const mxArray* mx_signal = prhs[0];
    const mxArray* mx_filter = prhs[1];
    const mxArray* mx_level  = prhs[2];
    const mxArray* mx_coeff  = NULL;    // Result for in-place operation
    if( nrhs>3 ) mx_coeff = prhs[3];

    // Fallback to CPU implementation if data type is single or double
    if( mxIsSingle(mx_signal) || mxIsDouble(mx_signal) )
    {
        mexCallMATLAB(nlhs, plhs, nrhs, const_cast<mxArray **>(prhs), "mrdwt");
        return;
    }

    GPUtype gpu_signal = gm->gputype.getGPUtype(mx_signal);
    GPUtype gpu_filter = gm->gputype.getGPUtype(mx_filter);

    // Check parameter types
    gpuTYPE_t type_signal = gm->gputype.getType( gpu_signal );
    gpuTYPE_t type_filter = gm->gputype.getType( gpu_filter );

    if( type_signal == gpuDOUBLE && !func.supportsDouble )
        mexErrMsgTxt("GPUdouble requires compute capability >= 2.0");
    if( type_signal != gpuDOUBLE && type_signal != gpuFLOAT )
        mexErrMsgTxt("Signal should be GPUsingle or GPUdouble");
    if( type_signal != type_filter )
        mexErrMsgTxt("Filter should be of the same type as the signal");

    const int* signal_dims = gm->gputype.getSize( gpu_signal );
    int m = signal_dims[0];
    int n = signal_dims[1];
    const int* filter_dims = gm->gputype.getSize( gpu_filter );
    int h_row = filter_dims[0];
    int h_col = filter_dims[1];
    int lh = max(h_col, h_row);

    int L;    // Number of levels

    if (nrhs >= 3){
        if (!mxIsDouble(mx_level))
        {
            mexErrMsgTxt("Input arguments are not double.");
        }
        L = (int) mxGetScalar(mx_level);
        if (L < 0)
            mexErrMsgTxt("The number of levels, L, must be a non-negative integer");
    }
    else /* Estimate L */ {
        int i=n;
        int j=0;
        while (even(i)){
            i=(i>>1);
            j++;
        }
        L=m;i=0;
        while (even(L)){
            L=(L>>1);
            i++;
        }
        if(min(m,n) == 1)
            L = max(i,j);
        else
            L = min(i,j);
        if (L==0){
            mexErrMsgTxt("Maximum number of levels is zero; no decomposition can be performed!");
            return;
        }
    }
    /* Check the ROW dimension of input */
    if(m > 1){
        double mtest = (double) m/pow(2.0, (double) L);
        if (!isint(mtest))
            mexErrMsgTxt("The matrix row dimension must be of size m*2^(L)");
    }
    /* Check the COLUMN dimension of input */
    if(n > 1){
        double ntest = (double) n/pow(2.0, (double) L);
        if (!isint(ntest))
            mexErrMsgTxt("The matrix column dimension must be of size n*2^(L)");
    }

    // Prepare output
    GPUtype gpu_yl, gpu_yh;
    if( mx_coeff == NULL )
    {
        // Allocate output if buffer is not provided by the user
        int dims[2] = {m, n};
        gpu_yl = gm->gputype.create( type_signal, 2, dims, NULL);
        if (min(m,n) == 1) {
            dims[0] = m;
            dims[1] = L*n;
        }
        else {
            dims[0] = m;
            dims[1] = 3*L*n;
        }
        gpu_yh = gm->gputype.create( type_signal, 2, dims, NULL);
    }
    else
    {
        // Grab elements of mx_coeff
        mxArray* mx_yl = mxGetCell( mx_coeff, 0 );
        gpu_yl = gm->gputype.getGPUtype( mx_yl );
        mxArray* mx_yh = mxGetCell( mx_coeff, 1 );
        gpu_yh = gm->gputype.getGPUtype( mx_yh );
    }
    if( nlhs > 2 )
    {
        plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
        double* Lr = mxGetPr(plhs[2]);
        *Lr = L;
    }

    const void* x = gm->gputype.getGPUptr( gpu_signal );
    const void* h = gm->gputype.getGPUptr( gpu_filter );

    void* yl = const_cast<void*>(gm->gputype.getGPUptr( gpu_yl ));
    void* yh = const_cast<void*>(gm->gputype.getGPUptr( gpu_yh ));

    mrdwt(x, m, n, h, lh, L, yl, yh, type_signal);

    if( nlhs > 0 )
    {
        plhs[0] = gm->gputype.createMxArray( gpu_yl );
        plhs[1] = gm->gputype.createMxArray( gpu_yh );
    }
}

void mrdwt(const void* x, int m, int n, const void* h, int lh, int L,
           void* yl, void* yh, gpuTYPE_t type_signal)
{
    if (n==1){
        n = m;
        m = 1;
    }

    int imageSize = m * n * (type_signal==gpuFLOAT ? sizeof(float) : sizeof(double));

    // Main loop
    for( int actual_L = 1; actual_L <= L; actual_L++ )
    {
        if( m == 1)
        {
            // Perform filtering along rows
            func.mrdwtRow((actual_L==1?x:yl), 1, n, h, lh, actual_L, yl, yh, type_signal);
        }
        else
        {
            int c_o_a = 3*(actual_L-1);

            //void* yll = yl;
            void* yhl = (char*)yh + c_o_a * imageSize;
            void* ylh = (char*)yh + (c_o_a + 1) * imageSize;
            void* yhh = (char*)yh + (c_o_a + 2) * imageSize;

            // Perform filtering along rows
            func.mrdwtRow((actual_L==1?x:yl), n, m, h, lh, actual_L, yl, yhl, type_signal);

            // Perform filtering along columns
            func.mrdwtCol(yl, n, m, h, lh, actual_L, yl, ylh, type_signal);
            func.mrdwtCol(yhl, n, m, h, lh, actual_L, yhl, yhh, type_signal);
        }
    }
}
