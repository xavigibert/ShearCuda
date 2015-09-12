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
void mirdwt(const void* x_inl, const void* x_inh, int m, int n,
           const void* h, int lh, int L, void* x_out, gpuTYPE_t type_signal);

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
    if (nrhs<3){
        mexErrMsgTxt("There are at least 3 input parameters required!");
        return;
    }

    const mxArray* mx_yl     = prhs[0];
    const mxArray* mx_yh     = prhs[1];
    const mxArray* mx_filter = prhs[2];
    const mxArray* mx_level  = prhs[3];

    // Fallback to CPU implementation if data type is single or double
    if( mxIsSingle(mx_yl) || mxIsDouble(mx_yl) )
    {
        mexCallMATLAB(nlhs, plhs, nrhs, const_cast<mxArray **>(prhs), "mirdwt");
        return;
    }

    GPUtype gpu_yl = gm->gputype.getGPUtype(mx_yl);
    GPUtype gpu_yh = gm->gputype.getGPUtype(mx_yh);
    GPUtype gpu_filter = gm->gputype.getGPUtype(mx_filter);

    // Check parameter types
    gpuTYPE_t type_yl = gm->gputype.getType( gpu_yl );
    gpuTYPE_t type_yh = gm->gputype.getType( gpu_yh );
    gpuTYPE_t type_filter = gm->gputype.getType( gpu_filter );

    if( type_yl == gpuDOUBLE && !func.supportsDouble )
        mexErrMsgTxt("GPUdouble requires compute capability >= 2.0");
    if( type_yl != gpuDOUBLE && type_yl != gpuFLOAT )
        mexErrMsgTxt("Signal should be GPUsingle or GPUdouble");
    if( type_yl != type_yh || type_yl != type_filter )
        mexErrMsgTxt("Filter should be of the same type as the signal");

    const int* yl_dims = gm->gputype.getSize( gpu_yl );
    int m = yl_dims[0];
    int n = yl_dims[1];
    const int* yh_dims = gm->gputype.getSize( gpu_yh );
    int mh = yh_dims[0];
    int nh = yh_dims[1];
    const int* filter_dims = gm->gputype.getSize( gpu_filter );
    int h_row = filter_dims[0];
    int h_col = filter_dims[1];
    int lh = max(h_col, h_row);

    int L;    // Number of levels

    if (nrhs == 4){
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

    /* check for consistency of rows and columns of yl, yh */
    if (min(m,n) > 1){
	if((m != mh) | (3*n*L != nh)){
	    mexErrMsgTxt("Dimensions of first two input matrices not consistent!");
	    return;
	}
    }
    else{
	if((m != mh) | (n*L != nh)){
	    mexErrMsgTxt("Dimensions of first two input vectors not consistent!");{
		return;
	    }
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

    int dims[2] = {m, n};
    GPUtype gpu_x = gm->gputype.create( type_yl, 2, dims, NULL);
    plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
    double* Lr = mxGetPr(plhs[1]);
    *Lr = L;

    void* x = const_cast<void*>(gm->gputype.getGPUptr( gpu_x ));
    const void* h = gm->gputype.getGPUptr( gpu_filter );

    const void* yl = gm->gputype.getGPUptr( gpu_yl );
    const void* yh = gm->gputype.getGPUptr( gpu_yh );

    mirdwt(yl, yh, m, n, h, lh, L, x, type_yl);

    plhs[0] = gm->gputype.createMxArray( gpu_x );
}

void mirdwt(const void* x_inl, const void* x_inh, int m, int n,
           const void* h, int lh, int L, void* x_out, gpuTYPE_t type_signal)
{
    if (n==1){
        n = m;
        m = 1;
    }

    int imageSize = m * n * (type_signal==gpuFLOAT ? sizeof(float) : sizeof(double));

    // Allocate temporary memory
    void* d_temp_xl = NULL;
    if( m > 1)
	cmexSafeCall( cudaMalloc( &d_temp_xl, imageSize ));

    // Main loop
    for( int actual_L = L; actual_L >= 1; actual_L-- )
    {
        if( m == 1)
        {
            // Perform filtering along rows
            func.mirdwtRow( (actual_L==L?x_inl:x_out), x_inh, 1, n, h, lh, actual_L, x_out, type_signal);
        }
        else
        {
            int c_o_a = 3*(actual_L-1);

            const void* yll = ( actual_L==L?x_inl:x_out );
            const void* yhl = (char*)x_inh + c_o_a * imageSize;
            const void* ylh = (char*)x_inh + (c_o_a + 1) * imageSize;
            const void* yhh = (char*)x_inh + (c_o_a + 2) * imageSize;

            // Perform reconstruction along columns
            func.mirdwtCol(yll, ylh, n, m, h, lh, actual_L, x_out, type_signal);
            func.mirdwtCol(yhl, yhh, n, m, h, lh, actual_L, d_temp_xl, type_signal);

            // Perform reconstruction along rows
            func.mirdwtRow(x_out, d_temp_xl, n, m, h, lh, actual_L, x_out, type_signal);

        }
    }

    // Delete temporary memory
    if( m > 1 )
	cmexSafeCall( cudaFree( d_temp_xl ));
}
