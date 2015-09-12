/******************************************************************
* atrousc.c -  Written by Arthur Cunha. This routine builds up on 
*               zconv2D_OS.c written by Jason Laska
*
*           Modified by Xavier Gibert Serra <gibert@umiacs.umd.edu>
*
* Inputs:   x - A 2D signal
*           h - 2D filter
*           m - separable upsampling matrix
*         
* Outputs:  y - 2D result of convolution with filter 
*           upsampled by a m, only the 'valid' part is returned.
*           Similar to conv2(x,h,'valid'), where h is the upsampled
*           filter.
*  
*          
*
* Usage:    y = zconv2D_O(x,h,m);
*
* Notes:    This function does not actually upsample the filter, 
*           it computes the convolution as if the filter had been 
*           upsampled. This is the ultimate optimized version.
*           Further optimized for separable (diagonal) upsampling matrices.
*
* This is a MEX-FILE for matlab
*
/********************************************************/

#include "mex.h"
#include "matrix.h"
#include <math.h>


#define OUT     plhs[0]
#define SIGNAL  prhs[0] 
#define FILTER  prhs[1] 
#define MMATRIX prhs[2]


#define LINPOS(row,col,collen) (row*collen)+col


template <class dataType> void convolutionLoop(const dataType* FArray,
        const dataType* SArray, dataType* outArray,
        int SColLength, int SRowLength, int FColLength, int FRowLength,
        int O_SColLength, int O_SRowLength, int M0, int M3, int sM0, int sM3)
{
    int SFColLength = FColLength-1;
    int SFRowLength = FRowLength-1;
    
	/* Convolution loop */
    for (int n1=0;n1<O_SRowLength;n1++){
		for (int n2=0;n2<O_SColLength;n2++){
			dataType sum=0;		    
		    int kk1 = n1 + sM0;
			for (int k1=0;k1<FRowLength;k1++){
  			    int kk2 = n2 + sM3;
                int f1 = SFRowLength - k1; /* flipped index */
                /* Precalculate column pointers */
                const dataType* pFCol = &FArray[LINPOS(f1,SFColLength,FColLength)];
                const dataType* pSCol = &SArray[LINPOS(kk1,0,SColLength)];
				for (int k2=0;k2<FColLength;k2++){
					 //f2 = SFColLength - k2;		
					 //sum+= FArray[LINPOS(f1,f2,FColLength)] * SArray[LINPOS(kk1,kk2,SColLength)];
                     sum+= pFCol[-k2] * pSCol[kk2];
					 kk2+=M3;
				}
				kk1+=M0;
			} 
		    outArray[LINPOS(n1,n2,O_SColLength)] = sum;
		}
	}    
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
/*  FArray   - Filter coefficients
    SArray   - Signal coefficients
    outArray - Output coefficients
    M        - upsampling matrix 	*/
    mxClassID mexDataType = mxDOUBLE_CLASS;

    int SColLength = mxGetM(SIGNAL); 
    int SRowLength = mxGetN(SIGNAL);
    int FColLength = mxGetM(FILTER); 
    int FRowLength = mxGetN(FILTER);

    if (nrhs<3){
        mexErrMsgTxt("There are at least 3 input parameters required!");
        return;
    }    
    
    if (!mxIsDouble(MMATRIX))
    {
        mexErrMsgIdAndTxt("MATLAB:atrousc:rhs",
                "Third input argument is not single.");
    }
    
    // Check input data types
    if (mxIsClass(FILTER, "single") && mxIsClass(SIGNAL, "single"))
    {
        mexDataType = mxSINGLE_CLASS;
    }
    else if (mxIsClass(FILTER, "double") && mxIsClass(SIGNAL, "double"))
    {
        mexDataType = mxDOUBLE_CLASS;
    }
    else
    {
        mexErrMsgIdAndTxt("MATLAB:atrousc:rhs",
                "Input arguments are neither single nor double.");
    }

    double *M = mxGetPr(MMATRIX);
            
    int M0 = (int)M[0];    
    int M3 = (int)M[3];   
    int sM0 = M0-1;
    int sM3 = M3-1;

	int O_SColLength = SColLength - M0*FColLength + 1;
	int O_SRowLength = SRowLength - M3*FRowLength + 1;
	
	mwSize dims[2] = {O_SColLength, O_SRowLength};
    OUT = mxCreateNumericArray(2, dims, mexDataType, mxREAL);
    
    if (mexDataType == mxSINGLE_CLASS)
    {
        const float* FArray = (float*)mxGetData(FILTER);
        const float* SArray = (float*)mxGetData(SIGNAL);    
        float* outArray = (float*)mxGetData(OUT);	
 
        convolutionLoop<float>(FArray, SArray, outArray,
                SColLength, SRowLength, FColLength, FRowLength,
                O_SColLength, O_SRowLength, M0, M3, sM0, sM3);
    }
    else
    {
        const double* FArray = (double*)mxGetData(FILTER);
        const double* SArray = (double*)mxGetData(SIGNAL);    
        double* outArray = (double*)mxGetData(OUT);	
 
        convolutionLoop<double>(FArray, SArray, outArray,
                SColLength, SRowLength, FColLength, FRowLength,
                O_SColLength, O_SRowLength, M0, M3, sM0, sM3);        
    }
    return;
}
