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

#include <stdio.h>
#include <string.h>

#ifdef UNIX
#include <stdint.h>
#endif

#include "mex.h"


#include "cufft.h"

#include "GPUcommon.hh"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	if (nrhs != 3)
		mexErrMsgTxt("Wrong number of arguments");

	/*%  cufftResult
	 %  cufftExecR2C(cufftHandle plan,
	 %               cufftReal *idata,
	 %               cufftComplex *odata);*/

	cufftHandle plan = (cufftHandle) mxGetScalar(prhs[0]);
	cufftReal *idata = (cufftReal*) (UINTPTR mxGetScalar(prhs[1]));
	cufftComplex *odata = (cufftComplex*) (UINTPTR mxGetScalar(prhs[2]));

	cufftResult status = cufftExecR2C(plan, idata, odata);

	plhs[0] = mxCreateDoubleScalar(status);

}
