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

	if (nrhs != 0)
		mexErrMsgTxt("Wrong number of arguments");

	cufftResult_t dummy;
	/*CUFFT_SUCCESS        = 0x0,
	 CUFFT_INVALID_PLAN   = 0x1,
	 CUFFT_ALLOC_FAILED   = 0x2,
	 CUFFT_INVALID_TYPE   = 0x3,
	 CUFFT_INVALID_VALUE  = 0x4,
	 CUFFT_INTERNAL_ERROR = 0x5,
	 CUFFT_EXEC_FAILED    = 0x6,
	 CUFFT_SETUP_FAILED   = 0x7,
	 CUFFT_INVALID_SIZE   = 0x8*/

	const char *field_names[] = { "CUFFT_SUCCESS", "CUFFT_INVALID_PLAN",
			"CUFFT_ALLOC_FAILED", "CUFFT_INVALID_TYPE", "CUFFT_INVALID_VALUE",
			"CUFFT_INTERNAL_ERROR", "CUFFT_EXEC_FAILED", "CUFFT_SETUP_FAILED",
			"CUFFT_INVALID_SIZE" };

	mxArray *r;
	mwSize dims[2] = { 1, 1 };
	mxArray *field_value;

	r = mxCreateStructArray(2, dims, 9, field_names);
	field_value = mxCreateDoubleScalar((unsigned int) CUFFT_SUCCESS);
	mxSetFieldByNumber(r, 0, 0, field_value);

	field_value = mxCreateDoubleScalar((unsigned int) CUFFT_INVALID_PLAN);
	mxSetFieldByNumber(r, 0, 1, field_value);

	field_value = mxCreateDoubleScalar((unsigned int) CUFFT_ALLOC_FAILED);
	mxSetFieldByNumber(r, 0, 2, field_value);

	field_value = mxCreateDoubleScalar((unsigned int) CUFFT_INVALID_TYPE);
	mxSetFieldByNumber(r, 0, 3, field_value);

	field_value = mxCreateDoubleScalar((unsigned int) CUFFT_INVALID_VALUE);
	mxSetFieldByNumber(r, 0, 4, field_value);

	field_value = mxCreateDoubleScalar((unsigned int) CUFFT_INTERNAL_ERROR);
	mxSetFieldByNumber(r, 0, 5, field_value);

	field_value = mxCreateDoubleScalar((unsigned int) CUFFT_EXEC_FAILED);
	mxSetFieldByNumber(r, 0, 6, field_value);

	field_value = mxCreateDoubleScalar((unsigned int) CUFFT_SETUP_FAILED);
	mxSetFieldByNumber(r, 0, 7, field_value);

	field_value = mxCreateDoubleScalar((unsigned int) CUFFT_INVALID_SIZE);
	mxSetFieldByNumber(r, 0, 8, field_value);

	plhs[0] = r;

}
