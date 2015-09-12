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



#if !defined(NUMERICS_HH_)
#define NUMERICS_HH_


/*************************************************
 * ERROR CODES
 *************************************************/
//
#define ERROR_NUMERICS_COMPNOTIMPLEMENTED  "Function compilation is not implemented. (ERROR code NUMERICS.1)"
#define WARNING_NUMERICS_COMPNOTIMPLEMENTED  "Function compilation is not implemented. (WARNING code NUMERICS.1)"

// mrdivide
#define ERROR_MRDIVIDE_MATRICES   "Division between matrices is not implemented. (ERROR code MRDIVIDE.1)"

// assign
#define ERROR_ASSIGN_FIRSTARG "First argument must be the 'direction'. (ERROR code ASSIGN.1)"

// gpufill
#define ERROR_GPUFILL_WRONGARGS "Wrong number of arguments.\nUsage: GPUfill(A, offset, incr, m, p, offsetp, type). (ERROR code GPUFILL.1)"

// memCpyDtoD
#define ERROR_MEMCPYDTOD_WRONGDSTINDEX "Wrong destination index. (ERROR code MEMCPYDTOD.1)"
#define ERROR_MEMCPYDTOD_TOOMANYEL  "Too many elements to copy. (ERROR code MEMCPYDTOD.2)"
#define ERROR_MEMCPYDTOD_DSTSRCSAME "DST and SRC should be of the same type. (ERROR code MEMCPYDTOD.3)"

// memCpyHtoD
#define ERROR_MEMCPYHTOD_WRONGDSTINDEX "Wrong destination index. (ERROR code MEMCPYHTOD.1)"
#define ERROR_MEMCPYHTOD_TOOMANYEL  "Too many elements to copy. (ERROR code MEMCPYHTOD.2)"
#define ERROR_MEMCPYHTOD_DSTSRCSAME "DST and SRC should be of the same type. (ERROR code MEMCPYHTOD.3)"
#define ERROR_MEMCPYHTOD_MXCOMPLEXNOTSUPP "Matlab complex arrays are not currently supported. (ERROR code MEMCPYHTOD.4)"

//mxColon
#define WARNING_MXCOLON_OPREALSCALARS "Colon operands must be real scalars. (Warning code MXCOLON.1)"


//mxPermute
#define ERROR_PERMUTE_INVALIDPERM "Invalid permutation vector. (ERROR code PERMUTE.1)"

// Casting
#define ERROR_CASTING_GPUSINGLE  "Wrong argument. Expected a GPUsingle. (ERROR code CAST.1)"
#define ERROR_CASTING_GPUDOUBLE  "Wrong argument. Expected a GPUdouble. (ERROR code CAST.2)"
#define ERROR_CASTING_GPUINT32   "Wrong argument. Expected a GPUint32.  (ERROR code CAST.3)"

/*************************************************
 * NUMERICS
 *************************************************/

/* EYE KERNEL */
#define N_EYEF  0
#define N_EYEC  1
#define N_EYED  2
#define N_EYEDC 3


/*************************************************
 * UTIL
 *************************************************/

void parseRange(int nrhs, const mxArray *prhs[], Range **rg, MyGCObj<Range> &mygc);

/// Creates a Range from prhs
/**
 * prhs is assumed to come from a Matlab subsref call
* @param[in]   rhsdim Dimensions of the RHS GPUtype
* @param[prhs] The Matlab subsref range to be parsed
*/
void parseMxRange(int rhsdim, const mxArray *prhs, Range **outrg, GPUmat *gm, MyGCObj<Range> &mygc1);

#endif
