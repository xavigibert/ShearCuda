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

#if !defined(GPUTYPEMAT_H_)
#define GPUTYPEMAT_H_

/*************** UTILS ***************/
/* toMxStruct */
//mxArray *
//toMxStruct(GPUtype *);

/* objToStruct */
//GPUtype *
//objToStruct(mxArray *p);

/* mxID */
void *
mxID (const mxArray *p);

/* mxToGPUtype */
GPUtype *
mxToGPUtype (const mxArray *prhs, GPUmanager *GPUman);

/* toMx */
mxArray *
toMx(GPUtype*, int=0);

/* MxNumericArrayToGPUtype */
GPUtype * mxNumericArrayToGPUtype(mxArray *res, GPUmanager *GPUman);

/* GPUtypeToMxNumericArray */
mxArray * GPUtypeToMxNumericArray(GPUtype &p);

/* mxNumericArrayToGPUtype */
void mxNumericArrayToGPUtype(mxArray *res, GPUtype *p);

/* mxCreateGPUtype */
GPUtype * mxCreateGPUtype(gpuTYPE_t type, GPUmanager *GPUman, int nrhs, const mxArray *prhs[]);
#endif
