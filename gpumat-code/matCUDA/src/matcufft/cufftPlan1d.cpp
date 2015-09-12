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


void
mexFunction( int nlhs, mxArray *plhs[],
             int nrhs, const  mxArray *prhs[] )
{


  if (nrhs!=4)
    mexErrMsgTxt("Wrong number of arguments");


  cufftHandle plan =  (cufftHandle) mxGetScalar(prhs[0]);
  int nx  =  (int) mxGetScalar(prhs[1]);
  cufftType_t type = (cufftType_t) ((int) mxGetScalar(prhs[2]));
  int batch  =  (int) mxGetScalar(prhs[3]);

  cufftResult status = cufftPlan1d(&plan, nx, type, batch);

  plhs[0] = mxCreateDoubleScalar(status);
  if (nlhs>1)
    plhs[1] = mxCreateDoubleScalar(plan);



}
