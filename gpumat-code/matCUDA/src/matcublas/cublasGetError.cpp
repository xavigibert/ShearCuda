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

#include "cublas.h"

#include "GPUcommon.hh"

void
mexFunction( int nlhs, mxArray *plhs[],
             int nrhs,  const mxArray *prhs[] )
{

  int status;

  if (nrhs!=0)
    mexErrMsgTxt("Wrong number of arguments");


  status = cublasGetError();

  plhs[0] = mxCreateDoubleScalar(status);

}
