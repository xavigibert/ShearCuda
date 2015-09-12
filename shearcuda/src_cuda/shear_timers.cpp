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
#include <unistd.h>

#include "mex.h"

#include "GpuTimes.h"

// static paramaters
static GpuTimes gt;     // Common GpuTimes object
static bool enabled = true;

// This function keeps track of timing used by computing functions
// The function has multiple subfunctions depending on the value of the parameter
// INPUT: val
//           0 - (or no value) display counters
//           1 - Reset counters
//           2 - Get pointer to global pointer object
//           3 - Enable timing (forces syncing at beginning and end of function
//           4 - Disable timing
//           5 - Run test
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int param = 0;

    if( nrhs > 0 )
    {
        param = (int)mxGetScalar(prhs[0]);
    }

    if( param < 0 && param > 4 )
        mexErrMsgTxt( "Invalid option");

    switch( param )
    {
    case 0:
        gt.displayTimers();
        break;
    case 1:
        gt.resetTimers();
        break;
    case 2:
    {
        int dims[2] = {1,1};
        plhs[0] = mxCreateNumericArray(2,dims,mxUINT64_CLASS,mxREAL);
        GpuTimes** pGT = (GpuTimes**)mxGetData(plhs[0]);
        *pGT = &gt;
        break;
    }
    case 3:
        gt.enable();
        mexPrintf("GPU timing enabled\n");
        break;
    case 4:
        gt.disable();
        mexPrintf("GPU timing disabled\n");
        break;
    case 5:
        mexPrintf("Sleeping 1 msec\n");
        gt.resetTimers();
        gt.startTimer(GpuTimes::testTimer);
        usleep(1000);
        gt.stopTimer(GpuTimes::testTimer);
        gt.displayTimers();
        break;
    }
}
