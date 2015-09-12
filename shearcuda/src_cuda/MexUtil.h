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

#ifndef MEXUTIL_H
#define MEXUTIL_H

#include <cufft.h>

/////////////////////////////////////////////////////////////
// The code below is extracted from cutil_inline_runtime.h //
// Copyright 1993-2012 NVIDIA Corporation                  //
/////////////////////////////////////////////////////////////

// Give a little more for Windows : the console window often disapears before we can read the message
#ifdef _WIN32
# if 1//ndef UNICODE
#  ifdef _DEBUG // Do this only in debug mode...
	inline void VSPrintf(FILE *file, LPCSTR fmt, ...)
	{
		size_t fmt2_sz	= 2048;
		char *fmt2	= (char*)malloc(fmt2_sz);
		va_list vlist;
		va_start(vlist, fmt);
		while((_vsnprintf(fmt2, fmt2_sz, fmt, vlist)) < 0) // means there wasn't anough room
		{
			fmt2_sz *= 2;
			if(fmt2) free(fmt2);
			fmt2 = (char*)malloc(fmt2_sz);
		}
		OutputDebugStringA(fmt2);
		fprintf(file, fmt2);
		free(fmt2);
	}
# 	define FPRINTF(a) VSPrintf a
#  else //debug
# 	define FPRINTF(a) fprintf a
// For other than Win32
#  endif //debug
# else //unicode
// Unicode case... let's give-up for now and keep basic printf
# 	define FPRINTF(a) fprintf a
# endif //unicode
#else //win32
# 	define FPRINTF(a) fprintf a
#endif //win32

/////////////////////////////////////////////////////////////
////////            END OF NVIDIA CODE               ////////
/////////////////////////////////////////////////////////////

#ifdef MATLAB_MEX_FILE

// Added support for mex calls
#include <mex.h>

inline void __cudaMexSafeCall( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
		FPRINTF((stderr, "%s(%i) : cudaSafeCall() Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString( err ) ));
        char strMsg[256];
		sprintf(strMsg, "%s(%i) : cudaSafeCall() Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString( err ) );
        //exit(-1);
        mexErrMsgIdAndTxt("MATLAB:CUDA:FatalError", strMsg);
    }
}

#define cmexSafeCall(err)           __cudaMexSafeCall      (err, __FILE__, __LINE__)

inline void __cufftmexSafeCall( cufftResult err, const char *file, const int line )
{
    if( CUFFT_SUCCESS != err) {
        char strMsg[256];
        FPRINTF((stderr, "%s(%i) : cufftmexSafeCall() CUFFT error %d: ",
                file, line, (int)err));
        switch (err) {
            case CUFFT_INVALID_PLAN:   sprintf(strMsg, "%s(%i) : CUFFT_INVALID_PLAN\n", file, line);
            case CUFFT_ALLOC_FAILED:   sprintf(strMsg, "%s(%i) : CUFFT_ALLOC_FAILED\n", file, line);
            case CUFFT_INVALID_TYPE:   sprintf(strMsg, "%s(%i) : CUFFT_INVALID_TYPE\n", file, line);
            case CUFFT_INVALID_VALUE:  sprintf(strMsg, "%s(%i) : CUFFT_INVALID_VALUE\n", file, line);
            case CUFFT_INTERNAL_ERROR: sprintf(strMsg, "%s(%i) : CUFFT_INTERNAL_ERROR\n", file, line);
            case CUFFT_EXEC_FAILED:    sprintf(strMsg, "%s(%i) : CUFFT_EXEC_FAILED\n", file, line);
            case CUFFT_SETUP_FAILED:   sprintf(strMsg, "%s(%i) : CUFFT_SETUP_FAILED\n", file, line);
            case CUFFT_INVALID_SIZE:   sprintf(strMsg, "%s(%i) : CUFFT_INVALID_SIZE\n", file, line);
            case CUFFT_UNALIGNED_DATA: sprintf(strMsg, "%s(%i) : CUFFT_UNALIGNED_DATA\n", file, line);
            default: sprintf(strMsg, "%s(%i) : CUFFT Unknown error code\n", file, line);
        }
        mexErrMsgIdAndTxt("MATLAB:CUDA:FatalError", strMsg);
        //exit(-1);
    }
}

#define cufftmexSafeCall(err)           __cufftmexSafeCall     (err, __FILE__, __LINE__)

#else //MATLAB_MEX_FILE

#define cmexSafeCall(err)           __cudaSafeCall      (err, __FILE__, __LINE__)
#define cufftmexSafeCall(err)       __cufftSafeCall     (err, __FILE__, __LINE__)

#endif //MATLAB_MEX_FILE

#endif //MEXUTIL_H
