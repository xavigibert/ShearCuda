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

#ifdef UNIX
#include <stdint.h>
#endif



#if !defined(GPUCOMMON_H_)
#define GPUCOMMON_H_

// The following is required by new GCC
#define STRINGCONST const


// POINTERS
#define UINTPTR (uintptr_t)

enum gpuTYPE {
  gpuFLOAT = 0, gpuCFLOAT = 1, gpuDOUBLE = 2, gpuCDOUBLE = 3, gpuINT32 = 4, gpuNOTDEF = 20
};

typedef enum gpuTYPE gpuTYPE_t;

#endif
