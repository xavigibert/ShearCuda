#ifndef DATATYPES_H
#define DATATYPES_H

#include <stdio.h>

enum gpuTYPE {
  gpuFLOAT = 0, gpuCFLOAT = 1, gpuDOUBLE = 2, gpuCDOUBLE = 3, gpuINT32 = 4, gpuUINT8 = 5, gpuNOTDEF = 20
};

typedef enum gpuTYPE gpuTYPE_t;

#endif // DATATYPES_H
