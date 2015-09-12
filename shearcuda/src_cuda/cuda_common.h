/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H



typedef unsigned int uint;

#ifdef __CUDACC__
    typedef float2 fComplex;
    typedef double2 dComplex;
#else
    typedef struct{
        float x;
        float y;
    } fComplex;
    typedef struct{
        double x;
        double y;
    } dComplex;
#endif


////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b){
    return (a % b != 0) ?  (a - a % b + b) : a;
}

#endif //CUDA_COMMON_H
