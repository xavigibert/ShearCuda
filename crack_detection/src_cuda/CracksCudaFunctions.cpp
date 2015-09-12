/*
 * Author: Xavier Gibert-Serra (gibert@umiacs.umd.edu)
 *
 * Copyright (C) 2013 University of Maryland. All rights reserved.
 *
 */

#include "CracksCudaFunctions.h"

// CUDA
#include "cuda.h"
#include "cuda_runtime.h"

#include "MexUtil.h"
#include "GPUkernel.hh"

CracksCudaFunctions::CracksCudaFunctions() :
    supportsFloat(false),
    supportsDouble(false),
    cudaLowLevelCrackFeatures(NULL),
    cudaDotProductsFeatures(NULL),
    cudaGraphAffinitiesFromDotProd(NULL),
    cudaGraphTerminals(NULL),
    m_timer(NULL)
{
   
}
    
bool CracksCudaFunctions::LoadGpuFunctions( const CUmodule *drvmod )
{
    bool bSuccessF = true, bSuccessD = false;
    
    // load GPU single precision functions
    bSuccessF = bSuccessF && (cuModuleGetFunction(&cudaLowLevelCrackFeatures, *drvmod, "lowLevelCrackFeatures") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&cudaDotProductsFeatures, *drvmod, "dotProductsFeatures") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&cudaGraphAffinitiesFromDotProd, *drvmod, "graphAffinitiesFromDotProd") == CUDA_SUCCESS);
    bSuccessF = bSuccessF && (cuModuleGetFunction(&cudaGraphTerminals, *drvmod, "graphTerminals") == CUDA_SUCCESS);

    supportsFloat = bSuccessF;
    supportsDouble = bSuccessD;
    
    return bSuccessF;
}

void CracksCudaFunctions::llCrackFeatures(
             const float * d_coeff, const float* d_theta, float w,
             int num_dirs, int dataLen, float* d_features ) const
{
    hostdrv_pars_t gpuprhs[6];
    int gpunrhs = 6;
    gpuprhs[0] = hostdrv_pars(&d_coeff,sizeof(d_coeff),__alignof(d_coeff));
    gpuprhs[1] = hostdrv_pars(&d_theta,sizeof(d_theta),__alignof(d_theta));
    gpuprhs[2] = hostdrv_pars(&w,sizeof(w),__alignof(w));
    gpuprhs[3] = hostdrv_pars(&num_dirs,sizeof(num_dirs),__alignof(num_dirs));
    gpuprhs[4] = hostdrv_pars(&dataLen,sizeof(dataLen),__alignof(dataLen));
    gpuprhs[5] = hostdrv_pars(&d_features,sizeof(d_features),__alignof(d_features));

    int N = dataLen;

    m_timer->startTimer( CracksGpuTimes::llCrackFeatures );
    hostGPUDRV(cudaLowLevelCrackFeatures, N, gpunrhs, gpuprhs);
    m_timer->stopTimer( CracksGpuTimes::llCrackFeatures );
}

void CracksCudaFunctions::dotProductsFeatures(
    const float* d_features, int fv_dims, int w, int h,
    float* d_sum_fv, float* d_dot_left, float* d_dot_top,
    float* d_dot_topleft, float* d_dot_topright ) const
{
    hostdrv_pars_t gpuprhs[9];
    int gpunrhs = 9;
    gpuprhs[0] = hostdrv_pars(&d_features,sizeof(d_features),__alignof(d_features));
    gpuprhs[1] = hostdrv_pars(&fv_dims,sizeof(fv_dims),__alignof(fv_dims));
    gpuprhs[2] = hostdrv_pars(&w,sizeof(w),__alignof(w));
    gpuprhs[3] = hostdrv_pars(&h,sizeof(h),__alignof(h));
    gpuprhs[4] = hostdrv_pars(&d_sum_fv,sizeof(d_sum_fv),__alignof(d_sum_fv));
    gpuprhs[5] = hostdrv_pars(&d_dot_left,sizeof(d_dot_left),__alignof(d_dot_left));
    gpuprhs[6] = hostdrv_pars(&d_dot_top,sizeof(d_dot_top),__alignof(d_dot_top));
    gpuprhs[7] = hostdrv_pars(&d_dot_topleft,sizeof(d_dot_topleft),__alignof(d_dot_topleft));
    gpuprhs[8] = hostdrv_pars(&d_dot_topright,sizeof(d_dot_topright),__alignof(d_dot_topright));

    int N = w * h;

    m_timer->startTimer( CracksGpuTimes::dotProductsFeatures );
    hostGPUDRV(cudaDotProductsFeatures, N, gpunrhs, gpuprhs);
    m_timer->stopTimer( CracksGpuTimes::dotProductsFeatures );
}

void CracksCudaFunctions::graphAffinitiesFromDotProd(
    const float* d_dot_left, const float* d_dot_top,
    const float* d_dot_topleft, const float* d_dot_topright,
    int* d_graph_left, int* d_graph_right, int* d_graph_top,
    int* d_graph_topleft, int* d_graph_topright, int* d_graph_bottom,
    int* d_graph_bottomleft, int* d_graph_bottomright,
    int w, int h, float gamma, float beta ) const
{
    hostdrv_pars_t gpuprhs[16];
    int gpunrhs = 16;
    gpuprhs[0] = hostdrv_pars(&d_dot_left,sizeof(d_dot_left),__alignof(d_dot_left));
    gpuprhs[1] = hostdrv_pars(&d_dot_top,sizeof(d_dot_top),__alignof(d_dot_top));
    gpuprhs[2] = hostdrv_pars(&d_dot_topleft,sizeof(d_dot_topleft),__alignof(d_dot_topleft));
    gpuprhs[3] = hostdrv_pars(&d_dot_topright,sizeof(d_dot_topright),__alignof(d_dot_topright));
    gpuprhs[4] = hostdrv_pars(&d_graph_left,sizeof(d_graph_left),__alignof(d_graph_left));
    gpuprhs[5] = hostdrv_pars(&d_graph_right,sizeof(d_graph_right),__alignof(d_graph_right));
    gpuprhs[6] = hostdrv_pars(&d_graph_top,sizeof(d_graph_top),__alignof(d_graph_top));
    gpuprhs[7] = hostdrv_pars(&d_graph_topleft,sizeof(d_graph_topleft),__alignof(d_graph_topleft));
    gpuprhs[8] = hostdrv_pars(&d_graph_topright,sizeof(d_graph_topright),__alignof(d_graph_topright));
    gpuprhs[9] = hostdrv_pars(&d_graph_bottom,sizeof(d_graph_bottom),__alignof(d_graph_bottom));
    gpuprhs[10] = hostdrv_pars(&d_graph_bottomleft,sizeof(d_graph_bottomleft),__alignof(d_graph_bottomleft));
    gpuprhs[11] = hostdrv_pars(&d_graph_bottomright,sizeof(d_graph_bottomright),__alignof(d_graph_bottomright));
    gpuprhs[12] = hostdrv_pars(&w,sizeof(w),__alignof(w));
    gpuprhs[13] = hostdrv_pars(&h,sizeof(h),__alignof(h));
    gpuprhs[14] = hostdrv_pars(&gamma,sizeof(gamma),__alignof(gamma));
    gpuprhs[15] = hostdrv_pars(&beta,sizeof(beta),__alignof(beta));

    int N = w * h;

    m_timer->startTimer( CracksGpuTimes::graphAffinitiesFromDotProd );
    hostGPUDRV(cudaGraphAffinitiesFromDotProd, N, gpunrhs, gpuprhs);
    m_timer->stopTimer( CracksGpuTimes::graphAffinitiesFromDotProd );
}

void CracksCudaFunctions::graphTerminals(
    const float* d_sum_fv, int* d_graph_terminals,
    int w, int h, float lambda, float bias ) const
{
    hostdrv_pars_t gpuprhs[4];
    int gpunrhs = 4;
    gpuprhs[0] = hostdrv_pars(&d_sum_fv,sizeof(d_sum_fv),__alignof(d_sum_fv));
    gpuprhs[1] = hostdrv_pars(&d_graph_terminals,sizeof(d_graph_terminals),__alignof(d_graph_terminals));
    gpuprhs[2] = hostdrv_pars(&lambda,sizeof(lambda),__alignof(lambda));
    gpuprhs[3] = hostdrv_pars(&bias,sizeof(bias),__alignof(bias));

    int N = w * h;

    m_timer->startTimer( CracksGpuTimes::graphTerminals );
    hostGPUDRV(cudaGraphTerminals, N, gpunrhs, gpuprhs);
    m_timer->stopTimer( CracksGpuTimes::graphTerminals );
}
