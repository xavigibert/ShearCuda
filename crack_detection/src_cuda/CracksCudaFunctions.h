/*
 * Author: Xavier Gibert-Serra (gibert@umiacs.umd.edu)
 *
 * Copyright (C) 2013 University of Maryland. All rights reserved.
 *
 */

#include "cuda.h"
#include <string.h>
#include <mex.h>
#include "GPUmat.hh"
#include "CracksGpuTimes.h"

// This class encapsulates function pointers for all kernels used by
// the cracks_cuda module
class CracksCudaFunctions
{
public:
    CracksCudaFunctions();
    
    bool LoadGpuFunctions( const CUmodule *drvmod );
    void setTimer(CracksGpuTimes* gt) { m_timer = gt; }

    bool supportsFloat;
    bool supportsDouble;

    // Function launchers
    // Low-level crack features from Shearlet coefficients
    void llCrackFeatures(
            const float * d_coeff, const float* d_theta, float w,
            int num_dirs, int dataLen, float* d_features ) const;

    void dotProductsFeatures(
        const float* d_features, int fv_dims, int w, int h,
        float* d_sum_fv, float* d_dot_left, float* d_dot_top,
        float* d_dot_topleft, float* d_dot_topright ) const;

    void graphAffinitiesFromDotProd(
        const float* d_dot_left, const float* d_dot_top,
        const float* d_dot_topleft, const float* d_dot_topright,
        int* d_graph_left, int* d_graph_right, int* d_graph_top,
        int* d_graph_topleft, int* d_graph_topright, int* d_graph_bottom,
        int* d_graph_bottomleft, int* d_graph_bottomright,
        int w, int h, float gamma, float beta ) const;

    void graphTerminals(
        const float* d_sum_fv, int* d_graph_terminals,
        int w, int h, float lambda, float bias ) const;

private:
    CUfunction cudaLowLevelCrackFeatures;
    CUfunction cudaDotProductsFeatures;
    CUfunction cudaGraphAffinitiesFromDotProd;
    CUfunction cudaGraphTerminals;

    CracksGpuTimes* m_timer;
};
