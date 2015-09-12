/*
 * Author: Xavier Gibert-Serra (gibert@umiacs.umd.edu)
 *
 * Copyright (C) 2013 University of Maryland. All rights reserved.
 *
 */

#include "cuda_common.h"
#include "GPUkernel.hh"
#include "complex_helper.h"

//#define BLOCK_DIM1D_LARGE 512
#define SQRT_BLOCK_DIM1D 16

// We support a maximum of 32 directional filters (for now)
#define MAX_DIRS 32

#define LINPOS(row,col,numcols) (((row)*(numcols))+(col))

__device__ inline float sqr(float val)
{
    return val*val;
}

// Feature extraction is done separately at each scale by finding the direction
// with maximum magnitude and then augmenting the magnitude into a 4D vector
// |C| (u(C) cos(\theta), u(C) sin(\theta), u(-C) cos(\theta), u(-C) sin(\theta)
// where u(C) is the step function and C is the value of the coefficient corresponding
// to direction \theta
// Input data:
//    coeff -- pointer to 3D vector (rows x columns x directions) for scale_idx
//    theta -- vector containing the angle theta (in radians) corresponding to each direction
//    w -- weight that multiplies feature vector
//    Nd -- Number of directions
//    dataLen -- Image dimension (w * h)
// Output data:
//    features -- 3D vector (rows x columns x 4)

////////////////////////////////////////////////////////////////////////////////
// Extracts low-level crack features
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void lowLevelCrackFeatures(
    int n,
    int offset,
    const float* d_coeff,
    const float* d_theta,
    float w,
    int Nd,
    int dataLen,
    float* d_features
){
    // Shared memory stores a copy of the data and the angle
    __shared__ float cos_theta[MAX_DIRS];
    __shared__ float sin_theta[MAX_DIRS];
    __shared__ float fv[BLOCK_DIM1D];
    __shared__ int fv_idx[BLOCK_DIM1D];
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    const int i = threadIdx.x;

    // Synchronized loading of the angle information
    if( i < Nd )
    {
        float ang = d_theta[i];
        cos_theta[i] = cos(ang);
        sin_theta[i] = sin(ang);
    }
    __syncthreads();
    
    if( (idx - offset) < n )
    {
        // Initialize features to first direction
        fv_idx[i] = 0;
        const float* ct_ptr = &d_coeff[idx];
        fv[i] = *ct_ptr;

        // Find maximum feature across all directions
        for( int j=1; j < Nd; j++ )
        {
            ct_ptr += dataLen;
            if( fabs(*ct_ptr) > abs(fv[i]) )
            {
                fv[i] = *ct_ptr; 
                fv_idx[i] = j;
            }
        }

        // Now we can generate the features
        float fv_pc = fv[i] > 0.f ? w * fv[i] * cos_theta[fv_idx[i]] : 0.f;
        float fv_ps = fv[i] > 0.f ? w * fv[i] * sin_theta[fv_idx[i]] : 0.f;
        float fv_nc = fv[i] < 0.f ? w * fv[i] * cos_theta[fv_idx[i]] : 0.f;
        float fv_ns = fv[i] < 0.f ? w * fv[i] * sin_theta[fv_idx[i]] : 0.f;
        d_features[idx] = fv_pc;
        d_features[idx + dataLen] = fv_ps;
        d_features[idx + 2*dataLen] = fv_nc;
        d_features[idx + 3*dataLen] = fv_ns;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Extracts dot products within an 8-neighborhood
// This function requires the image size to be a multiple of 16
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void dotProductsFeatures(
    int n,
    int offset,
    const float* d_features,
    int fv_dims,
    int width,
    int height,
    float* d_sum_fv,          // Sum of features
    float* d_dot_left,
    float* d_dot_top,
    float* d_dot_topleft,
    float* d_dot_topright
)
{
    // Shared memory stores a block of data shared by all threads in thread block
    __shared__ float data[SQRT_BLOCK_DIM1D+1][SQRT_BLOCK_DIM1D+1];

    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if( (idx - offset) >= n )
        return;     // Should not happen unless image size assumption is not met

    // Determine row and column blocks to operate on
    const int blockIdx = idx / BLOCK_DIM1D;
    //const int blockNumRows = height / SQRT_BLOCK_DIM1D;
    const int blockNumCols = width / SQRT_BLOCK_DIM1D;
    const int blockRow = blockIdx / blockNumCols;
    const int blockCol = blockIdx % blockNumCols;

    // Determine row and column within the block to operate on
    const int blockOffset =  threadIdx.x;
    const int row = blockOffset / SQRT_BLOCK_DIM1D;
    const int col = blockOffset % SQRT_BLOCK_DIM1D;

    // We can calculate absolute image row and column
    const int absRow = blockRow * SQRT_BLOCK_DIM1D + row;
    const int absCol = blockCol * SQRT_BLOCK_DIM1D + col;
    const int linIdx = absRow * width + absCol;

    // Results
    float sum_fv = 0.f;
    float dot_left = 0.f;
    float dot_top = 0.f;
    float dot_topleft = 0.f;
    float dot_topright = 0.f;

    // Accumulate across all features
    for( int j = 0; j < fv_dims; j++ )
    {
        // Fetch feature data
        data[row+1][col+1] = d_features[linIdx];
        // Fetch borders
        if( row == 0 )
            data[0][col+1] = absRow > 0 ? d_features[linIdx-width] : 0.f;
        if( col == 0 )
            data[row+1][0] = absCol > 0 ? d_features[linIdx-1] : 0.f;
        if( row == 0 && col == 0 )
            data[0][0] = absRow > 0 && absCol > 0 ? d_features[linIdx-width-1] : 0.f;
        __syncthreads();

        // Update partial dot products
        //sum_fv       += data[row+1][col+1];
        // TEST CODE
        if( j == 2 || j == 3 || j == 6 || j ==7 )
            sum_fv +=  data[row+1][col+1] * data[row+1][col+1];
        // END OF TEST CODE
//        dot_left     += data[row+1][col] * data[row+1][col+1];
//        dot_top      += data[row][col+1] * data[row+1][col+1];
//        dot_topleft  += data[row][col]   * data[row+1][col+1];
//        dot_topright += data[row+1][col] * data[row][col+1];        // Trick: we need to shift result by one pixel to the left
        dot_left     += sqr(data[row+1][col] - data[row+1][col+1]);
        dot_top      += sqr(data[row][col+1] - data[row+1][col+1]);
        dot_topleft  += sqr(data[row][col]   - data[row+1][col+1]);
        dot_topright += sqr(data[row+1][col] - data[row][col+1]);        // Trick: we need to shift result by one pixel to the left

        // Update feature pointer to next feature
        d_features += width * height;
    }

    // Save results
    d_sum_fv[linIdx] = -sum_fv;
    // TEST CODE
    //d_sum_fv[linIdx] = -sqrt(sum_fv);
    // END OF TEST CODE
    d_dot_left[linIdx] = dot_left;
    d_dot_top[linIdx] = dot_top;
    d_dot_topleft[linIdx] = dot_topleft;
    if( absCol > 0 )
        d_dot_topright[linIdx-1] = dot_topright;
    else
        d_dot_topright[linIdx-1+width] = dot_topright;
}

////////////////////////////////////////////////////////////////////////////////
// Generates graph structure from dot products within an 8-neighborhood
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void graphAffinitiesFromDotProd(
    int n,
    int offset,
    const float* d_dot_left,        // Dot products
    const float* d_dot_top,
    const float* d_dot_topleft,
    const float* d_dot_topright,
    int* d_graph_left,              // Graph affinities
    int* d_graph_right,
    int* d_graph_top,
    int* d_graph_topleft,
    int* d_graph_topright,
    int* d_graph_bottom,
    int* d_graph_bottomleft,
    int* d_graph_bottomright,
    int width,
    int height,
    float gamma,
    float beta
)
{
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if( (idx - offset) >= n )
        return;     // Should not happen unless image size assumption is not met
    // Determine coordinates to operate on
    const int row = idx / width;
    const int col = idx % width;

    float gamma_over_sqrt2 = gamma / sqrt(2.f);
    int val;

    // Left-right weights
    val = col > 0 ? int(gamma * exp(-beta * d_dot_left[idx])) : 0;
    d_graph_left[idx] = val;
    if( col > 0 )
        d_graph_right[idx-1] = val;
    else    // col == 0
        d_graph_right[idx-1+width] = 0;

    // Top-bottom weights
    val = row > 0 ? int(gamma * exp(-beta * d_dot_top[idx])) : 0;
    d_graph_top[idx] = val;
    if( row > 0 )
        d_graph_bottom[idx-width] = val;
    else    // row == 0
        d_graph_bottom[idx+(height-1)*width] = 0;

    // Top/Left-Bottom/Right weights
    val = (col > 0 && row > 0) ? int(gamma_over_sqrt2 * exp(-beta * d_dot_topleft[idx])) : 0;
    d_graph_topleft[idx] = val;
    if( col > 0 && row > 0 )
        d_graph_bottomright[idx-1-width] = val;
    else
        d_graph_bottomright[(height-1-row)*width+(width-1-col)] = 0;

    // Top/Right-Bottom/Left weights
    val = (col < width-1 && row > 0) ? int(gamma_over_sqrt2 * exp(-beta * d_dot_topright[idx])) : 0;
    d_graph_topright[idx] = val;
    if( col < width-1 && row > 0)
        d_graph_bottomleft[idx+1-width] = val;
    else                    // row == 0
        d_graph_bottomleft[(height-1-row)*width+(width-1-col)] = 0;
}

////////////////////////////////////////////////////////////////////////////////
// Generates graph terminals from sum of features
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void graphTerminals(
    int n,
    int offset,
    const float* d_sum_fv,          // Sum of features
    int* d_graph_terminals,
    float lambda,
    float bias
)
{
    const unsigned int idx = BLOCK_DIM1D * blockIdx.x + threadIdx.x + offset;
    if( (idx - offset) >= n )
        return;     // Should not happen unless image size assumption is not met

    d_graph_terminals[idx] = -int(lambda * d_sum_fv[idx] + bias);
}
