/*
 * Author: Xavier Gibert-Serra (gibert@umiacs.umd.edu)
 *
 * Copyright (C) 2013 University of Maryland. All rights reserved.
 *
 */

#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#ifdef UNIX
#include <stdint.h>
#endif

#include "mex.h"

#include <vector>

// Neigborhood definitions in ascending order
static const int s_cdisp_0 = 1;
static const int s_cdisp_1 = 9;
static const int s_cdisp_2 = 25;
static int xdisp[s_cdisp_2] = {0, -1,  0,  1,  0, -1,  1, -1,  1,
                                  -2,  0,  2,  0, -2,  2, -2,  2, -1,  1,  1, -1, -2, -2,  2,  2};
static int ydisp[s_cdisp_2] = {0,  0, -1,  0,  1, -1,  1,  1, -1,
                                   0, -2,  0,  2, -1,  1,  1, -1, -2,  2, -2,  2, -2,  2, -2,  2};

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // This function calculates the detection and false alarm rates for the
    // binary label assignment in "labels". Pixels in "mask_bg" are always
    // considered negative samples. Pixels in the complementary of "mask_bg"
    // are matched with those in "mask_crack" using bipartite graph matching.
    // The detection rate is the ratio between matched pixels to number of
    // nonzero elements in "mask_crack"

    // Check arguments
    if( nlhs != 2 )
        mexErrMsgTxt("Output arguments should be 2");
    if( nrhs != 4 )
        mexErrMsgTxt("Input arguments should be 4");

    const mxArray* mx_labels = prhs[0];
    const mxArray* mx_mask_bg = prhs[1];
    const mxArray* mx_mask_crack = prhs[2];
    int dist = int(mxGetScalar(prhs[3]));

    if( !mxIsLogical(mx_labels) || !mxIsLogical(mx_mask_bg) || !mxIsLogical(mx_mask_crack) )
        mexErrMsgTxt("Input arguments should be of 'logical' class");

    // Get image dimensions
    int rows = mxGetM(mx_labels);
    int cols = mxGetN(mx_labels);
    int imsize = rows * cols;

    if( rows != int(mxGetM(mx_mask_bg)) || cols != int(mxGetN(mx_mask_bg)) ||
        rows != int(mxGetM(mx_mask_crack)) || cols != int(mxGetN(mx_mask_crack)) )
        mexErrMsgTxt("Masks should be of the same dimensions as the input labels");

    mxLogical* labels = (mxLogical*)mxGetData( mx_labels );
    mxLogical* mask_bg = (mxLogical*)mxGetData( mx_mask_bg );
    mxLogical* mask_crack = (mxLogical*)mxGetData( mx_mask_crack );

    //pf = nnz(labels & mask_bg)/nnz(mask_bg);
    int false_neg = 0;
    int neg = 0;
    for( int i = 0; i < imsize; i++ )
    {
        if( mask_bg[i] ) neg++;
        if( labels[i] & mask_bg[i] ) false_neg++;
    }
    double pf = double(false_neg) / double(neg);

    // Define possible displacements
    int cdisp;          // Number of displacements in xdisp and ydisp
    switch( dist )
    {
    case 1:
        cdisp = s_cdisp_1;
        break;
    case 2:
        cdisp = s_cdisp_2;
        break;
    default:
        cdisp = s_cdisp_0;
    }

    // Label pixels to match
    //cracks = labels & ~mask_bg; % Unmatched cracks
    std::vector<bool> cracks( imsize, 0 );     // Image containing unmatched cracks
    //int nnz_cracks = 0;                     // Number of non-zero elements in cracks list
    //std::vector<int> cracks_idx( imsize );  // Indices for non-zero elements in cracks
    for( int i = 0; i < imsize; i++ ) {
        if( labels[i] && !mask_bg[i] )
        {
            cracks[i] = true;
            //cracks_idx[nnz_cracks++] = i;
        }
    }
    //num_gt_cracks = nnz(mask_crack);
    int num_gt_cracks = 0;
    for( int i = 0; i < imsize; i++ )
        if( mask_crack[i] ) num_gt_cracks++;
    //res_crack = mask_crack;     % Crack residual (unmatched gt)
    //std::vector<bool> res_crack( mask_crack, mask_crack + imsize );
    int nnz_res_cracks = 0;                     // Number of non-zero elements in res_cracks list
    std::vector<int> res_cracks_idx( imsize );  // Indices for non-zero elements in res_cracks
    for( int i = 0; i < imsize; i++ )
    {
        if( mask_crack[i] )
            res_cracks_idx[nnz_res_cracks++] = i;
    }
    int num_det_cracks = 0;
    bool changed = true;
    //temp_matched = zeros(nr,nc);
    //std::vector<bool> temp_matched( imsize, 0 );
    while( changed )
    {
        changed = false;
        std::vector<int> sum_crack( nnz_res_cracks );   // Number of matches for each crack in "cracks"
        std::vector<int> idx_match( nnz_res_cracks );   // Index in "crack" of first match
        int min_sum = 0xffff;
        for( int i = 0; i < nnz_res_cracks; i++ )
        {
            int num_matches = 0;
            int base_row = res_cracks_idx[i] % rows;
            int base_col = res_cracks_idx[i] / rows;

            for( int di = 0; di < cdisp; di++ )
            {
                int row = base_row + ydisp[di];
                int col = base_col + xdisp[di];
                if( row >= 0 && row < rows && col >= 0 && col < cols &&
                        cracks[col * rows + row])
                {
                    if( num_matches == 0 )
                            idx_match[i] = col * rows + row;
                    num_matches++;
                }
            }
            if( num_matches > 0 )
            {
                if( num_matches < min_sum ) min_sum = num_matches;
                if( min_sum == 1 )
                {
                    if( num_matches == 1 )
                    {
                        changed = true;
                        cracks[idx_match[i]] = false;    // We no not need to update sum_crack, as the match gets recorded immeditely
                        res_cracks_idx[i] = -1;                 // Mark it as "to be deleted"
                        num_det_cracks++;
                    }
                }
            }
            sum_crack[i] = num_matches;             // We will remove it later only if sum_crack[i]==min_sum
        }
        if( min_sum > 1 && min_sum < 0xffff )
        {
            for( int i = 0; i < nnz_res_cracks; i++ )
            {
                if( sum_crack[i] == min_sum && cracks[idx_match[i]] )
                {
                    changed = true;
                    cracks[idx_match[i]] = false;
                    res_cracks_idx[i] = -1;
                    num_det_cracks++;
                }
                else if( sum_crack[i] == 0 )
                {
                    res_cracks_idx[i] = -1;     // Not matched
                }
            }
        }
        // Compact "cracks" vector by removing already matched elements
        int old_nnz_cracks = nnz_res_cracks;
        nnz_res_cracks = 0;
        for( int i = 0; i < old_nnz_cracks; i++ )
        {
            if( res_cracks_idx[i] >= 0 )
                res_cracks_idx[nnz_res_cracks++] = res_cracks_idx[i];
        }
    }

    // The probability of detection is the ratio of pixels that have been
    // successfully matched
    double pd = double(num_det_cracks) / double(num_gt_cracks);

    plhs[0] = mxCreateDoubleScalar(pd);
    plhs[1] = mxCreateDoubleScalar(pf);

}
