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


#if !defined(KERNELTABLEINIT_H_)
#define KERNELTABLEINIT_H_
status = cuModuleGetFunction(&cuFunction[N_EXPF_KERNEL], cuModule, S_EXPF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (EXPF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_EXPC_KERNEL], cuModule, S_EXPC_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (EXPC_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_EXPD_KERNEL], cuModule, S_EXPD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (EXPD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_EXPCD_KERNEL], cuModule, S_EXPCD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (EXPCD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_SQRTF_KERNEL], cuModule, S_SQRTF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (SQRTF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_SQRTD_KERNEL], cuModule, S_SQRTD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (SQRTD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_LOGF_KERNEL], cuModule, S_LOGF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LOGF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LOGC_KERNEL], cuModule, S_LOGC_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LOGC_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LOGD_KERNEL], cuModule, S_LOGD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LOGD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LOGCD_KERNEL], cuModule, S_LOGCD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LOGCD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_LOG2F_KERNEL], cuModule, S_LOG2F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LOG2F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LOG2D_KERNEL], cuModule, S_LOG2D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LOG2D_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_LOG10F_KERNEL], cuModule, S_LOG10F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LOG10F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LOG10C_KERNEL], cuModule, S_LOG10C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LOG10C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LOG10D_KERNEL], cuModule, S_LOG10D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LOG10D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LOG10CD_KERNEL], cuModule, S_LOG10CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LOG10CD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_LOG1PF_KERNEL], cuModule, S_LOG1PF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LOG1PF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LOG1PD_KERNEL], cuModule, S_LOG1PD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LOG1PD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_SINF_KERNEL], cuModule, S_SINF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (SINF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_SIND_KERNEL], cuModule, S_SIND_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (SIND_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_COSF_KERNEL], cuModule, S_COSF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (COSF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_COSD_KERNEL], cuModule, S_COSD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (COSD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_TANF_KERNEL], cuModule, S_TANF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TANF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_TAND_KERNEL], cuModule, S_TAND_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TAND_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_ASINF_KERNEL], cuModule, S_ASINF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ASINF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_ASIND_KERNEL], cuModule, S_ASIND_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ASIND_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_ACOSF_KERNEL], cuModule, S_ACOSF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ACOSF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_ACOSD_KERNEL], cuModule, S_ACOSD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ACOSD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_ATANF_KERNEL], cuModule, S_ATANF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ATANF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_ATAND_KERNEL], cuModule, S_ATAND_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ATAND_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_SINHF_KERNEL], cuModule, S_SINHF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (SINHF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_SINHD_KERNEL], cuModule, S_SINHD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (SINHD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_COSHF_KERNEL], cuModule, S_COSHF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (COSHF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_COSHD_KERNEL], cuModule, S_COSHD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (COSHD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_TANHF_KERNEL], cuModule, S_TANHF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TANHF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_TANHD_KERNEL], cuModule, S_TANHD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TANHD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_ASINHF_KERNEL], cuModule, S_ASINHF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ASINHF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_ASINHD_KERNEL], cuModule, S_ASINHD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ASINHD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_ACOSHF_KERNEL], cuModule, S_ACOSHF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ACOSHF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_ACOSHD_KERNEL], cuModule, S_ACOSHD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ACOSHD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_ATANHF_KERNEL], cuModule, S_ATANHF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ATANHF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_ATANHD_KERNEL], cuModule, S_ATANHD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ATANHD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_ROUNDF_KERNEL], cuModule, S_ROUNDF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ROUNDF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_ROUNDD_KERNEL], cuModule, S_ROUNDD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ROUNDD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_CEILF_KERNEL], cuModule, S_CEILF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (CEILF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_CEILD_KERNEL], cuModule, S_CEILD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (CEILD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_FLOORF_KERNEL], cuModule, S_FLOORF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FLOORF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_FLOORD_KERNEL], cuModule, S_FLOORD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FLOORD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_ABSF_KERNEL], cuModule, S_ABSF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ABSF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_ABSC_KERNEL], cuModule, S_ABSC_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ABSC_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_ABSD_KERNEL], cuModule, S_ABSD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ABSD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_ABSCD_KERNEL], cuModule, S_ABSCD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ABSCD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_ZEROSF_KERNEL], cuModule, S_ZEROSF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ZEROSF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_ZEROSC_KERNEL], cuModule, S_ZEROSC_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ZEROSC_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_ZEROSD_KERNEL], cuModule, S_ZEROSD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ZEROSD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_ZEROSCD_KERNEL], cuModule, S_ZEROSCD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ZEROSCD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_ONESF_KERNEL], cuModule, S_ONESF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ONESF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_ONESC_KERNEL], cuModule, S_ONESC_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ONESC_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_ONESD_KERNEL], cuModule, S_ONESD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ONESD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_ONESCD_KERNEL], cuModule, S_ONESCD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (ONESCD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_UMINUSF_KERNEL], cuModule, S_UMINUSF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (UMINUSF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_UMINUSC_KERNEL], cuModule, S_UMINUSC_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (UMINUSC_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_UMINUSD_KERNEL], cuModule, S_UMINUSD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (UMINUSD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_UMINUSCD_KERNEL], cuModule, S_UMINUSCD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (UMINUSCD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_CONJUGATEF_KERNEL], cuModule, S_CONJUGATEF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (CONJUGATEF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_CONJUGATEC_KERNEL], cuModule, S_CONJUGATEC_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (CONJUGATEC_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_CONJUGATED_KERNEL], cuModule, S_CONJUGATED_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (CONJUGATED_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_CONJUGATECD_KERNEL], cuModule, S_CONJUGATECD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (CONJUGATECD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_NOTF_KERNEL], cuModule, S_NOTF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (NOTF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_NOTD_KERNEL], cuModule, S_NOTD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (NOTD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_TRANSPOSEF_KERNEL], cuModule, S_TRANSPOSEF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TRANSPOSEF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_TRANSPOSEF_TEX_KERNEL], cuModule, S_TRANSPOSEF_TEX_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TRANSPOSEF_TEX_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_TRANSPOSEC_TEX_KERNEL], cuModule, S_TRANSPOSEC_TEX_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TRANSPOSEC_TEX_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_TRANSPOSED_TEX_KERNEL], cuModule, S_TRANSPOSED_TEX_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TRANSPOSED_TEX_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_TRANSPOSECD_TEX_KERNEL], cuModule, S_TRANSPOSECD_TEX_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TRANSPOSECD_TEX_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_PACKFC2C_KERNEL], cuModule, S_PACKFC2C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PACKFC2C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_PACKDC2C_KERNEL], cuModule, S_PACKDC2C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PACKDC2C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_UNPACKFC2C_KERNEL], cuModule, S_UNPACKFC2C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (UNPACKFC2C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_UNPACKDC2C_KERNEL], cuModule, S_UNPACKDC2C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (UNPACKDC2C_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_SUBSINDEXF_KERNEL], cuModule, S_SUBSINDEXF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (SUBSINDEXF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_SUBSINDEXC_KERNEL], cuModule, S_SUBSINDEXC_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (SUBSINDEXC_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_SUBSINDEXD_KERNEL], cuModule, S_SUBSINDEXD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (SUBSINDEXD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_SUBSINDEXCD_KERNEL], cuModule, S_SUBSINDEXCD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (SUBSINDEXCD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_SUBSINDEX1F_KERNEL], cuModule, S_SUBSINDEX1F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (SUBSINDEX1F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_SUBSINDEX1C_KERNEL], cuModule, S_SUBSINDEX1C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (SUBSINDEX1C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_SUBSINDEX1D_KERNEL], cuModule, S_SUBSINDEX1D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (SUBSINDEX1D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_SUBSINDEX1CD_KERNEL], cuModule, S_SUBSINDEX1CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (SUBSINDEX1CD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_PERMSUBSINDEX1F_KERNEL], cuModule, S_PERMSUBSINDEX1F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PERMSUBSINDEX1F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_PERMSUBSINDEX1C_KERNEL], cuModule, S_PERMSUBSINDEX1C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PERMSUBSINDEX1C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_PERMSUBSINDEX1D_KERNEL], cuModule, S_PERMSUBSINDEX1D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PERMSUBSINDEX1D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_PERMSUBSINDEX1CD_KERNEL], cuModule, S_PERMSUBSINDEX1CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PERMSUBSINDEX1CD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_SUMF_TEX_KERNEL], cuModule, S_SUMF_TEX_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (SUMF_TEX_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_SUMC_TEX_KERNEL], cuModule, S_SUMC_TEX_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (SUMC_TEX_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_SUMD_TEX_KERNEL], cuModule, S_SUMD_TEX_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (SUMD_TEX_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_SUMCD_TEX_KERNEL], cuModule, S_SUMCD_TEX_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (SUMCD_TEX_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_SUMF_KERNEL], cuModule, S_SUMF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (SUMF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_SUM1F_TEX_KERNEL], cuModule, S_SUM1F_TEX_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (SUM1F_TEX_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_COPYMEMORY_KERNEL], cuModule, S_COPYMEMORY_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (COPYMEMORY_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_FILLVECTORF_KERNEL], cuModule, S_FILLVECTORF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FILLVECTORF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_FILLVECTORC_KERNEL], cuModule, S_FILLVECTORC_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FILLVECTORC_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_FILLVECTORD_KERNEL], cuModule, S_FILLVECTORD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FILLVECTORD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_FILLVECTORCD_KERNEL], cuModule, S_FILLVECTORCD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FILLVECTORCD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_FILLVECTOR1F_KERNEL], cuModule, S_FILLVECTOR1F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FILLVECTOR1F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_FILLVECTOR1C_KERNEL], cuModule, S_FILLVECTOR1C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FILLVECTOR1C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_FILLVECTOR1D_KERNEL], cuModule, S_FILLVECTOR1D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FILLVECTOR1D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_FILLVECTOR1CD_KERNEL], cuModule, S_FILLVECTOR1CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FILLVECTOR1CD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_FFTSYMMC_KERNEL], cuModule, S_FFTSYMMC_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FFTSYMMC_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_FFTSYMMCD_KERNEL], cuModule, S_FFTSYMMCD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FFTSYMMCD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_CHECKTEXTURE_KERNEL], cuModule, S_CHECKTEXTURE_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (CHECKTEXTURE_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_REALIMAGF_KERNEL], cuModule, S_REALIMAGF_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (REALIMAGF_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_REALIMAGD_KERNEL], cuModule, S_REALIMAGD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (REALIMAGD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_FLOAT_TO_DOUBLE_KERNEL], cuModule, S_FLOAT_TO_DOUBLE_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FLOAT_TO_DOUBLE_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_FLOAT_TO_INTEGER_KERNEL], cuModule, S_FLOAT_TO_INTEGER_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FLOAT_TO_INTEGER_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_DOUBLE_TO_FLOAT_KERNEL], cuModule, S_DOUBLE_TO_FLOAT_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (DOUBLE_TO_FLOAT_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_DOUBLE_TO_INTEGER_KERNEL], cuModule, S_DOUBLE_TO_INTEGER_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (DOUBLE_TO_INTEGER_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_INTEGER_TO_FLOAT_KERNEL], cuModule, S_INTEGER_TO_FLOAT_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (INTEGER_TO_FLOAT_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_INTEGER_TO_DOUBLE_KERNEL], cuModule, S_INTEGER_TO_DOUBLE_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (INTEGER_TO_DOUBLE_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_PLUS_F_F_KERNEL], cuModule, S_PLUS_F_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PLUS_F_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_PLUS_F_C_KERNEL], cuModule, S_PLUS_F_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PLUS_F_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_PLUS_F_D_KERNEL], cuModule, S_PLUS_F_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PLUS_F_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_PLUS_F_CD_KERNEL], cuModule, S_PLUS_F_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PLUS_F_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_PLUS_C_F_KERNEL], cuModule, S_PLUS_C_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PLUS_C_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_PLUS_C_C_KERNEL], cuModule, S_PLUS_C_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PLUS_C_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_PLUS_C_D_KERNEL], cuModule, S_PLUS_C_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PLUS_C_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_PLUS_C_CD_KERNEL], cuModule, S_PLUS_C_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PLUS_C_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_PLUS_D_F_KERNEL], cuModule, S_PLUS_D_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PLUS_D_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_PLUS_D_C_KERNEL], cuModule, S_PLUS_D_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PLUS_D_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_PLUS_D_D_KERNEL], cuModule, S_PLUS_D_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PLUS_D_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_PLUS_D_CD_KERNEL], cuModule, S_PLUS_D_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PLUS_D_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_PLUS_CD_F_KERNEL], cuModule, S_PLUS_CD_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PLUS_CD_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_PLUS_CD_C_KERNEL], cuModule, S_PLUS_CD_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PLUS_CD_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_PLUS_CD_D_KERNEL], cuModule, S_PLUS_CD_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PLUS_CD_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_PLUS_CD_CD_KERNEL], cuModule, S_PLUS_CD_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (PLUS_CD_CD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_TIMES_F_F_KERNEL], cuModule, S_TIMES_F_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TIMES_F_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_TIMES_F_C_KERNEL], cuModule, S_TIMES_F_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TIMES_F_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_TIMES_F_D_KERNEL], cuModule, S_TIMES_F_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TIMES_F_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_TIMES_F_CD_KERNEL], cuModule, S_TIMES_F_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TIMES_F_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_TIMES_C_F_KERNEL], cuModule, S_TIMES_C_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TIMES_C_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_TIMES_C_C_KERNEL], cuModule, S_TIMES_C_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TIMES_C_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_TIMES_C_D_KERNEL], cuModule, S_TIMES_C_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TIMES_C_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_TIMES_C_CD_KERNEL], cuModule, S_TIMES_C_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TIMES_C_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_TIMES_D_F_KERNEL], cuModule, S_TIMES_D_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TIMES_D_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_TIMES_D_C_KERNEL], cuModule, S_TIMES_D_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TIMES_D_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_TIMES_D_D_KERNEL], cuModule, S_TIMES_D_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TIMES_D_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_TIMES_D_CD_KERNEL], cuModule, S_TIMES_D_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TIMES_D_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_TIMES_CD_F_KERNEL], cuModule, S_TIMES_CD_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TIMES_CD_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_TIMES_CD_C_KERNEL], cuModule, S_TIMES_CD_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TIMES_CD_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_TIMES_CD_D_KERNEL], cuModule, S_TIMES_CD_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TIMES_CD_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_TIMES_CD_CD_KERNEL], cuModule, S_TIMES_CD_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (TIMES_CD_CD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_RDIVIDE_F_F_KERNEL], cuModule, S_RDIVIDE_F_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (RDIVIDE_F_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_RDIVIDE_F_C_KERNEL], cuModule, S_RDIVIDE_F_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (RDIVIDE_F_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_RDIVIDE_F_D_KERNEL], cuModule, S_RDIVIDE_F_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (RDIVIDE_F_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_RDIVIDE_F_CD_KERNEL], cuModule, S_RDIVIDE_F_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (RDIVIDE_F_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_RDIVIDE_C_F_KERNEL], cuModule, S_RDIVIDE_C_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (RDIVIDE_C_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_RDIVIDE_C_C_KERNEL], cuModule, S_RDIVIDE_C_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (RDIVIDE_C_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_RDIVIDE_C_D_KERNEL], cuModule, S_RDIVIDE_C_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (RDIVIDE_C_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_RDIVIDE_C_CD_KERNEL], cuModule, S_RDIVIDE_C_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (RDIVIDE_C_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_RDIVIDE_D_F_KERNEL], cuModule, S_RDIVIDE_D_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (RDIVIDE_D_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_RDIVIDE_D_C_KERNEL], cuModule, S_RDIVIDE_D_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (RDIVIDE_D_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_RDIVIDE_D_D_KERNEL], cuModule, S_RDIVIDE_D_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (RDIVIDE_D_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_RDIVIDE_D_CD_KERNEL], cuModule, S_RDIVIDE_D_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (RDIVIDE_D_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_RDIVIDE_CD_F_KERNEL], cuModule, S_RDIVIDE_CD_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (RDIVIDE_CD_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_RDIVIDE_CD_C_KERNEL], cuModule, S_RDIVIDE_CD_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (RDIVIDE_CD_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_RDIVIDE_CD_D_KERNEL], cuModule, S_RDIVIDE_CD_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (RDIVIDE_CD_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_RDIVIDE_CD_CD_KERNEL], cuModule, S_RDIVIDE_CD_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (RDIVIDE_CD_CD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_LDIVIDE_F_F_KERNEL], cuModule, S_LDIVIDE_F_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LDIVIDE_F_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LDIVIDE_F_C_KERNEL], cuModule, S_LDIVIDE_F_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LDIVIDE_F_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LDIVIDE_F_D_KERNEL], cuModule, S_LDIVIDE_F_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LDIVIDE_F_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LDIVIDE_F_CD_KERNEL], cuModule, S_LDIVIDE_F_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LDIVIDE_F_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LDIVIDE_C_F_KERNEL], cuModule, S_LDIVIDE_C_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LDIVIDE_C_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LDIVIDE_C_C_KERNEL], cuModule, S_LDIVIDE_C_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LDIVIDE_C_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LDIVIDE_C_D_KERNEL], cuModule, S_LDIVIDE_C_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LDIVIDE_C_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LDIVIDE_C_CD_KERNEL], cuModule, S_LDIVIDE_C_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LDIVIDE_C_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LDIVIDE_D_F_KERNEL], cuModule, S_LDIVIDE_D_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LDIVIDE_D_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LDIVIDE_D_C_KERNEL], cuModule, S_LDIVIDE_D_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LDIVIDE_D_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LDIVIDE_D_D_KERNEL], cuModule, S_LDIVIDE_D_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LDIVIDE_D_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LDIVIDE_D_CD_KERNEL], cuModule, S_LDIVIDE_D_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LDIVIDE_D_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LDIVIDE_CD_F_KERNEL], cuModule, S_LDIVIDE_CD_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LDIVIDE_CD_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LDIVIDE_CD_C_KERNEL], cuModule, S_LDIVIDE_CD_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LDIVIDE_CD_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LDIVIDE_CD_D_KERNEL], cuModule, S_LDIVIDE_CD_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LDIVIDE_CD_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LDIVIDE_CD_CD_KERNEL], cuModule, S_LDIVIDE_CD_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LDIVIDE_CD_CD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_POWER_F_F_KERNEL], cuModule, S_POWER_F_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (POWER_F_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_POWER_F_C_KERNEL], cuModule, S_POWER_F_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (POWER_F_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_POWER_F_D_KERNEL], cuModule, S_POWER_F_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (POWER_F_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_POWER_F_CD_KERNEL], cuModule, S_POWER_F_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (POWER_F_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_POWER_C_F_KERNEL], cuModule, S_POWER_C_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (POWER_C_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_POWER_C_C_KERNEL], cuModule, S_POWER_C_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (POWER_C_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_POWER_C_D_KERNEL], cuModule, S_POWER_C_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (POWER_C_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_POWER_C_CD_KERNEL], cuModule, S_POWER_C_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (POWER_C_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_POWER_D_F_KERNEL], cuModule, S_POWER_D_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (POWER_D_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_POWER_D_C_KERNEL], cuModule, S_POWER_D_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (POWER_D_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_POWER_D_D_KERNEL], cuModule, S_POWER_D_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (POWER_D_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_POWER_D_CD_KERNEL], cuModule, S_POWER_D_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (POWER_D_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_POWER_CD_F_KERNEL], cuModule, S_POWER_CD_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (POWER_CD_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_POWER_CD_C_KERNEL], cuModule, S_POWER_CD_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (POWER_CD_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_POWER_CD_D_KERNEL], cuModule, S_POWER_CD_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (POWER_CD_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_POWER_CD_CD_KERNEL], cuModule, S_POWER_CD_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (POWER_CD_CD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_MINUS_F_F_KERNEL], cuModule, S_MINUS_F_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (MINUS_F_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_MINUS_F_C_KERNEL], cuModule, S_MINUS_F_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (MINUS_F_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_MINUS_F_D_KERNEL], cuModule, S_MINUS_F_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (MINUS_F_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_MINUS_F_CD_KERNEL], cuModule, S_MINUS_F_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (MINUS_F_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_MINUS_C_F_KERNEL], cuModule, S_MINUS_C_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (MINUS_C_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_MINUS_C_C_KERNEL], cuModule, S_MINUS_C_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (MINUS_C_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_MINUS_C_D_KERNEL], cuModule, S_MINUS_C_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (MINUS_C_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_MINUS_C_CD_KERNEL], cuModule, S_MINUS_C_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (MINUS_C_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_MINUS_D_F_KERNEL], cuModule, S_MINUS_D_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (MINUS_D_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_MINUS_D_C_KERNEL], cuModule, S_MINUS_D_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (MINUS_D_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_MINUS_D_D_KERNEL], cuModule, S_MINUS_D_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (MINUS_D_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_MINUS_D_CD_KERNEL], cuModule, S_MINUS_D_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (MINUS_D_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_MINUS_CD_F_KERNEL], cuModule, S_MINUS_CD_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (MINUS_CD_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_MINUS_CD_C_KERNEL], cuModule, S_MINUS_CD_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (MINUS_CD_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_MINUS_CD_D_KERNEL], cuModule, S_MINUS_CD_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (MINUS_CD_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_MINUS_CD_CD_KERNEL], cuModule, S_MINUS_CD_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (MINUS_CD_CD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_LT_F_F_KERNEL], cuModule, S_LT_F_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LT_F_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LT_F_C_KERNEL], cuModule, S_LT_F_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LT_F_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LT_F_D_KERNEL], cuModule, S_LT_F_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LT_F_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LT_F_CD_KERNEL], cuModule, S_LT_F_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LT_F_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LT_C_F_KERNEL], cuModule, S_LT_C_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LT_C_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LT_C_C_KERNEL], cuModule, S_LT_C_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LT_C_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LT_C_D_KERNEL], cuModule, S_LT_C_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LT_C_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LT_C_CD_KERNEL], cuModule, S_LT_C_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LT_C_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LT_D_F_KERNEL], cuModule, S_LT_D_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LT_D_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LT_D_C_KERNEL], cuModule, S_LT_D_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LT_D_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LT_D_D_KERNEL], cuModule, S_LT_D_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LT_D_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LT_D_CD_KERNEL], cuModule, S_LT_D_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LT_D_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LT_CD_F_KERNEL], cuModule, S_LT_CD_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LT_CD_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LT_CD_C_KERNEL], cuModule, S_LT_CD_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LT_CD_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LT_CD_D_KERNEL], cuModule, S_LT_CD_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LT_CD_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LT_CD_CD_KERNEL], cuModule, S_LT_CD_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LT_CD_CD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_GT_F_F_KERNEL], cuModule, S_GT_F_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GT_F_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GT_F_C_KERNEL], cuModule, S_GT_F_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GT_F_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GT_F_D_KERNEL], cuModule, S_GT_F_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GT_F_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GT_F_CD_KERNEL], cuModule, S_GT_F_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GT_F_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GT_C_F_KERNEL], cuModule, S_GT_C_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GT_C_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GT_C_C_KERNEL], cuModule, S_GT_C_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GT_C_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GT_C_D_KERNEL], cuModule, S_GT_C_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GT_C_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GT_C_CD_KERNEL], cuModule, S_GT_C_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GT_C_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GT_D_F_KERNEL], cuModule, S_GT_D_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GT_D_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GT_D_C_KERNEL], cuModule, S_GT_D_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GT_D_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GT_D_D_KERNEL], cuModule, S_GT_D_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GT_D_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GT_D_CD_KERNEL], cuModule, S_GT_D_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GT_D_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GT_CD_F_KERNEL], cuModule, S_GT_CD_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GT_CD_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GT_CD_C_KERNEL], cuModule, S_GT_CD_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GT_CD_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GT_CD_D_KERNEL], cuModule, S_GT_CD_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GT_CD_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GT_CD_CD_KERNEL], cuModule, S_GT_CD_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GT_CD_CD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_LE_F_F_KERNEL], cuModule, S_LE_F_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LE_F_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LE_F_C_KERNEL], cuModule, S_LE_F_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LE_F_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LE_F_D_KERNEL], cuModule, S_LE_F_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LE_F_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LE_F_CD_KERNEL], cuModule, S_LE_F_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LE_F_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LE_C_F_KERNEL], cuModule, S_LE_C_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LE_C_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LE_C_C_KERNEL], cuModule, S_LE_C_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LE_C_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LE_C_D_KERNEL], cuModule, S_LE_C_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LE_C_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LE_C_CD_KERNEL], cuModule, S_LE_C_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LE_C_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LE_D_F_KERNEL], cuModule, S_LE_D_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LE_D_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LE_D_C_KERNEL], cuModule, S_LE_D_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LE_D_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LE_D_D_KERNEL], cuModule, S_LE_D_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LE_D_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LE_D_CD_KERNEL], cuModule, S_LE_D_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LE_D_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LE_CD_F_KERNEL], cuModule, S_LE_CD_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LE_CD_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LE_CD_C_KERNEL], cuModule, S_LE_CD_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LE_CD_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LE_CD_D_KERNEL], cuModule, S_LE_CD_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LE_CD_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_LE_CD_CD_KERNEL], cuModule, S_LE_CD_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (LE_CD_CD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_GE_F_F_KERNEL], cuModule, S_GE_F_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GE_F_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GE_F_C_KERNEL], cuModule, S_GE_F_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GE_F_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GE_F_D_KERNEL], cuModule, S_GE_F_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GE_F_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GE_F_CD_KERNEL], cuModule, S_GE_F_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GE_F_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GE_C_F_KERNEL], cuModule, S_GE_C_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GE_C_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GE_C_C_KERNEL], cuModule, S_GE_C_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GE_C_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GE_C_D_KERNEL], cuModule, S_GE_C_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GE_C_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GE_C_CD_KERNEL], cuModule, S_GE_C_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GE_C_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GE_D_F_KERNEL], cuModule, S_GE_D_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GE_D_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GE_D_C_KERNEL], cuModule, S_GE_D_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GE_D_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GE_D_D_KERNEL], cuModule, S_GE_D_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GE_D_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GE_D_CD_KERNEL], cuModule, S_GE_D_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GE_D_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GE_CD_F_KERNEL], cuModule, S_GE_CD_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GE_CD_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GE_CD_C_KERNEL], cuModule, S_GE_CD_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GE_CD_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GE_CD_D_KERNEL], cuModule, S_GE_CD_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GE_CD_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_GE_CD_CD_KERNEL], cuModule, S_GE_CD_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (GE_CD_CD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_NE_F_F_KERNEL], cuModule, S_NE_F_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (NE_F_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_NE_F_C_KERNEL], cuModule, S_NE_F_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (NE_F_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_NE_F_D_KERNEL], cuModule, S_NE_F_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (NE_F_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_NE_F_CD_KERNEL], cuModule, S_NE_F_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (NE_F_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_NE_C_F_KERNEL], cuModule, S_NE_C_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (NE_C_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_NE_C_C_KERNEL], cuModule, S_NE_C_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (NE_C_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_NE_C_D_KERNEL], cuModule, S_NE_C_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (NE_C_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_NE_C_CD_KERNEL], cuModule, S_NE_C_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (NE_C_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_NE_D_F_KERNEL], cuModule, S_NE_D_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (NE_D_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_NE_D_C_KERNEL], cuModule, S_NE_D_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (NE_D_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_NE_D_D_KERNEL], cuModule, S_NE_D_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (NE_D_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_NE_D_CD_KERNEL], cuModule, S_NE_D_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (NE_D_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_NE_CD_F_KERNEL], cuModule, S_NE_CD_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (NE_CD_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_NE_CD_C_KERNEL], cuModule, S_NE_CD_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (NE_CD_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_NE_CD_D_KERNEL], cuModule, S_NE_CD_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (NE_CD_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_NE_CD_CD_KERNEL], cuModule, S_NE_CD_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (NE_CD_CD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_EQ_F_F_KERNEL], cuModule, S_EQ_F_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (EQ_F_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_EQ_F_C_KERNEL], cuModule, S_EQ_F_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (EQ_F_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_EQ_F_D_KERNEL], cuModule, S_EQ_F_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (EQ_F_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_EQ_F_CD_KERNEL], cuModule, S_EQ_F_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (EQ_F_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_EQ_C_F_KERNEL], cuModule, S_EQ_C_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (EQ_C_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_EQ_C_C_KERNEL], cuModule, S_EQ_C_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (EQ_C_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_EQ_C_D_KERNEL], cuModule, S_EQ_C_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (EQ_C_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_EQ_C_CD_KERNEL], cuModule, S_EQ_C_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (EQ_C_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_EQ_D_F_KERNEL], cuModule, S_EQ_D_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (EQ_D_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_EQ_D_C_KERNEL], cuModule, S_EQ_D_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (EQ_D_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_EQ_D_D_KERNEL], cuModule, S_EQ_D_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (EQ_D_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_EQ_D_CD_KERNEL], cuModule, S_EQ_D_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (EQ_D_CD_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_EQ_CD_F_KERNEL], cuModule, S_EQ_CD_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (EQ_CD_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_EQ_CD_C_KERNEL], cuModule, S_EQ_CD_C_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (EQ_CD_C_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_EQ_CD_D_KERNEL], cuModule, S_EQ_CD_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (EQ_CD_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_EQ_CD_CD_KERNEL], cuModule, S_EQ_CD_CD_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (EQ_CD_CD_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_AND_F_F_KERNEL], cuModule, S_AND_F_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (AND_F_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_AND_F_D_KERNEL], cuModule, S_AND_F_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (AND_F_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_AND_D_F_KERNEL], cuModule, S_AND_D_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (AND_D_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_AND_D_D_KERNEL], cuModule, S_AND_D_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (AND_D_D_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_OR_F_F_KERNEL], cuModule, S_OR_F_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (OR_F_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_OR_F_D_KERNEL], cuModule, S_OR_F_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (OR_F_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_OR_D_F_KERNEL], cuModule, S_OR_D_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (OR_D_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_OR_D_D_KERNEL], cuModule, S_OR_D_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (OR_D_D_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_FMAX_F_F_KERNEL], cuModule, S_FMAX_F_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FMAX_F_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_FMAX_F_D_KERNEL], cuModule, S_FMAX_F_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FMAX_F_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_FMAX_D_F_KERNEL], cuModule, S_FMAX_D_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FMAX_D_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_FMAX_D_D_KERNEL], cuModule, S_FMAX_D_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FMAX_D_D_KERNEL)");
//
status = cuModuleGetFunction(&cuFunction[N_FMIN_F_F_KERNEL], cuModule, S_FMIN_F_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FMIN_F_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_FMIN_F_D_KERNEL], cuModule, S_FMIN_F_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FMIN_F_D_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_FMIN_D_F_KERNEL], cuModule, S_FMIN_D_F_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FMIN_D_F_KERNEL)");
status = cuModuleGetFunction(&cuFunction[N_FMIN_D_D_KERNEL], cuModule, S_FMIN_D_D_KERNEL);
if (CUDA_SUCCESS != status)
  mexErrMsgTxt("Error cuModuleGetFunction (FMIN_D_D_KERNEL)");
#endif
