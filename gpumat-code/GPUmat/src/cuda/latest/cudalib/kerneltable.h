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

#if !defined(KERNELTABLE_H_)
#define KERNELTABLE_H_
//cuFunction = (CUfunction **) Mymalloc(325 * sizeof(CUfunction *));
cuFunction[N_EXPF_KERNEL] = NULL;
cuFunction[N_EXPC_KERNEL] = NULL;
cuFunction[N_EXPD_KERNEL] = NULL;
cuFunction[N_EXPCD_KERNEL] = NULL;
//
cuFunction[N_SQRTF_KERNEL] = NULL;
cuFunction[N_SQRTD_KERNEL] = NULL;
//
cuFunction[N_LOGF_KERNEL] = NULL;
cuFunction[N_LOGC_KERNEL] = NULL;
cuFunction[N_LOGD_KERNEL] = NULL;
cuFunction[N_LOGCD_KERNEL] = NULL;
//
cuFunction[N_LOG2F_KERNEL] = NULL;
cuFunction[N_LOG2D_KERNEL] = NULL;
//
cuFunction[N_LOG10F_KERNEL] = NULL;
cuFunction[N_LOG10C_KERNEL] = NULL;
cuFunction[N_LOG10D_KERNEL] = NULL;
cuFunction[N_LOG10CD_KERNEL] = NULL;
//
cuFunction[N_LOG1PF_KERNEL] = NULL;
cuFunction[N_LOG1PD_KERNEL] = NULL;
//
cuFunction[N_SINF_KERNEL] = NULL;
cuFunction[N_SIND_KERNEL] = NULL;
//
cuFunction[N_COSF_KERNEL] = NULL;
cuFunction[N_COSD_KERNEL] = NULL;
//
cuFunction[N_TANF_KERNEL] = NULL;
cuFunction[N_TAND_KERNEL] = NULL;
//
cuFunction[N_ASINF_KERNEL] = NULL;
cuFunction[N_ASIND_KERNEL] = NULL;
//
cuFunction[N_ACOSF_KERNEL] = NULL;
cuFunction[N_ACOSD_KERNEL] = NULL;
//
cuFunction[N_ATANF_KERNEL] = NULL;
cuFunction[N_ATAND_KERNEL] = NULL;
//
cuFunction[N_SINHF_KERNEL] = NULL;
cuFunction[N_SINHD_KERNEL] = NULL;
//
cuFunction[N_COSHF_KERNEL] = NULL;
cuFunction[N_COSHD_KERNEL] = NULL;
//
cuFunction[N_TANHF_KERNEL] = NULL;
cuFunction[N_TANHD_KERNEL] = NULL;
//
cuFunction[N_ASINHF_KERNEL] = NULL;
cuFunction[N_ASINHD_KERNEL] = NULL;
//
cuFunction[N_ACOSHF_KERNEL] = NULL;
cuFunction[N_ACOSHD_KERNEL] = NULL;
//
cuFunction[N_ATANHF_KERNEL] = NULL;
cuFunction[N_ATANHD_KERNEL] = NULL;
//
cuFunction[N_ROUNDF_KERNEL] = NULL;
cuFunction[N_ROUNDD_KERNEL] = NULL;
//
cuFunction[N_CEILF_KERNEL] = NULL;
cuFunction[N_CEILD_KERNEL] = NULL;
//
cuFunction[N_FLOORF_KERNEL] = NULL;
cuFunction[N_FLOORD_KERNEL] = NULL;
//
cuFunction[N_ABSF_KERNEL] = NULL;
cuFunction[N_ABSC_KERNEL] = NULL;
cuFunction[N_ABSD_KERNEL] = NULL;
cuFunction[N_ABSCD_KERNEL] = NULL;
//
cuFunction[N_ZEROSF_KERNEL] = NULL;
cuFunction[N_ZEROSC_KERNEL] = NULL;
cuFunction[N_ZEROSD_KERNEL] = NULL;
cuFunction[N_ZEROSCD_KERNEL] = NULL;
//
cuFunction[N_ONESF_KERNEL] = NULL;
cuFunction[N_ONESC_KERNEL] = NULL;
cuFunction[N_ONESD_KERNEL] = NULL;
cuFunction[N_ONESCD_KERNEL] = NULL;
//
cuFunction[N_UMINUSF_KERNEL] = NULL;
cuFunction[N_UMINUSC_KERNEL] = NULL;
cuFunction[N_UMINUSD_KERNEL] = NULL;
cuFunction[N_UMINUSCD_KERNEL] = NULL;
//
cuFunction[N_CONJUGATEF_KERNEL] = NULL;
cuFunction[N_CONJUGATEC_KERNEL] = NULL;
cuFunction[N_CONJUGATED_KERNEL] = NULL;
cuFunction[N_CONJUGATECD_KERNEL] = NULL;
//
cuFunction[N_NOTF_KERNEL] = NULL;
cuFunction[N_NOTD_KERNEL] = NULL;
//
cuFunction[N_TRANSPOSEF_KERNEL] = NULL;
cuFunction[N_TRANSPOSEF_TEX_KERNEL] = NULL;
cuFunction[N_TRANSPOSEC_TEX_KERNEL] = NULL;
cuFunction[N_TRANSPOSED_TEX_KERNEL] = NULL;
cuFunction[N_TRANSPOSECD_TEX_KERNEL] = NULL;
//
cuFunction[N_PACKFC2C_KERNEL] = NULL;
cuFunction[N_PACKDC2C_KERNEL] = NULL;
cuFunction[N_UNPACKFC2C_KERNEL] = NULL;
cuFunction[N_UNPACKDC2C_KERNEL] = NULL;
//
cuFunction[N_SUBSINDEXF_KERNEL] = NULL;
cuFunction[N_SUBSINDEXC_KERNEL] = NULL;
cuFunction[N_SUBSINDEXD_KERNEL] = NULL;
cuFunction[N_SUBSINDEXCD_KERNEL] = NULL;
//
cuFunction[N_SUBSINDEX1F_KERNEL] = NULL;
cuFunction[N_SUBSINDEX1C_KERNEL] = NULL;
cuFunction[N_SUBSINDEX1D_KERNEL] = NULL;
cuFunction[N_SUBSINDEX1CD_KERNEL] = NULL;
//
cuFunction[N_PERMSUBSINDEX1F_KERNEL] = NULL;
cuFunction[N_PERMSUBSINDEX1C_KERNEL] = NULL;
cuFunction[N_PERMSUBSINDEX1D_KERNEL] = NULL;
cuFunction[N_PERMSUBSINDEX1CD_KERNEL] = NULL;
//
cuFunction[N_SUMF_TEX_KERNEL] = NULL;
cuFunction[N_SUMC_TEX_KERNEL] = NULL;
cuFunction[N_SUMD_TEX_KERNEL] = NULL;
cuFunction[N_SUMCD_TEX_KERNEL] = NULL;
cuFunction[N_SUMF_KERNEL] = NULL;
cuFunction[N_SUM1F_TEX_KERNEL] = NULL;
//
cuFunction[N_COPYMEMORY_KERNEL] = NULL;
//
cuFunction[N_FILLVECTORF_KERNEL] = NULL;
cuFunction[N_FILLVECTORC_KERNEL] = NULL;
cuFunction[N_FILLVECTORD_KERNEL] = NULL;
cuFunction[N_FILLVECTORCD_KERNEL] = NULL;
//
cuFunction[N_FILLVECTOR1F_KERNEL] = NULL;
cuFunction[N_FILLVECTOR1C_KERNEL] = NULL;
cuFunction[N_FILLVECTOR1D_KERNEL] = NULL;
cuFunction[N_FILLVECTOR1CD_KERNEL] = NULL;
//
cuFunction[N_FFTSYMMC_KERNEL] = NULL;
cuFunction[N_FFTSYMMCD_KERNEL] = NULL;
//
cuFunction[N_CHECKTEXTURE_KERNEL] = NULL;
//
cuFunction[N_REALIMAGF_KERNEL] = NULL;
cuFunction[N_REALIMAGD_KERNEL] = NULL;
//
cuFunction[N_FLOAT_TO_DOUBLE_KERNEL] = NULL;
cuFunction[N_FLOAT_TO_INTEGER_KERNEL] = NULL;
cuFunction[N_DOUBLE_TO_FLOAT_KERNEL] = NULL;
cuFunction[N_DOUBLE_TO_INTEGER_KERNEL] = NULL;
cuFunction[N_INTEGER_TO_FLOAT_KERNEL] = NULL;
cuFunction[N_INTEGER_TO_DOUBLE_KERNEL] = NULL;
//
cuFunction[N_PLUS_F_F_KERNEL] = NULL;
cuFunction[N_PLUS_F_C_KERNEL] = NULL;
cuFunction[N_PLUS_F_D_KERNEL] = NULL;
cuFunction[N_PLUS_F_CD_KERNEL] = NULL;
cuFunction[N_PLUS_C_F_KERNEL] = NULL;
cuFunction[N_PLUS_C_C_KERNEL] = NULL;
cuFunction[N_PLUS_C_D_KERNEL] = NULL;
cuFunction[N_PLUS_C_CD_KERNEL] = NULL;
cuFunction[N_PLUS_D_F_KERNEL] = NULL;
cuFunction[N_PLUS_D_C_KERNEL] = NULL;
cuFunction[N_PLUS_D_D_KERNEL] = NULL;
cuFunction[N_PLUS_D_CD_KERNEL] = NULL;
cuFunction[N_PLUS_CD_F_KERNEL] = NULL;
cuFunction[N_PLUS_CD_C_KERNEL] = NULL;
cuFunction[N_PLUS_CD_D_KERNEL] = NULL;
cuFunction[N_PLUS_CD_CD_KERNEL] = NULL;
//
cuFunction[N_TIMES_F_F_KERNEL] = NULL;
cuFunction[N_TIMES_F_C_KERNEL] = NULL;
cuFunction[N_TIMES_F_D_KERNEL] = NULL;
cuFunction[N_TIMES_F_CD_KERNEL] = NULL;
cuFunction[N_TIMES_C_F_KERNEL] = NULL;
cuFunction[N_TIMES_C_C_KERNEL] = NULL;
cuFunction[N_TIMES_C_D_KERNEL] = NULL;
cuFunction[N_TIMES_C_CD_KERNEL] = NULL;
cuFunction[N_TIMES_D_F_KERNEL] = NULL;
cuFunction[N_TIMES_D_C_KERNEL] = NULL;
cuFunction[N_TIMES_D_D_KERNEL] = NULL;
cuFunction[N_TIMES_D_CD_KERNEL] = NULL;
cuFunction[N_TIMES_CD_F_KERNEL] = NULL;
cuFunction[N_TIMES_CD_C_KERNEL] = NULL;
cuFunction[N_TIMES_CD_D_KERNEL] = NULL;
cuFunction[N_TIMES_CD_CD_KERNEL] = NULL;
//
cuFunction[N_RDIVIDE_F_F_KERNEL] = NULL;
cuFunction[N_RDIVIDE_F_C_KERNEL] = NULL;
cuFunction[N_RDIVIDE_F_D_KERNEL] = NULL;
cuFunction[N_RDIVIDE_F_CD_KERNEL] = NULL;
cuFunction[N_RDIVIDE_C_F_KERNEL] = NULL;
cuFunction[N_RDIVIDE_C_C_KERNEL] = NULL;
cuFunction[N_RDIVIDE_C_D_KERNEL] = NULL;
cuFunction[N_RDIVIDE_C_CD_KERNEL] = NULL;
cuFunction[N_RDIVIDE_D_F_KERNEL] = NULL;
cuFunction[N_RDIVIDE_D_C_KERNEL] = NULL;
cuFunction[N_RDIVIDE_D_D_KERNEL] = NULL;
cuFunction[N_RDIVIDE_D_CD_KERNEL] = NULL;
cuFunction[N_RDIVIDE_CD_F_KERNEL] = NULL;
cuFunction[N_RDIVIDE_CD_C_KERNEL] = NULL;
cuFunction[N_RDIVIDE_CD_D_KERNEL] = NULL;
cuFunction[N_RDIVIDE_CD_CD_KERNEL] = NULL;
//
cuFunction[N_LDIVIDE_F_F_KERNEL] = NULL;
cuFunction[N_LDIVIDE_F_C_KERNEL] = NULL;
cuFunction[N_LDIVIDE_F_D_KERNEL] = NULL;
cuFunction[N_LDIVIDE_F_CD_KERNEL] = NULL;
cuFunction[N_LDIVIDE_C_F_KERNEL] = NULL;
cuFunction[N_LDIVIDE_C_C_KERNEL] = NULL;
cuFunction[N_LDIVIDE_C_D_KERNEL] = NULL;
cuFunction[N_LDIVIDE_C_CD_KERNEL] = NULL;
cuFunction[N_LDIVIDE_D_F_KERNEL] = NULL;
cuFunction[N_LDIVIDE_D_C_KERNEL] = NULL;
cuFunction[N_LDIVIDE_D_D_KERNEL] = NULL;
cuFunction[N_LDIVIDE_D_CD_KERNEL] = NULL;
cuFunction[N_LDIVIDE_CD_F_KERNEL] = NULL;
cuFunction[N_LDIVIDE_CD_C_KERNEL] = NULL;
cuFunction[N_LDIVIDE_CD_D_KERNEL] = NULL;
cuFunction[N_LDIVIDE_CD_CD_KERNEL] = NULL;
//
cuFunction[N_POWER_F_F_KERNEL] = NULL;
cuFunction[N_POWER_F_C_KERNEL] = NULL;
cuFunction[N_POWER_F_D_KERNEL] = NULL;
cuFunction[N_POWER_F_CD_KERNEL] = NULL;
cuFunction[N_POWER_C_F_KERNEL] = NULL;
cuFunction[N_POWER_C_C_KERNEL] = NULL;
cuFunction[N_POWER_C_D_KERNEL] = NULL;
cuFunction[N_POWER_C_CD_KERNEL] = NULL;
cuFunction[N_POWER_D_F_KERNEL] = NULL;
cuFunction[N_POWER_D_C_KERNEL] = NULL;
cuFunction[N_POWER_D_D_KERNEL] = NULL;
cuFunction[N_POWER_D_CD_KERNEL] = NULL;
cuFunction[N_POWER_CD_F_KERNEL] = NULL;
cuFunction[N_POWER_CD_C_KERNEL] = NULL;
cuFunction[N_POWER_CD_D_KERNEL] = NULL;
cuFunction[N_POWER_CD_CD_KERNEL] = NULL;
//
cuFunction[N_MINUS_F_F_KERNEL] = NULL;
cuFunction[N_MINUS_F_C_KERNEL] = NULL;
cuFunction[N_MINUS_F_D_KERNEL] = NULL;
cuFunction[N_MINUS_F_CD_KERNEL] = NULL;
cuFunction[N_MINUS_C_F_KERNEL] = NULL;
cuFunction[N_MINUS_C_C_KERNEL] = NULL;
cuFunction[N_MINUS_C_D_KERNEL] = NULL;
cuFunction[N_MINUS_C_CD_KERNEL] = NULL;
cuFunction[N_MINUS_D_F_KERNEL] = NULL;
cuFunction[N_MINUS_D_C_KERNEL] = NULL;
cuFunction[N_MINUS_D_D_KERNEL] = NULL;
cuFunction[N_MINUS_D_CD_KERNEL] = NULL;
cuFunction[N_MINUS_CD_F_KERNEL] = NULL;
cuFunction[N_MINUS_CD_C_KERNEL] = NULL;
cuFunction[N_MINUS_CD_D_KERNEL] = NULL;
cuFunction[N_MINUS_CD_CD_KERNEL] = NULL;
//
cuFunction[N_LT_F_F_KERNEL] = NULL;
cuFunction[N_LT_F_C_KERNEL] = NULL;
cuFunction[N_LT_F_D_KERNEL] = NULL;
cuFunction[N_LT_F_CD_KERNEL] = NULL;
cuFunction[N_LT_C_F_KERNEL] = NULL;
cuFunction[N_LT_C_C_KERNEL] = NULL;
cuFunction[N_LT_C_D_KERNEL] = NULL;
cuFunction[N_LT_C_CD_KERNEL] = NULL;
cuFunction[N_LT_D_F_KERNEL] = NULL;
cuFunction[N_LT_D_C_KERNEL] = NULL;
cuFunction[N_LT_D_D_KERNEL] = NULL;
cuFunction[N_LT_D_CD_KERNEL] = NULL;
cuFunction[N_LT_CD_F_KERNEL] = NULL;
cuFunction[N_LT_CD_C_KERNEL] = NULL;
cuFunction[N_LT_CD_D_KERNEL] = NULL;
cuFunction[N_LT_CD_CD_KERNEL] = NULL;
//
cuFunction[N_GT_F_F_KERNEL] = NULL;
cuFunction[N_GT_F_C_KERNEL] = NULL;
cuFunction[N_GT_F_D_KERNEL] = NULL;
cuFunction[N_GT_F_CD_KERNEL] = NULL;
cuFunction[N_GT_C_F_KERNEL] = NULL;
cuFunction[N_GT_C_C_KERNEL] = NULL;
cuFunction[N_GT_C_D_KERNEL] = NULL;
cuFunction[N_GT_C_CD_KERNEL] = NULL;
cuFunction[N_GT_D_F_KERNEL] = NULL;
cuFunction[N_GT_D_C_KERNEL] = NULL;
cuFunction[N_GT_D_D_KERNEL] = NULL;
cuFunction[N_GT_D_CD_KERNEL] = NULL;
cuFunction[N_GT_CD_F_KERNEL] = NULL;
cuFunction[N_GT_CD_C_KERNEL] = NULL;
cuFunction[N_GT_CD_D_KERNEL] = NULL;
cuFunction[N_GT_CD_CD_KERNEL] = NULL;
//
cuFunction[N_LE_F_F_KERNEL] = NULL;
cuFunction[N_LE_F_C_KERNEL] = NULL;
cuFunction[N_LE_F_D_KERNEL] = NULL;
cuFunction[N_LE_F_CD_KERNEL] = NULL;
cuFunction[N_LE_C_F_KERNEL] = NULL;
cuFunction[N_LE_C_C_KERNEL] = NULL;
cuFunction[N_LE_C_D_KERNEL] = NULL;
cuFunction[N_LE_C_CD_KERNEL] = NULL;
cuFunction[N_LE_D_F_KERNEL] = NULL;
cuFunction[N_LE_D_C_KERNEL] = NULL;
cuFunction[N_LE_D_D_KERNEL] = NULL;
cuFunction[N_LE_D_CD_KERNEL] = NULL;
cuFunction[N_LE_CD_F_KERNEL] = NULL;
cuFunction[N_LE_CD_C_KERNEL] = NULL;
cuFunction[N_LE_CD_D_KERNEL] = NULL;
cuFunction[N_LE_CD_CD_KERNEL] = NULL;
//
cuFunction[N_GE_F_F_KERNEL] = NULL;
cuFunction[N_GE_F_C_KERNEL] = NULL;
cuFunction[N_GE_F_D_KERNEL] = NULL;
cuFunction[N_GE_F_CD_KERNEL] = NULL;
cuFunction[N_GE_C_F_KERNEL] = NULL;
cuFunction[N_GE_C_C_KERNEL] = NULL;
cuFunction[N_GE_C_D_KERNEL] = NULL;
cuFunction[N_GE_C_CD_KERNEL] = NULL;
cuFunction[N_GE_D_F_KERNEL] = NULL;
cuFunction[N_GE_D_C_KERNEL] = NULL;
cuFunction[N_GE_D_D_KERNEL] = NULL;
cuFunction[N_GE_D_CD_KERNEL] = NULL;
cuFunction[N_GE_CD_F_KERNEL] = NULL;
cuFunction[N_GE_CD_C_KERNEL] = NULL;
cuFunction[N_GE_CD_D_KERNEL] = NULL;
cuFunction[N_GE_CD_CD_KERNEL] = NULL;
//
cuFunction[N_NE_F_F_KERNEL] = NULL;
cuFunction[N_NE_F_C_KERNEL] = NULL;
cuFunction[N_NE_F_D_KERNEL] = NULL;
cuFunction[N_NE_F_CD_KERNEL] = NULL;
cuFunction[N_NE_C_F_KERNEL] = NULL;
cuFunction[N_NE_C_C_KERNEL] = NULL;
cuFunction[N_NE_C_D_KERNEL] = NULL;
cuFunction[N_NE_C_CD_KERNEL] = NULL;
cuFunction[N_NE_D_F_KERNEL] = NULL;
cuFunction[N_NE_D_C_KERNEL] = NULL;
cuFunction[N_NE_D_D_KERNEL] = NULL;
cuFunction[N_NE_D_CD_KERNEL] = NULL;
cuFunction[N_NE_CD_F_KERNEL] = NULL;
cuFunction[N_NE_CD_C_KERNEL] = NULL;
cuFunction[N_NE_CD_D_KERNEL] = NULL;
cuFunction[N_NE_CD_CD_KERNEL] = NULL;
//
cuFunction[N_EQ_F_F_KERNEL] = NULL;
cuFunction[N_EQ_F_C_KERNEL] = NULL;
cuFunction[N_EQ_F_D_KERNEL] = NULL;
cuFunction[N_EQ_F_CD_KERNEL] = NULL;
cuFunction[N_EQ_C_F_KERNEL] = NULL;
cuFunction[N_EQ_C_C_KERNEL] = NULL;
cuFunction[N_EQ_C_D_KERNEL] = NULL;
cuFunction[N_EQ_C_CD_KERNEL] = NULL;
cuFunction[N_EQ_D_F_KERNEL] = NULL;
cuFunction[N_EQ_D_C_KERNEL] = NULL;
cuFunction[N_EQ_D_D_KERNEL] = NULL;
cuFunction[N_EQ_D_CD_KERNEL] = NULL;
cuFunction[N_EQ_CD_F_KERNEL] = NULL;
cuFunction[N_EQ_CD_C_KERNEL] = NULL;
cuFunction[N_EQ_CD_D_KERNEL] = NULL;
cuFunction[N_EQ_CD_CD_KERNEL] = NULL;
//
cuFunction[N_AND_F_F_KERNEL] = NULL;
cuFunction[N_AND_F_D_KERNEL] = NULL;
cuFunction[N_AND_D_F_KERNEL] = NULL;
cuFunction[N_AND_D_D_KERNEL] = NULL;
//
cuFunction[N_OR_F_F_KERNEL] = NULL;
cuFunction[N_OR_F_D_KERNEL] = NULL;
cuFunction[N_OR_D_F_KERNEL] = NULL;
cuFunction[N_OR_D_D_KERNEL] = NULL;
//
cuFunction[N_FMAX_F_F_KERNEL] = NULL;
cuFunction[N_FMAX_F_D_KERNEL] = NULL;
cuFunction[N_FMAX_D_F_KERNEL] = NULL;
cuFunction[N_FMAX_D_D_KERNEL] = NULL;
//
cuFunction[N_FMIN_F_F_KERNEL] = NULL;
cuFunction[N_FMIN_F_D_KERNEL] = NULL;
cuFunction[N_FMIN_D_F_KERNEL] = NULL;
cuFunction[N_FMIN_D_D_KERNEL] = NULL;
#endif
