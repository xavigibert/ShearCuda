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

char *
/* expf_kernel */
/* expc_kernel */
CGEN_FUN_1D_IN1(expf, float *, float *);
CGEN_FUN_1D_IN1(expc, Complex *, Complex *);

/* sqrtf_kernel */
CGEN_FUN_1D_IN1(sqrtf, float *, float *);

/* logf_kernel */
CGEN_FUN_1D_IN1(logf, float *, float *);

/* log2f_kernel */
CGEN_FUN_1D_IN1(log2f, float *, float *);

/* log10f_kernel */
CGEN_FUN_1D_IN1(log10f, float *, float *);

/* log1pf_kernel */
CGEN_FUN_1D_IN1(log1pf, float *, float *);

/* sinf_kernel */
CGEN_FUN_1D_IN1(sinf, float *, float *);

/* cosf_kernel */
CGEN_FUN_1D_IN1(cosf, float *, float *);

/* tanf_kernel */
CGEN_FUN_1D_IN1(tanf, float *, float *);

/* asinf_kernel */
CGEN_FUN_1D_IN1(asinf, float *, float *);

/* acosf_kernel */
CGEN_FUN_1D_IN1(acosf, float *, float *);

/* atanf_kernel */
CGEN_FUN_1D_IN1(atanf, float *, float *);


/* sinhf_kernel */
CGEN_FUN_1D_IN1(sinhf, float *, float *);

/* coshf_kernel */
CGEN_FUN_1D_IN1(coshf, float *, float *);

/* tanhf_kernel */
CGEN_FUN_1D_IN1(tanhf, float *, float *);

/* asinhf_kernel */
CGEN_FUN_1D_IN1(asinhf, float *, float *);

/* acoshf_kernel */
CGEN_FUN_1D_IN1(acoshf, float *, float *);

/* atanhf_kernel */
CGEN_FUN_1D_IN1(atanhf, float *, float *);


/* truncf_kernel */
//CGEN_FUN_1D_IN1(truncf, float *, float *);

/* roundf_kernel */
CGEN_FUN_1D_IN1(roundf, float *, float *);

/* ceilf_kernel */
CGEN_FUN_1D_IN1(ceilf, float *, float *);

/* floorf_kernel */
CGEN_FUN_1D_IN1(floorf, float *, float *);

/* fabsf_kernel */
CGEN_FUN_1D_IN1(fabsf, float *, float *);
CGEN_FUN_1D_IN1(fabsc, Complex *, Complex *);


/* zerosf_kernel */
CGEN_FUN_1D_IN1(zerosf, float *, float *);

/* onesf_kernel */
CGEN_FUN_1D_IN1(onesf, float *, float *);


/*
* mat_uminusf
* mat_uminusc
*/
CGEN_FUN_1D_IN1(uminusf, float *, float *);
CGEN_FUN_1D_IN1(uminusc, Complex *, Complex *);

/*
* mat_conjugatec
*/
CGEN_FUN_1D_IN1(conjugatec, Complex *, Complex *);

/*
* mat_notf
*/
CGEN_FUN_1D_IN1(notf, float *, float *);

/*
* mat_timesf
* mat_timesc
* mat_timesf_scalar
* mat_timesc_scalar
*/
CGEN_FUN_1D_IN1_IN2(timesf, float *, float *, float *);
CGEN_FUN_1D_IN1_IN2(timesc, Complex *, Complex *, Complex *);
CGEN_FUN_1DSCALAR_IN1_IN2(timesf_scalar, float *, float *, float *);
CGEN_FUN_1DSCALAR_IN1_IN2(timesc_scalar, Complex *, Complex *, Complex *);
/*
* mat_ldividef
* mat_ldividec
* mat_ldividef_scalar
* mat_ldividec_scalar
*/
CGEN_FUN_1D_IN1_IN2(ldividef, float *, float *, float *);
CGEN_FUN_1D_IN1_IN2(ldividec, Complex *, Complex *, Complex *);
CGEN_FUN_1DSCALAR_IN1_IN2(ldividef_scalar, float *, float *, float *);
CGEN_FUN_1DSCALAR_IN1_IN2(ldividec_scalar, Complex *, Complex *, Complex *);

/*
* mat_rdividef
* mat_rdividec
* mat_rdividef_scalar
* mat_rdividec_scalar
*/
CGEN_FUN_1D_IN1_IN2(rdividef, float *, float *, float *);
CGEN_FUN_1D_IN1_IN2(rdividec, Complex *, Complex *, Complex *);
CGEN_FUN_1DSCALAR_IN1_IN2(rdividef_scalar, float *, float *, float *);
CGEN_FUN_1DSCALAR_IN1_IN2(rdividec_scalar, Complex *, Complex *, Complex *);

/*
* mat_plusf
* mat_plusc
* mat_plusf_nocache
* mat_plusf_scalar
* mat_plusc_scalar
*/
CGEN_FUN_1D_IN1_IN2(plusf, float *, float *, float *);
CGEN_FUN_1D_IN1_IN2(plusc, Complex *, Complex *, Complex *);
//CGEN_FUN_1D_IN1_IN2(plusf_nocache, float *, float *, float *);
CGEN_FUN_1DPITCH_IN1_IN2(plusf_pitch, float *, float *, float *);
CGEN_FUN_1D_IN1_IN2_STREAM(plusf, float *, float *, float *);
CGEN_FUN_1DSCALAR_IN1_IN2(plusf_scalar, float *, float *, float *);
CGEN_FUN_1DSCALAR_IN1_IN2(plusc_scalar, Complex *, Complex *, Complex *);
CGEN_FUN_2D_IN1_IN2(plusf_2d, float *, float *, float *);

/*
* mat_powerf
* mat_powerc
* mat_rpowerf_scalar
* mat_lpowerf_scalar
* mat_rpowerc_scalar
* mat_lpowerc_scalar
*/
CGEN_FUN_1D_IN1_IN2(powerf, float *, float *, float *);
CGEN_FUN_1D_IN1_IN2(powerc, Complex *, Complex *, Complex *);
CGEN_FUN_1DSCALAR_IN1_IN2(rpowerf_scalar, float *, float *, float *);
CGEN_FUN_1DSCALAR_IN1_IN2(lpowerf_scalar, float *, float *, float *);
CGEN_FUN_1DSCALAR_IN1_IN2(rpowerc_scalar, Complex *, Complex *, Complex *);
CGEN_FUN_1DSCALAR_IN1_IN2(lpowerc_scalar, Complex *, Complex *, Complex *);

/*
* mat_minusf
* mat_minusc
* mat_rminusf_scalar
* mat_lminusf_scalar
* mat_rminusc_scalar
* mat_lminusc_scalar
*/
CGEN_FUN_1D_IN1_IN2(minusf, float *, float *, float *);
CGEN_FUN_1D_IN1_IN2(minusc, Complex *, Complex *, Complex *);
CGEN_FUN_1DSCALAR_IN1_IN2(rminusf_scalar, float *, float *, float *);
CGEN_FUN_1DSCALAR_IN1_IN2(lminusf_scalar, float *, float *, float *);
CGEN_FUN_1DSCALAR_IN1_IN2(rminusc_scalar, Complex *, Complex *, Complex *);
CGEN_FUN_1DSCALAR_IN1_IN2(lminusc_scalar, Complex *, Complex *, Complex *);

/*
* mat_ltf
* mat_ltf_scalar
*/
CGEN_FUN_1D_IN1_IN2(ltf, float *, float *, float *);
CGEN_FUN_1DSCALAR_IN1_IN2(ltf_scalar, float *, float *, float *);

/*
* mat_gtf
* mat_gtf_scalar
*/
CGEN_FUN_1D_IN1_IN2(gtf, float *, float *, float *);
CGEN_FUN_1DSCALAR_IN1_IN2(gtf_scalar, float *, float *, float *);

/*
* mat_lef
* mat_lef_scalar
*/
CGEN_FUN_1D_IN1_IN2(lef, float *, float *, float *);
CGEN_FUN_1DSCALAR_IN1_IN2(lef_scalar, float *, float *, float *);

/*
* mat_gef
* mat_gef_scalar
*/
CGEN_FUN_1D_IN1_IN2(gef, float *, float *, float *);
CGEN_FUN_1DSCALAR_IN1_IN2(gef_scalar, float *, float *, float *);

/*
* mat_nef
* mat_nec
* mat_nef_scalar
* mat_nec_scalar
*/
CGEN_FUN_1D_IN1_IN2(nef, float *, float *, float *);
CGEN_FUN_1D_IN1_IN2(nec, Complex *, Complex *, Complex *);
CGEN_FUN_1DSCALAR_IN1_IN2(nef_scalar, float *, float *, float *);
CGEN_FUN_1DSCALAR_IN1_IN2(nec_scalar, Complex *, Complex *, Complex *);

/*
* mat_eqf
* mat_eqc
* mat_eqf_scalar
* mat_eqc_scalar
*/
CGEN_FUN_1D_IN1_IN2(eqf, float *, float *, float *);
CGEN_FUN_1D_IN1_IN2(eqc, Complex *, Complex *, Complex *);
CGEN_FUN_1DSCALAR_IN1_IN2(eqf_scalar, float *, float *, float *);
CGEN_FUN_1DSCALAR_IN1_IN2(eqc_scalar, Complex *, Complex *, Complex *);


/*
* mat_andf
* mat_andf_scalar
*/
CGEN_FUN_1D_IN1_IN2(andf, float *, float *, float *);
CGEN_FUN_1DSCALAR_IN1_IN2(andf_scalar, float *, float *, float *);


/*
* mat_orf
* mat_orf_scalar
*/
CGEN_FUN_1D_IN1_IN2(orf, float *, float *, float *);
CGEN_FUN_1DSCALAR_IN1_IN2(orf_scalar, float *, float *, float *);

/*
* mat_fmaxf
* mat_fmaxf_scalar
*/
CGEN_FUN_1D_IN1_IN2(fmaxf, float *, float *, float *);
CGEN_FUN_1DSCALAR_IN1_IN2(fmaxf_scalar, float *, float *, float *);

/*
* mat_fminf
* mat_fminf_scalar
*/
CGEN_FUN_1D_IN1_IN2(fminf, float *, float *, float *);
CGEN_FUN_1DSCALAR_IN1_IN2(fminf_scalar, float *, float *, float *);
