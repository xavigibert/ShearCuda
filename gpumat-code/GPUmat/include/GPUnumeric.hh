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

#if !defined(GPUNUMERIC_H_)
#define GPUNUMERIC_H_


/*************************************************************************
 * Utilities to check results consistency
 *************************************************************************/
#define NGPUTYPE 5

void checkResult(GPUtype &p, GPUtype &q, GPUtype &r, gpuTYPE_t *type);
void checkResult(GPUtype &p, GPUtype &r, gpuTYPE_t *type);

/* GPUsetKernelTextureA */
GPUmatResult_t GPUsetKernelTextureA(GPUtype &p, CUfunction *drvfun, int nsize);

/* GPUsetKernelTextureB */
GPUmatResult_t GPUsetKernelTextureB(GPUtype &p, CUfunction *drvfun, int nsize);


/* arg1op_drv */
GPUtype*
arg1op_drv(gpuTYPE_t *, GPUtype &p, GPUmatResult_t(*funf)(GPUtype&, GPUtype&));

/* arg1op_common */
GPUmatResult_t arg1op_common(gpuTYPE_t*, GPUtype &p, GPUtype &r, int F_KERNEL, int CF_KERNEL, int D_KERNEL, int CD_KERNEL);

/* arg2op_drv */
//GPUtype*
//arg2op_drv(int rreal, GPUtype &p, GPUtype &q, GPUmatResult_t(*funf)(GPUtype&, GPUtype&, GPUtype&));

/* arg3op_drv2 */
GPUtype*
arg3op_drv(gpuTYPE_t *, GPUtype &p, GPUtype &q, GPUmatResult_t(*funf)(GPUtype&, GPUtype&, GPUtype&));

/* arg3op2_common */
GPUmatResult_t arg3op2_common(gpuTYPE_t *, GPUtype &p,
		GPUtype &q, GPUtype &r,
		int F_F_KERNEL,
		int F_C_KERNEL,
		int F_D_KERNEL,
		int F_CD_KERNEL,
		int C_F_KERNEL,
		int C_C_KERNEL,
		int C_D_KERNEL,
		int C_CD_KERNEL,
		int D_F_KERNEL,
		int D_C_KERNEL,
		int D_D_KERNEL,
		int D_CD_KERNEL,
		int CD_F_KERNEL,
		int CD_C_KERNEL,
		int CD_D_KERNEL,
		int CD_CD_KERNEL
		);



/* fftcommon */
GPUtype *
fftcommon(GPUtype &p, int direction, int batch);

/* fftcommon2 */
GPUtype *
fftcommon2(GPUtype &p, int direction);

/* mtimes */
GPUmatResult_t
mtimes(GPUtype &p, GPUtype &q, GPUtype &r);
GPUtype*
mtimes_drv(GPUtype &p, GPUtype &q);


#endif

