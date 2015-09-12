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

#if !defined(CUDALIB_H_)
#define CUDALIB_H_

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#ifndef EXPORTDLL
#ifdef _WIN32
#define EXPORTDLL __declspec(dllexport)
#else
#define EXPORTDLL
#endif
#endif

typedef struct hostdrv_pars {
	hostdrv_pars() {
		par = NULL;
		psize = 0;
		align = __alignof(int);
	}
	void * par;
	unsigned int psize;
	size_t align;
} hostdrv_pars_t;

typedef struct gpukernelconfig {
	gpukernelconfig() {
		maxthreads = 65000;
		blockx = 0;
		blocky = 0;
		blockz = 0;
		// gpuexecute
		// 1 -> run code on GPU
		// 0 -> do not run code on GPU
		//gpuexecute = 1;
	}
	unsigned int maxthreads;
	unsigned int blockx;
	unsigned int blocky;
	unsigned int blockz;
	//unsigned int gpuexecute;

} gpukernelconfig_t;

  /* Prototypes */
  /**************************************************************
  * CUDA Runtime FUNCTIONS
  **************************************************************/


  /**************************************************************
  * KERNELS
  **************************************************************/

  /* mat_CRYPT */
  CUDALIBResult
    mat_CRYPT(const unsigned int N, CUdeviceptr d_idata,  CUdeviceptr d_odata, CUdeviceptr d_ipos, CUfunction *drvfun);

  /* mat_POS */
  CUDALIBResult
    mat_POS(const unsigned int N, const unsigned int M, CUdeviceptr d_ipos, CUfunction *drvfun);


	/* mat_transposef_tex */
	/* mat_transposec_tex */

  CUDALIBResult mat_HOSTDRV_TRANSPOSE(gpukernelconfig *kconf, const unsigned int M, const unsigned int N,
  		CUdeviceptr d_odata, int complex, int mysize, CUfunction *drvfun);

  /* GENERIC KERNELS */
  CUDALIBResult mat_HOSTDRV_A(
  		int N,
  		gpukernelconfig_t *kconf,
  		int nrhs, hostdrv_pars_t *prhs,
  		CUfunction *drvfun);
  /* mat_packfC2C */
  CUDALIBResult
    mat_PACKFC2C(gpukernelconfig *kconf, const unsigned int N, int onlyreal, CUdeviceptr d_re_idata, CUdeviceptr d_im_idata, CUdeviceptr d_odata, CUfunction *drvfun);

  /* mat_unpackfC2C */
  CUDALIBResult
    mat_UNPACKFC2C(gpukernelconfig *kconf, const unsigned int N, int onlyreal, CUdeviceptr d_idata, CUdeviceptr d_re_odata, CUdeviceptr d_im_odata, CUfunction *drvfun );

  /* mat_packC2C */
	CUDALIBResult
		mat_PACKC2C(gpukernelconfig *kconf, const unsigned int maxthreads,
				        const unsigned int N, int onlyreal,
				        CUdeviceptr d_re_idata, unsigned int,
				        CUdeviceptr d_im_idata, unsigned int,
				        CUdeviceptr d_odata, unsigned int,
				        CUfunction *drvfun);

	/* mat_unpackC2C */
	CUDALIBResult
		mat_UNPACKC2C(gpukernelconfig *kconf, const unsigned int maxthreads, const unsigned int N,
				          int onlyreal,
				          CUdeviceptr d_idata, unsigned int,
				          CUdeviceptr d_re_odata, unsigned int,
				          CUdeviceptr d_im_odata, unsigned int,
				          CUfunction *drvfun );


  /* mat_SUBSINDEXF */
  /*EXPORTDLL CUDALIBResult
    mat_SUBSINDEXF(const unsigned int N, CUdeviceptr d_idata, const unsigned int M,  const float idxshift,
    CUdeviceptr d_ix, CUdeviceptr d_odata, CUfunction *drvfun, CUtexref *drvtex);*/

  /* mat_SUBSINDEXF */
 /* EXPORTDLL CUDALIBResult
    mat_SUBSINDEXC(const unsigned int N, CUdeviceptr d_idata, const unsigned int M,  const float idxshift,
    CUdeviceptr d_ix, CUdeviceptr d_odata, CUfunction *drvfun, CUtexref *drvtex);*/

  /* mat_SUBSINDEXF */
  CUDALIBResult mat_SUBSINDEXF(gpukernelconfig *kconf, const unsigned int N, CUdeviceptr d_idata,
    const int idxshift, const unsigned int K, CUdeviceptr d_pars,
  		const unsigned int M, CUdeviceptr d_odata, CUfunction *drvfun,
  		CUtexref *drvtex_pars, CUtexref *drvtex_idata);

  /* mat_SUBSINDEXC */
  CUDALIBResult mat_SUBSINDEXC(gpukernelconfig *kconf, const unsigned int N, CUdeviceptr d_idata,
    const int idxshift, const unsigned int K, CUdeviceptr d_pars,
  		const unsigned int M, CUdeviceptr d_odata, CUfunction *drvfun,
  		CUtexref *drvtex_pars, CUtexref *drvtex_idata);

  /* mat_FTTSYMM */
  CUDALIBResult mat_FFTSYMM(gpukernelconfig *kconf, int M, int N, int Q, CUdeviceptr d_idata,
		CUdeviceptr d_odata, int batch, CUfunction *drvfun,
		CUtexref *drvtex_idata);


  /* mat_SUMF_TEX */
  CUDALIBResult
    mat_SUMF_TEX(gpukernelconfig *kconf, CUdeviceptr d_idata, const unsigned int Nthreads, const unsigned int M,
    const unsigned int GroupSize, const unsigned int GroupOffset, CUdeviceptr d_odata, CUfunction *drvfun, CUtexref *drvtex);

  /* mat_SUM1F_TEX */
   CUDALIBResult
      mat_SUM1F_TEX(gpukernelconfig *kconf, CUdeviceptr d_idata, const unsigned int Nthreads, const unsigned int M,
      const unsigned int GroupSize, const unsigned int GroupOffset, CUdeviceptr d_odata, CUfunction *drvfun, CUtexref *drvtex);

  /* mat_SUMC_TEX */
  CUDALIBResult
    mat_SUMC_TEX(gpukernelconfig *kconf, CUdeviceptr d_idata, const unsigned int Nthreads, const unsigned int M,
    const unsigned int GroupSize, const unsigned int GroupOffset, CUdeviceptr d_odata, CUfunction *drvfun, CUtexref *drvtex);


  /* mat_REALIMAG */
  CUDALIBResult
    mat_REALIMAG(gpukernelconfig *kconf, const unsigned int N, CUdeviceptr data,
		CUdeviceptr re, CUdeviceptr im , int dir,
		int mode, CUfunction *drvfun);


  /**************************************************************
  * UTILITIES KERNELS
  **************************************************************/

  /* mat_CHECKTEXTURE */
  CUDALIBResult
    mat_CHECKTEXTURE(
        gpukernelconfig *kconf,
        const unsigned int N,
        CUdeviceptr d_idata,
        const unsigned int M,
        CUdeviceptr d_odata,
		    const unsigned int offset,
		    CUfunction *drvfun,
        CUtexref *drvtex_idata);

  CUDALIBResult
    mat_LOADMODULE(gpukernelconfig *kconf, CUdevice dv, CUcontext *ctx, CUmodule *md, char *buffer);

  /* mat_FILLVECTORF */
  CUDALIBResult
    mat_FILLVECTORF(gpukernelconfig *kconf, const unsigned int N, const int offs, const float incr, CUdeviceptr d_odata, CUfunction *drvfun);

  /* mat_COPYMEMORY */
  CUDALIBResult
    mat_COPYMEMORY(gpukernelconfig *kconf, const unsigned int N, CUdeviceptr d_idata, CUdeviceptr d_odata, CUfunction *drvfun);


#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* !defined(CUDALIB_H_) */



