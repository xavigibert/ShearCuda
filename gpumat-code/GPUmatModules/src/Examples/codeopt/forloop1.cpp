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

#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#ifdef UNIX
#include <stdint.h>
#endif
#include "mex.h"
// CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "GPUmat.hh"

/*
 * The C code below is equivalent to the following Matlab script:
 *
 * for i=1:size(kernel_fft_rot,4)
 *   for j=1:size(vis_fft,3)
 *     out=real(ifft2( vis_fft(:,:,j).*kernel_fft_rot(:,:,j,i) ));
 *     C(:,:,i)= C(:,:,i) + out(1:end,1:end);
 *   end
 * end
 *
 * vis_fft        -> NxNxM
 * kernel_fft_rot -> NxNxMxK
 * C              -> NxNxM

end*/

// static paramaters
static CUfunction drvfuns[4];
static int init = 0;
static GPUmat *gm;
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // At least 2 arguments expected
  // Input and result
  if (nrhs!=3)
     mexErrMsgTxt("Wrong number of arguments");
  if (init == 0) {
    // Initialize function
    //mexLock();
    // load GPUmat
    gm = gmGetGPUmat();
    init = 1;
  }
  // mex parameters are:
  // kernel_fft_rot
  // vis_fft
  // C

  GPUtype kernel_fft_rot  = gm->gputype.getGPUtype(prhs[0]);
  GPUtype vis_fft         = gm->gputype.getGPUtype(prhs[1]);
  GPUtype C               = gm->gputype.getGPUtype(prhs[2]);

  // SIZE
  const int *ks = gm->gputype.getSize(kernel_fft_rot);
  const int *vs = gm->gputype.getSize(vis_fft);
  const int *cs = gm->gputype.getSize(C);

  // DIMENSIONS
  int kn = gm->gputype.getNdims(kernel_fft_rot);
  int vn = gm->gputype.getNdims(vis_fft);
  int cn = gm->gputype.getNdims(C);

  // check size
  // first 2 dimensions must agree
  // kn is 4
  // vn is 3
  // cn is 3
  if ((kn!=4)||(vn!=3)||(cn!=3))
    mexErrMsgTxt("Check input parameters dimensions.");

  if ((ks[0]!=vs[0])||(ks[0]!=cs[0])||(ks[1]!=vs[1])||(ks[1]!=cs[1]))
  	mexErrMsgTxt("Check input parameters size.");

  if (ks[2]!=vs[2])
      mexErrMsgTxt("Check input parameters size.");

  if (ks[2]!=cs[2])
        mexErrMsgTxt("Check input parameters size.");

  for (int i=0;i<ks[3];i++) {
  	for (int j=0;j<vs[2];j++) {
  	  GPUtype tmp1 = gm->gputype.slice(vis_fft,
  				                          Range(0,1,END,
  				                          Range(0,1,END,
  				                          Range(j,0,0))));
  		GPUtype tmp2 = gm->gputype.slice(kernel_fft_rot,
  				                              Range(0,1,END,
  				                              Range(0,1,END,
  				                              Range(j,0,0,
  				                              Range(i,0,0)))));
  		gm->numerics.Times(tmp1,tmp2,tmp1);
  		gm->numerics.Real(gm->fft.IFFT2Drv(tmp1),tmp2);

  		tmp1 = gm->gputype.slice(C,
                               Range(0,1,END,
                               Range(0,1,END,
                               Range(j,0,0))));

  		gm->numerics.Plus(tmp1,tmp2,tmp1);
  		gm->gputype.assign(C, tmp1,
                         Range(0,1,END,
                         Range(0,1,END,
                         Range(j,0,0))),1);


  	}
  }



}
