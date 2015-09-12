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

gmGPUtype gmGPUopFFT1Drv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    GPUtype *ptmp =  (GPUtype *) p.ptrCounter->ptr;
    r = GPUopFFTDrv(*(ptmp), 1, CUFFT_FORWARD);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}

gmGPUtype gmGPUopFFT2Drv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    GPUtype *ptmp =  (GPUtype *) p.ptrCounter->ptr;
    r = GPUopFFTDrv(*(ptmp), 2, CUFFT_FORWARD);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}

gmGPUtype gmGPUopFFT3Drv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    GPUtype *ptmp =  (GPUtype *) p.ptrCounter->ptr;
    r = GPUopFFTDrv(*(ptmp), 3, CUFFT_FORWARD);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}

gmGPUtype gmGPUopIFFT1Drv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    GPUtype *ptmp =  (GPUtype *) p.ptrCounter->ptr;
    r = GPUopFFTDrv(*(ptmp), 1, CUFFT_INVERSE);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}

gmGPUtype gmGPUopIFFT2Drv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    GPUtype *ptmp =  (GPUtype *) p.ptrCounter->ptr;
    r = GPUopFFTDrv(*(ptmp), 2, CUFFT_INVERSE);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}

gmGPUtype gmGPUopIFFT3Drv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    GPUtype *ptmp =  (GPUtype *) p.ptrCounter->ptr;
    r = GPUopFFTDrv(*(ptmp), 3, CUFFT_INVERSE);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}

