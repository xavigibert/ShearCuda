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

// FFT1Drv
gm.fft.FFT1Drv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("FFT1Drv");

// FFT2Drv
gm.fft.FFT2Drv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("FFT2Drv");

// FFT3Drv
gm.fft.FFT3Drv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("FFT3Drv");

// IFFT1Drv
gm.fft.IFFT1Drv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("IFFT1Drv");

// IFFT2Drv
gm.fft.IFFT2Drv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("IFFT2Drv");

// IFFT3Drv
gm.fft.IFFT3Drv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("IFFT3Drv");


