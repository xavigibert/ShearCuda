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

gm.fun.registerFunction("FFT1Drv", (void *) gmGPUopFFT1Drv);
gm.fun.registerFunction("FFT2Drv", (void *) gmGPUopFFT2Drv);
gm.fun.registerFunction("FFT3Drv", (void *) gmGPUopFFT3Drv);
gm.fun.registerFunction("IFFT1Drv", (void *) gmGPUopIFFT1Drv);
gm.fun.registerFunction("IFFT2Drv", (void *) gmGPUopIFFT2Drv);
gm.fun.registerFunction("IFFT3Drv", (void *) gmGPUopIFFT3Drv);
