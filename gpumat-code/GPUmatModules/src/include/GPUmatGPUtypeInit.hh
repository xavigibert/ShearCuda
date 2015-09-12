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

// getType
gm.gputype.getType = (gpuTYPE_t (*)(const GPUtype &)) gmat->fun.getFunctionByName("getType");

// getSize
gm.gputype.getSize = (const int * (*)(const GPUtype &)) gmat->fun.getFunctionByName("getSize");

// getNdims
gm.gputype.getNdims = (int (*)(const GPUtype &)) gmat->fun.getFunctionByName("getNdims");

// getNumel
gm.gputype.getNumel = (int (*)(const GPUtype &)) gmat->fun.getFunctionByName("getNumel");

// getGPUptr
gm.gputype.getGPUptr = (const void * (*)(const GPUtype &)) gmat->fun.getFunctionByName("getGPUptr");

// getDataSize
gm.gputype.getDataSize = (int (*)(const GPUtype &)) gmat->fun.getFunctionByName("getDataSize");

// setSize
gm.gputype.setSize = (void (*)(const GPUtype &p, int n, const int *s)) gmat->fun.getFunctionByName("setSize");

// isScalar
gm.gputype.isScalar = (int (*)(const GPUtype &)) gmat->fun.getFunctionByName("isScalar");

// isComplex
gm.gputype.isComplex = (int (*)(const GPUtype &)) gmat->fun.getFunctionByName("isComplex");

// isEmpty
gm.gputype.isEmpty = (int (*)(const GPUtype &)) gmat->fun.getFunctionByName("isEmpty");

// isFloat
gm.gputype.isFloat = (int (*)(const GPUtype &)) gmat->fun.getFunctionByName("isFloat");

// isDouble
gm.gputype.isDouble = (int (*)(const GPUtype &)) gmat->fun.getFunctionByName("isDouble");

// create
gm.gputype.create = (GPUtype (*)(gpuTYPE_t type, int ndims, const int *size, void * init)) gmat->fun.getFunctionByName("create");

// colon
gm.gputype.colon = (GPUtype (*)(gpuTYPE_t type, double , double, double)) gmat->fun.getFunctionByName("colon");

// clone
gm.gputype.clone = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("clone");

// toMxArray
gm.gputype.toMxArray = (mxArray * (*)(const GPUtype &)) gmat->fun.getFunctionByName("toMxArray");

// createMx
gm.gputype.createMx = (GPUtype (*)(gpuTYPE_t type, int nrhs, const mxArray *prhs[])) gmat->fun.getFunctionByName("createMx");

// createMxArray
gm.gputype.createMxArray = (mxArray * (*)(const GPUtype &)) gmat->fun.getFunctionByName("createMxArray");

// createMxArrayPtr
gm.gputype.createMxArrayPtr = (mxArray * (*)(const mxArray*, const GPUtype &)) gmat->fun.getFunctionByName("createMxArrayPtr");


// mxToGPUtype
gm.gputype.mxToGPUtype = (GPUtype (*)(const mxArray *)) gmat->fun.getFunctionByName("mxToGPUtype");

// getGPUtype
gm.gputype.getGPUtype = (GPUtype (*)(const mxArray *)) gmat->fun.getFunctionByName("getGPUtype");

// fill
gm.gputype.fill = (void (*)(const GPUtype &q, double offset, double incr, int m, int p, int offsetp, int type)) gmat->fun.getFunctionByName("fill");

// slice
gm.gputype.slice = (GPUtype (*)(const GPUtype &p, const Range &r)) gmat->fun.getFunctionByName("slice");

// mxSlice
gm.gputype.mxSlice = (GPUtype (*)(const GPUtype &p, const Range &r)) gmat->fun.getFunctionByName("mxSlice");

// assign
gm.gputype.assign = (void (*)(const GPUtype &p, const GPUtype &q, const Range &r, int dir)) gmat->fun.getFunctionByName("assign");

// mxAssign
gm.gputype.mxAssign = (void (*)(const GPUtype &p, const GPUtype &q, const Range &r, int dir)) gmat->fun.getFunctionByName("mxAssign");

// permute
gm.gputype.permute = (void (*)(const GPUtype &p, const GPUtype &q, const Range &r, int dir, int*perm)) gmat->fun.getFunctionByName("permute");

// mxPermute
gm.gputype.mxPermute = (void (*)(const GPUtype &p, const GPUtype &q, const Range &r, int dir, int*perm)) gmat->fun.getFunctionByName("mxPermute");

// realimag
gm.gputype.realimag = (void (*)(const GPUtype &cpx, const GPUtype &re, const GPUtype &im, int dir, int mode)) gmat->fun.getFunctionByName("realimag");

// floatToDouble
gm.gputype.floatToDouble = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("floatToDouble");

// doubleToFloat
gm.gputype.doubleToFloat = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("doubleToFloat");

// realToComplex
gm.gputype.realToComplex = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("realToComplex");

// realImagToComplex
gm.gputype.realImagToComplex = (GPUtype (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("realImagToComplex");


