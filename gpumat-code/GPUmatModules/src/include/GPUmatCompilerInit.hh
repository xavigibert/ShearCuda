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

// getCompileMode
gm.comp.getCompileMode = (int (*)(void)) gmat->fun.getFunctionByName("getCompileMode");

// pushGPUtype
gm.comp.pushGPUtype = (void (*)(void *)) gmat->fun.getFunctionByName("compPushGPUtype");

// pushMx
gm.comp.pushMx = (void (*)(const mxArray *)) gmat->fun.getFunctionByName("compPushMx");

// registerInstruction
gm.comp.registerInstruction = (void (*)(char *)) gmat->fun.getFunctionByName("compRegisterInstruction");

// abort
gm.comp.abort = (void (*)(STRINGCONST char *)) gmat->fun.getFunctionByName("compAbort");

// getContextGPUtype
gm.comp.getContextGPUtype = (int (*)(void *)) gmat->fun.getFunctionByName("compGetContextGPUtype");

// getContextMx
gm.comp.getContextMx = (int (*)(void *)) gmat->fun.getFunctionByName("compGetContextMx");

// CreateMxContext
gm.comp.createMxContext = (void (*)(mxArray *)) gmat->fun.getFunctionByName("compCreateMxContext");

// functionStart
gm.comp.functionStart = (void (*)(STRINGCONST char *)) gmat->fun.getFunctionByName("compFunctionStart");

// functionEnd
gm.comp.functionEnd = (void (*)(void )) gmat->fun.getFunctionByName("compFunctionEnd");

// functionSetParamInt
gm.comp.functionSetParamInt = (void (*)(int)) gmat->fun.getFunctionByName("compFunctionSetParamInt");

// functionSetParamFloat
gm.comp.functionSetParamFloat = (void (*)(float)) gmat->fun.getFunctionByName("compFunctionSetParamFloat");

// functionSetParamDouble
gm.comp.functionSetParamDouble = (void (*)(double)) gmat->fun.getFunctionByName("compFunctionSetParamDouble");


// functionSetParamGPUtype
gm.comp.functionSetParamGPUtype = (void (*)(const GPUtype *)) gmat->fun.getFunctionByName("compFunctionSetParamGPUtype");

// functionSetParamMx
gm.comp.functionSetParamMx = (void (*)(const mxArray *)) gmat->fun.getFunctionByName("compFunctionSetParamMx");

// functionSetParamMxMx
gm.comp.functionSetParamMxMx = (void (*)(int, const mxArray *[])) gmat->fun.getFunctionByName("compFunctionSetParamMxMx");


// log
gm.debug.log = (void (*)(STRINGCONST char *, int)) gmat->fun.getFunctionByName("debugLog");

// logPush
gm.debug.logPush = (void (*)(void)) gmat->fun.getFunctionByName("debugLogPush");

// logPop
gm.debug.logPop = (void (*)(void)) gmat->fun.getFunctionByName("debugLogPop");

// reset
gm.debug.reset = (void (*)(void)) gmat->fun.getFunctionByName("debugReset");

// getCompileMode
gm.debug.getDebugMode = (int (*) (void)) gmat->fun.getFunctionByName("getDebugMode");

// setCompileMode
gm.debug.setDebugMode = (void (*) (int)) gmat->fun.getFunctionByName("setDebugMode");

// debugPushInstructionStack
gm.debug.debugPushInstructionStack = (void (*) (int)) gmat->fun.getFunctionByName("debugPushInstructionStack");




