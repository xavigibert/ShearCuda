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


#if !defined(MATLABTEMPLATES_H_)
#define MATLABTEMPLATES_H_

/*************************************************************************
* MATEXPLIKE
*************************************************************************/
// I divide this template in two parts. The first part is error checking, the second part
// is execution. I separate so I can add custom checks in the check part

// EXPONENTIAL LIKE TEMPLATE

#define MATEXPLIKEPART1\
  mxArray *lhs[2];\
  GPUtype * p; \
  if (nrhs != 1) \
  mexErrMsgTxt("Wrong number of arguments"); \
  if (mxIsClass(prhs[0], "GPUsingle")) {\
  mexCallMATLAB(1, &lhs[0], 1, (mxArray**) &prhs[0], "struct");\
  p = objToStruct(lhs[0]);\
  mxDestroyArray(lhs[0]);\
  } else {\
  mexErrMsgTxt("Wrong argument. Expected a GPUsingle.");\
  }\
  GPUtype *r;\

#define MATEXPLIKEPART2F(FUNF)\
  try {\
  if ((!p->isComplex())) {\
  r = unaryop_common(*p, (GPUmatResult_t (*)( int, void*,void*,GPUmanager*)) FUNF);\
  }\
  } catch (GPUexception ex) {\
  mexErrMsgTxt(ex.getError());\
  }\

#define MATEXPLIKEPART2C(FUNC)\
  try {\
  if (p->isComplex()) {\
  r = unaryop_common(*p, (GPUmatResult_t (*)( int, void*,void*,GPUmanager*)) FUNC);\
  }\
  } catch (GPUexception ex) {\
  mexErrMsgTxt(ex.getError());\
  }\


//#define MATEXPLIKEPART3\
//  mxArray *tmpr = toMxStruct(r);\
//  mexCallMATLAB(1, plhs, 1, &tmpr, "GPUsingle");\
//  mxDestroyArray(tmpr);\

#define MATEXPLIKEPART3\
  plhs[0] = toMx(r);\

/*************************************************************************
* MATCHECKINPUT
*************************************************************************/
// the passed element should be a GPUsingle
#define MATCHECKINPUT(x) \
  if (!(mxIsClass(x, "GPUsingle")) && !(mxIsClass(x, "GPUdouble")) && !(mxIsClass(x, "GPUint32"))) \
  mexErrMsgTxt(ERROR_EXPECTED_GPUTYPE);\

/*************************************************************************
* MAT_INPUT_TO_GPUTYPE1
*************************************************************************/
// convert GPusingle to struct and then get the pointer
#define MAT_INPUT_TO_GPUTYPE1 \
  mexCallMATLAB(1, &lhs[0], 1, (mxArray**) &prhs[0], "struct");\
  GPUtype * p = objToStruct(lhs[0]);\
  mxDestroyArray(lhs[0]);\

/*************************************************************************
* MAT_INPUT_TO_GPUTYPE2
*************************************************************************/
// convert GPusingle to struct and then get the pointer
#define MAT_INPUT_TO_GPUTYPE2 \
  mexCallMATLAB(1, &lhs[0], 1, (mxArray**) &prhs[1], "struct");\
  GPUtype * q = objToStruct(lhs[0]);\
  mxDestroyArray(lhs[0]);\

/*************************************************************************
* MAT_INPUT_TO_GPUTYPE3
*************************************************************************/
// convert GPusingle to struct and then get the pointer
#define MAT_INPUT_TO_GPUTYPE3 \
  mexCallMATLAB(1, &lhs[0], 1, (mxArray**) &prhs[2], "struct");\
  GPUtype * r = objToStruct(lhs[0]);\
  mxDestroyArray(lhs[0]);\

/*************************************************************************
* MATCHECKINPUT1
*************************************************************************/
#define MATCHECKINPUT1 \
  gpuTYPE_t ptype = p->getType();\
  gpuTYPE_t qtype = q->getType();\
  int pel = p->getNumel();\
  int qel = q->getNumel();\
  if (ptype!=qtype) {\
  mexErrMsgTxt("Input variables should be of the same type");\
  }\
  if (pel!=qel) {\
  mexErrMsgTxt("Input variables should have the same number of elements");\
  }\

/*************************************************************************
* MATCHECKINPUT2
*************************************************************************/
#define MATCHECKINPUT2 \
  gpuTYPE_t rtype = r->getType();\
  gpuTYPE_t qtype = q->getType();\
  gpuTYPE_t ptype = p->getType();\
  int rel = r->getNumel();\
  int qel = q->getNumel();\
  int pel = p->getNumel();\
  if ((qel!=1)&&(rel!=qel)) {\
  mexErrMsgTxt("Input variables should have the same number of elements");\
  }\
  if ((pel!=1)&&(rel!=pel)) {\
  mexErrMsgTxt("Input variables should have the same number of elements");\
  }\
  if ((pel==1)&&(qel==1)&&(rel!=pel)) {\
  mexErrMsgTxt("Input variables should have the same number of elements");\
  }\


/*************************************************************************
* MAT_INPUT_TO_GPUTYPE2
*************************************************************************/
#define MAT_INPUT_TO_GPUTYPE2_REAL \
  if (p->isComplex())\
  mexErrMsgTxt(ERROR_ARG_REAL);\
  if (q->isComplex())\
  mexErrMsgTxt(ERROR_ARG_REAL);\

#define MAT_INPUT_TO_GPUTYPE2_A \
  mxArray *lhs[2];\
  GPUtype * p;\
  GPUtype * q;\
  \
  \
  GPUmanager * GPUman;\
  opTYPE_t op1 = opNOTDEF;\
  opTYPE_t op2 = opNOTDEF;\
  int cudacap;\
  \
  if (mxIsClass(prhs[0], "GPUsingle") || mxIsClass(prhs[0], "GPUdouble")) {\
  op1 = opGPUtype;\
  mexCallMATLAB(1, &lhs[0], 1, (mxArray**) &prhs[0], "struct");\
  p = objToStruct(lhs[0]);\
  mxDestroyArray(lhs[0]);\
  GPUman = p->getGPUmanager();\
  cudacap = GPUman->getCudaCapability();\
  }\
  if (mxIsClass(prhs[1], "GPUsingle") || mxIsClass(prhs[1], "GPUdouble")) {\
  op2 = opGPUtype;\
  mexCallMATLAB(1, &lhs[1], 1, (mxArray**) &prhs[1], "struct");\
  q = objToStruct(lhs[1]);\
  mxDestroyArray(lhs[1]);\
  GPUman = q->getGPUmanager();\
  cudacap = GPUman->getCudaCapability();\
  }\
  \
  if (op1 == opGPUtype || op2 == opGPUtype) {\
  } else {\
  mexErrMsgTxt(ERROR_EXPECTED_GPUTYPE);\
  }\
  \

#define MAT_INPUT_TO_GPUTYPE2_B \
  if (!(mxIsClass(prhs[0], "GPUsingle")) && !(mxIsClass(prhs[0], "GPUdouble"))) {\
  if (\
  (mxGetClassID(prhs[0]) == mxUNKNOWN_CLASS)||\
  (mxGetClassID(prhs[0]) == mxCELL_CLASS)||\
  (mxGetClassID(prhs[0]) == mxSTRUCT_CLASS)||\
  (mxGetClassID(prhs[0]) == mxCHAR_CLASS)||\
  (mxGetClassID(prhs[0]) == mxFUNCTION_CLASS)) {\
  mexErrMsgTxt(ERROR_NON_SUPPORTED_TYPE);\
    } else {\
    if (mxIsSingle(prhs[0])) {\
    if (mxGetNumberOfElements(prhs[0]) != 1) {\
    mexErrMsgTxt(ERROR_ONLY_MATLAB_SCALARS);\
    } else {\
    if (mxIsComplex(prhs[0])) {\
    Complex opCF;\
    float *tmp = (float *) mxGetPr(prhs[0]);\
    opCF.x = tmp[0];\
    tmp = (float *) mxGetPi(prhs[0]);\
    opCF.y = tmp[0];\
    op1 = opCFLOAT;\
    p = new GPUtype(opCF, GPUman);\
    mgc.setPtr(p);\
    \
    } else {\
    float opF;\
    float *tmp = (float *) mxGetPr(prhs[0]);\
    opF = tmp[0];\
    op1 = opFLOAT;\
    p = new GPUtype(opF, GPUman);\
    mgc.setPtr(p);\
    }\
    }\
    }	else if (mxIsDouble(prhs[0])) {\
    if (mxGetNumberOfElements(prhs[0]) != 1) {\
    mexErrMsgTxt(ERROR_ONLY_MATLAB_SCALARS);\
    } else {\
    if (mxIsComplex(prhs[0])) {\
    DoubleComplex opCD;\
    Complex opCF;\
    double *tmp = mxGetPr(prhs[0]);\
    opCD.x = (double) tmp[0];\
    opCF.x = (float) tmp[0];\
    tmp = mxGetPi(prhs[0]);\
    opCD.y = (double) tmp[0];\
    opCF.y = (float) tmp[0];\
    if (cudacap>=13) {\
    op1 = opCDOUBLE;\
    p = new GPUtype(opCD, GPUman);\
    } else {\
    op1 = opCFLOAT;\
    p = new GPUtype(opCF, GPUman);\
    }\
    mgc.setPtr(p);\
    } else {\
    double opD;\
    float opF;\
    double *tmp = (double *) mxGetPr(prhs[0]);\
    opD = (double) tmp[0];\
    opF = (float) tmp[0];\
    if (cudacap>=13) {\
    op1 = opDOUBLE;\
    p = new GPUtype(opD, GPUman);\
    } else {\
    op1 = opFLOAT;\
    p = new GPUtype(opF, GPUman);\
    }\
    mgc.setPtr(p);\
    }\
    }\
    } else {\
    mexErrMsgTxt(ERROR_NON_SUPPORTED_TYPE);\
      }\
    }\
  }\
  \
  if (!(mxIsClass(prhs[1], "GPUsingle")) && !(mxIsClass(prhs[1], "GPUdouble"))) {\
  if (\
  (mxGetClassID(prhs[1]) == mxUNKNOWN_CLASS)||\
  (mxGetClassID(prhs[1]) == mxCELL_CLASS)||\
  (mxGetClassID(prhs[1]) == mxSTRUCT_CLASS)||\
  (mxGetClassID(prhs[1]) == mxCHAR_CLASS)||\
  (mxGetClassID(prhs[1]) == mxFUNCTION_CLASS)) {\
  mexErrMsgTxt(ERROR_NON_SUPPORTED_TYPE);\
    } else {\
    if (mxIsSingle(prhs[1])) {\
    if (mxGetNumberOfElements(prhs[1]) != 1) {\
    mexErrMsgTxt(ERROR_ONLY_MATLAB_SCALARS);\
    } else {\
    if (mxIsComplex(prhs[1])) {\
    Complex opCF;\
    float *tmp = (float *) mxGetPr(prhs[1]);\
    opCF.x = tmp[0];\
    tmp = (float *) mxGetPi(prhs[1]);\
    opCF.y = tmp[0];\
    op2 = opCFLOAT;\
    q = new GPUtype(opCF, GPUman);\
    mgc.setPtr(q);\
    } else {\
    float opF;\
    float *tmp = (float *) mxGetPr(prhs[1]);\
    opF = tmp[0];\
    op2 = opFLOAT;\
    q = new GPUtype(opF, GPUman);\
    mgc.setPtr(q);\
    \
    }\
    }\
    }	else if (mxIsDouble(prhs[1])) {\
    if (mxGetNumberOfElements(prhs[1]) != 1) {\
    mexErrMsgTxt(ERROR_ONLY_MATLAB_SCALARS);\
    } else {\
    if (mxIsComplex(prhs[1])) {\
    DoubleComplex opCD;\
    Complex opCF;\
    double *tmp = mxGetPr(prhs[1]);\
    opCD.x = (double) tmp[0];\
    opCF.x = (float) tmp[0];\
    tmp = mxGetPi(prhs[1]);\
    opCD.y = (double) tmp[0];\
    opCF.y = (float) tmp[0];\
    if (cudacap>=13) {\
    op2 = opCDOUBLE;\
    q = new GPUtype(opCD, GPUman);\
    } else {\
    op2 = opCFLOAT;\
    q = new GPUtype(opCF, GPUman);\
    }\
    mgc.setPtr(q);\
    \
    } else {\
    double opD;\
    float opF;\
    double *tmp = (double *) mxGetPr(prhs[1]);\
    opD = (double) tmp[0];\
    opF = (float) tmp[0];\
    if (cudacap>=13) {\
    op2 = opDOUBLE;\
    q = new GPUtype(opD, GPUman);\
    } else {\
    op2 = opFLOAT;\
    q = new GPUtype(opF, GPUman);\
    }\
    mgc.setPtr(q);\
    \
    }\
    }\
    } else {\
    mexErrMsgTxt(ERROR_NON_SUPPORTED_TYPE);\
      }\
    }\
  }\

/*************************************************************************
* MATPLUSLIKE
*************************************************************************/

// PLUS LIKE TEMPLATE

#define MATPLUSLIKEPART1\
  mxArray *lhs[2];\
  Complex op1C;\
  float op1F;\
  Complex op2C;\
  float op2F;\
  GPUtype * p;\
  GPUtype * q;\
  if (nrhs != 2)\
  mexErrMsgTxt(ERROR_WRONG_NUMBER_ARGS);\
  opTYPE_t op1 = opGPUsingle;\
  opTYPE_t op2 = opGPUsingle;\
  if (mxIsClass(prhs[0], "GPUsingle")) {\
  op1 = opGPUsingle;\
  mexCallMATLAB(1, &lhs[0], 1, (mxArray**) &prhs[0], "struct");\
  p = objToStruct(lhs[0]);\
  mxDestroyArray(lhs[0]);\
  if (p->getNumel() == 1) {\
  mexErrMsgTxt(\
  ERROR_FIRST_SCALAR);\
  }\
  }\
  if (mxIsClass(prhs[1], "GPUsingle")) {\
  op2 = opGPUsingle;\
  mexCallMATLAB(1, &lhs[1], 1, (mxArray**) &prhs[1], "struct");\
  q = objToStruct(lhs[1]);\
  mxDestroyArray(lhs[1]);\
  if (q->getNumel() == 1) {\
  mexErrMsgTxt(\
  ERROR_SECOND_SCALAR);\
  }\
  }\
  if (op1 == opGPUsingle || op2 == opGPUsingle) {\
  } else {\
  mexErrMsgTxt(ERROR_EXPECTED_GPUSINGLE);\
  }\
  if (!(mxIsClass(prhs[0], "GPUsingle"))) {\
  if (\
  (mxGetClassID(prhs[0]) == mxUNKNOWN_CLASS)||\
  (mxGetClassID(prhs[0]) == mxCELL_CLASS)||\
  (mxGetClassID(prhs[0]) == mxSTRUCT_CLASS)||\
  (mxGetClassID(prhs[0]) == mxCHAR_CLASS)||\
  (mxGetClassID(prhs[0]) == mxFUNCTION_CLASS)) {\
  mexErrMsgTxt(ERROR_NON_SUPPORTED_TYPE);\
    } else {\
    if (mxIsSingle(prhs[0])) {\
    if (mxGetNumberOfElements(prhs[0]) != 1) {\
    mexErrMsgTxt(ERROR_ONLY_MATLAB_SCALARS);\
    } else {\
    if (mxIsComplex(prhs[0])) {\
    op1 = opCFLOAT;\
    float *tmp = (float *) mxGetPr(prhs[0]);\
    op1C.x = tmp[0];\
    tmp = (float *) mxGetPi(prhs[0]);\
    op1C.y = tmp[0];\
    } else {\
    op1 = opFLOAT;\
    float *tmp = (float *) mxGetPr(prhs[0]);\
    op1F = tmp[0];\
    }\
    }\
    }	else if (mxIsDouble(prhs[0])) {\
    if (mxGetNumberOfElements(prhs[0]) != 1) {\
    mexErrMsgTxt(ERROR_ONLY_MATLAB_SCALARS);\
    } else {\
    if (mxIsComplex(prhs[0])) {\
    op1 = opCFLOAT;\
    double *tmp = mxGetPr(prhs[0]);\
    op1C.x = (float) tmp[0];\
    tmp = mxGetPi(prhs[0]);\
    op1C.y = (float) tmp[0];\
    } else {\
    op1 = opFLOAT;\
    double *tmp = (double *) mxGetPr(prhs[0]);\
    op1F = (float) tmp[0];\
    }\
    }\
    } else { \
    mexErrMsgTxt(ERROR_NON_SUPPORTED_TYPE);\
      }\
    }\
  }\
  if (!(mxIsClass(prhs[1], "GPUsingle"))) {\
  if (\
  (mxGetClassID(prhs[1]) == mxUNKNOWN_CLASS)||\
  (mxGetClassID(prhs[1]) == mxCELL_CLASS)||\
  (mxGetClassID(prhs[1]) == mxSTRUCT_CLASS)||\
  (mxGetClassID(prhs[1]) == mxCHAR_CLASS)||\
  (mxGetClassID(prhs[1]) == mxFUNCTION_CLASS)) {\
  mexErrMsgTxt(ERROR_NON_SUPPORTED_TYPE);\
    } else {\
    if (mxIsSingle(prhs[1])) {\
    if (mxGetNumberOfElements(prhs[1]) != 1) {\
    mexErrMsgTxt(ERROR_ONLY_MATLAB_SCALARS);\
    } else {\
    if (mxIsComplex(prhs[1])) {\
    op2 = opCFLOAT;\
    float *tmp = (float *) mxGetPr(prhs[1]);\
    op2C.x = tmp[0];\
    tmp = (float *) mxGetPi(prhs[1]);\
    op2C.y = tmp[0];\
    } else {\
    op2 = opFLOAT;\
    float *tmp = (float *) mxGetPr(prhs[1]);\
    op2F = tmp[0];\
    }\
    }\
    } else if (mxIsDouble(prhs[1])) {\
    if (mxGetNumberOfElements(prhs[1]) != 1) {\
    mexErrMsgTxt(ERROR_ONLY_MATLAB_SCALARS);\
    } else {\
    if (mxIsComplex(prhs[1])) {\
    op2 = opCFLOAT;\
    double *tmp = mxGetPr(prhs[1]);\
    op2C.x = (float) tmp[0];\
    tmp = mxGetPi(prhs[1]);\
    op2C.y = (float) tmp[0];\
    } else {\
    op2 = opFLOAT;\
    double *tmp = mxGetPr(prhs[1]);\
    op2F = (float) tmp[0];\
    }\
    }\
    } else { \
    mexErrMsgTxt(ERROR_NON_SUPPORTED_TYPE);\
      }\
    }\
  }\
  GPUtype *r;\


#define MATPLUSLIKEPART2(FUNF,FUNC, RFUNSCALARF, LFUNSCALARF, RFUNSCALARC, LFUNSCALARC)\
  try {\
  if (op1 == opGPUsingle && op2 == opGPUsingle) {\
  r = plusminus_common(*p, *q, FUNF, FUNC);\
  } else if (op1 == opGPUsingle && op2 == opFLOAT) {\
  r = plusminus_common(*p, op2F, RFUNSCALARF, RFUNSCALARC);\
  } else if (op1 == opGPUsingle && op2 == opCFLOAT) {\
  r = plusminus_common(*p, op2C, RFUNSCALARF, RFUNSCALARC);\
  } else if (op1 == opFLOAT && op2 == opGPUsingle) {\
  r = plusminus_common(op1F, *q, LFUNSCALARF, LFUNSCALARC);\
  } else if (op1 == opCFLOAT && op2 == opGPUsingle) {\
  r = plusminus_common(op1C, *q, LFUNSCALARF, LFUNSCALARC);\
  } else {\
  mexErrMsgTxt("Unknown operations. Please check arguments.");\
  }\
} catch (GPUexception ex) {\
  mexErrMsgTxt(ex.getError());\
}\


#define MATPLUSLIKEPART3\
  mxArray *tmpr = toMxStruct(r);\
  mexCallMATLAB(1, plhs, 1, &tmpr, "GPUsingle");\
  mxDestroyArray(tmpr);\


#endif
