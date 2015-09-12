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

#ifdef UNIX
#include <stdint.h>
#endif

#include "mex.h"
#ifndef MATLAB
#define MATLAB
#endif

//#include "cutil.h"
#include "cublas.h"
#include "cuda_runtime.h"
#include "cuda.h"
#include "cufft.h"

#include "cudalib_common.h"
#include "cudalib_error.h"
#include "cudalib.h"

#include "GPUcommon.hh"
#include "GPUerror.hh"
#include "Queue.hh"
#include "GPUstream.hh"
#include "GPUmanager.hh"
#include "GPUtype.hh"

#include "GPUop.hh"
#include "GPUmat.hh"

#include "GPUtypeMat.hh"

#include "kernelnames.h"

static GPUmanager p;
static CUdevice cuDevice;
static CUcontext cuContext;
static CUmodule cuModule;
static CUfunction cuFunction[MAXCUFUNCTION];
static CUtexref cuTexref[MAXCUTEXREF];
static char *fatbin;


/* buffer is used to print strings */
#define BUFFERSIZE 300
#define CLEARBUFFER memset(STRINGBUFFER1,0,BUFFERSIZE);
char STRINGBUFFER1[BUFFERSIZE];

// maximum number of elements for automatic mx conversion
#define MAXELMXCONVERT 50

/* GPUmatInterface structure */
static GPUmatInterface gm;

static void test() {
  GPUmanager *r = &p;
}


//****************************************************
// GPU MANAGER
//****************************************************
void gmGMcacheClean() {
  p.cacheClean();
}

//****************************************************
// DEBUG
//****************************************************

int gmGetDebugMode () {
  return p.getDebugMode();
}

void gmSetDebugMode (int mode) {
  p.setDebugMode(mode);
}

void gmDebugPushInstructionStack (int instr) {
  p.debugPushInstructionStack(instr);
}

void gmDebugLog (STRINGCONST char *str, int v) {
  if (v<p.debugGetVerbose()) {
    for (int i=0;i<p.debugGetIndent();i++)
      mexPrintf("  ");
    mexPrintf(str);
  }
}

void gmDebugLogPush() {
  p.debugPushIndent();
}

void gmDebugLogPop() {
  p.debugPopIndent();
}

void gmDebugReset() {
  p.debugReset();
}
//****************************************************
// COMPILER
//****************************************************



int gmGetCompileMode () {
  return p.getCompileMode();
}

void gmCompPush(void *ptmp, int type) {
  try {
    p.compPush(ptmp, type);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmCompPushGPUtype (void *gt) {
  gmGPUtype *p0 = (gmGPUtype *) gt;
  GPUtype *r = (GPUtype *) p0->ptrCounter->ptr;
  try {
    p.compPush((void *)r, STACKGPUTYPE);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}

void gmCompPushMx (void *mx) {
  try {
    p.compPush(mx, STACKMX);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}

void gmCompRegisterInstruction (STRINGCONST char *str) {
  p.compRegisterInstruction(str);
}

void gmCompAbort (STRINGCONST char *str) {
  try {
    p.compAbort(str);
  } catch (GPUexception ex) {
    CLEARBUFFER;
    sprintf(STRINGBUFFER1,"Compilation aborted. %s\n", ex.getError());
    mexErrMsgTxt(STRINGBUFFER1);
  }
}

int gmCompGetContextGPUtype (void *gt) {
  gmGPUtype *p0 = (gmGPUtype *) gt;
  GPUtype *r = (GPUtype *) p0->ptrCounter->ptr;
  return p.compGetContext((void *)r, STACKGPUTYPE);
}

int gmCompGetContextMx (void *mx) {
  return p.compGetContext(mx, STACKMX);
}


void gmCompCreateMxContext(mxArray *mx) {
  int nrhs = 1;

  // Garbage collector
  MyGC mygc = MyGC();
  gmDebugLog("> mxArray is not available in the compilation context.\n",3);
  gmDebugLogPush();

  // push mx variable so it is available in compilation context
  // 10/04/19 I have to push as a temp variable
  gmCompPush(mx, STACKMX);

  // get id
  int mx_id = p.compGetContext(mx, STACKMX);

  CLEARBUFFER
    sprintf(STRINGBUFFER1,"CREATE MXARG%d\n", mx_id);
  gmDebugLog(STRINGBUFFER1,3);

  //CLEARBUFFER
  //sprintf(STRINGBUFFER1,"DECLARE_MXNID(%d,%d)", mx_id, nrhs);
  //p.compRegisterInstruction(STRINGBUFFER1);
  //CLEARBUFFER
  //sprintf(STRINGBUFFER1,"DECLARE_MXID(%d,%d)", mx_id, nrhs);
  //p.compRegisterInstruction(STRINGBUFFER1);

  int i = 0;
  if (mxGetClassID(mx)==mxDOUBLE_CLASS){
    int nel = mxGetNumberOfElements(mx);
    int iscomplex = mxIsComplex(mx);
    CLEARBUFFER
      sprintf(STRINGBUFFER1,"Converting DOUBLE array\n");
    gmDebugLog(STRINGBUFFER1,3);
    CLEARBUFFER
      sprintf(STRINGBUFFER1,"Number of elements -> %d\n",nel);
    gmDebugLog(STRINGBUFFER1,3);

    if (nel > MAXELMXCONVERT) {
      CLEARBUFFER
        sprintf(STRINGBUFFER1,"Matlab array cannot be converted. Number of elements (%d) exceeds limits (%d). %s\n",nel, MAXELMXCONVERT, ERROR_GPUMANAGER_MAXELMXCONVERT);
      gmCompAbort(STRINGBUFFER1);
    }

    CLEARBUFFER
      sprintf(STRINGBUFFER1,"Complexity -> %d\n",iscomplex);
    gmDebugLog(STRINGBUFFER1,3);

    CLEARBUFFER
      if (!iscomplex)
        sprintf(STRINGBUFFER1,"CREATE_MXID_DOUBLEARRAY(%d,%d,%d, mxREAL)", mx_id, i, nel);
      else
        sprintf(STRINGBUFFER1,"CREATE_MXID_DOUBLEARRAY(%d,%d,%d, mxCOMPLEX)", mx_id, i, nel);
    p.compRegisterInstruction(STRINGBUFFER1);

    CLEARBUFFER
      sprintf(STRINGBUFFER1,"DECLARE_MXID_DOUBLEPTR_REAL(%d,%d)", mx_id, i);
    p.compRegisterInstruction(STRINGBUFFER1);

    CLEARBUFFER
      sprintf(STRINGBUFFER1,"DECLARE_MXID_DOUBLEPTR_IMAG(%d,%d)", mx_id, i);
    p.compRegisterInstruction(STRINGBUFFER1);


    double *tmpR = mxGetPr(mx);
    double *tmpI = mxGetPi(mx);

    for(int j=0;j<nel;j++) {
      double scR = tmpR[j];
      double scI = 0;
      CLEARBUFFER
        sprintf(STRINGBUFFER1,"ASSIGN_MXID_DOUBLEARRAY_REAL(%d,%d,%d,%f)", mx_id, i, j, scR);
      p.compRegisterInstruction(STRINGBUFFER1);

      if (iscomplex) {
        scI = tmpI[j];
        CLEARBUFFER
          sprintf(STRINGBUFFER1,"ASSIGN_MXID_DOUBLEARRAY_IMAG(%d,%d,%d,%f)", mx_id, i, j, scI);
        p.compRegisterInstruction(STRINGBUFFER1);
      }
      CLEARBUFFER
        sprintf(STRINGBUFFER1,"   - Converting scalar (re,im) -> (%f,%f)\n", scR, scI);
      gmDebugLog(STRINGBUFFER1,3);
    }
  } else if (mxGetClassID(mx)==mxSINGLE_CLASS){
    int nel = mxGetNumberOfElements(mx);
    int iscomplex = mxIsComplex(mx);
    CLEARBUFFER
      sprintf(STRINGBUFFER1,"Converting SINGLE array\n");
    gmDebugLog(STRINGBUFFER1,3);
    CLEARBUFFER
      sprintf(STRINGBUFFER1,"Number of elements -> %d\n",nel);
    gmDebugLog(STRINGBUFFER1,3);

    if (nel > MAXELMXCONVERT) {
      CLEARBUFFER
        sprintf(STRINGBUFFER1,"Matlab array cannot be converted. Number of elements (%d) exceeds limits (%d). %s\n",nel, MAXELMXCONVERT, ERROR_GPUMANAGER_MAXELMXCONVERT);
      gmCompAbort(STRINGBUFFER1);
    }

    CLEARBUFFER
      sprintf(STRINGBUFFER1,"Complexity -> %d\n",iscomplex);
    gmDebugLog(STRINGBUFFER1,3);

    CLEARBUFFER
      if (!iscomplex)
        sprintf(STRINGBUFFER1,"CREATE_MXID_SINGLEARRAY(%d,%d,%d, mxREAL)", mx_id, i, nel);
      else
        sprintf(STRINGBUFFER1,"CREATE_MXID_SINGLEARRAY(%d,%d,%d, mxCOMPLEX)", mx_id, i, nel);
    p.compRegisterInstruction(STRINGBUFFER1);

    CLEARBUFFER
      sprintf(STRINGBUFFER1,"DECLARE_MXID_SINGLEPTR_REAL(%d,%d)", mx_id, i);
    p.compRegisterInstruction(STRINGBUFFER1);

    CLEARBUFFER
      sprintf(STRINGBUFFER1,"DECLARE_MXID_SINGLEPTR_IMAG(%d,%d)", mx_id, i);
    p.compRegisterInstruction(STRINGBUFFER1);


    float *tmpR = (float *) mxGetPr(mx);
    float *tmpI = (float *) mxGetPi(mx);

    for(int j=0;j<nel;j++) {
      float scR = tmpR[j];
      float scI = 0;
      CLEARBUFFER
        sprintf(STRINGBUFFER1,"ASSIGN_MXID_SINGLEARRAY_REAL(%d,%d,%d,%f)", mx_id, i, j, scR);
      p.compRegisterInstruction(STRINGBUFFER1);

      if (iscomplex) {
        scI = tmpI[j];
        CLEARBUFFER
          sprintf(STRINGBUFFER1,"ASSIGN_MXID_SINGLEARRAY_IMAG(%d,%d,%d,%f)", mx_id, i, j, scI);
        p.compRegisterInstruction(STRINGBUFFER1);
      }
      CLEARBUFFER
        sprintf(STRINGBUFFER1,"   - Converting scalar (re,im) -> (%f,%f)\n", scR, scI);
      gmDebugLog(STRINGBUFFER1,3);
    }


  } else if (mxGetClassID(mx)==mxCELL_CLASS){

    int nel = mxGetNumberOfElements(mx);
    CLEARBUFFER
      sprintf(STRINGBUFFER1,"CREATE_MXID_CELL(%d,%d,%d)", mx_id, 0, nel);
    p.compRegisterInstruction(STRINGBUFFER1);

    for (int i = 0; i < nel; i++) {
      mxArray *tmpmx = mxGetCell(mx, i);
      int rid = p.compGetContext(tmpmx,STACKMX);
      if (rid==-1) {
        gmCompCreateMxContext(tmpmx);
        rid = p.compGetContext(tmpmx,STACKMX);
      }
      CLEARBUFFER
        sprintf(STRINGBUFFER1,"ASSIGN_MXID_CELL_MXID(%d,%d,%d,%d,%d)", mx_id, 0, i, rid, 0);
      p.compRegisterInstruction(STRINGBUFFER1);
    }
    /*} else if (mxGetClassID(mx)==mxSTRUCT_CLASS){
    if (mxGetNumberOfElements(mx) > 1)
    gmCompAbort(ERROR_GPUMANAGER_COMPINVALIDSTRUCT);

    mxArray *field = mxGetField(mx, 0, "type");
    if (field==NULL) {
    gmCompAbort(ERROR_GPUMANAGER_COMPINVALIDSTRUCT);
    }
    if (mxIsChar(field)) {
    char buffer[10];
    mxGetString(field, buffer, 10);
    if (strcmp(buffer, "()") == 0) {
    mxArray * subs = mxGetField(mx, 0, "subs");
    int rid = p.compGetContext(subs,1);
    if (rid==-1) {
    gmCompCreateMxContext(subs);
    rid = p.compGetContext(subs,1);
    }
    CLEARBUFFER
    sprintf(STRINGBUFFER1,"ASSIGN_MXID_SUBSREF_STRUCT(%d,%d,%d,%d)", mx_id, 0, rid, 0);
    p.compRegisterInstruction(STRINGBUFFER1);
    } else {
    gmCompAbort(ERROR_GPUMANAGER_COMPINVALIDSTRUCT);
    }
    } else {
    gmCompAbort(ERROR_GPUMANAGER_COMPINVALIDSTRUCT);
    }
    */
  } else if (mxGetClassID(mx)==mxCHAR_CLASS){

    unsigned int strlen = mxGetM(mx);
    if (mxGetN(mx) > strlen)
      strlen = mxGetN(mx);
    char *str = (char*) Mymalloc((strlen+1)*sizeof(char),&mygc);
    memset(str,0,strlen+1);
    mxGetString(mx, str, strlen+1);
    sprintf(STRINGBUFFER1,"ASSIGN_MXID_CHAR(%d,%d,\"%s\")", mx_id, i, str);
    p.compRegisterInstruction(STRINGBUFFER1);

    CLEARBUFFER
      sprintf(STRINGBUFFER1,"Converting CHAR '%s'\n",str);
    gmDebugLog(STRINGBUFFER1,3);
  } else {
    CLEARBUFFER
      sprintf(STRINGBUFFER1,"Cannot assign '%s' to the compilation context.\n",mxGetClassName(mx));
    gmCompAbort(STRINGBUFFER1);
  }
  gmDebugLogPop();
}


void gmCompFunctionStart(STRINGCONST char *name) {
  p.compFunctionStart(name);
  //CLEARBUFFER
  //sprintf(STRINGBUFFER1, "START FUNCTION %s \n", name);
  //gmDebugLog(STRINGBUFFER1, 3);
}

void gmCompFunctionEnd(void) {
  p.compFunctionEnd();

  //CLEARBUFFER
  //sprintf(STRINGBUFFER1, "END FUNCTION \n");
  //gmDebugLog(STRINGBUFFER1, 3);
}

void gmCompFunctionSetParamGPUtype(void *gt) {
  gmGPUtype *p0 = (gmGPUtype *) gt;
  GPUtype *r = (GPUtype *) p0->ptrCounter->ptr;
  try {
    p.compFunctionSetParamGPUtype(r);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }

  /*int ridx = p.compGetContext(r,0);
  if (ridx==-1) {
  gmCompAbort(ERROR_GPUMANAGER_COMPINCONSGPUTYPE);
  }
  CLEARBUFFER
  sprintf(STRINGBUFFER1, "  GPUARG%d\n", ridx);
  gmDebugLog(STRINGBUFFER1, 3);


  CLEARBUFFER
  sprintf(STRINGBUFFER1,"GPUTYPEID(%d)", ridx);
  p.compFunctionSetParam(STRINGBUFFER1);*/

}

void gmCompFunctionSetParamInt(int par) {
  /*CLEARBUFFER
  sprintf(STRINGBUFFER1, "  INT %d\n", par);
  gmDebugLog(STRINGBUFFER1, 3);

  CLEARBUFFER
  sprintf(STRINGBUFFER1,"%d", par);
  p.compFunctionSetParam(STRINGBUFFER1);*/
  try {
    p.compFunctionSetParamInt(par);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }

}

void gmCompFunctionSetParamFloat(float par) {
  try {
    p.compFunctionSetParamFloat(par);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }

  /*CLEARBUFFER
  sprintf(STRINGBUFFER1, "  FLOAT %d\n", par);
  gmDebugLog(STRINGBUFFER1, 3);

  CLEARBUFFER
  sprintf(STRINGBUFFER1,"%f", par);
  p.compFunctionSetParam(STRINGBUFFER1);*/

}

void gmCompFunctionSetParamDouble(double par) {
  try {
    p.compFunctionSetParamDouble(par);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  /*CLEARBUFFER
  sprintf(STRINGBUFFER1, "  DOUBLE %d\n", par);
  gmDebugLog(STRINGBUFFER1, 3);

  CLEARBUFFER
  sprintf(STRINGBUFFER1,"%f", par);
  p.compFunctionSetParam(STRINGBUFFER1);*/

}

void gmCompFunctionSetParamMx(void *mx) {
  int ridx = p.compGetContext(mx,STACKMX);
  if (ridx==-1) {
    gmCompCreateMxContext((mxArray *)mx);
    ridx = p.compGetContext(mx,STACKMX);
    // any created param should be nulled
    p.compStackNullPtr(mx, STACKMX);
  }
  CLEARBUFFER
    sprintf(STRINGBUFFER1, "  MXARG%d\n", ridx);
  gmDebugLog(STRINGBUFFER1, 3);


  CLEARBUFFER
    sprintf(STRINGBUFFER1,"MXID(%d)[0]", ridx);
  p.compFunctionSetParam(STRINGBUFFER1);
}

void gmCompFunctionSetParamMxMx(int nrhs, mxArray *prhs[]) {
  // mxArray *prhs[] have to be added and deleted afterwards from the context

  gmCompPush(prhs,STACKMXMX);
  int mx_id = p.compGetContext(prhs,STACKMXMX);
  p.compStackNullPtr(prhs,STACKMXMX);

  CLEARBUFFER
    sprintf(STRINGBUFFER1,"DECLARE_MXNID(%d,%d)", mx_id, nrhs);
  p.compRegisterInstruction(STRINGBUFFER1);

  CLEARBUFFER
    sprintf(STRINGBUFFER1,"DECLARE_MXID(%d,%d)", mx_id, nrhs);
  p.compRegisterInstruction(STRINGBUFFER1);

  for (int i=0;i<nrhs;i++) {
    int rid = p.compGetContext(prhs[i],STACKMX);
    if (rid==-1) {
      gmCompCreateMxContext(prhs[i]);
      rid = p.compGetContext(prhs[i],STACKMX);
      p.compStackNullPtr(prhs[i],STACKMX);
    }
    CLEARBUFFER
      sprintf(STRINGBUFFER1,"ASSIGN_MXID_MXID(%d,%d,%d,%d)", mx_id, i, rid, 0);
    p.compRegisterInstruction(STRINGBUFFER1);

    CLEARBUFFER
      sprintf(STRINGBUFFER1, "  MXARG%d\n", rid);
    gmDebugLog(STRINGBUFFER1, 3);
  }

  CLEARBUFFER
    sprintf(STRINGBUFFER1,"MXNID(%d), MXID(%d)", mx_id, mx_id);
  p.compFunctionSetParam(STRINGBUFFER1);

  p.compClearContext(prhs,2);
}


//****************************************************
// CONFIG
//****************************************************
void gmGetMajorMinor (int *major, int *minor) {
  *major = GPUMATVERSION_MAJOR;
  *minor = GPUMATVERSION_MINOR;
}

int gmGetActiveDeviceNumber () {
  return cuDevice;
}

//****************************************************
// FUNCTIONS
//****************************************************
int gmRegisterFunction (STRINGCONST char *name, void *f) {
  int r = -1;
  try {
    r = p.funRegisterFunction(name, f);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return r;
}

void * gmGetFunctionByName(STRINGCONST char *name) {
  void *r = 0;
  try {
    r = p.funGetFunctionByName(name);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return r;

}

int gmGetFunctionNumber(STRINGCONST char *name) {
  int r = -1;
  try {
    r = p.funGetFunctionNumber(name);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return r;
}

void * gmGetFunctionByNumber(int findex) {
  void *r = 0;
  try {
    r = p.funGetFunctionByNumber(findex);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return r;
}

//****************************************************
// GPUTYPE
//****************************************************

void gmDeleteGPUtype (void *p) {
  delete (GPUtype*) p;
}

gmGPUtype gmGetGPUtype(const mxArray *mx) {

  // The returned gmGPUtype will take care of deleting the pointed GPUtype
  // when necessary. The gmGPUtype is a smart pointer.

  mxArray *lhs[2];

  GPUmanager * GPUman = &p;

  // CUDA capability
  int cudacap = GPUman->getCudaCapability();

  GPUtype *gp;
  //I have to allow also Matlab scalars

  // check input
  if ((mxIsClass(mx, "GPUsingle")) || (mxIsClass(mx, "GPUdouble")) ) {

    GPUtype *r = (GPUtype *) mxToGPUtype(mx, GPUman);

    // creates copy of the GPUtype
    gp = new GPUtype(*r,0);
  } else {
    if (mxIsSingle(mx)) {
      if (mxGetNumberOfElements(mx) != 1) {
        mexErrMsgTxt(ERROR_ONLY_MATLAB_SCALARS);
      } else {
        if (mxIsComplex(mx)) {
          Complex opCF;
          float *tmp = (float *) mxGetPr(mx);
          opCF.x = tmp[0];
          tmp = (float *) mxGetPi(mx);
          opCF.y = tmp[0];
          gp = new GPUtype(opCF, GPUman);
        } else {
          float opF;
          float *tmp = (float *) mxGetPr(mx);
          opF = tmp[0];
          gp = new GPUtype(opF, GPUman);
        }
        if (gmGetCompileMode()==1) {
          try {
            p.compPush((void *)gp, STACKGPUTYPE);
          } catch (GPUexception ex) {
            mexErrMsgTxt(ex.getError());
          }
          gmCompFunctionStart("GPUMAT_mxToGPUtype");
          p.compFunctionSetParamGPUtype(gp);
          gmCompFunctionSetParamMx((void*)mx);
          gmCompFunctionEnd();


        }
      }
    } else if (mxIsDouble(mx)) {
      if (mxGetNumberOfElements(mx) != 1) {
        mexErrMsgTxt(ERROR_ONLY_MATLAB_SCALARS);
      } else {
        if (mxIsComplex(mx)) {
          DoubleComplex opCD;
          Complex opCF;
          double *tmp = mxGetPr(mx);
          opCD.x = (double) tmp[0];
          opCF.x = (float) tmp[0];
          tmp = mxGetPi(mx);
          opCD.y = (double) tmp[0];
          opCF.y = (float) tmp[0];
          if (cudacap>=13) {
            gp = new GPUtype(opCD, GPUman);
          } else {
            gp = new GPUtype(opCF, GPUman);
          }

        } else {
          double opD;
          float opF;
          double *tmp = (double *) mxGetPr(mx);
          opD = (double) tmp[0];
          opF = (float) tmp[0];
          if (cudacap>=13) {
            gp = new GPUtype(opD, GPUman);
          } else {
            gp = new GPUtype(opF, GPUman);
          }

        }
        if (gmGetCompileMode()==1) {
          try {
            p.compPush((void *)gp, STACKGPUTYPE);
          } catch (GPUexception ex) {
            mexErrMsgTxt(ex.getError());
          }
          gmCompFunctionStart("GPUMAT_mxToGPUtype");
          p.compFunctionSetParamGPUtype(gp);
          gmCompFunctionSetParamMx((void*)mx);
          gmCompFunctionEnd();

        }
      }
    } else {
      mexErrMsgTxt(ERROR_NON_SUPPORTED_TYPE);
    }
  }

  return gmGPUtype(gp,gmDeleteGPUtype);
}

//****************************************************
// GETTERS
//****************************************************

static gpuTYPE_t gmGetType(const gmGPUtype &p) {
  GPUtype *r = (GPUtype *) p.ptrCounter->ptr;
  return r->getType();
}

static const int * gmGetSize(const gmGPUtype &p) {
  GPUtype *r = (GPUtype *) p.ptrCounter->ptr;
  return r->getSize();
}

static int gmGetNdims(const gmGPUtype &p) {
  GPUtype *r = (GPUtype *) p.ptrCounter->ptr;
  return r->getNdims();

}

static int  gmGetNumel(const gmGPUtype &p) {
  GPUtype *r = (GPUtype *) p.ptrCounter->ptr;
  return r->getNumel();
}

static const void * gmGetGPUptr(const gmGPUtype &p) {
  GPUtype *r = (GPUtype *) p.ptrCounter->ptr;
  return r->getGPUptr();
}

static int gmGetDataSize(const gmGPUtype &p) {
  GPUtype *r = (GPUtype *) p.ptrCounter->ptr;
  return r->getMySize();
}

//****************************************************
// PROPERTIES
//****************************************************

static int gmIsFloat(const gmGPUtype &p) {
  GPUtype *r = (GPUtype *) p.ptrCounter->ptr;
  return r->isFloat();
}

static int gmIsDouble(const gmGPUtype &p) {
  GPUtype *r = (GPUtype *) p.ptrCounter->ptr;
  return r->isDouble();
}

static int gmIsComplex(const gmGPUtype &p) {
  GPUtype *r = (GPUtype *) p.ptrCounter->ptr;
  return r->isComplex();
}

static int gmIsEmpty(const gmGPUtype &p) {
  GPUtype *r = (GPUtype *) p.ptrCounter->ptr;
  return r->isEmpty();
}

static int gmIsScalar(const gmGPUtype &p) {
  GPUtype *r = (GPUtype *) p.ptrCounter->ptr;
  return r->isScalar();
}

//****************************************************
// SETTERS
//****************************************************

static void gmSetSize(const gmGPUtype &p, int n, const int *s) {
  GPUtype *r = (GPUtype *) p.ptrCounter->ptr;
  return r->setSize(n,(int *) s);
}

/* GPUtype creation */
static gmGPUtype gmCloneGPUtype(const gmGPUtype &p) {
  GPUtype *r = (GPUtype *) p.ptrCounter->ptr;
  GPUtype *r1 = r->clone();
  gmGPUtype gmR = gmGPUtype(r1,gmDeleteGPUtype);
  return gmR;
}

gmGPUtype  gmCreateGPUtype (gpuTYPE_t type, int ndims, const int *size, void *init) {
  // My garbage collector
  MyGC mgc = MyGC();    // Garbage collector for Malloc
  MyGCObj<GPUtype> mgcobj; // Garbage collector for GPUtype

  GPUtype *r;
  if ((size==NULL)||(ndims==0)) {
    r = new GPUtype(&p);
    r->setType(type);
    mgcobj.setPtr(r); // should delete this pointer
  } else {
    r = new GPUtype(type, ndims, size, &p);
    mgcobj.setPtr(r); // should delete this pointer
    try {
      GPUopAllocVector(*r);

      if (init)
        GPUopCudaMemcpy(r->getGPUptr(), init,
        r->getMySize() * r->getNumel(), cudaMemcpyHostToDevice,
        &p);

    } catch (GPUexception ex) {
      mexErrMsgTxt(ex.getError());
    }
  }
  mgcobj.remPtr(r);
  gmGPUtype gmR = gmGPUtype(r,gmDeleteGPUtype);
  return gmR;
}



gmGPUtype  gmMxCreateGPUtype (gpuTYPE_t type, int nrhs, const mxArray *prhs[]) {
  GPUtype *r;
  try {
    r = mxCreateGPUtype (type, &p, nrhs, prhs);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r,gmDeleteGPUtype);
}

gmGPUtype  gmColon (gpuTYPE_t type, double j, double d, double k) {
  // My garbage collector
  MyGC mgc = MyGC();       // Garbage collector for Malloc
  MyGCObj<GPUtype> mgcobj; // Garbage collector for GPUtype

  GPUtype *r;

  try {
    // create an empty dummy GPUtype
    GPUtype *dummy = new GPUtype(&p);
    dummy->setType(type);
    mgcobj.setPtr(dummy); // should delete this pointer

    r = GPUopColonDrv(j,k,d,*dummy);

  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r,gmDeleteGPUtype);
}

mxArray * gmToMxArray (const gmGPUtype &p) {
  //GPUtype *r = new GPUtype(*((GPUtype *) p.ptrCounter->ptr));
  GPUtype *r = (GPUtype *) p.ptrCounter->ptr;
  return GPUtypeToMxNumericArray(*r);
}

mxArray * gmCreateMxArray (const gmGPUtype &p) {
  GPUtype *r = new GPUtype(*((GPUtype *) p.ptrCounter->ptr));
  return toMx(r);
}

mxArray * gmCreateMxArrayPtr (mxArray *mx, const gmGPUtype &ptr) {
  GPUtype *r = new GPUtype(*((GPUtype *) ptr.ptrCounter->ptr));
  int slot = 0;
  // request a free slot in cache
  void *mxtmp = p.extCacheGetFreeSlot(&slot, gpuNOTDEF); // looking for non cached elements
  if ((slot<0)||(mxtmp!=NULL)) {
    // internal error
    mexErrMsgTxt(ERROR_MXID_CACHEINTERNAL);
  }
  p.extCacheRegisterPtrBySlot(slot, mxID(mx), mx, r, gpuNOTDEF);

  return mxCreateDoubleScalar(slot);
}

// Fill
void gmGPUtypeFill (const gmGPUtype &p, double offset, double incr, int m, int q, int offsq, int type) {
  GPUtype *r = (GPUtype *) p.ptrCounter->ptr;
  try {
    GPUopFillVector1(offset, incr,  *r, m, q, offsq, type);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}

// From Matlab to GPUtype
gmGPUtype  gmMxToGPUtype(const mxArray *mx) {
  GPUtype *r;
  try {
    r=mxNumericArrayToGPUtype((mxArray*)mx, &p);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r,gmDeleteGPUtype);
}

// real imag
void gmRealImag (const gmGPUtype &data, const gmGPUtype &re, const gmGPUtype &im, int dir, int mode) {
  GPUtype *d = (GPUtype *) data.ptrCounter->ptr;
  GPUtype *r = (GPUtype *) re.ptrCounter->ptr;
  GPUtype *i = (GPUtype *) im.ptrCounter->ptr;

  try {
    GPUopRealImag(*d, *r, *i, dir, mode);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}

//****************************************************
// CASTING
//****************************************************

static gmGPUtype gmGPUtypeDoubleToFloat(const gmGPUtype &p) {
  GPUtype *r = (GPUtype *) p.ptrCounter->ptr;
  if (!r->isDouble()) {
    mexErrMsgTxt("Wrong GPUtype casting. Expected DOUBLE.");
  }
  GPUtype *r1 = r->DOUBLEtoFLOAT();
  gmGPUtype gmR = gmGPUtype(r1,gmDeleteGPUtype);
  return gmR;
}


static gmGPUtype gmGPUtypeFloatToDouble(const gmGPUtype &p) {
  GPUtype *r = (GPUtype *) p.ptrCounter->ptr;
  if (!r->isFloat()) {
    mexErrMsgTxt("Wrong GPUtype casting. Expected FLOAT.");
  }
  GPUtype *r1 = r->FLOATtoDOUBLE();
  gmGPUtype gmR = gmGPUtype(r1,gmDeleteGPUtype);
  return gmR;
}



static gmGPUtype gmGPUtypeFloatToComplexFloat(const gmGPUtype &p) {
  GPUtype *r = (GPUtype *) p.ptrCounter->ptr;
  if (r->getType()!=gpuFLOAT) {
    mexErrMsgTxt("Wrong GPUtype casting. Expected gpuFLOAT.");
  }
  GPUtype *r1 = r->FLOATtoCFLOAT();
  gmGPUtype gmR = gmGPUtype(r1,gmDeleteGPUtype);
  return gmR;
}

static gmGPUtype gmGPUtypeRealToComplex(const gmGPUtype &p) {
  GPUtype *r = (GPUtype *) p.ptrCounter->ptr;
  if (r->isComplex()) {
    mexErrMsgTxt("Wrong GPUtype casting. Expected a REAL GPUtype.");
  }
  GPUtype *r1 = r->REALtoCOMPLEX();
  gmGPUtype gmR = gmGPUtype(r1,gmDeleteGPUtype);
  return gmR;
}

static gmGPUtype gmGPUtypeRealImagToComplex(const gmGPUtype &re, const gmGPUtype &im) {
  GPUtype *retmp = (GPUtype *) re.ptrCounter->ptr;
  GPUtype *imtmp = (GPUtype *) im.ptrCounter->ptr;
  if (retmp->isComplex()) {
    mexErrMsgTxt("Wrong GPUtype casting. Expected a REAL GPUtype.");
  }
  if (imtmp->isComplex()) {
    mexErrMsgTxt("Wrong GPUtype casting. Expected a REAL GPUtype.");
  }

  GPUtype *r1 = retmp->REALtoCOMPLEX(*imtmp);
  gmGPUtype gmR = gmGPUtype(r1,gmDeleteGPUtype);
  return gmR;
}


// ASSIGN
void  gmGPUtypeAssign (const gmGPUtype &p, const gmGPUtype &q, const Range &r, int dir) {
  try {
    GPUtype *ptmp =  (GPUtype *) p.ptrCounter->ptr;
    GPUtype *qtmp =  (GPUtype *) q.ptrCounter->ptr;
    GPUtype *res = GPUopAssign(*(ptmp),*(qtmp), r, dir, 0, 0);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}

// MXASSIGN
void  gmGPUtypeMxAssign (const gmGPUtype &p, const gmGPUtype &q, const Range &r, int dir) {
  try {
    GPUtype *ptmp =  (GPUtype *) p.ptrCounter->ptr;
    GPUtype *qtmp =  (GPUtype *) q.ptrCounter->ptr;
    GPUtype *res = GPUopAssign(*(ptmp),*(qtmp), r, dir, 0, 1);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}

// PERMUTE
void  gmGPUtypePermute (const gmGPUtype &p, const gmGPUtype &q, const Range &r, int dir, int*perm) {
  try {
    GPUtype *ptmp =  (GPUtype *) p.ptrCounter->ptr;
    GPUtype *qtmp =  (GPUtype *) q.ptrCounter->ptr;
    GPUtype *res = GPUopPermute(*(ptmp),*(qtmp), r, dir, 0, 0, perm);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}

// MXPERMUTE
void  gmGPUtypeMxPermute (const gmGPUtype &p, const gmGPUtype &q, const Range &r, int dir, int*perm) {
  try {
    GPUtype *ptmp =  (GPUtype *) p.ptrCounter->ptr;
    GPUtype *qtmp =  (GPUtype *) q.ptrCounter->ptr;
    GPUtype *res = GPUopPermute(*(ptmp),*(qtmp), r, dir, 0, 1, perm);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}

// SLICE
gmGPUtype  gmCreateGPUtypeSlice (const gmGPUtype &p, const Range &r) {
  GPUtype *res;
  try {
    GPUtype *ptmp =  (GPUtype *) p.ptrCounter->ptr;
    res = GPUopAssign(*(ptmp),*(ptmp), r, 0, 1, 0);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(res,gmDeleteGPUtype);
}

// MX SLICE
gmGPUtype  gmCreateGPUtypeMxSlice (const gmGPUtype &p, const Range &r) {
  GPUtype *res;
  try {
    GPUtype *ptmp =  (GPUtype *) p.ptrCounter->ptr;
    res = GPUopAssign(*(ptmp),*(ptmp), r, 0, 1, 1);
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(res,gmDeleteGPUtype);
}

// NUMERICS FUNCTIONS
#include "GPUmatInterfaceFunctions.hh"

// FTT FUNCTIONS
#include "GPUmatFFTFunctions.hh"


// init
// 0: still to init
// 1: already initialized. Should delete
static int init=0;
static int modulesLoaded = 0;

static void GPUmanagerClose() {
  if (init==0)
    return;
  CUresult status;

  if (modulesLoaded==1) {
    status = cuModuleUnload(cuModule);
    //if (CUDA_SUCCESS != status) {
    //	mexErrMsgTxt("Unable to unload CUDA module");
    //}

    // cuCtxDetach is obsolete from CUDA 4.0

    //status = cuCtxDetach(cuContext);
    //if (status != CUDA_SUCCESS) {
    //	mexErrMsgTxt(
    //			"Error in CUDA context cleanup.");
    //}
    // The behaviour of cuCtxDestroy is strange and unexpected
    status = cuCtxDestroy(cuContext);
    /*if (status != CUDA_SUCCESS) {
    mexErrMsgTxt(
    "Error in CUDA context cleanup.");
    }*/
    //p.cleanup();
  }

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  GPUmatResult_t gpustatus;
  cudaError_t cudastatus = cudaSuccess;


  if (nrhs != 7)
    mexErrMsgTxt("Wrong number of arguments");

  int nstreams = (int) mxGetScalar(prhs[0]);
  char buffer[2048];
  mxGetString(prhs[1], buffer, 2049);
  int dev = (int) mxGetScalar(prhs[2]);
  int today = (int) mxGetScalar(prhs[3]);
  int del   = (int) mxGetScalar(prhs[4]); // if 1 cleanup GPUmanager

  int major   = (int) mxGetScalar(prhs[5]);
  int minor   = (int) mxGetScalar(prhs[6]);


  if (del==0) {

    cuDevice = dev;

    if (init==0) {
      mexLock();
      p = GPUmanager(nstreams);
    }

    gpustatus = p.initStreams();
    if (gpustatus != GPUmatSuccess)
      mexErrMsgTxt(
      "Unable to initialize GPUmanager streams. This is an internal error. Please report a bug to gp-you@gp-you.org.");

    //
    CUresult status;

#include "kerneltable.h"

    int size = 0;
    FILE *f = fopen(buffer, "rb");
    if (f == NULL) {
      fatbin = NULL;
      mexErrMsgTxt(
        "Error opening KERNELS file. This is an internal error. Please report a bug to gp-you@gp-you.org.");
    }
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    fatbin = (char *) Mymalloc((size + 1) * sizeof(char));
    if (size != fread(fatbin, sizeof(char), size, f)) {
      Myfree(fatbin);
      mexErrMsgTxt(
        "Error reading KERNELS  file. This is an internal error. Please report a bug to gp-you@gp-you.org.");
    }
    fclose(f);

    fatbin[size] = 0;
    size = size + 1; // adding null to the end of the string

    // create context

    int ret_error = 0;
    cuContext = 0;
    status = cuCtxCreate(&cuContext, 0, cuDevice);
    if (status != CUDA_SUCCESS) {
      mexErrMsgTxt(
        "Error creating KERNELS context. This is an internal error. Please report a bug to gp-you@gp-you.org.");
    }

    status = cuModuleLoadData(&cuModule, fatbin);
    if (CUDA_SUCCESS != status)
      goto error;

    // update GPUmanager
    p.setCuDevice(&cuDevice);
    p.setCuContext(&cuContext);
    p.setCuModule(&cuModule);
    p.setCuFunction(&cuFunction[0]);
    p.setCuTexref(&cuTexref[0]);

    // set cuda compatibility
    if (major==1) {
      if (minor==0) {
        p.setCudaCapability(10);
      } else if (minor==1) {
        p.setCudaCapability(11);
      } else if (minor==2) {
        p.setCudaCapability(12);
      } else if (minor==3) {
        p.setCudaCapability(13);
      } else {
        mexErrMsgTxt(
          "Unable to recognize the GPU CUDA capability");
      }
    } else if (major==2) {
      if (minor==0) {
        p.setCudaCapability(20);
      } else if (minor==1) {
        p.setCudaCapability(21);
      } else if (minor==2) {
        p.setCudaCapability(22);
      } else if (minor==3) {
        p.setCudaCapability(23);
      } else {
        mexErrMsgTxt(
          "Unable to recognize the GPU CUDA capability");
      }
    } else if (major==3) {
      if (minor==0) {
        p.setCudaCapability(30);
      } else if (minor==5) {
        p.setCudaCapability(35);
      } else {
        mexErrMsgTxt(
          "Unable to recognize the GPU CUDA capability");
      }
    } else {
      mexErrMsgTxt(
        "Unable to recognize the GPU CUDA capability");

    }

#include "kerneltableinit.h"

    // load textures
    status = cuModuleGetTexRef(&(cuTexref[N_TEXREF_F1_A]), cuModule, "texref_f1_a");
    if (CUDA_SUCCESS != status)
      goto error;
    status = cuModuleGetTexRef(&(cuTexref[N_TEXREF_F1_B]), cuModule, "texref_f1_b");
    if (CUDA_SUCCESS != status)
      goto error;
    status = cuModuleGetTexRef(&(cuTexref[N_TEXREF_C1_A]), cuModule, "texref_c1_a");
    if (CUDA_SUCCESS != status)
      goto error;
    status = cuModuleGetTexRef(&(cuTexref[N_TEXREF_C1_B]), cuModule, "texref_c1_b");
    if (CUDA_SUCCESS != status)
      goto error;
    status = cuModuleGetTexRef(&(cuTexref[N_TEXREF_D1_A]), cuModule, "texref_d1_a");
    if (CUDA_SUCCESS != status)
      goto error;
    status = cuModuleGetTexRef(&(cuTexref[N_TEXREF_D1_B]), cuModule, "texref_d1_b");
    if (CUDA_SUCCESS != status)
      goto error;
    status = cuModuleGetTexRef(&(cuTexref[N_TEXREF_CD1_A]), cuModule, "texref_cd1_a");
    if (CUDA_SUCCESS != status)
      goto error;
    status = cuModuleGetTexRef(&(cuTexref[N_TEXREF_CD1_B]), cuModule, "texref_cd1_b");
    if (CUDA_SUCCESS != status)
      goto error;



    status = cuModuleGetTexRef(&(cuTexref[N_TEXREF_I1_A]), cuModule, "texref_i1_a");
    if (CUDA_SUCCESS != status)
      goto error;

    status = cuModuleGetTexRef(&(cuTexref[N_TEXREF_F2_A]), cuModule, "texref_f2_a");
    if (CUDA_SUCCESS != status)
      goto error;
    status = cuModuleGetTexRef(&(cuTexref[N_TEXREF_C2_A]), cuModule, "texref_c2_a");
    if (CUDA_SUCCESS != status)
      goto error;
    status = cuModuleGetTexRef(&(cuTexref[N_TEXREF_D2_A]), cuModule, "texref_d2_a");
    if (CUDA_SUCCESS != status)
      goto error;
    status = cuModuleGetTexRef(&(cuTexref[N_TEXREF_CD2_A]), cuModule, "texref_cd2_a");
    if (CUDA_SUCCESS != status)
      goto error;


    init =1;
    modulesLoaded = 1;
    plhs[0] = mxCreateDoubleScalar(UINTPTR &p);

    /* gpu manager control */
    gm.control.cacheClean = gmGMcacheClean;

    /* functions */
    gm.fun.registerFunction   = gmRegisterFunction;
    gm.fun.getFunctionByName   = gmGetFunctionByName;
    gm.fun.getFunctionNumber   = gmGetFunctionNumber;
    gm.fun.getFunctionByNumber   = gmGetFunctionByNumber;

    /* config */
    gm.config.getMajorMinor   = gmGetMajorMinor;
    gm.config.getActiveDeviceNumber = gmGetActiveDeviceNumber;





    /* set gm function pointers */
    /* getters */

    gm.fun.registerFunction("getGPUptr", (void*) gmGetGPUptr);
    gm.fun.registerFunction("getSize", (void*) gmGetSize);
    gm.fun.registerFunction("getType", (void*) gmGetType);
    gm.fun.registerFunction("getNdims", (void*) gmGetNdims);
    gm.fun.registerFunction("getNumel", (void*) gmGetNumel);
    gm.fun.registerFunction("getDataSize", (void*) gmGetDataSize);

    //gm.gputype.getGPUptr   = gmGetGPUptr;
    //gm.gputype.getSize     = gmGetSize;
    //gm.gputype.getType     = gmGetType;
    //gm.gputype.getNdims    = gmGetNdims;
    //gm.gputype.getNumel    = gmGetNumel;
    //gm.gputype.getDataSize = gmGetDataSize;

    /* setters */
    gm.fun.registerFunction("setSize", (void*) gmSetSize);
    //gm.gputype.setSize     = gmSetSize;

    /* properties */
    gm.fun.registerFunction("isFloat",   (void*) gmIsFloat);
    gm.fun.registerFunction("isDouble",  (void*) gmIsDouble);
    gm.fun.registerFunction("isComplex", (void*) gmIsComplex);
    gm.fun.registerFunction("isEmpty",   (void*) gmIsEmpty);
    gm.fun.registerFunction("isScalar",  (void*) gmIsScalar);

    //gm.gputype.isFloat     = gmIsFloat;
    //gm.gputype.isDouble    = gmIsDouble;
    //gm.gputype.isComplex   = gmIsComplex;

    gm.fun.registerFunction("clone", (void*) gmCloneGPUtype);
    gm.fun.registerFunction("create", (void*) gmCreateGPUtype);
    gm.fun.registerFunction("colon", (void*) gmColon);
    gm.fun.registerFunction("createMx", (void*) gmMxCreateGPUtype);
    gm.fun.registerFunction("toMxArray", (void*) gmToMxArray);
    gm.fun.registerFunction("createMxArray", (void*) gmCreateMxArray);
    gm.fun.registerFunction("createMxArrayPtr", (void*) gmCreateMxArrayPtr);
    gm.fun.registerFunction("mxToGPUtype", (void*) gmMxToGPUtype);
    gm.fun.registerFunction("getGPUtype", (void*) gmGetGPUtype);
    gm.fun.registerFunction("slice", (void*) gmCreateGPUtypeSlice);
    gm.fun.registerFunction("mxSlice", (void*) gmCreateGPUtypeMxSlice);
    gm.fun.registerFunction("assign", (void*) gmGPUtypeAssign);
    gm.fun.registerFunction("mxAssign", (void*) gmGPUtypeMxAssign);
    gm.fun.registerFunction("fill", (void*) gmGPUtypeFill);
    gm.fun.registerFunction("realimag", (void*) gmRealImag);

    gm.fun.registerFunction("permute", (void*) gmGPUtypePermute);
    gm.fun.registerFunction("mxPermute", (void*) gmGPUtypeMxPermute);


    /* casting */
    gm.fun.registerFunction("floatToDouble", (void*) gmGPUtypeFloatToDouble);
    gm.fun.registerFunction("doubleToFloat", (void*) gmGPUtypeDoubleToFloat);
    gm.fun.registerFunction("realToComplex", (void*) gmGPUtypeRealToComplex);
    gm.fun.registerFunction("realImagToComplex", (void*) gmGPUtypeRealImagToComplex);

    /* compiler */
    gm.fun.registerFunction("getCompileMode", (void*) gmGetCompileMode);
    gm.fun.registerFunction("compPushGPUtype", (void*) gmCompPushGPUtype);
    gm.fun.registerFunction("compPushMx", (void*) gmCompPushMx);

    gm.fun.registerFunction("compRegisterInstruction", (void*) gmCompRegisterInstruction);
    gm.fun.registerFunction("compAbort", (void*) gmCompAbort);

    gm.fun.registerFunction("compGetContextGPUtype", (void*) gmCompGetContextGPUtype);
    gm.fun.registerFunction("compGetContextMx", (void*) gmCompGetContextMx);

    gm.fun.registerFunction("compCreateMxContext", (void*) gmCompCreateMxContext);

    gm.fun.registerFunction("compFunctionEnd", (void*) gmCompFunctionEnd);
    gm.fun.registerFunction("compFunctionStart", (void*) gmCompFunctionStart);

    gm.fun.registerFunction("compFunctionSetParamInt", (void*) gmCompFunctionSetParamInt);
    gm.fun.registerFunction("compFunctionSetParamFloat", (void*) gmCompFunctionSetParamFloat);
    gm.fun.registerFunction("compFunctionSetParamDouble", (void*) gmCompFunctionSetParamDouble);

    gm.fun.registerFunction("compFunctionSetParamGPUtype", (void*) gmCompFunctionSetParamGPUtype);
    gm.fun.registerFunction("compFunctionSetParamMx", (void*) gmCompFunctionSetParamMx);
    gm.fun.registerFunction("compFunctionSetParamMxMx", (void*) gmCompFunctionSetParamMxMx);



    gm.fun.registerFunction("debugPushInstructionStack", (void*) gmDebugPushInstructionStack);
    gm.fun.registerFunction("getDebugMode", (void*) gmGetDebugMode);
    gm.fun.registerFunction("setDebugMode", (void*) gmSetDebugMode);
    gm.fun.registerFunction("debugLog", (void*) gmDebugLog);
    gm.fun.registerFunction("debugLogPop", (void*) gmDebugLogPop);
    gm.fun.registerFunction("debugLogPush", (void*) gmDebugLogPush);
    gm.fun.registerFunction("debugReset", (void*) gmDebugReset);




#include "GPUmatInterfaceFunctions_init.hh"
#include "GPUmatFFTFunctions_init.hh"

    if (nlhs > 1) {
      plhs[1] = mxCreateDoubleScalar(UINTPTR &gm);
    }

    // It is necessary to register mexAtExit for the following reasons
    // 1) I get an error at Matlab exit if GPUstart is the last command, but not if GPUstop
    //    is the last command. It means that the modules should be released, maybe it is a CUDA
    //    problem

    mexAtExit(GPUmanagerClose);

    goto cleanup;

error: ret_error = 1;

    status = cuCtxDestroy(cuContext);
    if (status != CUDA_SUCCESS) {
      mexErrMsgTxt(
        "Error context destroy. This is an internal error. Please report a bug to gp-you@gp-you.org.");
    }

cleanup:
    // clean up
    Myfree(fatbin);


    if (ret_error)
      mexErrMsgTxt(
      "Error in GPUmanager. This is an internal error. Please report a bug to gp-you@gp-you.org.");
  } else {
    if (init==0)
      return;
    CUresult status;


    status = cuModuleUnload(cuModule);
    if (CUDA_SUCCESS != status) {
      mexErrMsgTxt("Unable to unload CUDA module");
    }
    // cuCtxDetach is obsolete in CUDA 4.0
    /*status = cuCtxDetach(cuContext);
    if (status != CUDA_SUCCESS) {
    mexErrMsgTxt(
    "Error in CUDA context cleanup.");
    }*/
    // The behaviour of cuCtxDestroy is strange and unexpected
    status = cuCtxDestroy(cuContext);
    /*if (status != CUDA_SUCCESS) {
    mexErrMsgTxt(
    "Error in CUDA context cleanup.");
    }*/
    p.cleanup();
    modulesLoaded = 0;

  }

}
