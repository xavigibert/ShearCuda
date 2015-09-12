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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifdef UNIX
#include <stdint.h>
#endif

#include "cublas.h"
#include "cuda_runtime.h"
#include "cuda.h"

#include "cudalib_common.h"
#include "cudalib_error.h"
#include "cudalib.h"

#include "GPUcommon.hh"
#include "GPUerror.hh"
#include "Queue.hh"
#include "GPUstream.hh"
#include "GPUmanager.hh"
#include "GPUtype.hh"

#include "kernelnames.h"

#define BUFFERSIZE 300
#define CLEARBUFFER memset(STRINGBUFFER,0,BUFFERSIZE);
#define CLEARCOMPINSTBUFFER memset(this->comp.instbuffer,0,MAXCOMPINSTBUFFER);
char STRINGBUFFER[BUFFERSIZE];
/* MATLAB dependent section*/
struct mxArray_tag {
    void    *reserved;
    int      reserved1[2];
    void    *reserved2;
    size_t  number_of_dims;
    unsigned int reserved3;
    struct {
        unsigned int    flag0 : 1;
        unsigned int    flag1 : 1;
        unsigned int    flag2 : 1;
        unsigned int    flag3 : 1;
        unsigned int    flag4 : 1;
        unsigned int    flag5 : 1;
        unsigned int    flag6 : 1;
        unsigned int    flag7 : 1;
        unsigned int    flag7a: 1;
        unsigned int    flag8 : 1;
        unsigned int    flag9 : 1;
        unsigned int    flag10 : 1;
        unsigned int    flag11 : 4;
        unsigned int    flag12 : 8;
        unsigned int    flag13 : 8;
    }   flags;
    size_t reserved4[2];
    union {
        struct {
            void  *pdata;
            void  *pimag_data;
            void  *reserved5;
            size_t reserved6[3];
        }   number_array;
    }   data;
};

static void printMx(struct mxArray_tag *tmp) {
  mexPrintf("%p\n",tmp->reserved);
  mexPrintf("%d\n",tmp->reserved1[0]);
  mexPrintf("%d\n",tmp->reserved1[1]);
  mexPrintf("%p\n",tmp->reserved2);
  mexPrintf("%d\n",tmp->number_of_dims);
  mexPrintf("%d\n",tmp->reserved3);
  mexPrintf("%d\n",tmp->reserved4[0]);
  mexPrintf("%d\n",tmp->reserved4[1]);

  mexPrintf("%p\n",tmp->data.number_array.pdata);
  mexPrintf("%p\n",tmp->data.number_array.pimag_data);
  mexPrintf("%p\n",tmp->data.number_array.reserved5);
  mexPrintf("%d\n",tmp->data.number_array.reserved6[0]);
  mexPrintf("%d\n",tmp->data.number_array.reserved6[1]);
  mexPrintf("%d\n",tmp->data.number_array.reserved6[2]);

  mexPrintf("%d\n",tmp->flags.flag0);
  mexPrintf("%d\n",tmp->flags.flag1);
  mexPrintf("%d\n",tmp->flags.flag2);
  mexPrintf("%d\n",tmp->flags.flag3);
  mexPrintf("%d\n",tmp->flags.flag4);
  mexPrintf("%d\n",tmp->flags.flag5);
  mexPrintf("%d\n",tmp->flags.flag6);
  mexPrintf("%d\n",tmp->flags.flag7);
  mexPrintf("%d\n",tmp->flags.flag8);
  mexPrintf("%d\n",tmp->flags.flag9);
  mexPrintf("%d\n",tmp->flags.flag10);
  mexPrintf("%d\n",tmp->flags.flag11);
  mexPrintf("%d\n",tmp->flags.flag12);
  mexPrintf("%d\n",tmp->flags.flag13);
}
/* end Matlab section */

GPUmanager::GPUmanager(int n) :
	nStreams(0), executiondelayed(0), cudaStreams(NULL), cuDevice(0),
			cuContext(0), cuModule(0), cuFunction(0), cuTexref(0), gpuKernelConfig(gpukernelconfig()), capability(10), memcache(cache()), extcache(extcaches()) {

	if (n > 0) {
		stream = (Queue<GPUstream> **) Mymalloc(n * sizeof(Queue<GPUstream> *));
		nStreams = n;

		// init streams
		for (int i = 0; i < nStreams; i++)
			stream[i] = new Queue<GPUstream> ();

	}

	// allocate textures reference
	//cuTexref = (CUtexref **) Mymalloc(10 * sizeof(CUtexref*));

#ifdef DEBUG
	FILE *fout=fopen("GPUmanager.dat","a+");
	fprintf(fout,"%p\n",this);
	fclose(fout);
#endif
	//#ifdef DEBUG
	//  debugout = fopen ("log.out","w+");
	//#endif
//#include "kerneltable.h"

	// INIT USER Module structures
	nModules = 0;
	for (int i=0;i<MAXMODULES;i++) {
		userModuleName[i] = 0;
		userModules[i] = 0;
	}

	// CACHE
	memcache.maxel = MAXCACHE;
	for (int i=0;i<MAXCACHE;i++) {
    memcache.ptrs[i] = 0;
    memcache.size[i] = 0;
  }

	// EXT POINTERS CACHE
	extcache.maxel = 0;
	extcache.ptr = NULL;
	extcache.cachemiss = 0;
	extcache.totalbyptr = 0;
	extcache.totalbyslot = 0;
	extcache.totalcleanup = 0;
	extcache.totalfreeslot = 0;
	extcache.totalfreeslotcached = 0;
	extcache.totalfreeslotcachedmiss = 0;


}

/* destructor */
GPUmanager::~GPUmanager() {

  this->cleanup();

}

void GPUmanager::cleanup() {
	// should flush all streams?
#ifdef DEBUG
	FILE *fout=fopen("GPUmanagerdelete.dat","a+");
	fprintf(fout,"%p\n",this);
	fclose(fout);
#endif

	if (cudaStreams != NULL) {
		for (int i = 0; i < nStreams; i++)
			cudaStreamDestroy(cudaStreams[i]);
		Myfree(cudaStreams);
		cudaStreams =NULL;
	}
	// clean up streams here before free(stream)

	if (nStreams > 0) {
		Myfree(stream);
		nStreams = 0;
	}

	// MODULES
  for (int i=0;i<MAXMODULES;i++) {
  	struct userModule *mymod = userModules[i];
  	if (mymod!=0) {

			delete userModules[i];
		}

  	// clean up name
  	if (userModuleName[i]!=0) {
  		Myfree(userModuleName[i]);
  	}

  	userModules[i] = 0;
  	userModuleName[i] = 0;

  }

  // USER FUNCTIONS
  for (int i=0;i<user.nFunctions;i++) {
    if (user.userFunctionName!=0)
      if (user.userFunctionName[i]!=0)
        Myfree(user.userFunctionName[i]);

  }
  if (user.userFunctions!=0)
    Myfree(user.userFunctions);
  if (user.userFunctionName!=0)
    Myfree(user.userFunctionName);

  user.userFunctions = 0;
  user.userFunctionName = 0;
  user.nFunctions = 0;

  // Clean up cache
  try {
    cacheClean();
  } catch (GPUexception ex) {
    //mexErrMsgTxt(ex.getError());
  }

  // External cache pointers clean up
  if (extcache.ptr!=NULL)
      Myfree(extcache.ptr);

  extcache.ptr = NULL;
  extcache.maxel = 0;


}

/*
 * Init CUDA module
 *
 * Load module and update all function symbols
 */

/*
 * Init cudaStreams
 */
GPUmatResult_t GPUmanager::initStreams() {
	GPUmatResult_t status = GPUmatSuccess;
	cudaError_t cudastatus;
	if (nStreams > 0 && cudaStreams == NULL) {
		cudaStreams = (cudaStream_t*) Mymalloc(nStreams * sizeof(cudaStream_t));
		for (int i = 0; i < nStreams; i++) {
			cudastatus = cudaStreamCreate(&(cudaStreams[i]));
			if (cudastatus != cudaSuccess)
				return GPUmatError;
		}
	}
	return status;
}
/* run */
GPUmatResult_t GPUmanager::run() {
	GPUmatResult status = GPUmatSuccess;
	int old = executiondelayed;
	executiondelayed = 0;

	for (int i = 0; i < nStreams; i++) {
#ifdef DEBUG
		OPENFOUT;
		fprintf(fout,"GPUmanager stream, %d\n",i);
		CLOSEFOUT;
#endif
		while (!stream[i]->isEmpty()) {
			GPUstream *s = stream[i]->dequeue();
			status = s->run();
			if (status != GPUmatSuccess)
				return status;
		}
	}

	executiondelayed = old;

	return status;
}
/* pushStream */
void GPUmanager::pushStream(GPUstream *s, int streamnumber) {

	stream[streamnumber]->enqueue(s);
}

/* print */
void GPUmanager::print() {
	for (int i = 0; i < nStreams; i++) {
		myPrintf("**********************************\n");
		myPrintf("Stream %d\n", i);
		int index = 0;
		GPUstream *s = stream[i]->getElementAt(index);
		while (s != NULL) {
			s->print();
			myPrintf("--------------------------------\n");

			index++;
			s = stream[i]->getElementAt(index);
		}

	}
}

/* resetError */
void GPUmanager::resetError() {
	strcpy(error.errbuffer, "");
	error.lasterror = GPUmatSuccess;
}

/* resetError */
void GPUmanager::setError(GPUmatResult_t errornumber, STRINGCONST char *str) {
	strcpy(error.errbuffer, str);
	error.lasterror = errornumber;
}

/* executionDelayed */
int GPUmanager::executionDelayed() {
	return executiondelayed;
}

/* setExecutionDelayed */
void GPUmanager::setExecutionDelayed() {
	executiondelayed = 1;
}
/* resetExecutionDelayed */
void GPUmanager::resetExecutionDelayed() {
	executiondelayed = 0;
}

/* registerGPUtype */
//template <class C>
//  void
//  GPUmanager::registerGPUtype(GPUtype<C> *) {};

/* set device */
void GPUmanager::setCuDevice(CUdevice *d) {
	cuDevice = d;
}

/* get device */
CUdevice *GPUmanager::getCuDevice() {
	return cuDevice;
}

/* set context */
void GPUmanager::setCuContext(CUcontext *c) {
	cuContext = c;
}

/* set cuFunction */
void GPUmanager::setCuFunction(CUfunction *c) {
	cuFunction = c;
}

/* set cuTexref */
void GPUmanager::setCuTexref(CUtexref *c) {
	cuTexref = c;
}

/* set module */
void GPUmanager::setCuModule(CUmodule *c) {
	cuModule = c;
}

/* set maxthreads */
void GPUmanager::setKernelMaxthreads(unsigned int mx) {
	this->gpuKernelConfig.maxthreads = mx;
}

/* get maxthreads */
unsigned int GPUmanager::getKernelMaxthreads() {
	return this->gpuKernelConfig.maxthreads;
}

/* get kernel config */
	gpukernelconfig_t * GPUmanager::getKernelConfig() {
		return &this->gpuKernelConfig;
	}


/* get context */
CUcontext *GPUmanager::getCuContext() {
	return cuContext;
}

/* get cuFunction */
CUfunction *GPUmanager::getCuFunction(int i) {
	return &(cuFunction[i]);
}

/* get cuTexref */
CUtexref *GPUmanager::getCuTexref(int i) {
	return &(cuTexref[i]);
}

/* get cuModule */
CUmodule *GPUmanager::getCuModule() {
	return cuModule;
}

/* CUDA capability */
void GPUmanager::setCudaCapability(int cp) {
	capability = cp;
}

int GPUmanager::getCudaCapability() {
	return capability;
}

/* throw cublas */
GPUmatResult_t GPUmanager::throwCublas(int cublasresult) {

	if (cublasresult == CUBLAS_STATUS_NOT_INITIALIZED) {
		this->setError(GPUmatCUBLAS_STATUS_NOT_INITIALIZED,
				"CUBLAS: environment not initialized.");

	} else if (cublasresult == CUBLAS_STATUS_ALLOC_FAILED) {
		this->setError(GPUmatCUBLAS_STATUS_ALLOC_FAILED,
				"CUBLAS: memory allocation failed.");

	} else if (cublasresult == CUBLAS_STATUS_INVALID_VALUE) {
		this->setError(GPUmatCUBLAS_STATUS_INVALID_VALUE, "CUBLAS: invalid value.");

	} else if (cublasresult == CUBLAS_STATUS_ARCH_MISMATCH) {
		this->setError(GPUmatCUBLAS_STATUS_ARCH_MISMATCH,
				"CUBLAS: architecture mismatch.");

	} else if (cublasresult == CUBLAS_STATUS_MAPPING_ERROR) {
		this->setError(GPUmatCUBLAS_STATUS_MAPPING_ERROR, "CUBLAS: mapping error.");

	} else if (cublasresult == CUBLAS_STATUS_EXECUTION_FAILED) {
		this->setError(GPUmatCUBLAS_STATUS_EXECUTION_FAILED,
				"CUBLAS: execution failed.");

	} else if (cublasresult == CUBLAS_STATUS_INTERNAL_ERROR) {
		this->setError(GPUmatCUBLAS_STATUS_INTERNAL_ERROR,
				"CUBLAS: internal error.");

	} else {
		this->setError(GPUmatCUBLAS_UNKNOWN_ERROR, "CUBLAS: unknown error.");
	}

	return error.lasterror;
}

/* throw cuda */
GPUmatResult_t GPUmanager::throwCuda(cudaError_t cudaresult) {
	this->setError(GPUmatCUDA_GENERIC_ERROR, "CUDA run-time error");
	return GPUmatError;
}

//**************************************************************
// FUNCTIONS
//**************************************************************
int GPUmanager::funRegisterFunction(STRINGCONST char *name, void *fh) {
  CLEARBUFFER

  // check if function is already present
  for (int i=0;i<user.nFunctions;i++) {
    if (strcmp(user.userFunctionName[i],name)==0) {
      CLEARBUFFER
      sprintf(STRINGBUFFER, "%s (%s)", ERROR_GPUMANAGER_FUNALREADYDEFINED,name );
      throw GPUexception(GPUmatError,STRINGBUFFER);
    }
  }
  // OK, add new function

  // First add new entry in the vector
  user.nFunctions++;
  void ** userFunctionsTmp = (void **) Mymalloc(user.nFunctions*sizeof(void*));
  for (int i=0;i<user.nFunctions-1;i++) {
    userFunctionsTmp[i] = user.userFunctions[i];
  }

  char ** userFunctionNameTmp = (char **) Mymalloc(user.nFunctions*sizeof(char *));
  // now I should allocate all names and copy
  for (int i=0;i<user.nFunctions-1;i++) {
    userFunctionNameTmp[i] = (char *) Mymalloc((strlen(user.userFunctionName[i])+1)*sizeof(char));
    strcpy(userFunctionNameTmp[i], user.userFunctionName[i]);
    Myfree(user.userFunctionName[i]);
  }
  userFunctionNameTmp[user.nFunctions-1] = (char *) Mymalloc((strlen(name)+1)*sizeof(char));
  strcpy(userFunctionNameTmp[user.nFunctions-1], name);

  // now sould delete old data and update with new
  if (user.userFunctions!=0)
    Myfree(user.userFunctions);
  if (user.userFunctionName!=0)
    Myfree(user.userFunctionName);

  user.userFunctions = userFunctionsTmp;
  user.userFunctionName = userFunctionNameTmp;

  user.userFunctions[user.nFunctions-1] = fh;

  return user.nFunctions-1;

}

void * GPUmanager::funGetFunctionByName(STRINGCONST char *name) {

  // check if function is present
  int funindex = -1;
  for (int i=0;i<user.nFunctions;i++) {
    if (strcmp(user.userFunctionName[i],name)==0)
      funindex = i;
  }
  if (funindex == -1) {
    CLEARBUFFER
    sprintf(STRINGBUFFER, "%s (%s)", ERROR_GPUMANAGER_FUNNOTDEFINED,name );
    throw GPUexception(GPUmatError,STRINGBUFFER);

  }

  // OK, I have the function
  return (user.userFunctions[funindex]);
}

int    GPUmanager::funGetFunctionNumber(STRINGCONST char *name) {

  // check if function is present
  int funindex = -1;
  for (int i=0;i<user.nFunctions;i++) {
    if (strcmp(user.userFunctionName[i],name)==0)
      funindex = i;
  }
  if (funindex == -1) {
    CLEARBUFFER
    sprintf(STRINGBUFFER, "%s (%s)", ERROR_GPUMANAGER_FUNNOTDEFINED,name );
    throw GPUexception(GPUmatError,STRINGBUFFER);
  }

  // OK, I have the function
  return funindex;
}

void * GPUmanager::funGetFunctionByNumber(int findex) {
  if (findex < user.nFunctions) {
    return user.userFunctions[findex];
  } else {
    throw GPUexception(GPUmatError,ERROR_GPUMANAGER_FUNNOTDEFINED);
  }
}

//**************************************************************
// MODULES
//**************************************************************

int GPUmanager::getNModules() {
  return nModules;
}

char * GPUmanager::getModuleName(int n) {
	if (n<MAXMODULES) {
    return userModuleName[n];
	} else {
		return 0;
	}
}

void GPUmanager::deleteUserModule(char * modname) {
	CUresult status;
	int modnumber = getUserModuleNumberByName(modname);
	if (modnumber<0) {
		throw GPUexception(GPUmatError,
								ERROR_GPUMANAGER_INVMODNAME);
	}
	struct userModule *mymod = getUserModule(modnumber);
	// unload module
	status = cuModuleUnload(mymod->cuModule);
	if (CUDA_SUCCESS != status) {
		// clean up
		throw GPUexception(GPUmatError,
												ERROR_GPUMANAGER_UNABLEUNLOADKER);
	}

	Myfree(userModuleName[modnumber]);
	userModuleName[modnumber]=0;
	delete mymod;
	userModules[modnumber] =0;
	nModules--;


}

void GPUmanager::registerUserModule(char * name, char *kernelname ) {
	CUresult status;
	int freemodindex = -1;
	if (nModules<MAXMODULES) {
		// check if module exists and also finds the first available

		for (int i=0;i<MAXMODULES;i++) {
			if (userModules[i]!=0) {
				if (strcmp(userModuleName[i],name)==0)
					throw GPUexception(GPUmatError,ERROR_GPUMANAGER_MODALREADYDEFINED);
			} else {
				// module is 0, it is available
				if (freemodindex==-1)
					freemodindex = i;
			}
		}
	  // load fatbin
		// load kernel from file
		 char *kernel;
		int size = 0;
		FILE *f = fopen(kernelname, "rb");
		if (f == NULL) {
			kernel = NULL;
			throw GPUexception(GPUmatError,
											ERROR_GPUMANAGER_OPENKERFILE);
		}
		fseek(f, 0, SEEK_END);
		size = ftell(f);
		fseek(f, 0, SEEK_SET);
		kernel = (char *) Mymalloc((size + 1) * sizeof(char));
		if (size != fread(kernel, sizeof(char), size, f)) {
			Myfree(kernel);
			throw GPUexception(GPUmatError,
													ERROR_GPUMANAGER_READKERFILE);
		}
		fclose(f);
		kernel[size] = 0;
		// load kernel
		// load kernel using cuModuleLoadData
		CUmodule cuModuleTmp;
		status = cuModuleLoadData(&cuModuleTmp, kernel);
		if (CUDA_SUCCESS != status) {
			// clean up
			Myfree(kernel);
			throw GPUexception(GPUmatError,
													ERROR_GPUMANAGER_UNABLELOADKER);
		}

		// clean up char*
		Myfree(kernel);

		userModuleName[freemodindex] = (char *) Mymalloc((strlen(name)+1)*sizeof(char));
		memset(userModuleName[freemodindex],0,strlen(name)+1);
		strcpy(userModuleName[freemodindex],name);

		// now allocate module
    userModules[freemodindex] = new userModule();
		// update module
		userModules[freemodindex]->cuModule = cuModuleTmp;

    nModules++;

	} else {
		throw GPUexception(GPUmatError,
								ERROR_GPUMANAGER_MAXMODNUM);
	}
}

int GPUmanager::getUserModuleNumberByName(char *name) {
	for (int i=0;i<MAXMODULES;i++) {
		if (userModules[i]!=0) {
			if (strcmp(userModuleName[i],name)==0)
				return i;
		}
	}
	return -1;
}

struct userModule * GPUmanager::getUserModule(int n) {
	if (n<MAXMODULES)
	  return userModules[n];
	else
		return 0;
}

void GPUmanager::registerUserFunction(char *modname, char *funname) {
	CUresult status;

  // get module number
	int modnumber = getUserModuleNumberByName(modname);
	if (modnumber<0) {
		throw GPUexception(GPUmatError,
								ERROR_GPUMANAGER_INVMODNAME);
	}
	struct userModule *mymod = getUserModule(modnumber);
  // check if function is already present
	for (int i=0;i<mymod->nFunctions;i++) {
		if (strcmp(mymod->cuFunctionName[i],funname)==0)
			throw GPUexception(GPUmatError,
									ERROR_GPUMANAGER_FUNALREADYDEFINED);
	}
	// OK, add new function
	CUfunction cuFunctionNew;
	status = cuModuleGetFunction(&cuFunctionNew, mymod->cuModule, funname);
	if (CUDA_SUCCESS != status) {
		throw GPUexception(GPUmatError,ERROR_GPUMANAGER_LOADFUN);
	}

	// First add new entry in the vector
	mymod->nFunctions++;
	CUfunction * cuFunctionTmp = (CUfunction *) Mymalloc(mymod->nFunctions*sizeof(CUfunction));
	for (int i=0;i<mymod->nFunctions-1;i++) {
		cuFunctionTmp[i] = mymod->cuFunction[i];
	}

	char ** cuFunctionNameTmp = (char **) Mymalloc(mymod->nFunctions*sizeof(char *));
  // now I should allocate all names and copy from mymod
	for (int i=0;i<mymod->nFunctions-1;i++) {
		cuFunctionNameTmp[i] = (char *) Mymalloc((strlen(mymod->cuFunctionName[i])+1)*sizeof(char));
		strcpy(cuFunctionNameTmp[i], mymod->cuFunctionName[i]);
		Myfree(mymod->cuFunctionName[i]);
	}
	cuFunctionNameTmp[mymod->nFunctions-1] = (char *) Mymalloc((strlen(funname)+1)*sizeof(char));
	strcpy(cuFunctionNameTmp[mymod->nFunctions-1], funname);

	// now sould delete old data and update with new
	if (mymod->cuFunction!=0)
	  Myfree(mymod->cuFunction);
	if (mymod->cuFunctionName!=0)
	  Myfree(mymod->cuFunctionName);

	mymod->cuFunction = cuFunctionTmp;
	mymod->cuFunctionName = cuFunctionNameTmp;

	mymod->cuFunction[mymod->nFunctions-1] = cuFunctionNew;
}

CUfunction * GPUmanager::getUserFunctionByName(char *modname, char *funname) {
	// get module number
	int modnumber = getUserModuleNumberByName(modname);
	if (modnumber<0) {
		throw GPUexception(GPUmatError,
								ERROR_GPUMANAGER_INVMODNAME);
	}
	struct userModule *mymod = getUserModule(modnumber);
	// check if function is present
	int funindex = -1;
	for (int i=0;i<mymod->nFunctions;i++) {
		if (strcmp(mymod->cuFunctionName[i],funname)==0)
			funindex = i;
	}
	if (funindex == -1)
			throw GPUexception(GPUmatError,ERROR_GPUMANAGER_FUNNOTDEFINED);

	// OK, I have the function
  return &(mymod->cuFunction[funindex]);
}

struct userModule ** GPUmanager::getUserModules() {
	return userModules;
}

//**************************************************************
// EXT MEMORY CACHE
//**************************************************************

void *GPUmanager::extCacheGetGPUtypePtr(void * p0) {
  this->extcache.totalbyptr++;
  // search for an empty slot
  for (int i=0;i<this->extcache.maxel;i++) {
    void * ptr = this->extcache.ptr[i].id;
    if (ptr==p0) {
      return this->extcache.ptr[i].gp;
    }
  }
  return NULL;
}

void *GPUmanager::extCacheGetGPUtypePtrBySlot(int slot) {
  this->extcache.totalbyslot++;
  if (slot>=0 && slot<extcache.maxel) {
    return this->extcache.ptr[slot].gp;
  } else {
    return NULL;
  }
}

void GPUmanager::extCacheFreeGPUtypePtr(int slot) {
  // NOT yet implemented
  return;
  gpuTYPE_t mxtype = this->extcache.ptr[slot].mxtype;
  if (mxtype != gpuNOTDEF) { // gpuNOTDEF means an object that I created internally
    void * ptr = this->extcache.ptr[slot].mx;
    struct mxArray_tag * tmp = (struct mxArray_tag *) ptr;
    if (tmp->reserved2==NULL) {
      void * gptr = this->extcache.ptr[slot].gp;
      if (gptr!=NULL) {
        GPUtype *todel = (GPUtype*) gptr;
        delete todel;
        this->extcache.ptr[slot].gp = NULL;
        this->extcache.ptr[slot].assigned = 0;
      }
    }
  }
}

/* This function searches for non cleaned cached elements */
void GPUmanager::extCacheCleanUp() {
  // non implemented yet
  return;
  this->extcache.totalcleanup++;

  for (int i=0;i<extcache.maxel;i++) {
    extCacheFreeGPUtypePtr(i);
  }
}

/*
 * The cache can have some elements defined as "cached" and some not. The "cached" elements are created
 * outside in Matlab and are persistent to some locked function. It means that they will never be deleted,
 * but we now that with some internal flags we can find out if the element is "alone" or linked. 
 *
 * The way we request for a free slot is the following
 * 1. We can request a slot with mxtype = gpuNOTDEF. It means that we are not looking for a cached GPUsingle/GPUdouble
 * 2. We can request a slot with mxtype != gpuNOTDEF. It means that we are looking for a cached GPUsingle/GPUdouble
 *
 * A slot is always returned. If mxtype!=gpuNOTDEF and the slot was not found then a slot with mxtype==gpuNOTDEF 
 * is returned (and the corresponding pointer returned will be NULL).   
 * 
 * The first iteration searches for an empty slot (The search is for both mxtype==gpuNOTDEF and mxtype!=gpuNOTDEF). 
 * The iteration performs also a memory cleanup. If no slot was found after the first iteration, then I have to extend
 * the memory and return the first element in the new allocated space. 
 *
 */

void * GPUmanager::extCacheGetFreeSlot(int *slot, gpuTYPE_t gtype) {
  int slotidx = -1;

	// slotcached and slotnoncached are used 
	// to store found slot for non cached and cached
	// elements
	
	int slotnoncached = -1;
  this->extcache.totalfreeslot++;

  if (gtype!=gpuNOTDEF) {
      this->extcache.totalfreeslotcached++;
  }
  // first clean up cache
  //this->extCacheCleanUp();

  for (int i=0;i<this->extcache.maxel;i++) {
		// cleanup pointer in this phase NOT yet implemented
		// extCacheFreeGPUtypePtr(i);

    void * ptr = this->extcache.ptr[i].gp;
    gpuTYPE_t mxtype = this->extcache.ptr[i].mxtype;
    if (ptr==0) {
			// I store a possible non cached candidate to use later
			if (mxtype==gpuNOTDEF) {
				if (slotnoncached==-1)
			    slotnoncached = i;
			} 

      // if the request is gpuNOTDEF, then I am not looking for cached
      // elements, I am just asking for a slot
      if (gtype==mxtype) {
				if (slotidx==-1) {
          slotidx = i;
          break;
				}
      }
    }
  }

	// keep some statistics
  if (gtype!=gpuNOTDEF) {
    if (slotidx==-1)
      this->extcache.totalfreeslotcachedmiss++;
  }
	//else
  //  this->extcache.totalfreeslotcachexxx++;

	// if slotidx was not found, try with non cached slot
  if (slotidx==-1)
	  slotidx = slotnoncached;

  if (slotidx==-1) {
    // didn't find anything, I have to allocate more space

    // oops, no space left, I have to increase the size
    struct extcacheel *tmp0 = (struct extcacheel *)Mymalloc((this->extcache.maxel+MAXEXTCACHE)*sizeof(struct extcacheel));

    memset(tmp0,0,(this->extcache.maxel+MAXEXTCACHE)*sizeof(struct extcacheel));

    for (int i=0;i<this->extcache.maxel+MAXEXTCACHE;i++) {
      if (i<this->extcache.maxel) {
				// copy existing elements
        tmp0[i].assigned = this->extcache.ptr[i].assigned;
        tmp0[i].gp = this->extcache.ptr[i].gp;
        tmp0[i].mx = this->extcache.ptr[i].mx;
        tmp0[i].id = this->extcache.ptr[i].id;
        tmp0[i].mxtype = this->extcache.ptr[i].mxtype;
      } else {
        tmp0[i].assigned = 0;
        tmp0[i].gp = 0;
        tmp0[i].mx = 0;
        tmp0[i].id = 0;
        tmp0[i].mxtype = gpuNOTDEF;
      }
    }

    int index = this->extcache.maxel; // last element is for sure the free slot
    this->extcache.maxel += MAXEXTCACHE;

    if (extcache.ptr!=NULL)
      Myfree(extcache.ptr);

    this->extcache.ptr = tmp0;

    slotidx = index;
  }
  *slot = slotidx;
  return extcache.ptr[slotidx].mx;
}


void GPUmanager::extCachePrint() {
  mexPrintf("*** Begin cache\n");
  mexPrintf("Cache slot requests         -> %d\n", this->extcache.totalfreeslot);
  mexPrintf("Cache slot requests (cached)-> %d\n", this->extcache.totalfreeslotcached);
  mexPrintf("Cache slot request miss     -> %d\n", this->extcache.totalfreeslotcachedmiss);
  mexPrintf("Cache by slot               -> %d\n", this->extcache.totalbyslot);
  mexPrintf("Cache by ptr                -> %d\n", this->extcache.totalbyptr);
  mexPrintf("Cache clean up              -> %d\n", this->extcache.totalcleanup);
  mexPrintf("Cache miss                  -> %d\n", this->extcache.cachemiss);

  for (int i=0;i<this->extcache.maxel;i++) {
    void * ptr = this->extcache.ptr[i].gp;
    void * id = this->extcache.ptr[i].id;
    void * mx =  this->extcache.ptr[i].mx;
    if (mx!=0) {
      struct mxArray_tag *tmp = (struct mxArray_tag *) mx;
      mexPrintf("N. %d reserved -> %p\n", i, tmp->reserved2);
      //printMx(tmp);
      //if (tmp->reserved2)
      //  mexPrintf("N. %d reserved -> %p\n", i, tmp->reserved2);
    }
    if (ptr!=0) {
      gpuTYPE_t mxtype = this->extcache.ptr[i].mxtype;
      if (mxtype != gpuNOTDEF)
        mexPrintf("(Cached) N. %d -> %p %p\n", i, id, this->extcache.ptr[i].gp);
      else
        mexPrintf("N. %d -> %p %p\n", i, id, this->extcache.ptr[i].gp);
    }
  }
  mexPrintf("*** End cache\n");


}


void GPUmanager::extCacheCleanPtrBySlot(int slot) {
  // I can't delete cached elements
  if (this->extcache.ptr[slot].mxtype==gpuNOTDEF) {
    this->extcache.ptr[slot].id = 0;
    this->extcache.ptr[slot].mx = 0;
    this->extcache.ptr[slot].mxtype = gpuNOTDEF;

  }
  // search for an empty slot
  this->extcache.ptr[slot].assigned = 0;
  this->extcache.ptr[slot].gp = 0;


}

/* 
 */
void GPUmanager::extCacheRegisterPtrBySlot(int idx, void *id, void * p1, void * p2, gpuTYPE_t mxtype) {

  // internal check
  if ((mxtype==gpuNOTDEF)&&(p1==NULL)) {

  }

  if ((mxtype!=gpuNOTDEF)&&(p1!=NULL)) {

  }

  // mxtype == gpuNOTDEF is a non cached element
  this->extcache.ptr[idx].mxtype = mxtype;

  if (id!=NULL)
      this->extcache.ptr[idx].id = id;

  if (mxtype==gpuNOTDEF) {
    this->extcache.ptr[idx].mx = p1;
  } else {
    this->extcache.ptr[idx].assigned = 1;
  }

  if (p2!=NULL)
    this->extcache.ptr[idx].gp = p2;
}

/*
 */
void GPUmanager::extCacheRegisterCachedPtrBySlot(int idx, void *id, void * p1, gpuTYPE_t mxtype) {


  // mxtype == gpuNOTDEF is a non cached element
  this->extcache.ptr[idx].mxtype = mxtype;
  this->extcache.ptr[idx].id = id;
  this->extcache.ptr[idx].mx = p1;
  this->extcache.ptr[idx].gp = NULL;
  this->extcache.ptr[idx].assigned = 0;

}


void GPUmanager::extCacheCacheMiss() {
  this->extcache.cachemiss++;
}


//**************************************************************
// MEMORY CACHE
//**************************************************************

// register a pointer to GPU memory
void GPUmanager::cacheRegisterPtr(void * p) {
  // rules. If the pointer has more than two links
  // then I cannot store in cache
  GPUtype *tmp = (GPUtype*) p;

  if (tmp->getPtrCount()==1) {
    // search for an empty slot
    for (int i=0;i<this->memcache.maxel;i++) {
      GPUtype * ptr = (GPUtype *) this->memcache.ptrs[i];
      if (ptr==0) {
        // register
        this->memcache.ptrs[i] = new GPUtype(*tmp);
        this->memcache.size[i] = tmp->getMySize()*tmp->getNumel();
        //mexPrintf("Cache in %p -> %p %d\n", tmp, this->memcache.ptrs[i], tmp->getMySize()*tmp->getNumel());
        return;
      }
    }
  }
}

// Request a pointer of specified size
void * GPUmanager::cacheRequestPtr(int size) {
  // Policy
  // 1) Search a slot with enough memory
  // 2) Between all possibilities with enough memory returns
  //    the option with less memory
  // 3) Do not return a pointer if the occupied memory is
  //    > some percentage of the requested size

  GPUtype *res=0;
  int sres = 0;
  double ratio = 1.5;
  int countptr = 0;
  for (int i=0;i<this->memcache.maxel;i++) {
    GPUtype * ptr = (GPUtype *) this->memcache.ptrs[i];
    if ((ptr!=0)&&(ptr->getPtrCount()==1)) {
      // check conditions
      // size cannot be double
      countptr++;
      int sptr = this->memcache.size[i];
      if ((size<=sptr)&&(size*ratio>sptr)) {
        if (res==0) {
          res = ptr;
          sres = sptr;
        } else {
          if (sptr<sres) {
            res = ptr;
            sres = sptr;
          }
        }
      }
    }
  }
  //if (res)
    //mexPrintf("Cache out <- %p %d\n", res, sres);
  // clean the cache if the returned pointer is NULL
  if ((res==NULL)&&(countptr==this->memcache.maxel)) {
    cacheClean();
  }
  return res;
}

// Empty the cache
void GPUmanager::cacheClean() {

  // first clean external cache
  //this->extCacheCleanUp();

  for (int i=0;i<this->memcache.maxel;i++) {
    GPUtype * ptr = (GPUtype *) this->memcache.ptrs[i];
    if (ptr!=0) {
      //mexPrintf("Cache clear <- %p %d %d\n", ptr, this->memcache.size[i], ptr->getPtrCount());
      // delete
      delete ptr;
      this->memcache.ptrs[i] = 0;
      this->memcache.size[i] = 0;
    }
  }
}

// Empty the cache
void GPUmanager::cachePrint() {
  for (int i=0;i<this->memcache.maxel;i++) {
    GPUtype * ptr = (GPUtype *) this->memcache.ptrs[i];
    if (ptr!=0) {
      mexPrintf("Cache N. %d -> %p %d\n", i, ptr, this->memcache.size[i]);
    }
  }
}

//**************************************************************
// DEBUG
//**************************************************************

/// Compilation mode
int GPUmanager::getDebugMode() {
  return debug.debugmode;
}

void GPUmanager::setDebugMode(int mode) {
  debug.debugmode = mode;
}

/// Pushes instruction in the debug stack
void GPUmanager::debugPushInstructionStack(int inst) {
  // the instruction stack is a FILO stack

  for (int i=MAXDEBUGINSTR-1;i>0;i--) {
    debug.instructionStack[i] = debug.instructionStack[i-1];
  }
  debug.instructionStack[0] = inst;
}

void GPUmanager::debugSetVerbose(int v) {
  this->debug.verbose = v;
}

int GPUmanager::debugGetVerbose() {
  return this->debug.verbose;
}

void GPUmanager::debugPopIndent() {
  if (debug.indent>0)
    debug.indent--;
}

void GPUmanager::debugPushIndent() {
  debug.indent++;
}

void GPUmanager::debugResetIndent() {
  debug.indent=0;
}

void GPUmanager::debugReset() {
  debug.indent =0;
  debug.verbose = 0;
  debug.debugmode = 0;
}


int GPUmanager::debugGetIndent() {
  return debug.indent;
}

void GPUmanager::debugLog (STRINGCONST char *str, int v) {
  if (v<this->debugGetVerbose()) {
    for (int i=0;i<this->debugGetIndent();i++)
      mexPrintf("  ");
    mexPrintf(str);
  }
}


//**************************************************************
// COMPILER
//**************************************************************

void GPUmanager::compForCountIncrease() {
  this->comp.forcount++;
}

void GPUmanager::compForCountDecrease() {
  this->comp.forcount--;
}

int GPUmanager::compGetFourCount() {
  return this->comp.forcount;
}

/// Write instruction to compiled file
/**
* @param[in] inst Instruction to be printed to file.
* @return
*/
void GPUmanager::compRegisterInstruction(STRINGCONST char *inst, int type) {
  if (this->comp.compilemode == 0) {
    this->compAbort(ERROR_GPUMANAGER_STARTCOMPILEMODE);
  }

  if (this->comp.filename == NULL)
    this->compAbort(ERROR_GPUMANAGER_NULLCOMPFILENAME);

  if (this->comp.buffer1 == NULL)
    this->compAbort(ERROR_GPUMANAGER_COMPNOTINITIALIZED);

  if (this->comp.buffer2 == NULL)
    this->compAbort(ERROR_GPUMANAGER_COMPNOTINITIALIZED);

  // type is used to manage different files or buffers
  if (type==0) {
    //FILE *fout=fopen(this->comp.filename,"a+");
    //fprintf(fout,"%s\n",inst);
    //fclose(fout);

    // register to buffer1
    int lsrc = strlen(inst)+strlen("\n");
    int ldst = strlen(this->comp.buffer1);

    // remember that teh last character is the NULL
    if ((ldst+lsrc) >= this->comp.bf1size) {
      //Have to allocate a new buffer
      char *tmp = (char *) Mymalloc(this->comp.bf1size*sizeof(char));
      strcpy(tmp,this->comp.buffer1);
      Myfree(this->comp.buffer1);

      this->comp.bf1size = this->comp.bf1size+lsrc+INITBUFFERSIZE;
      this->comp.buffer1 = (char *) Mymalloc(this->comp.bf1size*sizeof(char));
      memset(this->comp.buffer1,0,this->comp.bf1size);
      strcpy(this->comp.buffer1, tmp);
      Myfree(tmp);
    }
    sprintf(this->comp.buffer1,"%s%s\n",this->comp.buffer1,inst);
  }

  if (type==1) {

    // register to buffer2
    int lsrc = strlen(inst)+strlen("\n");
    int ldst = strlen(this->comp.buffer2);

    // remember that the last character is the NULL
    if ((ldst+lsrc) >= this->comp.bf2size) {
      //Have to allocate a new buffer
      char *tmp = (char *) Mymalloc(this->comp.bf2size*sizeof(char));
      strcpy(tmp,this->comp.buffer2);
      Myfree(this->comp.buffer2);

      this->comp.bf2size = this->comp.bf2size+lsrc+INITBUFFERSIZE;
      this->comp.buffer2 = (char *) Mymalloc(this->comp.bf2size*sizeof(char));
      memset(this->comp.buffer2,0,this->comp.bf2size);
      strcpy(this->comp.buffer2, tmp);
      Myfree(tmp);
    }
    sprintf(this->comp.buffer2,"%s%s\n",this->comp.buffer2,inst);

  }

}

void GPUmanager::compFunctionStart(STRINGCONST char *name) {
  CLEARCOMPINSTBUFFER
  // reset function parameters counter
  this->comp.par_id = 0;
  sprintf(this->comp.instbuffer,"%s (", name);

  CLEARBUFFER
  sprintf(STRINGBUFFER, "START FUNCTION %s \n", name);
  this->debugLog(STRINGBUFFER, 3);
}

void GPUmanager::compFunctionSetParamGPUtype(void *r) {
  int ridx = this->compGetContext(r,STACKGPUTYPE);
  if (ridx==-1) {
    this->compAbort(ERROR_GPUMANAGER_COMPINCONSGPUTYPE);
  }
  CLEARBUFFER
  sprintf(STRINGBUFFER, "  GPUARG%d\n", ridx);
  this->debugLog(STRINGBUFFER, 3);


  CLEARBUFFER
  sprintf(STRINGBUFFER,"GPUTYPEID(%d)", ridx);
  this->compFunctionSetParam(STRINGBUFFER);

}

void GPUmanager::compFunctionSetParamInt(int par) {
  CLEARBUFFER
  sprintf(STRINGBUFFER, "  INT %d\n", par);
  this->debugLog(STRINGBUFFER, 3);

  CLEARBUFFER
  sprintf(STRINGBUFFER,"%d", par);
  this->compFunctionSetParam(STRINGBUFFER);
}

void GPUmanager::compFunctionSetParamFloat(float par) {
  CLEARBUFFER
  sprintf(STRINGBUFFER, "  FLOAT %d\n", par);
  this->debugLog(STRINGBUFFER, 3);

  CLEARBUFFER
  sprintf(STRINGBUFFER,"%f", par);
  this->compFunctionSetParam(STRINGBUFFER);
}

void GPUmanager::compFunctionSetParamDouble(double par) {
  CLEARBUFFER
  sprintf(STRINGBUFFER, "  DOUBLE %d\n", par);
  this->debugLog(STRINGBUFFER, 3);

  CLEARBUFFER
  sprintf(STRINGBUFFER,"%f", par);
  this->compFunctionSetParam(STRINGBUFFER);
}


void GPUmanager::compFunctionSetParam(STRINGCONST char *inst) {
  if (this->comp.par_id==0)
    sprintf(this->comp.instbuffer,"%s%s", this->comp.instbuffer, inst);
  else
    sprintf(this->comp.instbuffer,"%s, %s", this->comp.instbuffer, inst);
  this->comp.par_id++;
}

void GPUmanager::compFunctionEnd() {
  sprintf(this->comp.instbuffer,"%s)", this->comp.instbuffer);
  this->compRegisterInstruction(this->comp.instbuffer);

  CLEARBUFFER
  sprintf(STRINGBUFFER, "END FUNCTION \n");
  this->debugLog(STRINGBUFFER, 3);
}




/// Set compiled filename
/**
* @param[in] filename File name
* @return
*/
void GPUmanager::compSetFilename(STRINGCONST char *filename, int type) {
  if (this->comp.filename!=NULL) {
    Myfree(this->comp.filename);
    this->comp.filename=NULL;
  }
  // allocate file name
  this->comp.filename = (char *) Mymalloc((strlen(filename)+1)*sizeof(char));
  memset(this->comp.filename,0,strlen(filename)+1);
  strcpy(this->comp.filename,filename);
}

/// Returned compiled filename
/**
* @return File name
*/
char * GPUmanager::compGetFilename(int type) {
  return (this->comp.filename);
}

void GPUmanager::compClear() {
  // clear file names
  if (this->comp.filename!=NULL) {
    Myfree(this->comp.filename);
  }
  this->comp.filename=NULL;

  if (this->comp.filename1!=NULL) {
    Myfree(this->comp.filename1);
  }
  this->comp.filename1=NULL;

  // clear buffers
  
  if (this->comp.instbuffer!=NULL) {
    Myfree(this->comp.instbuffer);
  }
  this->comp.instbuffer=NULL;

  if (this->comp.buffer1!=NULL) {
    Myfree(this->comp.buffer1);
  }
  this->comp.buffer1=NULL;
  this->comp.bf1size = INITBUFFERSIZE;

  if (this->comp.buffer2!=NULL) {
    Myfree(this->comp.buffer2);
  }
  this->comp.buffer2=NULL;
  this->comp.bf2size = INITBUFFERSIZE;

  // stack
  if (this->comp.stack!=NULL) {
    Myfree(this->comp.stack);
  }
  this->comp.stack=NULL;

}

void GPUmanager::compStart(STRINGCONST char *filename, int header) {

  // reset indent
  this->debug.indent = 0;

  // set compilemode
  this->comp.compilemode = 1;

  // skip gpu execution
  //this->gpuKernelConfig.gpuexecute = 0;

  // var_id is used to tag the GPUtype or any other pointer
  // comp_id is an incremental index. It identifies the compilation number
  // When the compilation starts I have to update these indexes
  this->comp.var_id = 0;
  this->comp.gt_id = 0;
  this->comp.comp_id++;

  // resst for loop count
  this->comp.forcount = 0;

  if (this->comp.stack==NULL)
    this->comp.stack= (struct userCompileStack*) Mymalloc(MAXSTACK*sizeof(struct userCompileStack));
  if (this->comp.instbuffer==NULL)
    this->comp.instbuffer = (char *) Mymalloc(MAXCOMPINSTBUFFER*sizeof(char));
  CLEARCOMPINSTBUFFER

  // clean up stack
  for (int i=0;i<MAXSTACK;i++) {
    this->comp.stack[i].ptr=0;
    this->comp.stack[i].type=0;
    this->comp.stack[i].var_id=0;
    this->comp.stack[i].comp_id=0;
  }


  // initialize buffers
  if (this->comp.buffer1==NULL) {
    this->comp.buffer1 = (char*) Mymalloc(this->comp.bf1size*sizeof(char));
  }
  memset(this->comp.buffer1,0,this->comp.bf1size);


  if (this->comp.buffer2==NULL) {
    this->comp.buffer2 = (char*) Mymalloc(this->comp.bf2size*sizeof(char));
  }
  memset(this->comp.buffer2,0,this->comp.bf2size);




  // set the file name
  // open it, write header and close it
  this->compSetFilename(filename);
  FILE *fout=fopen(this->comp.filename,"w+");

  struct tm *current;
  time_t now;
  time(&now);
  current = localtime(&now);
  char dateStr [9];
  char timeStr [9];
  //_strdate( dateStr);
  //_strtime( timeStr );
  if (header) {
    fprintf(fout, "//*****************************************************************\n");
    fprintf(fout, "// File automatically generated with GPUmat (http://gp-you.org)\n");
    //fprintf(fout, "// Date: %i/%i/%i\n", current->tm_year, current->tm_mon, current->tm_mday);
    //fprintf(fout, "// Time: %i:%i:%i\n", current->tm_hour, current->tm_min, current->tm_sec);
    //fprintf(fout, "// Date: %s\n", dateStr);
    //fprintf(fout, "// Time: %s\n", timeStr);

    fprintf(fout, "//*****************************************************************\n");
  }
  fclose(fout);

}

void GPUmanager::compFlush() {
  if (this->comp.compilemode == 0) {
    this->compAbort(ERROR_GPUMANAGER_STARTCOMPILEMODE);
  }

  if (this->comp.filename == NULL)
    this->compAbort(ERROR_GPUMANAGER_NULLCOMPFILENAME);

  if (this->comp.buffer1 == NULL)
    this->compAbort(ERROR_GPUMANAGER_COMPNOTINITIALIZED);

  if (this->comp.buffer2 == NULL)
    this->compAbort(ERROR_GPUMANAGER_COMPNOTINITIALIZED);


  FILE *fout=fopen(this->comp.filename,"a+");
  fprintf(fout,"%s",this->comp.buffer2);
  fprintf(fout,"%s",this->comp.buffer1);
  fclose(fout);
}

void GPUmanager::compStop() {
  // unset compilemode
  if (this->comp.compilemode==1) {
    this->comp.compilemode = 0;
    //this->gpuKernelConfig.gpuexecute = 1;

    compClear();
  }

  debugReset();

}

void GPUmanager::compAbort(STRINGCONST char *msg) {
  debugReset();


  // unset compilemode
  if (this->comp.compilemode==1) {
    this->comp.compilemode = 0;
    //this->gpuKernelConfig.gpuexecute = 1;

    if (this->comp.filename!=NULL) {
      FILE *fout=fopen(this->comp.filename,"w+");
      fprintf(fout, "// COMPILATION aborted\n");
      fclose(fout);
    }

    if (this->comp.filename1!=NULL) {
      FILE *fout=fopen(this->comp.filename1,"w+");
      fprintf(fout, "// COMPILATION aborted\n");
      fclose(fout);
    }

    compClear();
    if (msg!=NULL)
      throw GPUexception(GPUmatError,msg);
  }

}

int GPUmanager::getCompileMode() {
  return this->comp.compilemode;
}

void GPUmanager::compStackNullPtr(void *ptmp, int type) {
  // I have to search first if the pointer is in the stack
  int exist = -1;
  for (int i=0;i<MAXSTACK;i++) {
    struct userCompileStack tmp= this->comp.stack[i];
    if ((tmp.type == type)&&(tmp.comp_id == this->comp.comp_id)&&(tmp.ptr == ptmp))
       exist = i;
  }
  if (exist!=-1) {
    // set the corresponding pointer to null
    this->comp.stack[exist].ptr = NULL;
  } else {
    // nothing
  }
}
void GPUmanager::compPush(void *ptmp, int type) {

  CLEARBUFFER
  sprintf(STRINGBUFFER,"PUSH (%p, %d)\n", ptmp, type);
  this->debugLog(STRINGBUFFER,30);

  if (type==0) {
    GPUtype *p =(GPUtype *) ptmp;
    // id1 is the compilation index
    // if the GPUtype has different compilation index
    // than current, the it should be updated
    if (p->getID1()==this->comp.comp_id) {

    } else {
      p->setID1(this->comp.comp_id);
      p->setID0(this->comp.gt_id);
      this->comp.gt_id++;

      CLEARBUFFER
      sprintf(STRINGBUFFER,"DECLARE_GPUTYPEID(%d)", p->getID0());
      this->compRegisterInstruction(STRINGBUFFER,1);
    }
  } else {

    // I have to search first if the pointer is already in the stack
    // The pointer is in the stack if:
    // 1) The pointer with specified type is in this->comp.stack
    // 2) The corresponding comp_id is equal to the current comp_id
    int exist = -1;
    for (int i=0;i<MAXSTACK;i++) {
      struct userCompileStack tmp= this->comp.stack[i];
      if ((tmp.type == type)&&(tmp.comp_id == this->comp.comp_id)&&(tmp.ptr == ptmp))
        exist = i;
    }
    if (exist==-1) {
      if (this->comp.var_id>=MAXSTACK) {
        throw GPUexception(GPUmatError,ERROR_GPUMANAGER_COMPSTACKOVERFLOW);
      }
      // OK, I can push the pointer
      int idx = this->comp.var_id;
      this->comp.stack[idx].ptr = ptmp;
      this->comp.stack[idx].var_id = idx;
      this->comp.stack[idx].comp_id = this->comp.comp_id;
      this->comp.stack[idx].type = type;

      if (type==1) {
        CLEARBUFFER
        sprintf(STRINGBUFFER,"DECLARE_MXNID(%d, 1)", idx);
        this->compRegisterInstruction(STRINGBUFFER,1);

        CLEARBUFFER
        sprintf(STRINGBUFFER,"DECLARE_MXID(%d, 1)", idx);
        this->compRegisterInstruction(STRINGBUFFER,1);
      }
      this->comp.var_id++;
    }
  }

}

int GPUmanager::compGetContext(void *ptmp, int type) {

  CLEARBUFFER
  sprintf(STRINGBUFFER,"GET CONTEXT (%p, %d)\n", ptmp, type);
  this->debugLog(STRINGBUFFER,30);

  if (type==0) {
    GPUtype *p =(GPUtype *) ptmp;
    // id1 is the compilation index
    // if the GPUtype has different compilation index
    // than current, the it should be updated
    if (p->getID1()==this->comp.comp_id) {
      return p->getID0();
    } else {
      return -1;
    }
  } else {
    // I have to search first if the pointer is in the stack
    // The pointer is in the stack if:
    // 1) The pointer is in this->comp.stack
    // 2) The corresponding comp_id is equal to the current comp_id
    int exist = -1;
    for (int i=0;i<MAXSTACK;i++) {
      struct userCompileStack tmp= this->comp.stack[i];
      if ((tmp.type == type)&&(tmp.comp_id == this->comp.comp_id)&&(tmp.ptr == ptmp))
         exist = i;
    }
    if (exist!=-1) {
      return this->comp.stack[exist].var_id;
    } else {
      return -1;
    }
  }
}

void GPUmanager::compClearContext(void *ptmp, int type) {

  CLEARBUFFER
  sprintf(STRINGBUFFER,"CLEAR CONTEXT (%p, %d)\n", ptmp, type);
  this->debugLog(STRINGBUFFER,30);

  if (type==0) {
    GPUtype *p =(GPUtype *) ptmp;

    p->setID0(0);
    p->setID1(0);

  } else {
    // I have to search first if the pointer is in the stack
    // The pointer is in the stack if:
    // 1) The pointer is in this->comp.stack
    // 2) The corresponding comp_id is equal to the current comp_id
    int exist = -1;
    for (int i=0;i<MAXSTACK;i++) {
      struct userCompileStack tmp= this->comp.stack[i];
      if ((tmp.type == type)&&(tmp.comp_id == this->comp.comp_id)&&(tmp.ptr == ptmp))
         exist = i;
    }
    if (exist!=-1) {
      this->comp.stack[exist].var_id = 0;
      this->comp.stack[exist].ptr = 0;
      this->comp.stack[exist].comp_id = 0;
      this->comp.stack[exist].type = 0;

    }
  }
}






