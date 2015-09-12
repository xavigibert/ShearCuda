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

#if !defined(GPUMANAGER_H_)
#define GPUMANAGER_H_

#define MAXSTREAMS 10

// CUDA Textures
#define N_TEXREF_F1_A   0
#define N_TEXREF_F1_B   1
#define N_TEXREF_C1_A   2
#define N_TEXREF_C1_B   3
#define N_TEXREF_D1_A   4
#define N_TEXREF_D1_B   5
#define N_TEXREF_CD1_A  6
#define N_TEXREF_CD1_B  7

#define N_TEXREF_I1_A  8

#define N_TEXREF_F2_A   9
#define N_TEXREF_C2_A   10
#define N_TEXREF_D2_A   11
#define N_TEXREF_CD2_A  12

#define MAXCUFUNCTION 600
#define MAXCUTEXREF   15

// maximum number of modules
#define MAXMODULES 100

// Compiler stuff
// debug. maximum number of instruction in the stack
#define MAXDEBUGINSTR 500
#define STACKGPUTYPE 0
#define STACKMX      1
#define STACKMXMX    2




// initial buffer size
#define INITBUFFERSIZE 1024


// maximum cache
#define MAXCACHE 50

// extcache blocks

// MAXEXTCACHE is the size of the non cached elements
// the cache is increased by MAXEXTCACHE
#define MAXEXTCACHE 1000

//**********************************************
// COMPILER
//**********************************************

// maximum cache
#define MAXSTACK 150
#define MAXCOMPINSTBUFFER 2048

struct userCompileStack {
	userCompileStack() : ptr(0), var_id(0), comp_id(0), type(0) {}
	void *ptr;
	int var_id;
	int comp_id;
	int type;
};

struct userCompile {

	userCompile(): filename(NULL), filename1(NULL), filehandle(NULL), var_id(0), gt_id(0), comp_id(0), par_id(0),
		forcount(0), compilemode(0), bf1size(INITBUFFERSIZE), buffer1(NULL), bf2size(INITBUFFERSIZE), buffer2(NULL),
		stack(0), instbuffer(0), compiletype(0) {

	}
	// file0 being generated
	char *filename;

	// file1 being generated
	char *filename1;

	// handle to the file
	FILE *filehandle;

	// Used as an index to tag any other variable than GPUtype
	unsigned int var_id;

	// Used as an index to tag GPUtype variables
	unsigned int gt_id;

	// Used as an index for current compilation
	unsigned int comp_id;

	// count function parameters
	unsigned int par_id;

	// count for-loops
	unsigned int forcount;

	// compilation mode
	int compilemode;

	// buffers
	int bf1size;
	char *buffer1;

	int bf2size;
	char *buffer2;

	// stack (array of compStack)
	struct userCompileStack * stack;

	// buffer
	char *instbuffer;

	// compilation type
	// 0 - Matlab
	int compiletype;

	~userCompile() {
		if (filename!=NULL) {
			Myfree(filename);
			filename = NULL;
		}

		if (filename1!=NULL) {
			Myfree(filename1);
			filename1 = NULL;
		}

		if (buffer1!=NULL) {
			Myfree(buffer1);
			buffer1 = NULL;
		}

		if (buffer2!=NULL) {
			Myfree(buffer2);
			buffer2 = NULL;
		}

		if (stack!=NULL) {
			Myfree(stack);
			stack = NULL;
		}

		if (instbuffer!=NULL) {
			Myfree(instbuffer);
			instbuffer = NULL;
		}
	}

};

//**********************************************
// USER MODULES
//**********************************************

struct userModule {
	userModule() {
		cuDevice   = 0;
		cuContext  = 0;
		cuModule   = 0;
		cuFunction = 0;
		cuTexref   = 0;
		cuFunctionName = 0;
		nFunctions = 0;
	}
	~userModule() {
		// clean up module
		for (int i=0;i<nFunctions-1;i++) {
			Myfree(cuFunctionName[i]);
		}
		// now should delete old data and update with new
		if (cuFunction!=0)
			Myfree(cuFunction);
		if (cuFunctionName!=0)
			Myfree(cuFunctionName);

	}
	CUdevice cuDevice;
	CUcontext cuContext;
	CUmodule cuModule;
	CUfunction *cuFunction;
	CUtexref *cuTexref;
	char ** cuFunctionName; // store function names
	int nFunctions;

};


//**********************************************
// DEBUG
//**********************************************

struct debugStruct {
	debugStruct() : debugmode(0), verbose(0), indent(0) {
		for (int i=0;i<MAXDEBUGINSTR;i++)
			instructionStack[i] = 0;
	}
	int instructionStack[MAXDEBUGINSTR];
	int debugmode;
	// verbose
	int verbose;

	// indent using in log
	int indent;

};

//**********************************************
// USER functions
//**********************************************

struct userFunction {
	userFunction() {
		userFunctions = 0;
		userFunctionName = 0;
		nFunctions = 0;
	}
	~userFunction() {
		// clean up
		for (int i=0;i<nFunctions;i++) {
			if (userFunctionName!=0)
				if (userFunctionName[i]!=0)
					Myfree(userFunctionName[i]);
		}
		if (userFunctionName!=0)
			Myfree(userFunctionName);
		if (userFunctions!=0)
			Myfree(userFunctions);

	}
	void ** userFunctions; // array of pointer to functions
	char ** userFunctionName; // store function names
	int nFunctions;

};


struct cache {
	void * ptrs[MAXCACHE];
	int maxel;
	int size[MAXCACHE];

	cache() {
		maxel = MAXCACHE;
		for (int i=0;i<MAXCACHE;i++) {
			ptrs[i] = 0;
			size[i] = 0;
		}
	}

};

/* extcache is used to store any external pointer. For example, in Matlab
* we store the mxArray pointers and associate to them a GPUtype
*/
struct extcacheel {
	void * id;
	void * mx;
	int   assigned; // 1 if assigned
	// gp stores the pointer to GPUtype
	void * gp;

	// stores the GPUtype type. We use the following rule:
	// 1. If mxtype is GPUNOTDEF, then the cached element is not prestored.
	//    Prestored elements are created at the beginning in Matlab and stored in
	//    the cache. For this elements I have a particular way of cleaning up pointed
	//    GPUtypes, because the destructor will never be called explicitly from Matlab but
	//    I have to understand from an internal flag if the object was cleaned.
	//
	gpuTYPE_t mxtype;

	extcacheel() : id(0), mx(0), assigned (0), gp(0), mxtype(gpuNOTDEF) {
	}

};
struct extcaches {

	struct extcacheel *ptr;

	int maxel;
	// anytime we have a cache miss
	int cachemiss;
	// total cache access by ptr
	int totalbyptr;
	// total cache access by slot
	int totalbyslot;
	// cache cleanup
	int totalcleanup;
	// free slot cached pointers request
	int totalfreeslotcached;

	// free slot cached miss
	int totalfreeslotcachedmiss;

	// free slot request
	int totalfreeslot;




	// we initialize the cache with EXTCACHE_MXCACHED elements
	extcaches() : maxel(0), ptr(NULL), cachemiss(0), totalbyptr(0), totalbyslot(0),
		totalcleanup(0), totalfreeslotcachedmiss(0), totalfreeslotcached(0), totalfreeslot(0) {
	}
};




class GPUmanager {
	Queue<GPUstream> **stream;
	int nStreams;
	GPUmatError_t error;

	int executiondelayed;

	cudaStream_t *cudaStreams;

	CUdevice *cuDevice;

	CUcontext *cuContext;

	CUmodule *cuModule;

	CUfunction *cuFunction;

	CUtexref *cuTexref;

	char *fatbin;

	gpukernelconfig_t gpuKernelConfig;

	// user modules
	char *userModuleName[MAXMODULES];
	struct userModule *userModules[MAXMODULES];
	int nModules; // number of loaded modules

	// cuda capability
	int capability;

	// pointers cache
	struct cache memcache;

	// external pointers cache
	struct extcaches extcache;

	// compiler
	struct userCompile comp;

	// user functions
	struct userFunction user;

	// debug
	struct debugStruct debug;

public:
#ifdef DEBUG
	FILE *debugout;
#endif

	/* constructor */
	GPUmanager(int = 0);

	/* destructor */
	~GPUmanager();

	/* cleanup */
	void cleanup();

	/* Init cudaStreams */
	GPUmatResult_t initStreams();


	/* empty */
	int empty();

	/* run */
	GPUmatResult_t run();

	/* print */
	void print();

	/* pushStream */
	void pushStream(GPUstream *, int);

	/* popStream */
	GPUstream * popStream(int);

	/* getStream */
	int getStream();

	/* front */
	GPUstream *
		front();

	/* resetError */
	void resetError();

	/* setError */
	void setError(GPUmatResult_t error, const char * str);

	/* executionDelayed */
	int executionDelayed();

	/* setExecutionDelayed */
	void setExecutionDelayed();

	/* resetExecutionDelayed */
	void resetExecutionDelayed();

	/* set Device */
	void setCuDevice(CUdevice *d);

	/* get Device */
	CUdevice *getCuDevice();

	/* set context */
	void setCuContext(CUcontext *c);

	/* get context */
	CUcontext *getCuContext();

	/* set module */
	void setCuModule(CUmodule *c);

	/* get cuModule */
	CUmodule *getCuModule();

	/* set cuFunction */
	void setCuFunction(CUfunction *c);

	/* set cuTexref */
	void setCuTexref(CUtexref *c);

	/* get cuFunction */
	CUfunction *getCuFunction(int);

	/* get cuTexref */
	CUtexref *getCuTexref(int);

	/* set maxthreads */
	void setKernelMaxthreads(unsigned int mx);

	/* get maxthreads */
	unsigned int getKernelMaxthreads();

	/* get kernel config */
	gpukernelconfig_t * getKernelConfig();


	/* registerGPUtype */

	//void
	//registerGPUtype(GPUtype<C> *);

	/* throw cublas */
	GPUmatResult_t throwCublas(int cublasresult);

	/* throw cuda */
	GPUmatResult_t throwCuda(cudaError_t cudaresult);

	//**************************************************************
	// MODULES
	//**************************************************************

	int getNModules();
	char * getModuleName(int);
	struct userModule ** getUserModules();
	void registerUserModule(char * name, char *kernelname);
	void deleteUserModule(char * name);

	int getUserModuleNumberByName(char *name);
	struct userModule * getUserModule(int n);
	void registerUserFunction(char *modname, char *funname);
	CUfunction * getUserFunctionByName(char *modname, char *funname);

	//**************************************************************
	// FUNCTIONS
	//**************************************************************
	int   funRegisterFunction(STRINGCONST char *name, void *f);
	void * funGetFunctionByName(STRINGCONST char *name);
	int    funGetFunctionNumber(STRINGCONST char *name);
	void * funGetFunctionByNumber(int findex);

	//**************************************************************
	// CUDA capability
	//**************************************************************

	void setCudaCapability(int cp);
	int getCudaCapability();


	//**************************************************************
	// External Memory cache
	//**************************************************************

	void extCacheFreeGPUtypePtr(int);
	void * extCacheGetGPUtypePtr(void *);
	void * extCacheGetGPUtypePtrBySlot(int);
	void   extCacheCleanUp();
	void * extCacheGetFreeSlot(int *, gpuTYPE_t mxtype = gpuNOTDEF);
	void   extCachePrint();
	void   extCacheCleanPtrBySlot(int);
	void   extCacheRegisterPtrBySlot(int , void *, void *, void *, gpuTYPE_t mxtype = gpuNOTDEF);
	void   extCacheRegisterCachedPtrBySlot(int , void *, void *, gpuTYPE_t mxtype);

	void extCacheCacheMiss();
	//**************************************************************
	// Memory cache
	//**************************************************************

	// register a pointer to GPU memory
	void cacheRegisterPtr(void *);
	// Request a pointer of specified size
	void * cacheRequestPtr(int size);
	// Empty the cache
	void cacheClean();
	void cachePrint();

	//**************************************************************
	// DEBUG
	//**************************************************************

	void debugPushInstructionStack(int);
	int getDebugMode();
	void setDebugMode(int);

	void debugSetVerbose(int);
	int  debugGetVerbose();

	void debugLog(STRINGCONST char *, int);
	void debugReset();
	void debugPushIndent();
	void debugPopIndent();
	void debugResetIndent();

	int debugGetIndent();


	//**************************************************************
	// COMPILER
	//**************************************************************
	void compRegisterInstruction(STRINGCONST char *inst, int type=0);
	void compFunctionStart(STRINGCONST char *);
	void compFunctionSetParam(STRINGCONST char *);
	void compFunctionSetParamGPUtype(void *);
	void compFunctionSetParamInt(int);
	void compFunctionSetParamFloat(float);
	void compFunctionSetParamDouble(double);
	void compFunctionEnd();

	void compForCountIncrease();
	void compForCountDecrease();
	int compGetFourCount();

	void compStart(STRINGCONST char *, int header = 1);
	void compStop();
	void compFlush();
	void compAbort(STRINGCONST char *);
	int getCompileMode();
	void compStackNullPtr(void *ptmp, int type);
	void compPush(void *, int type);
	int compGetContext(void *, int type);
	void compClearContext(void *, int type);

	void compSetFilename(STRINGCONST char *filename, int type=0);
	void compClear();
	char * compGetFilename(int type=0);



};

#endif

