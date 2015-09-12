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

#if !defined(GPUERROR_H_)
#define GPUERROR_H_

#define OPENFOUT {FILE *fout=fopen("log.csv","a+");
#define CLOSEFOUT fclose(fout);}

/*
 * ERROR codes
 */

#define ERROR_EXPECTED_GPUSINGLE  "Wrong argument. Expected a GPUsingle. (ERROR code A.1)"
#define ERROR_EXPECTED_GPUDOUBLE  "Wrong argument. Expected a GPUdouble. (ERROR code A.1)"
#define ERROR_EXPECTED_GPUINT32   "Wrong argument. Expected a GPUint32.  (ERROR code A.1)"
#define ERROR_MXTOGPUTYPE         "Unable to convert mxArray to GPUtype. (ERROR code A.2)"
#define ERROR_MXID_NOTCONSISTENT  "Matlab GPUtype is not mapped. (ERROR code A.3)"
#define ERROR_MXID_CACHEINTERNAL  "GPUmat internal error. (ERROR code A.4)"
#define ERROR_MXID_CACHEINTERNAL1 "GPUmat internal error. (ERROR code A.5)"
#define ERROR_MXID_REGISTERGT     "GPUmat internal error. (ERROR code A.6)"

#define ERROR_WRONG_NUMBER_ARGS   "Wrong number of arguments. (ERROR code 2)"
#define ERROR_FIRST_SCALAR        "First input argument is a scalar. Please do not use a scalar GPUsingle, use instead a Matlab scalar. (ERROR code 3)"
#define ERROR_SECOND_SCALAR       "Second input argument is a scalar. Please do not use a scalar GPUsingle, use instead a Matlab scalar. (ERROR code 4)"
#define ERROR_ONLY_MATLAB_SCALARS "Operations between Matlab arrays and GPU arrays are not supported. (ERROR code 5)"
#define ERROR_NON_SUPPORTED_TYPE  "Operation between a GPU type and an unsupported Matlab type. (ERROR code 6)"
#define ERROR_EXPECTED_GPUTYPE    "Wrong argument. Expected a GPUtype (GPUsingle, GPUdouble, GPUint32). (ERROR code 8)"
#define ERROR_ARG_SCALARS         "Operations between scalars must be performed on the CPU. (ERROR code 9)"
#define ERROR_CAST_WRONG_ARG      "Wrong argument in casting. (ERROR code 10)"

#define ERROR_NOTIMPL_FLOAT       "Function not implemented for input argument of type 'GPUsingle'. (ERROR code 11)"
#define ERROR_NOTIMPL_CFLOAT      "Function not implemented for input argument of type 'complex GPUsingle'. (ERROR code 11)"
#define ERROR_NOTIMPL_DOUBLE      "Function not implemented for input argument of type 'GPUdouble'. (ERROR code 11)"
#define ERROR_NOTIMPL_CDOUBLE     "Function not implemented for input argument of type 'complex GPUdouble'. (ERROR code 11)"

#define ERROR_ARG_REAL            "Operands must be real. (ERROR code 12)"
#define ERROR_MRDIVIDE_MATRICES   "Division between matrices not implemented. (ERROR code 13)"
#define ERROR_POWER_REALEXP       "Power implemented for real exponents only. (ERROR code 14)"

#define ERROR_ARG2OP_DIMENSIONS   "Number of array dimensions must match for binary array operation. (ERROR code 15)"
#define ERROR_ARG2OP_2D           "Input arguments must be 2-D. (ERROR code 16)"
#define ERROR_ARG2OP_INNER        "Inner matrix dimensions must agree. (ERROR code 17)"
#define ERROR_ARG2OP_ELEMENTS   "Number of elements of returned variable is incorrect. (ERROR code 15)"

#define ERROR_TRANSPOSE_INPUT2D   "Input argument must be 2-D. (ERROR code 18)"
#define ERROR_NOTIMPL_GENERIC     "Function not implemented for specified GPU types. (ERROR code 19)"

#define ERROR_GPUTYPE_INT32CPX    "Undefined GPUint32 COMPLEX. (ERROR code GPUTYPE.1)"
#define ERROR_GPU_KERNELEXECUTION "Kernel execution error. (ERROR code GPUTYPE.2)"
#define ERROR_GPUTYPE_FLOATTOCOMPLEX_SAMETYPE "Imaginary and real part should be of the same type. (ERROR code GPUTYPE.3)"
#define ERROR_GPUTYPE_FLOATTOCOMPLEX_COMPLEX  "Conversion from complex type to complex type. (ERROR code GPUTYPE.4)"
#define ERROR_GPUTYPE_FLOATTOCOMPLEX_SAMENUMBER  "Imaginary and complex part should have the same number of elements. (ERROR code GPUTYPE.5)"
#define ERROR_GPUTYPE_FLOATTOCOMPLEX_FLOAT  "Expected float type. (ERROR code GPUTYPE.6)"
#define ERROR_GPUTYPE_DOUBLETOCDOUBLE_DOUBLE  "Expected double precision type. (ERROR code GPUTYPE.7)"
#define ERROR_GPUTYPE_DOUBLENOTSUPPORTED "DOUBLE PRECISION IS SUPPORTED BY GPUs WITH CUDA CAPABILITY >=13. (ERROR code GPUTYPE.8)"
#define ERROR_GPUTYPE_ISCOMPILEDGPUPTR "Trying to access a GPUtype that was created in COMPILATION mode. (ERROR code GPUTYPE.9)"

#define ERROR_GPUOPCOMPLEX_WRONGOUT "Wrong output type. Output must be complex. (ERROR code GPUCOMPLEX.1)"
#define ERROR_GPUOPCOMPLEX_WRONGNEL "Wrong number of elements. (ERROR code GPUCOMPLEX.2)"
#define ERROR_GPUOPCOMPLEX_WRONGIN "Wrong input type. Input must be real and output complex. (ERROR code GPUCOMPLEX.2)"




#define ERROR_SUBSREF_FIELDNOTSUPP "Field name indexing not supported by GPUtype objects. (ERROR code SUBSREF.1)"
#define ERROR_SUBSREF_CELLNOTSUPP  "Cell array indexing not supported by GPUtype objects. (ERROR code SUBSREF.2)"
#define ERROR_SUBSREF_MAXIND       "GPUtype indexing is supported for GPUtype with dimensions <= 5. (ERROR code SUBSREF.3)"
#define ERROR_SUBSREF_INDEXC       "Number of indexes exceeds GPUtype dimensions. (ERROR code SUBSREF.4)"
#define ERROR_SUBSREF_INDEXMAXDIM  "Index exceeds matrix dimensions. (ERROR code SUBSREF.5)"
#define ERROR_SUBSREF_SUBSINTEGER  "Subscript indices must be real positive integers. (ERROR code SUBSREF.6)"
#define ERROR_SUBSREF_INDEXREAL    "Index must be a real. (ERROR code SUBSREF.7)"
#define ERROR_SUBSREF_INDEXWRONGTYPE "Index must be either a double or GPUtype. (ERROR code SUBSREF.8)"
#define ERROR_SUBSREF_WRONGRANGE    "Specified range is wrong. (ERROR code SUBSREF.9)"
#define ERROR_SUBSREF_EMPTYRESULT   "Empty result. Check specified range. (ERROR code SUBSREF.10)"

#define ERROR_PERMUTE_INVALIDPERM   "Invalid permutation vector. (ERROR code PERMUTE.1)"
#define ERROR_PERMUTE_INVALIDORDER   "ORDER must have at least N elements for an N-D array. (ERROR code PERMUTE.2)"

#define ERROR_SUBSASGN_RHS          "Right hand side in the assignment must be a GPUtype or a Matlab scalar (ERROR code SUBSASGN.1)"
#define ERROR_SUBSASGN_ONELEVEL     "Two many indexes. Only one level is supported (ERROR code SUBSASGN.2)"
#define ERROR_SUBSASGN_ONESUBS      "An indexing expression on the left side of an assignment must have at least one subscript. (ERROR code SUBSASGN.3)"
#define ERROR_SUBSASGN_SAMEEL      "In an assignment  A(:) = B, the number of elements in A and B must be the same. (ERROR code SUBSASGN.4)"
#define ERROR_SUBSASGN_NONSING      "Assignment must have the same non-singleton rhs dimensions as non-singleton subscripts. (ERROR code SUBSASGN.5)"
#define ERROR_SUBSASGN_DIMMIS      "Subscripted assignment dimension mismatch. (ERROR code SUBSASGN.6)"
#define ERROR_SUBSASGN_SAMETYPE    "Right and left hand side in the assignment must be of the same type. (ERROR code SUBSASGN.7)"
#define ERROR_SUBSASGN_INDEXC       "Number of indexes exceeds GPUtype dimensions. (ERROR code SUBSASGN.8)"

#define ERROR_GPUMANAGER_MAXMODNUM   "Unable to register module: maximum number of modules reached. (ERROR code GPUMANAGER.1)"
#define ERROR_GPUMANAGER_INVMODNAME  "Invalid module name. (ERROR code GPUMANAGER.2)"
#define ERROR_GPUMANAGER_FUNALREADYDEFINED  "Function is already defined. (ERROR code GPUMANAGER.3)"
#define ERROR_GPUMANAGER_LOADFUN     "Unable to load function from module. (ERROR code GPUMANAGER.4)"
#define ERROR_GPUMANAGER_FUNNOTDEFINED  "Function is not defined. (ERROR code GPUMANAGER.5)"
#define ERROR_GPUMANAGER_MODALREADYDEFINED  "Module is already defined. (ERROR code GPUMANAGER.6)"
#define ERROR_GPUMANAGER_OPENKERFILE  "Error opening KERNEL file.(ERROR code GPUMANAGER.7)"
#define ERROR_GPUMANAGER_READKERFILE  "Error reading KERNEL  file. (ERROR code GPUMANAGER.8)"
#define ERROR_GPUMANAGER_UNABLELOADKER  "Unable to load module. (ERROR code GPUMANAGER.9)"
#define ERROR_GPUMANAGER_UNABLEUNLOADKER  "Unable to unload module. (ERROR code GPUMANAGER.10)"
#define ERROR_GPUMANAGER_NULLCOMPFILENAME  "Compilation file name is not set. (ERROR code GPUMANAGER.11)"
#define ERROR_GPUMANAGER_STARTCOMPILEMODE  "Trying to set file name, but compilation not started. (ERROR code GPUMANAGER.12)"
#define ERROR_GPUMANAGER_COMPINCONSGPUTYPE  "GPUtype variable is not available in compilation context. (ERROR code GPUMANAGER.13)"
#define ERROR_GPUMANAGER_COMPINCONSMX       "Matlab variable is not available in compilation context. (ERROR code GPUMANAGER.13)"
#define ERROR_GPUMANAGER_COMPNOTINITIALIZED  "Compilation has not been properly initialized. (ERROR code GPUMANAGER.14)"
#define ERROR_GPUMANAGER_COMPINVALIDSTRUCT  "Unable to process Matlab STRUCT during compilation. (ERROR code GPUMANAGER.15)"
#define ERROR_GPUMANAGER_MAXELMXCONVERT  "(ERROR code GPUMANAGER.16)"

#define ERROR_GPUMANAGER_COMPNOTIMPLEMENTED  "Function compilation is not implemented. (ERROR code GPUMANAGER.14)"
#define WARNING_GPUMANAGER_COMPNOTIMPLEMENTED  "Function compilation is not implemented. (Warning code GPUMANAGER.14)"
#define ERROR_GPUMANAGER_COMPSTACKOVERFLOW  "Compilation stack overflow. (ERROR code GPUMANAGER.15)"


#define ERROR_GPUCOMPILER_INVALIDARGUMENT  "Invalid argument. Arguments can be either a GPUtype or a Matlab cell array. (ERROR code GPUCOMPILER.1)"


#define ERROR_VERTCAT_SAMETYPE      "VERTCAT variables should have the same type. (ERROR code HORZCAT/VERTCAT.1)"
#define ERROR_VERTCAT_DIM2          "CAT implemented only for dimensions <= 2. (ERROR code HORZCAT/VERTCAT.2)"
#define ERROR_VERTCAT_DIMNOTCONSISTENT  "CAT arguments dimensions are not consistent. (ERROR code HORZCAT/VERTCAT.3)"
#define ERROR_VERTCAT_SCALAR        "CAT not supported for scalar arguments. (ERROR code HORZCAT/VERTCAT.4)"


#define ERROR_FFT_MAXDIM "Maximum number of dimensions for 1D FFT/IFFT is 3. (ERROR code FFT.1)"
#define ERROR_FFT_MAXEL1D "Maximum number of elements for 1D FFT/IFFT is 8e6. (ERROR code FFT.2)"
#define ERROR_FFT_ONLY2D "FFT/IFFT implemented only for 2D arrays. (ERROR code FFT.3)"
#define ERROR_FFT_ONLY3D "FFT/IFFT implemented only for 3D arrays. (ERROR code FFT.4)"
#define ERROR_FFT_MEMCOPY "Error in memcpy2D. (ERROR code FFT.5)"

#define ERROR_SUM_POSINDEX "Dimension argument must be a positive integer scalar within indexing range. (ERROR code SUM.1)"

#define ERROR_REAL_COMPLEXOUT    "Returned variable must be real (ERROR code REAL/COMPLEX.1)"


#define ERROR_GPUFOR_ITDOUBLE    "GPUfor iterator must be a Matlab double precision variable. (ERROR code GPUFOR.3)"
#define ERROR_GPUFOR_ITWRONG     "GPUfor iterator is not an array like [start:step:stop]. (ERROR code GPUFOR.4)"

#ifndef MATLAB
#define myPrintf printf
#else
#define myPrintf mexPrintf
extern "C" int mexPrintf(
    const char  *fmt, /* printf style format */
    ...       /* any additional arguments */
    );
#endif

enum GPUmatResult {
  GPUmatSuccess = 0,
  GPUmatError = 1,

  /* known cublas errors */
  GPUmatCUBLAS_STATUS_NOT_INITIALIZED = 102,
  GPUmatCUBLAS_STATUS_ALLOC_FAILED = 103,
  GPUmatCUBLAS_STATUS_INVALID_VALUE = 104,
  GPUmatCUBLAS_STATUS_ARCH_MISMATCH = 105,
  GPUmatCUBLAS_STATUS_MAPPING_ERROR = 106,
  GPUmatCUBLAS_STATUS_EXECUTION_FAILED = 107,
  GPUmatCUBLAS_STATUS_INTERNAL_ERROR = 108,
  GPUmatCUBLAS_UNKNOWN_ERROR = 109,

  GPUmatCUDA_ERROR_INVALID_VALUE = 401,
  GPUmatCUDA_ERROR_OUT_OF_MEMORY = 402,
  GPUmatCUDA_ERROR_NOT_INITIALIZED = 403,
  GPUmatCUDA_ERROR_DEINITIALIZED = 404,

  GPUmatCUDA_ERROR_NO_DEVICE = 405,
  GPUmatCUDA_ERROR_INVALID_DEVICE = 406,

  GPUmatCUDA_ERROR_INVALID_IMAGE = 200,
  GPUmatCUDA_ERROR_INVALID_CONTEXT = 201,
  GPUmatCUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
  GPUmatCUDA_ERROR_MAP_FAILED = 205,
  GPUmatCUDA_ERROR_UNMAP_FAILED = 206,
  GPUmatCUDA_ERROR_ARRAY_IS_MAPPED = 207,
  GPUmatCUDA_ERROR_ALREADY_MAPPED = 208,
  GPUmatCUDA_ERROR_NO_BINARY_FOR_GPU = 209,
  GPUmatCUDA_ERROR_ALREADY_ACQUIRED = 210,
  GPUmatCUDA_ERROR_NOT_MAPPED = 211,

  GPUmatCUDA_ERROR_INVALID_SOURCE = 300,
  GPUmatCUDA_ERROR_FILE_NOT_FOUND = 301,

  GPUmatCUDA_ERROR_INVALID_HANDLE = 400,

  GPUmatCUDA_ERROR_NOT_FOUND = 500,

  GPUmatCUDA_ERROR_NOT_READY = 600,

  GPUmatCUDA_ERROR_LAUNCH_FAILED = 700,
  GPUmatCUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
  GPUmatCUDA_ERROR_LAUNCH_TIMEOUT = 702,
  GPUmatCUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
  GPUmatCUDA_GENERIC_ERROR = 704,

  GPUmatCUDA_ERROR_UNKNOWN = 999
};

typedef enum GPUmatResult GPUmatResult_t;

typedef struct {
  char errbuffer[300];
  GPUmatResult lasterror;

} GPUmatError_t;




class GPUexception {

  char buffer[300];
  GPUmatResult_t error;

public:

  GPUexception() {
    strcpy(buffer, "Unknown error.");
    error = GPUmatError;
  }

  GPUexception(GPUmatResult_t errnumber, const char *err) {
    strcpy(buffer, err);
    error = errnumber;
  }

  char * getError() {
    return buffer;
  }

  GPUmatResult_t getErrorNumber() {
    return error;
  }

};


class MyGC {

	void **ptr;
	int mysize;
	int idx;

public:

	MyGC();

	void setPtr(void *p);

	void remPtr(void *p);

	~MyGC();

};

void *Mymalloc(size_t size, MyGC *m=0);
void Myfree(void *p, MyGC *m=0);

// Simple garbage collector
template<class C>
class MyGCObj {

	C **ptr;
	int mysize;
	int idx;

public:

	MyGCObj();

	void setPtr(C *p);

	void remPtr(C *p);

	~MyGCObj();

};

// Garbage collector 1
template <class C>
MyGCObj<C>::MyGCObj() :
	ptr(NULL), mysize(10), idx(0) {

	ptr = (C **) Mymalloc(mysize * sizeof(C *));

	for (int i = 0; i < mysize; i++)
		ptr[i] = NULL;
}

template <class C>
void MyGCObj<C>::setPtr(C *p) {
	if (idx == mysize) {
		// increase size
		int newmysize = mysize + 10;
		C **tmpptr = (C **) Mymalloc(newmysize * sizeof(C *));
		for (int i = 0; i < newmysize; i++)
			tmpptr[i] = NULL;

		memcpy(tmpptr, ptr, mysize * sizeof(C *));
		Myfree(ptr);
		mysize = newmysize;
		ptr = tmpptr;
	}
	ptr[idx] = p;
	idx++;
}

template <class C>
void MyGCObj<C>::remPtr(C *p) {
	for (int i = mysize - 1; i >= 0; i--) {
		if (ptr[i] == p) {
			ptr[i] = NULL;
			break;
		}
	}
}

template <class C>
MyGCObj<C>::~MyGCObj() {
	for (int i = 0; i < mysize; i++) {
		if (ptr[i] != NULL) {
			delete ptr[i];
		}
	}
	Myfree(ptr);

}


#ifndef GPUerror_CPP
#define malloc Mymalloc
#define free Myfree
#endif


#endif
