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

#if !defined(GPUMAT_HH_)
#define GPUMAT_HH_

// The following is required by new GCC
#define STRINGCONST const

// GPUmat version
#define GPUMATVERSION_MAJOR 0
#define GPUMATVERSION_MINOR 280

#define ALIGN_UP(offset, alignment) \
      (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)
/*************************************************
 * GPUmatInterface
 *************************************************/

struct GPUmatInterfaceFunction {
  /// registerFunction
  /**
  * Register a user defined function
  * @param[in] name Function name.
  * @param[in] f Function pointer.
  * @return The index assigned to the function. Can be used to get the function pointer using getFunctionByNumber
  */
  int (*registerFunction)(STRINGCONST char *name, void *f);

  /// getFunctionByName
  /**
  * Returns the function pointer
  * @param[in] name Function name.
  * @return Function pointer
  */
  void * (*getFunctionByName)(STRINGCONST char *name);

  /// getFunctionNumber
  /**
  * Returns the function index
  * @param[in] name Function name.
  * @return Function index
  */
  int (*getFunctionNumber)(STRINGCONST char *name);

  /// getFunctionByNumber
  /**
  * Returns the function pointer
  * @param[in] findex Function index (returned using registerFunction or getFunctionNumber).
  * @return Function pointer
  */
  void * (*getFunctionByNumber)(int findex);

};

struct GPUmatInterfaceConfig {
  /// getMajorMinor
  /**
  * Returns the GPUmat major and minor version
  * @param[out] major Returns the major number
  * @param[out] minor Returns the minor number
  */

  void (*getMajorMinor)(int *major, int *minor);

  /// getActiveDeviceNumber
  /**
  * Returns the GPU device being used
  * @return The device number being used
  */

  int (*getActiveDeviceNumber)();

};

struct GPUmatManager {
  /// cacheClean
  /**
  * Clean GPU memory cache
  */

  void (*cacheClean)(void);

};

typedef struct {

  // manage functions
  struct GPUmatInterfaceFunction fun;

  // manage configuration parameter
  struct GPUmatInterfaceConfig config;

  // manage low level GPU manager
  struct GPUmatManager control;

} GPUmatInterface;


#ifndef GPUMAT
/*************************************************
 * POINTERS
 **************************************************/

// The following is required by new GCC
#define STRINGCONST const

#define UINTPTR (uintptr_t)


/*************************************************
 * GPUTYPE
 *************************************************/

enum gpuTYPE {
  gpuFLOAT = 0, gpuCFLOAT = 1, gpuDOUBLE = 2, gpuCDOUBLE = 3, gpuINT32 = 4, gpuNOTDEF = 20
};

typedef enum gpuTYPE gpuTYPE_t;

#endif


typedef struct GPUtypeS {

  void (*ptr0)(void*);


  struct counter {
    counter(void* p = 0, unsigned c = 1) :
      ptr(p), count(c) {}
    void* ptr;
    unsigned count;
  }* ptrCounter;


  /*GPUtypeS(void (*p2)(void*) = NULL);
  void acquireGPUptr(counter* c);
  void releaseGPUptr();*/

  /* Constructor */
  GPUtypeS(void * p = NULL, void (*p0)(void*) = NULL) : ptr0(p0), ptrCounter(new counter(p)) {
  }

  /* Constructor */
  GPUtypeS(const GPUtypeS &p) : ptr0(NULL), ptrCounter(new counter()) {
    *this = p;
  }

  /* Destructor */
  ~GPUtypeS() {
    this->releasePtr();
  }

  /// Increment the count
  /**
   * acquirePtr
   **/
  void acquirePtr(counter* c) {
    ptrCounter = c;
    if (c)
      ++c->count;
  }

  /// Decrement the count, delete if 0
  /**
   * releasePtr
   * */
  void releasePtr() {

    if (--ptrCounter->count == 0) {
      if (ptrCounter->ptr) {
        this->ptr0(ptrCounter->ptr);
      }
      delete ptrCounter;
    }
    ptrCounter = 0;
  }

  /* Operator = */
  const GPUtypeS & operator=(const GPUtypeS &p) {
    if (this != &p) {

      this->ptr0 = p.ptr0;

      this->releasePtr();
      this->acquirePtr(p.ptrCounter);
    }
    return *this;
  }


} GPUtype;



#ifndef GPUMAT

/*************************************************
 * Simple garbage collector
 **************************************************/
/*
 * MyGC is a garbage collector for objects allocated with malloc.
 * An object can be registered to the GC after being created with 'malloc'
 * statement. When the GC is deleted from stack, registered elements
 * will be deleted also. An element can also be unregistered.
 *
 * Example:
 * MyGC mygc;
 * int *tmp = (int*) malloc(10*sizeof(int));
 * mygc.setPtr(tmp);
 *
 */
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


/*
 * MyGCObj is a template garbage collector for objects of class C
 * An can can be registered to the GC after being created with 'new'
 * statement. When the GC is deleted from stack, registered elements
 * will be deleted also. An element can also be unregistered.
 *
 * Example:
 * MyGCObj<C> mygc;
 * C *tmp = new C();
 * mygc.setPtr(tmp);
 *
 */

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

  ptr = (C **) malloc(mysize * sizeof(C *));

  for (int i = 0; i < mysize; i++)
    ptr[i] = NULL;
}

template <class C>
void MyGCObj<C>::setPtr(C *p) {
  if (idx == mysize) {
    // increase size
    int newmysize = mysize + 10;
    C **tmpptr = (C **) malloc(newmysize * sizeof(C *));
    for (int i = 0; i < newmysize; i++)
      tmpptr[i] = NULL;

    memcpy(tmpptr, ptr, mysize * sizeof(C *));
    free(ptr);
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
  free(ptr);

}


/*************************************************
 * Range
 **************************************************/
/// Range is used in slices
/**
 * Use Range only in slices.
 * A Range defines the range to be used in slices. The range can be specified as:
 * 1) [inf:stride:sup]. Rnage from inf to sup with specified stride.
 *    Example 1:10:20 (from 1 to 20 with stride 10)
 * 2) Array of indexes in iindx (CPU). For example iindx = {0,1,3,4,5}.
 *    When using iindx sup is used to store the size of the array. Last element index of iindx
 *    should be assigned to sup. For example, iindx = {0,1,3,4}, sup = 3.
 * 3) Array of indexes in gindx (GPU). Same as (2), but indexes are stored on GPU variable.
 *
 * All indexes are assumed 0 based, means first element has index 0 (also in iindx and gindx)
 */

#define BEGIN 0
#define END   -1

typedef struct RangeS{
  int inf;
  int sup;
  int stride;
  int begin;
  int end;

  // we allow different type of indexes
  // iindx -> integer
  // findx -> float
  // dindx -> double

  int * iindx;
  float  *findx;
  double *dindx;

  GPUtype * gindx;
  void * gindxptr; //internal pointer

  RangeS *next;
  RangeS(int a) : inf(a), sup(a), stride(0),
                  begin(BEGIN), end(END), next(NULL),
                  iindx(NULL), findx(NULL), dindx(NULL),
                  gindx(NULL), gindxptr(NULL)  {}

  RangeS(int a, const RangeS &r) : inf(a), sup(a), stride(0),
                                   begin(BEGIN), end(END),
                                   iindx(NULL), findx(NULL), dindx(NULL),
                                   gindx(NULL), gindxptr(NULL){
    RangeS *tmp = (RangeS*) &r;
    next = tmp;

  }

  RangeS(int a, int b, int c) : inf(a), sup(c), stride(b),
                                begin(BEGIN), end(END), next(NULL),
                                iindx(NULL), findx(NULL), dindx(NULL),
                                gindx(NULL), gindxptr(NULL) {}

  RangeS(int a, int b, int c, const RangeS &r) :
     inf(a), sup(c), stride(b),
     begin(BEGIN), end(END),
     iindx(NULL), findx(NULL), dindx(NULL),
     gindx(NULL), gindxptr(NULL){

    RangeS *tmp = (RangeS*) &r;
    next = tmp;

  }

  // IINDX
  RangeS(int s, int* c) : inf(0), sup(s), stride(1),
                          begin(BEGIN), end(END), next(NULL),
                          iindx(NULL), findx(NULL), dindx(NULL),
                          gindx(NULL), gindxptr(NULL) {
    // make a local copy of the array
    iindx = (int*) malloc((s+1)*sizeof(int));
    memcpy(iindx, c, (s+1)*sizeof(int));
  }

  RangeS(int s, int* c, const RangeS &r) : inf(0), sup(s), stride(1),
                            begin(BEGIN), end(END), next(NULL),
                            iindx(NULL), findx(NULL), dindx(NULL),
                            gindx(NULL), gindxptr(NULL) {
    // make a local copy of the array
    iindx = (int*) malloc((s+1)*sizeof(int));
    memcpy(iindx, c, (s+1)*sizeof(int));

    RangeS *tmp = (RangeS*) &r;
    next = tmp;
  }

  // FINDX
  RangeS(int s, float* c) : inf(0), sup(s), stride(1),
                          begin(BEGIN), end(END), next(NULL),
                          iindx(NULL), findx(NULL), dindx(NULL),
                          gindx(NULL), gindxptr(NULL) {
    // make a local copy of the array
    findx = (float*) malloc((s+1)*sizeof(float));
    memcpy(findx, c, (s+1)*sizeof(float));
  }

  RangeS(int s, float* c, const RangeS &r) : inf(0), sup(s), stride(1),
                            begin(BEGIN), end(END), next(NULL),
                            iindx(NULL), findx(NULL), dindx(NULL),
                            gindx(NULL), gindxptr(NULL) {
    // make a local copy of the array
    findx = (float*) malloc((s+1)*sizeof(float));
    memcpy(findx, c, (s+1)*sizeof(float));

    RangeS *tmp = (RangeS*) &r;
    next = tmp;
  }

  // DINDX
  RangeS(int s, double* c) : inf(0), sup(s), stride(1),
                          begin(BEGIN), end(END), next(NULL),
                          iindx(NULL), findx(NULL), dindx(NULL),
                          gindx(NULL), gindxptr(NULL) {
    // make a local copy of the array
    dindx = (double*) malloc((s+1)*sizeof(double));
    memcpy(dindx, c, (s+1)*sizeof(double));
  }

  RangeS(int s, double* c, const RangeS &r) : inf(0), sup(s), stride(1),
                            begin(BEGIN), end(END), next(NULL),
                            iindx(NULL), findx(NULL), dindx(NULL),
                            gindx(NULL), gindxptr(NULL) {
    // make a local copy of the array
    dindx = (double*) malloc((s+1)*sizeof(double));
    memcpy(dindx, c, (s+1)*sizeof(double));

    RangeS *tmp = (RangeS*) &r;
    next = tmp;
  }



  // GINDX
  RangeS(GPUtype& c) : inf(0), sup(0), stride(1),
                              begin(BEGIN), end(END), next(NULL),
                              iindx(NULL), findx(NULL), dindx(NULL),
                              gindx(NULL), gindxptr(NULL) {
    gindx = new GPUtype(c);
    gindxptr = c.ptrCounter->ptr;
  }

  RangeS(GPUtype& c, const RangeS &r) : inf(0), sup(0), stride(1),
                                begin(BEGIN), end(END), next(NULL),
                                iindx(NULL), findx(NULL), dindx(NULL),
                                gindx(NULL), gindxptr(NULL) {
    gindx = new GPUtype(c);
    gindxptr = c.ptrCounter->ptr;

    RangeS *tmp = (RangeS*) &r;
    next = tmp;
  }

  // Destructor
  ~RangeS() {
    if (iindx!=NULL) {
      free(iindx);
    }

    if (dindx!=NULL) {
      free(dindx);
    }

    if (findx!=NULL) {
      free(findx);
    }

    if (gindx!=NULL) {
      delete gindx;
      gindxptr = 0;
    }

  }

} Range;

#endif

#ifndef GPUMAT
/*************************************************
 * GPUMAT
 *************************************************/

/*
 * GPUmat struct
 * Interface to different GPUmat functions
 *
 * Notes about names
 * - The keyword 'mx' should appear in the name if the function is related to Matlab
 * - The keyword 'drv' is used in general for functions that return a GPUtype. Usually
 *   there is an equivalent function without the keyword 'drv' that perform the same
 *   operation but in-place (for example, Exp, ExpDrv)
 */
#include "GPUmatNumerics.hh"

typedef struct {

    // GETTERS

    /// Returns the type (gpuTYPE_t) of a GPUtype object
    /**
    * @param[in] p GPUtype input variable.
    * @return gpuTYPE_t
    */
    gpuTYPE_t (*getType)(const GPUtype  &p);

    /// Returns the dimensions array of a GPUtype object
    /**
    * @param[in] p GPUtype input variable.
    * @return The dimensions array
    */
    const int * (*getSize)(const GPUtype &p);

    /// Returns the number of dimensions of a GPUtype object
    /**
    * @param[in] p GPUtype input variable.
    * @return The number of dimensions
    */
    int  (*getNdims)(const GPUtype &p);

    /// Returns the number of elements of a GPUtype object
    /**
    * @param[in] p GPUtype input variable.
    * @return The number of elements
    */
    int  (*getNumel)(const GPUtype &p);

    /// Returns the pointer to the GPU memory
    /**
    * @param[in] p GPUtype input variable.
    * @return The pointer to the GPU memory
    */
    const void * (*getGPUptr)(const GPUtype &p);

    /// Returns the size of the elements on the GPU memory
    /**
    * @param[in] p GPUtype input variable.
    * @return The size of the elements on GPU memory
    */
    int (*getDataSize)(const GPUtype &p);


    // SETTERS
    /// Set the size of the GPUtype
    /**
    * @param[in] p GPUtype input variable.
    * @param[in] n The number of elements of the array s
    * @param[in] s Array with dimensions
    */
    void (*setSize)(const GPUtype &p, int n, const int *s);

    // Properties

    /// Returns 1 if the GPUtype is SCALAR
    /**
    * @param[in] p GPUtype input variable.
    * @return 1 if SCALAR
    */
    int (*isScalar)(const GPUtype  &p);


    /// Returns 1 if the GPUtype is COMPLEX
    /**
    * @param[in] p GPUtype input variable.
    * @return 1 if COMPLEX
    */
    int (*isComplex)(const GPUtype  &p);

    /// Returns 1 if the GPUtype is EMPTY
    /**
    * @param[in] p GPUtype input variable.
    * @return 1 if EMPTY
    */
    int (*isEmpty)(const GPUtype  &p);

    /// Returns 1 if the GPUtype is FLOAT (either REAL or COMPLEX)
    /**
    * @param[in] p GPUtype input variable.
    * @return 1 if FLOAT
    */
    int (*isFloat)(const GPUtype  &p);


    /// Returns 1 if the GPUtype is DOUBLE (either REAL or COMPLEX)
    /**
    * @param[in] p GPUtype input variable.
    * @return 1 if DOUBLE
    */
    int (*isDouble)(const GPUtype  &p);

    /* GPUtype creation */

    /// Creates a GPUtype with specified properties: type, number of dimensions, size.
   /**
    * Creates a GPUtype with specified properties: type, number of dimensions, size. If
    * ndims = 0 or size=NULL the GPUtype is an empty GPUtype.
    * @param[in] type GPUtype type (gpuFLOAT, gpuDOUBLE, ...).
    * @param[in] ndims Number of dimensions.
    * @param[in] size Dimensions array.
    * @param[in] initialization vector. Set to NULL if initialization is not required.
    * @return The created GPUtype
    */
    GPUtype (*create) (gpuTYPE_t type, int ndims, const int *size, void * init);

    /// Clones a GPUtype.
    /**
    * @param[in] p GPUtype to clone.
    * @return The cloned GPUtype
    */
    GPUtype (*clone) (const GPUtype &p);

    /// Creates a GPUtype with specified type.
    /**
    * Dimensions are constructed from input arguments nrhs and prhs. For
    * example, we have the following expression in Matlab:
    *
    * A = eye(3,4,5,GPUsingle)
    *
    * The function eye should create an output GPUtype variable
    * with dimensions (3,4,5). The code to perform such operation
    * is the following:
    *
    * @code
    * GPUtype IN = gm->gputype.getGPUtype(prhs[nrhs-1]);
    * gpuTYPE_t tin = gm->gputype.getType(IN);
    * GPUtype r = gm->gputype.createMx(tin, nrhs-1, prhs);
    * @endcode
    *
    * @param[in] type GPUtype type (gpuFLOAT, gpuDOUBLE, ...).
    * @param[in] nrhs Number of elements of array prhs[].
    * @param[in] prhs Each element specifies a dimension.
    * @return    The created GPUtype
    */
    GPUtype (*createMx) (gpuTYPE_t type, int nrhs, const mxArray *prhs[]);


    /// Creates a GPUsingle or GPUdouble object to be returned to Matlab from a given GPUtype.
    /**
    * @param[in] p GPUtype input variable.
    * @return    Matlab mxArray pointer (GPUsingle or GPUdouble object)
    */
    mxArray * (*createMxArray) (const GPUtype &p);

    /// Creates an mxArray from a given GPUtype.
    /**
     * This function is different from createMxArray. It creates a Matlab array with
     * the same type (double, single) of the input GPUtype. createMxArray creates an mxArray
     * of type GPUsingle, GPUdouble or any other GPUmat variable.
     * @param[in] p GPUtype input variable.
     * @return    Matlab mxArray pointer (different from createMxArray)
     */
    mxArray * (*toMxArray) (const GPUtype &p);

    /// Internal function
    mxArray * (*createMxArrayPtr) (const mxArray *, const GPUtype &p);

    /// Creates a GPUtype from a Matlab array.
    /**
    * @param[in] mx Matlab array.
    * @return    GPUtype
    */
    GPUtype (*mxToGPUtype) (const mxArray *mx);


    /// Creates a GPUtype from a Matlab GPUmat variable
    /**
    * @param[in] mx GPUmat variable (GPUsingle, GPUdouble).
    * @return GPUtype object
    */
    GPUtype (*getGPUtype) (const mxArray *mx);

    /// Fills a GPUtype with a sequence of values
    /**
    * The element of q in position i will have the following value
    * incr*(i % m) + offset if (((i+offsetp) % p)==0);
    *
    * @param[in] q GPUtype.
    * @param[in] offset (see formula above)
    * @param[in] incr (see formula above)
    * @param[in] m (see formula above)
    * @param[in] p (see formula above)
    * @param[in] offsetp (see formula above)
    * @param[in] type=0 only real part is modified
    *            type=1 only imaginary part is modified
    *            type=2 both real and imaginary are modified
    */
    void (*fill) (const GPUtype &q, double offset, double incr, int m, int p, int offsetp, int type);

    /// Similar to the Matlab colon command
    /**
     * J:K    is the same as [J, J+1, ..., K].
     * J:K    is empty if J > K.
     * J:D:K  is the same as [J, J+D, ..., J+m*D] where m = fix((K-J)/D).
     * J:D:K  is empty if D == 0, if D > 0 and J > K, or if D < 0 and J < K.
     * @param[in] type GPUtype type.
     */
    GPUtype (*colon) (gpuTYPE_t type, double j, double d, double k);


    /// Creates a slice from a GPUtype using specified Range
    /**
    * @param[in] p GPUtype input variable
    * @param[in] r Range used for the slice
    * @return A new GPUtype object
    */

    GPUtype (*slice) (const GPUtype &p, const Range &r);

    /// Creates a slice from a GPUtype using specified Range
    /**
    * Indexes are considered as in Matlab/Fortran (starting from 1)
    * @param[in] p GPUtype input variable
    * @param[in] r Range used for the slice
    * @return A new GPUtype object
    */

    GPUtype (*mxSlice) (const GPUtype &p, const Range &r);

    /// Assigns a GPUtype to another.
    /**
    * Range and dir are used to apply the Range to left or right hand side
    * @param[in] p GPUtype input variable
    * @param[in] q GPUtype input variable
    * @param[in] r Range used for the slice
    * @param[in] dir If 0 Range is applied to q, if 1 Range is applied to p
    * @return A new GPUtype object
    */
    void (*assign) (const GPUtype &p, const GPUtype &q, const Range &r, int dir);

    /// Assigns a GPUtype to another.
    /**
    * Indexes are considered as in Matlab/Fortran (starting from 1)
    * Range and dir are used to apply the Range to left or right hand side
    * @param[in] p GPUtype input variable
    * @param[in] q GPUtype input variable
    * @param[in] r Range used for the slice
    * @param[in] dir If 0 Range is applied to q, if 1 Range is applied to p
    * @return A new GPUtype object
    */
    void (*mxAssign) (const GPUtype &p, const GPUtype &q, const Range &r, int dir);

    /// Assigns with permuted indexes a GPUtype to another.
    /**
    * Range and dir are used to apply the Range to left or right hand side
    * @param[in] p GPUtype input variable
    * @param[in] q GPUtype input variable
    * @param[in] r Range used for the slice
    * @param[in] dir If 0 Range is applied to q, if 1 Range is applied to p
    * @param[in] perm Array with permutation indexes
    * @return A new GPUtype object
    */
    void (*permute) (const GPUtype &p, const GPUtype &q, const Range &r, int dir, int *perm);


    /// Assigns with permuted indexes a GPUtype to another.
    /**
    * Indexes are considered as in Matlab/Fortran (starting from 1)
    * Range and dir are used to apply the Range to left or right hand side
    * @param[in] p GPUtype input variable
    * @param[in] q GPUtype input variable
    * @param[in] r Range used for the slice
    * @param[in] dir If 0 Range is applied to q, if 1 Range is applied to p
    * @param[in] perm Array with permutation indexes
    * @return A new GPUtype object
    */
    void (*mxPermute) (const GPUtype &p, const GPUtype &q, const Range &r, int dir, int *perm);

    /// Converts real to complex and complex to real
    /**
    * Depending on dir and mode the following operations are performed:
    * dir
    * 0 - REAL to COMPLEX
    * 1 - COMPLEX to REAL
    * mode
    * 0 - REAL, IMAG
    * 1 - REAL
    * 2 - IMAG
    * The following operations are done depending on the combination dir/mode
    * dir mode operation
    * 0   0    re and im -> cpx
    * 0   1    re -> cpx (imaginary part set to zero)
    * 0   2    im -> cpx (real part set to zero)
    * 1   0    cpx -> re and im
    * 1   1    cpx -> re (im is not considered)
    * 1   2    cpx -> im (re is not considered)
    * @param[in] cpx GPUtype complex
    * @param[in] re GPUtype real
    * @param[in] im GPUtype im
    * @param[in] dir Defines the type of the operation (real to complex, complex to real)
    * @param[in] mode Defines which GPUtype re or/and im is used
    */
    void (*realimag) (const GPUtype &cpx, const GPUtype &re, const GPUtype &im, int dir, int mode);

    // Casting functions
    /// Cast from FLOAT to DOUBLE
    /**
    *
    * @param[in] p GPUtype input variable (FLOAT)
    * @return A new GPUtype object (DOUBLE)
    */

    GPUtype (*floatToDouble) (const GPUtype &p);


    /// Cast from DOUBLE to FLOAT
    /**
    *
    * @param[in] p GPUtype input variable (DOUBLE)
    * @return A new GPUtype object (FLOAT)
    */

    GPUtype (*doubleToFloat) (const GPUtype &p);

    /// Cast from REAL to COMPLEX
    /**
    *
    * @param[in] p GPUtype input variable (REAL)
    * @return A new GPUtype object (COMPLEX)
    */

    GPUtype (*realToComplex) (const GPUtype &p);

    /// Cast from REAL to COMPLEX
    /**
    *
    * @param[in] re GPUtype input variable (REAL)
    * @param[in] im GPUtype input variable (IMAG)
    * @return A new GPUtype object (COMPLEX)
    */

    GPUtype (*realImagToComplex) (const GPUtype &re, const GPUtype &im);

    /// MXREPMAT
    /**
     * Defined in NUMERICS module
     */
    GPUtype (*mxRepmatDrv) (const GPUtype &p, int nrhs, const mxArray *prhs[]);

    /// mxPermuteDrv
    /**
     * Defined in NUMERICS module
     */
    GPUtype (*mxPermuteDrv) (const GPUtype &RHS, int nrhs, const mxArray *prhs[]);


    /// MXEYEDRV
    /**
     * Defined in NUMERICS module
     */
    GPUtype (*mxEyeDrv) (const GPUtype &p, int nrhs, const mxArray *prhs[]);

    /// MXZEROSDRV
    /**
     * Defined in NUMERICS module
     */
    GPUtype (*mxZerosDrv) (const GPUtype &p, int nrhs, const mxArray *prhs[]);

    /// MXONESDRV
    /**
     * Defined in NUMERICS module
     */
    GPUtype (*mxOnesDrv) (const GPUtype &p, int nrhs, const mxArray *prhs[]);


    /// EYE
    /**
     * Defined in NUMERICS module
     */
    void (*eye) (const GPUtype &p);

    /// ZEROS
    /**
     * Defined in NUMERICS module
     */
    void (*zeros) (const GPUtype &p);

    /// ONES
    /**
     * Defined in NUMERICS module
     */
    void (*ones) (const GPUtype &p);


    /// MXFILL
    /**
     * Wrapper to the fill function
     * Defined in NUMERICS module
     */
    void (*mxFill) (const GPUtype &p, int nrhs, const mxArray *prhs[]);

    /// MXCOLON
    /**
     * Wrapper to the colon function
     * Defined in NUMERICS module
     */
    GPUtype (*mxColonDrv) (const GPUtype &p, int nrhs, const mxArray *prhs[]);


    /// MXMEMCPYDTOD
    /**
     * Wrapper to the memCpyDtoD function
     * Defined in NUMERICS module
     */
    void (*mxMemCpyDtoD) (const GPUtype &dst, const GPUtype &src, int nrhs, const mxArray *prhs[]);

    /// MXMEMCPYHTOD
    /**
     * Wrapper to the memCpyHtoD function
     * Defined in NUMERICS module
     */
    void (*mxMemCpyHtoD) (const GPUtype &dst, int nrhs, const mxArray *prhs[]);

} GPUmatGPUtype;


typedef struct {
  /// FFT1D
  /**
  * @param[in] p GPUtype input variable.
  * @return A new GPUtype object
  */
  GPUtype (*FFT1Drv)(const GPUtype  &p);

  /// FFT2D
  /**
  * @param[in] p GPUtype input variable.
  * @return A new GPUtype object
  */
  GPUtype (*FFT2Drv)(const GPUtype  &p);

  /// FFT3D
  /**
  * @param[in] p GPUtype input variable.
  * @return A new GPUtype object
  */
  GPUtype (*FFT3Drv)(const GPUtype  &p);

  /// IFFT1D
  /**
  * @param[in] p GPUtype input variable.
  * @return A new GPUtype object
  */
  GPUtype (*IFFT1Drv)(const GPUtype  &p);

  /// IFFT2D
  /**
  * @param[in] p GPUtype input variable.
  * @return A new GPUtype object
  */
  GPUtype (*IFFT2Drv)(const GPUtype  &p);

  /// IFFT3D
  /**
  * @param[in] p GPUtype input variable.
  * @return A new GPUtype object
  */
  GPUtype (*IFFT3Drv)(const GPUtype  &p);

} GPUmatFFT;

typedef struct  {
  /// getCompileMode
  /**
  * Returns 1 if GPUmat is running in compilation mode.
  * If the returned value is 1, then the function should
  * either abort or generate the necessary code for compilation
  */

  int (*getCompileMode)();

  /// pushGPUtype
  /**
  * GPUtype must be pushed into the compilation context to make it available
  * to other functions.
  */
  void (*pushGPUtype) (void *);

  /// pushMx
  /**
  * Push an mxArray to the compiler context (stack).
  */

  void (*pushMx) (const mxArray *);

  /// createMxContext
  /**
  * Add an mxArray to the compilation context.
  */

  void (*createMxContext) (mxArray *mx);


  /// registerInstruction
  /**
  * @param[in] str Register the string str in the compilation buffer.
  *            Basically it writes to the generated file the string str.
  */
  void (*registerInstruction) (char * str);

  /// abort
  /**
  * Aborts compilation and writes the specified string error
  * @param[in] str Aborts the compilation and writes the specified error message.
  */
  void (*abort) (STRINGCONST char * str);

  /// getContextGPUtype
  /**
  * Returns the context ID of the GPUtype or -1 if it is not in the compilation context
  * @param[in] p pointer to GPUtype (casted to void *)
  * @return -1 if the GPUtype is not it the compilation context, it's ID otherwise
  */
  int (*getContextGPUtype) (void * p);

  /// getContextMx
  /**
  * Returns the context ID of the mxArray or -1 if it is not in the compilation context
  * @param[in] p pointer to mxArray (casted to void *)
  * @return -1 if the mxArray is not it the compilation context, it's ID otherwise
  */
  int (*getContextMx) (void * p);


  /// functionStart
  /**
   *
   */
  void (*functionStart) (STRINGCONST char *);

  /// setParamInt
  /**
   *
   */
  void (*functionSetParamInt) (int);

  /// setParamFloat
  /**
   *
   */
  void (*functionSetParamFloat) (float);

  /// setParamDouble
  /**
   *
   */
  void (*functionSetParamDouble) (double);



  /// setParamGPUtype
  /**
   *
   */
  void (*functionSetParamGPUtype) (const GPUtype *);

  /// setParamMx
  /**
   *
   */
  void (*functionSetParamMx) (const mxArray *);

  /// setParamMxMx
  /**
   *
   */
  void (*functionSetParamMxMx) (int nrhs, const mxArray *[]);


  /// registerFunction
  /**
   *
   */
  void (*functionEnd) (void);


} GPUmatCompiler;

struct GPUmatModules{
  GPUmatModules() : gpumat(0), modules(0), numerics(0), rand(0) {}
  /// native GPUmat functions loaded
  int gpumat;

  /// main module
  int modules;

  /// numerics module
  int numerics;

  /// rand module
  int rand;


};


// RAND struct

typedef struct {

    /// RAND
    /**
     * Defined in RAND module
     */
    void (*rand) (const GPUtype &p);

    /// MXRANDDRV
    /**
     * Defined in RAND module
     */
    GPUtype (*mxRandDrv) (const GPUtype &p, int nrhs, const mxArray *prhs[]);

    /**
     * Defined in RAND module
     */
    void (*randn) (const GPUtype &p);

    /// MXRANDNDRV
    /**
     * Defined in RAND module
     */
    GPUtype (*mxRandnDrv) (const GPUtype &p, int nrhs, const mxArray *prhs[]);

} GPUmatRAND;

typedef struct {

  /// mxAssign
  /**
   * Defined in NUMERICS module. Used for 'assign' function compilation
   * mxAssign is used in assign.cpp. It is a wrapper to the GPUmat native
   * mxAssign function, with additional checks and functionality
   */
  void (*mxAssign) (const GPUtype &LHS, const GPUtype &RHS, int dir,  int nrhs, const mxArray *prhs[]);

  /// mxSlice
  /**
   * Defined in NUMERICS module. Used for 'slice' function compilation
   * mxSlice is used in slice.cpp. It is a wrapper to the GPUmat native
   * mxSlice function, with additional checks and functionality
   */
  GPUtype (*mxSliceDrv) (const GPUtype &RHS, int nrhs, const mxArray *prhs[]);

} GPUmatAux;

typedef struct {

  /// getDebugMode
  /**
   * Returns the debug mode flag.
   */
  int (*getDebugMode)();

  /// setDebugMode
  /**
   * Set the debug mode flag.
   */
  void (*setDebugMode)(int);

  /// debugPushInstructionStack
  /**
   * Pushes the instruction to the instruction stack for debugging.
   */
  void (*debugPushInstructionStack) (int);

  /// log
  /**
  * Writes the specified text to the log.
  * @param[in] str Text string to be logged
  */

  void (*log) (STRINGCONST char *str, int level);


  /// logPush
  /**
   * Log indent push.
   */
  void (*logPush) ();

  /// logPop
  /**
   * Log indent pop.
   */
  void (*logPop) ();

  /// reset
  void (*reset) ();


} GPUmatDebug;

typedef struct  {

  /* GPUtype properties */
  GPUmatGPUtype gputype;

  /* Numerics functions*/
  GPUmatNumerics numerics;

  /* FFT */
  GPUmatFFT fft;

  /* GPUmat native functions interface */
  GPUmatInterface* gmat;

  /* compiler functions */
  GPUmatCompiler comp;

  /* modules configuration*/
  struct GPUmatModules mod;

  /* aux functions are special functions I need */
  GPUmatAux aux;

  /* debug */
  GPUmatDebug debug;

  /* rand */
  GPUmatRAND rand;


} GPUmat;

 /// gmGetGPUmat
/**
 * Returns pointer to GPUmat structure
 */
GPUmat * gmGetGPUmat();


 /// gmGetModule
 /**
 * Returns a pointer to the CUDA module
 */
CUmodule * gmGetModule(STRINGCONST char *modname);

 /// gmCheckGPUmat
/**
 * Check GPUmat consistency
 */
void gmCheckGPUmat(GPUmat *gm);


/*************************************************
 * UTILS
 *************************************************/
//Round a / b to nearest higher integer value
int iDivUp(int a, int b);

/*************************************************
 * HOST DRIVERS
 *************************************************/
typedef struct hostdrv_pars {
  hostdrv_pars() {
    par = NULL;
    psize = 0;
    align = __alignof(int);
  }
  hostdrv_pars(void *p, int s) {
    par = p;
    psize = s;
    align = __alignof(int);
  }
  hostdrv_pars(void *p, int s, size_t t) {
    par = p;
    psize = s;
    align = t;
  }
  void * par;
  unsigned int psize;
  size_t align;
} hostdrv_pars_t;


void hostGPUDRV(CUfunction drvfun, int N, int nrhs, hostdrv_pars_t *prhs);




/*************************************************
 * ERROR
 *************************************************/

class GPUexception {
  char *buffer;
public:

  /* CONSTRUCTOR */
  GPUexception(const char *err):buffer(NULL) {
    buffer = (char *) malloc((strlen(err)+1)*sizeof(char));
    strcpy(buffer, err);
  }

  /* DESTRUCTOR */

  /* DESTRUCTOR */
  ~GPUexception() {
    if (this->buffer != NULL)
        free(this->buffer);
  }

  char * getError() {
    return buffer;
  }

};
#endif

#endif



