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

#if !defined(GPUTYPE_H_)
#define GPUTYPE_H_

/******** DEFINES / TYPEDEF ************/
#define GPU_SIZE_OF_FLOAT   4
#define GPU_SIZE_OF_CFLOAT  8
#define GPU_SIZE_OF_DOUBLE  8
#define GPU_SIZE_OF_CDOUBLE 16
#define GPU_SIZE_OF_INT32   4

///enum gpuTYPE {
//  gpuFLOAT = 0, gpuCFLOAT = 1, gpuDOUBLE = 2, gpuCDOUBLE = 3, gpuINT32 = 4, gpuNOTDEF = 20
//};

//typedef enum gpuTYPE gpuTYPE_t;

enum opTYPE {
  opGPUtype = 0, opFLOAT = 1, opCFLOAT = 2, opDOUBLE = 3, opCDOUBLE = 4, opINT32 = 5, opNOTDEF = 20
};

typedef enum opTYPE opTYPE_t;


// Complex defined in "cuda_runtime.h"
//typedef float2 Complex;

/* GPUtype
* Defaults
* 1. Size cannot be NULL.
* 2. stream 0 by default
* 3. GPUptr NULL
* 4. GPUman cannot be NULL. Although the empty constructor
*    creates a NULL GPUman, this should be set afterwards
* 5. ndims = 1
* 6. size = [0]
* 7. mysize = sizeof(C)
*
* cloned
* 1. A GPUtype is cloned when created from another GPUtype using
*    one of the available constructors or = operator
* 2. A cloned GPUtype cannot delete the pointer to GPU
*/

class GPUtype {
  int *size;
  int ndims;
  int trans;
  int numel;
  int mysize;
  /* mytype is used to store the type of the GPUtype. This might be useful when using
  * the GPUtype in an external applications, such as Matlab, to tell Matlab what is
  * the  type of the pointed object. The compiler doesn't need this info of course
  */
  gpuTYPE_t mytype;
  int stream;
  GPUmanager * GPUman;

  // ids. ids are used in compilation mode
  unsigned int id0;
  unsigned int id1;
  unsigned int id2;


  // iscompiled is set to 1 when the GPUtype was created in
  // compilation mode. If the GPUtype is created in compilation mode, then
  // GPU memory is not allocated and any call to getGPUptr should throw an exception
  unsigned int iscompiled;

private:

  struct counter {
    counter(void* p = 0, unsigned c = 1) :
  ptr(p), count(c) {
#ifdef DEBUG
    OPENFOUT;
    //if (itsCounter)
    fprintf(fout,"counter create, %p, \n",this);
    CLOSEFOUT;
    //else
    //fprintf(fout,"counter create, %p, \n",this);

#endif
  }
  // The state with itsCounter=0 is not allowed
  void* ptr;
  unsigned count;
  }* itsCounter;



  struct myscalar {

    myscalar() {
      myfloat    = 0.0;
      myint32    = 0;
      mycfloat.x = 0.0;
      mycfloat.y = 0.0;
      mydouble   = 0.0;
      mycdouble.x = 0.0;
      mycdouble.y = 0.0;
      isset = 0;

    }
    int     myint32;
    float   myfloat;
    Complex mycfloat;
    double  mydouble;
    DoubleComplex mycdouble;
    int isset;

  } myScalar;



public:
  void acquireGPUptr(counter* c);
  void acquireGPUptr(GPUtype *p);
  void releaseGPUptr();

  /* setGPUptr */
  /* Set the GPU pointer, do not clean existing pointer */
  void setGPUptr(void* p = 0) // allocate a new counter
  {
    // if the pointer exist I should release it before setting the new one
    //if (p) {

    //if (itsCounter) {
    // itsCounter cannot be NULL
    // if (itsCounter->ptr) {
    // delete previous pointer
    //this->releaseGPUptr();
    //   GPUopFree(*this);
    //itsCounter = new counter(p);
    // }
    itsCounter->ptr = p;

    //}



  }

  /* clone */
  //GPUtype * clone();
  /* Scalar constructors */

  /* Contructor */
  GPUtype(int , GPUmanager *GPUman);

  /* Contructor */
  GPUtype(float , GPUmanager *GPUman);

  /* Contructor */
  GPUtype(Complex , GPUmanager *GPUman);

  /* Contructor */
  GPUtype(double , GPUmanager *GPUman);

  /* Contructor */
  GPUtype(DoubleComplex , GPUmanager *GPUman);

  /* Destructor */
  ~GPUtype();

  /* Contructor */
  GPUtype(gpuTYPE_t type, int ndims, const int *size, GPUmanager *GPUman);

  /* Contructor */
  GPUtype(GPUmanager *gp = NULL);

  /* Contructor */
  GPUtype(const GPUtype &, int clone = 0);

  /*************** OPERATORS ***************/

  /* Operator = */
  const GPUtype & operator=(const GPUtype &p) {
#ifdef DEBUG
    OPENFOUT;
    fprintf(fout,"op =, %p, %p\n",this, &p);
    CLOSEFOUT;
#endif
    if (this != &p) {

      this->ndims = p.ndims;
      this->trans = p.trans;
      this->numel = p.numel;
      this->mysize = p.mysize;
      this->mytype = p.mytype;
      this->stream = p.stream;
      this->GPUman = p.GPUman;
      this->id0 = p.id0;
      this->id1 = p.id1;
      this->id2 = p.id2;
      this->iscompiled = p.iscompiled;

      this->myScalar = p.myScalar;


      // now I should register myself
      //GPUman->registerGPUtype(this);

      /* size
      * I should init size to the new size
      */
      if (this->size != NULL)
        Myfree(size);

      this->size = (int *) Mymalloc(p.ndims * sizeof(int));
      this->numel = 1;
      for (int i = 0; i < p.ndims; i++) {
        this->size[i] = p.size[i];
        this->numel *= p.size[i];
      }

      // counter GPUptr

      this->releaseGPUptr();
      this->acquireGPUptr(p.itsCounter);

    }
    return *this;
  }

  /*************** SETTERS ***************/

  /* setID */
  void
    setID(unsigned int i0, unsigned int i1, unsigned int i2);

  void
    setID0(unsigned int i0);

  void
    setID1(unsigned int i1);

  void
    setID2(unsigned int i2);

  /* setType */
  void
    setType(gpuTYPE_t type);

  /* setComplex */
  void
    setComplex();

  /* setReal */
  void setReal();



  /* setSize */
  void setSize(int ndims, int *size);

  /* setStream */
  void setStream(int);

  /* setTrans */
  void setTrans(int);


  /*************** GETTERS ***************/

  /* get IDs */
  unsigned int getID0();
  unsigned int getID1();
  unsigned int getID2();

  /* get iscompiled */
  unsigned int isCompiled();


  int getPtrCount();

  //struct counter * getPtrCounter();

  /* getScalarPtr*/
  /*void * getScalarPtr();*/

  /* getGPUmanager */
  GPUmanager * getGPUmanager();


  /* get ScalarF */
  float getScalarF();

  /* get ScalarCF */
  Complex getScalarCF();

  /* get ScalarD */
  double getScalarD();

  /* get ScalarCD */
  DoubleComplex getScalarCD();

  /* get ScalarI32 */
  int getScalarI32();


  /* isFloat */
  int
    isFloat();

  /* isComplex */
  int
    isComplex();

  /* isDouble */
  int
    isDouble();

  /* isInt32 */
  int
    isInt32();

  /* isScalar */
  int
    isScalar();

  /* isTrans */
  int
    isTrans();

  /* getGPUptr */
  void *
    getGPUptr();

  /* getGPUptrptr */
  void **
    getGPUptrptr();

  /* getNumel */
  int getNumel();

  /* getNdims */
  int getNdims();

  /* getSize */
  int *
    getSize();

  /* isempty */
  int isEmpty();



  /* inStream */
  int getStream();

  /* getMySize() */
  int getMySize();

  /* get type
  * Specify a getType for each particular C */

  gpuTYPE_t getType();

  /* getEnd() */
  int getEnd(int k, int n);


  /*************** Util ***************/

  /* dump */
  void dump();

  /* print */
  void print();

  /* printShort */
  void printShort();

  /*************** Casting ***************/

  /* clone */
  GPUtype * clone();

  /* REALtoCOMPLEX */
  GPUtype * REALtoCOMPLEX();
  GPUtype * REALtoCOMPLEX(GPUtype &im);


  /* INT32toFLOAT */
  GPUtype * INT32toFLOAT();

  /* INT32toCFLOAT */
  GPUtype * INT32toCFLOAT();

  /* INT32toDOUBLE */
  GPUtype * INT32toDOUBLE();

  /* INT32toCDOUBLE */
  GPUtype * INT32toCDOUBLE();

  /* FLOATtoCFLOAT */
  GPUtype * FLOATtoCFLOAT();
  GPUtype * FLOATtoCFLOAT(GPUtype &im);


  /* FLOATtoDOUBLE */
  GPUtype * FLOATtoDOUBLE();

  /* FLOATtoCDOUBLE */
  GPUtype * FLOATtoCDOUBLE();

  /* DOUBLEtoFLOAT */
  GPUtype * DOUBLEtoFLOAT();

  /* DOUBLEtoCFLOAT */
  GPUtype * DOUBLEtoCFLOAT();

  /* DOUBLEtoCDOUBLE */
  GPUtype * DOUBLEtoCDOUBLE();
  GPUtype * DOUBLEtoCDOUBLE(GPUtype &im);

  /* float * */
  /*template<class C>
  operator C *() {
  // allocate destination
  if (this->numel == 0)
  throw GPUexception(GPUmatError, "Casting an empty object");

  if (this->getGPUptr() == 0)
  throw GPUexception(GPUmatError,
  "Casting an object not allocated on GPU memory");
  unsigned int memsize = this->numel * sizeof(C);
  C * dest = (C*) Mymalloc(memsize);
  // no streaming operation in this case
  // TODO handle executionDelayed


  // copy from GPU to CPU
  // copy result from device to host
  GPUmatResult_t status = GPUopCudaMemcpy(dest, this->getGPUptr(), memsize,
  cudaMemcpyDeviceToHost, this->GPUman);
  if (status != GPUmatSuccess)
  throw GPUexception(GPUmatError,
  "Unable to copy from GPU memory to CPU memory.");

  return dest;
  }*/

};

#endif

