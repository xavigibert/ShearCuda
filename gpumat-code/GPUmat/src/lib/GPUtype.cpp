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


#ifdef UNIX
#include <stdint.h>
#endif

//#include "cutil.h"
#include "cublas.h"
#include "cuda_runtime.h"
#include "cuda.h"
//extern "C" cudaError_t  cudaFree(void *devPtr);
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
/**************** CONSTRUCTORS    **************/
/* Destructor */
GPUtype::~GPUtype() {
	//this->print();
#ifdef DEBUG
	FILE *fout=fopen("GPUtypedelete.dat","a+");
	fprintf(fout,"%p\n",this);
	fclose(fout);
#endif

	// release GPU memory
	this->releaseGPUptr();

	// should be done at the end, otherwise the GPUtype
	// is dirty and any scheduled operation from releaseGPUptr
	// will be wrong


	if (this->size != NULL)
		Myfree(this->size);

	// I should unregister myself from GPUmanager
	//this->GPUman->unregister (this);
}

/* Scalar constructors */
/* Constructor */
/* Some defaults
 *  Some defaults
 * 1. Size cannot be NULL.
 * 2. stream 0 by default
 * 3. GPUptr NULL
 * 4. GPUman NULL
 * 5. ndims = 1
 * 6. size = [0]
 */

GPUtype::GPUtype(float myf, GPUmanager *GPUman) :
	size(NULL), ndims(2), trans(0), numel(1), mysize(GPU_SIZE_OF_FLOAT), mytype(
			gpuFLOAT), stream(0), itsCounter(new counter()), myScalar(myscalar()),
			id0(0), id1(0), id2(0), iscompiled(0) {

#ifdef DEBUG
	FILE *fout=fopen("GPUtype.dat","a+");
	fprintf(fout,"%p\n",this);
	fclose(fout);
#endif

#ifdef DEBUG
	OPENFOUT;
	fprintf(fout,"constructor1, %p\n",this);
	CLOSEFOUT;
#endif
	int i;

	/* default values */
  int ndims = 2;
  int size[2] ={1,1};

	/* now set the right values */
	this->ndims = ndims;
	this->GPUman = GPUman;

	// in this case I have to use the CPU size?
	//this->mysize = GPU_SIZE_OF_FLOAT;
	this->mysize = sizeof(float);

	this->mytype = gpuFLOAT;
	this->myScalar.myfloat = myf;
	this->myScalar.isset = 1;

	/* size is set according to ndims
	 * length(size) = ndims
	 */
	if (ndims > 0) {
		this->size = (int *) Mymalloc(ndims * sizeof(int));
		this->numel = 1;
		for (i = 0; i < ndims; i++) {
			this->size[i] = size[i];
			this->numel *= size[i];
		}

	} else {
		this->ndims = 1;
		this->size = (int *) Mymalloc(this->ndims * sizeof(int));
		/* consistent size, numel */
		this->size[0] = 0;
		this->numel = 0;

	}

	// iscompiled
	if (this->GPUman->getCompileMode()==1)
	  this->iscompiled = 1;

	GPUopAllocVector(*this);
	GPUopCudaMemcpy(this->getGPUptr(), &myf, this->getMySize()
					* 1, cudaMemcpyHostToDevice, this->getGPUmanager());

}

GPUtype::GPUtype(Complex mycf, GPUmanager *GPUman) :
	size(NULL), ndims(2), trans(0), numel(1), mysize(GPU_SIZE_OF_CFLOAT), mytype(
			gpuCFLOAT), stream(0), itsCounter(new counter()), myScalar(myscalar()),
			id0(0), id1(0), id2(0), iscompiled(0) {

#ifdef DEBUG
	FILE *fout=fopen("GPUtype.dat","a+");
	fprintf(fout,"%p\n",this);
	fclose(fout);
#endif

#ifdef DEBUG
	OPENFOUT;
	fprintf(fout,"constructor1, %p\n",this);
	CLOSEFOUT;
#endif
	int i;

	/* default values */
  int ndims = 2;
  int size[2] ={1,1};

	/* now set the right values */
	this->ndims = ndims;
	this->GPUman = GPUman;

	//this->mysize = GPU_SIZE_OF_CFLOAT;
	this->mysize = sizeof(Complex);

	this->mytype = gpuCFLOAT;
	this->myScalar.mycfloat = mycf;
	this->myScalar.isset = 1;

	/* size is set according to ndims
	 * length(size) = ndims
	 */
	if (ndims > 0) {
		this->size = (int *) Mymalloc(ndims * sizeof(int));
		this->numel = 1;
		for (i = 0; i < ndims; i++) {
			this->size[i] = size[i];
			this->numel *= size[i];
		}

	} else {
		this->ndims = 1;
		this->size = (int *) Mymalloc(this->ndims * sizeof(int));
		/* consistent size, numel */
		this->size[0] = 0;
		this->numel = 0;

	}

	// iscompiled
  if (this->GPUman->getCompileMode()==1)
    this->iscompiled = 1;

	GPUopAllocVector(*this);
	GPUopCudaMemcpy(this->getGPUptr(), &mycf, this->getMySize()
					* 1, cudaMemcpyHostToDevice, this->getGPUmanager());

}

GPUtype::GPUtype(double myd, GPUmanager *GPUman) :
	size(NULL), ndims(2), trans(0), numel(1), mysize(GPU_SIZE_OF_DOUBLE), mytype(
			gpuDOUBLE), stream(0), itsCounter(new counter()), myScalar(myscalar()),
			id0(0), id1(0), id2(0), iscompiled(0) {

#ifdef DEBUG
	FILE *fout=fopen("GPUtype.dat","a+");
	fprintf(fout,"%p\n",this);
	fclose(fout);
#endif

#ifdef DEBUG
	OPENFOUT;
	fprintf(fout,"constructor1, %p\n",this);
	CLOSEFOUT;
#endif

	// check if supported
	if (GPUman->getCudaCapability()<13) {
		throw GPUexception(GPUmatError,
						ERROR_GPUTYPE_DOUBLENOTSUPPORTED);
	}
	int i;

	/* default values */
  int ndims = 2;
  int size[2] ={1,1};

	/* now set the right values */
	this->ndims = ndims;
	this->GPUman = GPUman;

	//this->mysize = GPU_SIZE_OF_DOUBLE;
	this->mysize = sizeof(double);

	this->mytype = gpuDOUBLE;
	this->myScalar.mydouble = myd;
	this->myScalar.isset = 1;

	/* size is set according to ndims
	 * length(size) = ndims
	 */
	if (ndims > 0) {
		this->size = (int *) Mymalloc(ndims * sizeof(int));
		this->numel = 1;
		for (i = 0; i < ndims; i++) {
			this->size[i] = size[i];
			this->numel *= size[i];
		}

	} else {
		this->ndims = 1;
		this->size = (int *) Mymalloc(this->ndims * sizeof(int));
		/* consistent size, numel */
		this->size[0] = 0;
		this->numel = 0;

	}

	if (this->GPUman->getCompileMode()==1)
    this->iscompiled = 1;

	GPUopAllocVector(*this);
	GPUopCudaMemcpy(this->getGPUptr(), &myd, this->getMySize()
					* 1, cudaMemcpyHostToDevice, this->getGPUmanager());
}

GPUtype::GPUtype(DoubleComplex mycd, GPUmanager *GPUman) :
	size(NULL), ndims(2), trans(0), numel(1), mysize(GPU_SIZE_OF_CDOUBLE), mytype(
			gpuCDOUBLE), stream(0), itsCounter(new counter()), myScalar(myscalar()),
			id0(0), id1(0), id2(0), iscompiled(0) {

#ifdef DEBUG
	FILE *fout=fopen("GPUtype.dat","a+");
	fprintf(fout,"%p\n",this);
	fclose(fout);
#endif

#ifdef DEBUG
	OPENFOUT;
	fprintf(fout,"constructor1, %p\n",this);
	CLOSEFOUT;
#endif
	if (GPUman->getCudaCapability()<13) {
		throw GPUexception(GPUmatError,
						ERROR_GPUTYPE_DOUBLENOTSUPPORTED);
	}
	int i;

	/* default values */
  int ndims = 2;
  int size[2] ={1,1};

	/* now set the right values */
	this->ndims = ndims;
	this->GPUman = GPUman;

	//this->mysize = GPU_SIZE_OF_CDOUBLE;
	this->mysize = sizeof(DoubleComplex);

	this->mytype = gpuCDOUBLE;
	this->myScalar.mycdouble = mycd;
	this->myScalar.isset = 1;

	/* size is set according to ndims
	 * length(size) = ndims
	 */
	if (ndims > 0) {
		this->size = (int *) Mymalloc(ndims * sizeof(int));
		this->numel = 1;
		for (i = 0; i < ndims; i++) {
			this->size[i] = size[i];
			this->numel *= size[i];
		}

	} else {
		this->ndims = 1;
		this->size = (int *) Mymalloc(this->ndims * sizeof(int));
		/* consistent size, numel */
		this->size[0] = 0;
		this->numel = 0;

	}

  if (this->GPUman->getCompileMode()==1)
    this->iscompiled = 1;

	GPUopAllocVector(*this);
	GPUopCudaMemcpy(this->getGPUptr(), &mycd, this->getMySize()
					* 1, cudaMemcpyHostToDevice, this->getGPUmanager());
}

GPUtype::GPUtype(int myi, GPUmanager *GPUman) :
	size(NULL), ndims(2), trans(0), numel(1), mysize(GPU_SIZE_OF_INT32), mytype(
			gpuINT32), stream(0), itsCounter(new counter()), myScalar(myscalar()),
			id0(0), id1(0), id2(0), iscompiled(0) {

#ifdef DEBUG
	FILE *fout=fopen("GPUtype.dat","a+");
	fprintf(fout,"%p\n",this);
	fclose(fout);
#endif

#ifdef DEBUG
	OPENFOUT;
	fprintf(fout,"constructor1, %p\n",this);
	CLOSEFOUT;
#endif
	int i;

	/* default values */
  int ndims = 2;
  int size[2] ={1,1};

	/* now set the right values */
	this->ndims = ndims;
	this->GPUman = GPUman;

	// in this case I have to use the CPU size?
	//this->mysize = GPU_SIZE_OF_FLOAT;
	this->mysize = sizeof(int);

	this->mytype = gpuINT32;
	this->myScalar.myint32 = myi;
	this->myScalar.isset = 1;

	/* size is set according to ndims
	 * length(size) = ndims
	 */
	if (ndims > 0) {
		this->size = (int *) Mymalloc(ndims * sizeof(int));
		this->numel = 1;
		for (i = 0; i < ndims; i++) {
			this->size[i] = size[i];
			this->numel *= size[i];
		}

	} else {
		this->ndims = 1;
		this->size = (int *) Mymalloc(this->ndims * sizeof(int));
		/* consistent size, numel */
		this->size[0] = 0;
		this->numel = 0;

	}

  if (this->GPUman->getCompileMode()==1)
    this->iscompiled = 1;

	GPUopAllocVector(*this);
	GPUopCudaMemcpy(this->getGPUptr(), &myi, this->getMySize()
					* 1, cudaMemcpyHostToDevice, this->getGPUmanager());

}


/* Constructor */
/* Some defaults
 * 1. Size cannot be NULL.
 * 2. stream 0 by default
 * 3. GPUptr NULL
 * 4. GPUman NULL
 * 5. ndims = 1
 * 6. size = [0]
 */

GPUtype::GPUtype(GPUmanager *gp) :
	size(NULL), ndims(1), trans(0), numel(0), mysize(GPU_SIZE_OF_FLOAT), mytype(
			gpuFLOAT), stream(0), GPUman(NULL), itsCounter(new counter()), myScalar(myscalar()),
			id0(0), id1(0), id2(0), iscompiled(0) {

	/* default values */

	// now I should register myself
	//GPUman->registerGPUtype(this);

	// default size
	// numel = 0
	// size = [0]
	// ndims = 1

	this->size = (int *) Mymalloc(sizeof(int));
	/* consistent size, numel */
	this->size[0] = 0;
	this->GPUman = gp;

  if (this->GPUman->getCompileMode()==1)
    this->iscompiled = 1;


#ifdef DEBUG
	FILE *fout=fopen("GPUtype.dat","a+");
	fprintf(fout,"%p\n",this);
	fclose(fout);
#endif

}

/* Constructor */
/* Some defaults
 *  Some defaults
 * 1. Size cannot be NULL.
 * 2. stream 0 by default
 * 3. GPUptr NULL
 * 4. GPUman NULL
 * 5. ndims = 1
 * 6. size = [0]
 */

GPUtype::GPUtype(gpuTYPE_t type, int ndims, const int *size, GPUmanager *GPUman) :
	size(NULL), ndims(1), trans(0), numel(0), mysize(GPU_SIZE_OF_FLOAT), mytype(
			gpuFLOAT), stream(0), itsCounter(new counter()), myScalar(myscalar()),
			id0(0), id1(0), id2(0), iscompiled(0) {

#ifdef DEBUG
	FILE *fout=fopen("GPUtype.dat","a+");
	fprintf(fout,"%p\n",this);
	fclose(fout);
#endif

#ifdef DEBUG
	OPENFOUT;
	fprintf(fout,"constructor1, %p\n",this);
	CLOSEFOUT;
#endif
	if ((type==gpuDOUBLE)||(type==gpuCDOUBLE)) {
		if (GPUman->getCudaCapability()<13) {
			throw GPUexception(GPUmatError,
							ERROR_GPUTYPE_DOUBLENOTSUPPORTED);
		}
	}
	int i;

	/* default values */

	/* now set the right values */
	this->ndims = ndims;

	this->GPUman = GPUman;

	this->mytype = type;
	/* mysize */
	if (type == gpuFLOAT) {
		mysize = GPU_SIZE_OF_FLOAT;
	} else if (type == gpuCFLOAT) {
		mysize = GPU_SIZE_OF_CFLOAT;
	} else if (type == gpuDOUBLE) {
		mysize = GPU_SIZE_OF_DOUBLE;
	} else if (type == gpuCDOUBLE) {
		mysize = GPU_SIZE_OF_CDOUBLE;
	} else if (type == gpuINT32) {
		mysize = GPU_SIZE_OF_INT32;
	}

	// now I should register myself
	//GPUman->registerGPUtype(this);

	/* size is set according to ndims
	 * length(size) = ndims
	 */
	if (ndims > 0) {
		this->size = (int *) Mymalloc(ndims * sizeof(int));
		this->numel = 1;
		for (i = 0; i < ndims; i++) {
			this->size[i] = size[i];
			this->numel *= size[i];
		}

	} else {
		this->ndims = 1;
		this->size = (int *) Mymalloc(this->ndims * sizeof(int));
		/* consistent size, numel */
		this->size[0] = 0;
		this->numel = 0;

	}

  if (this->GPUman->getCompileMode()==1)
    this->iscompiled = 1;


}

/* Constructor */

GPUtype::GPUtype(const GPUtype &p, int clone) :
	size(NULL), ndims(1), trans(0), numel(0), mysize(GPU_SIZE_OF_FLOAT), mytype(
			gpuFLOAT), stream(0), itsCounter(new counter()), myScalar(myscalar()),
			id0(0), id1(0), id2(0), iscompiled(0) {
#ifdef DEBUG
	OPENFOUT;
	fprintf(fout,"constructor2, %p\n",this);
	CLOSEFOUT;
#endif

#ifdef DEBUG
	FILE *fout=fopen("GPUtype.dat","a+");
	fprintf(fout,"%p\n",this);
	fclose(fout);
#endif

	if (clone) {
		this->ndims = p.ndims;
		this->trans = p.trans;
		this->numel = p.numel;
		this->mysize = p.mysize;
		this->mytype = p.mytype;
		this->stream = p.stream;
		this->GPUman = p.GPUman;
		this->id0    = p.id0;
		this->id1    = p.id1;
		this->id2    = p.id2;
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
	} else {
		*this = p;

	}

	if (this->GPUman->getCompileMode()==1)
    this->iscompiled = 1;

}

/* acquireGPUptr */
void GPUtype::acquireGPUptr(counter* c) { // increment the count

	itsCounter = c;
	if (c)
		++c->count;

#ifdef DEBUG
	OPENFOUT;
	if (itsCounter)
	fprintf(fout,"counter ++, %p, %p, %d\n",itsCounter, this, itsCounter->count);
	else
	fprintf(fout,"counter ++, %p, %p, %d\n",itsCounter, this, 0);

	CLOSEFOUT;
#endif
}

void GPUtype::acquireGPUptr(GPUtype * p) { // increment the count
  acquireGPUptr(p->itsCounter);
}
/* releaseGPUptr */
void GPUtype::releaseGPUptr() { // decrement the count, delete if it is 0
	// itsCoounter cannot be NULL
	//if (itsCounter) {

	//
	if (--itsCounter->count == 0) {

		if (itsCounter->ptr)
			GPUopFree(*this);

		delete itsCounter;
	}
#ifdef DEBUG
	OPENFOUT;
	if (itsCounter)
	fprintf(fout,"counter release, %p, %p, %d\n",itsCounter, this, itsCounter->count);
	else
	fprintf(fout,"counter release, %p, %p, %d\n",itsCounter, this, 0);

	CLOSEFOUT;
#endif
	itsCounter = 0;
	//}

}

/*************** GETTERS ***************/

  /* isCompiled */
  unsigned int GPUtype::isCompiled() {
    return this->iscompiled;
  }

  /* getID */

  unsigned int GPUtype::getID0() {
    return this->id0;
  }

  unsigned int GPUtype::getID1() {
    return this->id1;
  }

  unsigned int GPUtype::getID2() {
    return this->id2;
  }

	/* get ScalarF */
  float GPUtype::getScalarF() {
  	float tmp = 0.0;
  	if (this->myScalar.isset==1) {
  	  tmp = myScalar.myfloat;
  	} else {
  		GPUopCudaMemcpy(&tmp, this->getGPUptr(), this->getMySize()
  							* 1, cudaMemcpyDeviceToHost, this->getGPUmanager());
  	}
  	return tmp;
  }

  /* get ScalarCF */
  Complex GPUtype::getScalarCF(){
  	Complex tmp = {0.0,0.0};
  	if (this->myScalar.isset==1) {
  	  tmp = myScalar.mycfloat;
  	} else {
  		GPUopCudaMemcpy(&tmp, this->getGPUptr(), this->getMySize()
  							* 1, cudaMemcpyDeviceToHost, this->getGPUmanager());
  	}
  	return tmp;
  }

  /* get ScalarD */
  double GPUtype::getScalarD(){
  	double tmp = 0.0;
  	if (this->myScalar.isset==1) {
  	  tmp = myScalar.mydouble;
  	} else {
  		GPUopCudaMemcpy(&tmp, this->getGPUptr(), this->getMySize()
  							* 1, cudaMemcpyDeviceToHost, this->getGPUmanager());
  	}
  	return tmp;
  }

  /* get ScalarCD */
  DoubleComplex GPUtype::getScalarCD(){
  	DoubleComplex tmp = {0.0,0.0};
  	if (this->myScalar.isset==1) {
     	tmp =  myScalar.mycdouble;
  	} else {
		  GPUopCudaMemcpy(&tmp, this->getGPUptr(), this->getMySize()
							* 1, cudaMemcpyDeviceToHost, this->getGPUmanager());
  	}
  	return tmp;

  }

  /* get ScalarI32 */
	int GPUtype::getScalarI32() {
		int tmp = 0;
		if (this->myScalar.isset==1) {
			tmp = myScalar.myint32;
		} else {
			GPUopCudaMemcpy(&tmp, this->getGPUptr(), this->getMySize()
								* 1, cudaMemcpyDeviceToHost, this->getGPUmanager());
		}
		return tmp;
	}


/* getEnd()
 *
 * Same as Matlab function end. Check help end in Matlab
 *
 * k starts from 0 instead of 1 as in Matlab
 *
 * */
int GPUtype::getEnd(int k, int n) {

	/*s = size(A);

	 c = cumprod([1 s(1:end)]); % c(end) è il numero totale di elementi
	 ntot = c(end);

	 l = length(s);

	 if (k==n)
	 % se k == n
	 y = ntot/c(k);
	 else
	 y = s(k);
	 end*/
	int r = 0;
	if (k == (n - 1)) {
		// calculate cumulative
		int c = 1;
		for (int i = 0; i < k; i++) {
			if (i < this->ndims)
				c = c * this->size[i];
			// if we use an index with more dimensions
			// the last dimensions are considered to be 1 by default
		}
		r = this->numel / c - 1; // index from 0
	} else {
		if (k <  this->ndims)
			r = size[k] - 1;
		else
			r = 0;

	}

	return r;

}

/* isComplex */
int GPUtype::isComplex() {
	return ((this->mytype == gpuCFLOAT) || (this->mytype == gpuCDOUBLE));
}

/* isFloat */
int GPUtype::isFloat() {
	return ((this->mytype == gpuCFLOAT) || (this->mytype == gpuFLOAT));
}

/* isDouble */
int GPUtype::isDouble() {
	return ((this->mytype == gpuCDOUBLE) || (this->mytype == gpuDOUBLE));
}

/* isInt32 */
int GPUtype::isInt32() {
	return ((this->mytype == gpuINT32));
}

/* getTrans */
int GPUtype::isTrans() {
	return trans;
}

/* getGPUManager */

GPUmanager *
GPUtype::getGPUmanager() {
	return this->GPUman;
}

/* getMySize() */

int GPUtype::getMySize() {
	return mysize;
}

/* getStream */

int GPUtype::getStream() {
	return stream;
}

/* isscalar */
/* Scalar should have size = [1 1]
 *
 */

int GPUtype::isScalar() {
	int l = this->ndims;
	int r = 0;
	int * el = this->size;
	if (l == 2 && el[0] == 1 && el[1] == 1) {
		r = 1;
	}
	return r;

}

/* isempty */
int GPUtype::isEmpty() {
	return this->numel == 0;
}

/* getGPUptr */
void *
GPUtype::getGPUptr() {
  /*
  // If a GPUtype was created during the compilation mode
  // it is not possible to retrieve the GPU pointer.
  // In general this should not happen, but anyway we perform
  // here a check
  if (this->isCompiled()) {
    throw GPUexception(GPUmatError,
        ERROR_GPUTYPE_ISCOMPILEDGPUPTR);
  }*/
	return (itsCounter ? itsCounter->ptr : 0);
}
/* getGPUptrptr */

void **
GPUtype::getGPUptrptr() {
  /*if (this->isCompiled()) {
    throw GPUexception(GPUmatError,
        ERROR_GPUTYPE_ISCOMPILEDGPUPTR);
  }*/
	return (itsCounter ? &(itsCounter->ptr) : 0);
}
/* numel */

int GPUtype::getNumel() {
	return numel;
}

/* getNdims */

int GPUtype::getNdims() {
	return ndims;
}

/* getSize */

int *
GPUtype::getSize() {
	return size;
}

/* getSize */

gpuTYPE_t GPUtype::getType() {
	return mytype;
}

int GPUtype::getPtrCount() {
  return itsCounter->count;
}

//struct GPUtype::counter * GPUtype::getPtrCounter() {
//  return itsCounter;
//}

/*************** SETTERS ***************/

/* setID */
void GPUtype::setID(unsigned int i0, unsigned int i1, unsigned int i2) {
  this->id0 = i0;
  this->id1 = i1;
  this->id2 = i2;
}

void GPUtype::setID0(unsigned int i0) {
  this->id0 = i0;
}

void GPUtype::setID1(unsigned int i1) {
  this->id1 = i1;
}

void GPUtype::setID2(unsigned int i2) {
  this->id2 = i2;
}

/* setType */
void GPUtype::setType(gpuTYPE_t type) {
	if (type == gpuCFLOAT) {
		this->mysize = GPU_SIZE_OF_CFLOAT;
		this->mytype = gpuCFLOAT;

	} else if (type == gpuCDOUBLE) {
		this->mysize = GPU_SIZE_OF_CDOUBLE;
		this->mytype = gpuCDOUBLE;

	} else if (type == gpuFLOAT) {
		this->mysize = GPU_SIZE_OF_FLOAT;
		this->mytype = gpuFLOAT;

	} else if (type == gpuDOUBLE) {
		this->mysize = GPU_SIZE_OF_DOUBLE;
		this->mytype = gpuDOUBLE;

	} else if (type == gpuINT32) {
		this->mysize = GPU_SIZE_OF_INT32;
		this->mytype = gpuINT32;
	}

}

/* setComplex */
void GPUtype::setComplex() {
	if (mytype == gpuFLOAT) {
		mysize = GPU_SIZE_OF_CFLOAT;
		mytype = gpuCFLOAT;
	}
	if (mytype == gpuDOUBLE) {
		mysize = GPU_SIZE_OF_CDOUBLE;
		mytype = gpuCDOUBLE;
	}
	if (mytype == gpuINT32) {
		/* not possible */
		throw GPUexception(GPUmatError,
				ERROR_GPUTYPE_INT32CPX);
	}

}

/* setReal */
void GPUtype::setReal() {
	if (mytype == gpuCFLOAT) {
		mysize = GPU_SIZE_OF_FLOAT;
		mytype = gpuFLOAT;
	}
	if (mytype == gpuCDOUBLE) {
		mysize = GPU_SIZE_OF_DOUBLE;
		mytype = gpuDOUBLE;
	}

}


/* set trans */
void GPUtype::setTrans(int t) {
	trans = t;
}

/* setStream */

void GPUtype::setStream(int s) {
	stream = s;
}

/* setSize */

void GPUtype::setSize(int ndims, int *size) {
	int i;
	// be carefull here. size could be the same pointer as
	// this->size. In that case, we don't have to free it
	// here but just at the end. Thats why we make a local copy to free at the
	// end

	int *oldsize = this->size;
	// delete at the end
	//if (this->size != NULL)
	//  Myfree(this->size);

	/* size is set according to ndims
	 *
	 * empty GPUtype is allowed so I should
	 * manage this case
	 *
	 * length(size) = ndims
	 */
	if (ndims > 0) {
		this->size = (int *) Mymalloc(ndims * sizeof(int));
		this->numel = 1;
		this->ndims = ndims;
		for (i = 0; i < ndims; i++) {
			this->size[i] = size[i];
			this->numel *= size[i];
		}

	} else {
		this->ndims = 1;
		this->size = (int *) Mymalloc(this->ndims * sizeof(int));
		/* consistent size, numel */
		this->size[0] = 0;
		this->numel = 0;

	}

	// delete only here to avoid deleting a pointer
	// that is being used
	if (oldsize != NULL)
		Myfree(oldsize);

}

/*************** Casting ***************/

  /* REALtoCOMPLEX */
  GPUtype * GPUtype::REALtoCOMPLEX() {
    if (this->mytype==gpuFLOAT) {
    	return this->FLOATtoCFLOAT();

    } else if (this->mytype==gpuDOUBLE) {
    	return this->DOUBLEtoCDOUBLE();

    } else if (this->mytype==gpuCDOUBLE) {
    	return this->clone();

    } else if (this->mytype==gpuCFLOAT) {
        	return this->clone();

    } else if (this->mytype==gpuINT32) {
    	throw GPUexception(GPUmatError,
    					ERROR_GPUTYPE_INT32CPX);
    }
  }

  GPUtype * GPUtype::REALtoCOMPLEX(GPUtype &im) {
		if (this->mytype==gpuFLOAT) {
			return this->FLOATtoCFLOAT(im);

		} else if (this->mytype==gpuDOUBLE) {
			return this->DOUBLEtoCDOUBLE(im);

		} else if (this->mytype==gpuCDOUBLE) {
			return this->clone();

		} else if (this->mytype==gpuCFLOAT) {
					return this->clone();

		} else if (this->mytype==gpuINT32) {
			throw GPUexception(GPUmatError,
							ERROR_GPUTYPE_INT32CPX);
		}
	}

  /* FLOATtoCFLOAT */
  GPUtype * GPUtype::FLOATtoCFLOAT() {
		// garbage collector
		MyGCObj<GPUtype> mgc;

		if (!this->isFloat()) {
			throw GPUexception(GPUmatError,
										ERROR_GPUTYPE_FLOATTOCOMPLEX_FLOAT);
		}

		if (this->isComplex()) {
			throw GPUexception(GPUmatError,
			    					ERROR_GPUTYPE_FLOATTOCOMPLEX_COMPLEX);
		}

  	if (this->myScalar.isset==1) {
      Complex tmp = {this->getScalarF(),0.0};
      GPUtype *qtmp1 = new GPUtype(tmp,this->getGPUmanager());
      return qtmp1;
  	} else {
  		GPUtype *qtmp1 = new GPUtype(*this, 1);
  		mgc.setPtr(qtmp1);
			qtmp1->setComplex();
			GPUopAllocVector(*qtmp1);
			GPUopZeros(*qtmp1,*qtmp1);
			// pack data
			GPUopRealImag(*qtmp1, *this, *this, 0, 1);
			//GPUopPackC2C(1, *this, *this, *qtmp1); //1 is for onlyreal
			mgc.remPtr(qtmp1);
			return qtmp1;
  	}
  }

  /* FLOATtoCFLOAT with imaginary part*/
	GPUtype * GPUtype::FLOATtoCFLOAT(GPUtype &im) {
		// garbage collector
		MyGCObj<GPUtype> mgc;

		if (!this->isFloat()) {
			throw GPUexception(GPUmatError,
										ERROR_GPUTYPE_FLOATTOCOMPLEX_FLOAT);
		}

		if (this->isComplex()) {
			throw GPUexception(GPUmatError,
										ERROR_GPUTYPE_FLOATTOCOMPLEX_COMPLEX);
		}

		// check size and number of elements
		if (this->getType()!= im.getType()) {
			throw GPUexception(GPUmatError,
		    					ERROR_GPUTYPE_FLOATTOCOMPLEX_SAMETYPE);
		}

		if (this->getNumel()!= im.getNumel()) {
			throw GPUexception(GPUmatError,
									ERROR_GPUTYPE_FLOATTOCOMPLEX_SAMENUMBER);
		}

		if (this->myScalar.isset==1) {
			Complex tmp = {this->getScalarF(),im.getScalarF()};
			GPUtype *qtmp1 = new GPUtype(tmp,this->getGPUmanager());
			return qtmp1;
		} else {
			GPUtype *qtmp1 = new GPUtype(*this, 1);
			mgc.setPtr(qtmp1);
			qtmp1->setComplex();
			GPUopAllocVector(*qtmp1);
			GPUopZeros(*qtmp1,*qtmp1);
			// pack data
			GPUopRealImag(*qtmp1, *this, im, 0, 0);
			//GPUopPackC2C(1, *this, *this, *qtmp1); //1 is for onlyreal
			mgc.remPtr(qtmp1);
			return qtmp1;
		}
	}


  /* clone
   * Return a copy
   * */
	GPUtype * GPUtype::clone() {
		// garbage collector
		MyGCObj<GPUtype> mgc;

		if (this->myScalar.isset==1) {
			GPUtype *qtmp1;
			if (this->mytype==gpuFLOAT) {
				float tmp = this->getScalarF();
				qtmp1 = new GPUtype(tmp,this->getGPUmanager());

			} if (this->mytype==gpuCFLOAT) {
				Complex tmp = this->getScalarCF();
				qtmp1 = new GPUtype(tmp,this->getGPUmanager());

			} if (this->mytype==gpuDOUBLE) {
				double tmp = this->getScalarD();
				qtmp1 = new GPUtype(tmp,this->getGPUmanager());

			} if (this->mytype==gpuCDOUBLE) {
				DoubleComplex tmp = this->getScalarCD();
				qtmp1 = new GPUtype(tmp,this->getGPUmanager());
			}
			return qtmp1;
		} else {
			GPUtype *qtmp1 = new GPUtype(*this, 1);
			mgc.setPtr(qtmp1);
			GPUopAllocVector(*qtmp1);
			GPUopCudaMemcpy(qtmp1->getGPUptr(), this->getGPUptr(), this->getMySize()
								* this->getNumel(), cudaMemcpyDeviceToDevice, this->getGPUmanager());
			mgc.remPtr(qtmp1);
			return qtmp1;
		}
	}

  /* FLOATtoDOUBLE */
 	GPUtype * GPUtype::FLOATtoDOUBLE() {
 		// garbage collector
   	MyGCObj<GPUtype> mgc;

 		if (this->myScalar.isset==1) {
 			GPUtype *qtmp1;
 			if (this->getType() == gpuFLOAT) {
 				double tmp = (double) this->getScalarF();
 				qtmp1 = new GPUtype(tmp,this->getGPUmanager());
 			} else if (this->getType() == gpuCFLOAT) {
 			  Complex src = this->getScalarCF();
 				DoubleComplex tmp = {(double)src.x, (double)src.y};
 				qtmp1 = new GPUtype(tmp,this->getGPUmanager());
 			}

 			return qtmp1;
 		} else {
       GPUtype *r;
 			r = new GPUtype(*this, 1); // do not copy GPUptr
 			mgc.setPtr(r);

 			// should change type
 			if (this->getType() == gpuFLOAT) {
 			  r->setType(gpuDOUBLE);
 			} else if (this->getType() == gpuCFLOAT) {
 			  r->setType(gpuCDOUBLE);
 			}

 			GPUopAllocVector(*r);

 			GPUmatResult_t status = GPUmatSuccess;
 			status = GPUopFloatToDouble(*this, *r);

 			mgc.remPtr(r);
 			return r;
 		}
 	}
 	/* DOUBLEtoFLOAT */
	GPUtype * GPUtype::DOUBLEtoFLOAT() {
		// garbage collector
		MyGCObj<GPUtype> mgc;

		if (this->myScalar.isset==1) {
			GPUtype *qtmp1;
			if (this->getType() == gpuDOUBLE) {
				float tmp = (float) this->getScalarD();
				qtmp1 = new GPUtype(tmp,this->getGPUmanager());
			} else if (this->getType() == gpuCDOUBLE) {
				DoubleComplex src = this->getScalarCD();
				Complex tmp = {(float)src.x, (float)src.y};
				qtmp1 = new GPUtype(tmp,this->getGPUmanager());
			}

			return qtmp1;
		} else {
			GPUtype *r;
			r = new GPUtype(*this, 1); // do not copy GPUptr
			mgc.setPtr(r);

			// should change type
			if (this->getType() == gpuDOUBLE) {
				r->setType(gpuFLOAT);
			} else if (this->getType() == gpuCDOUBLE) {
				r->setType(gpuCFLOAT);
			}

			GPUopAllocVector(*r);

			GPUmatResult_t status = GPUmatSuccess;
			status = GPUopDoubleToFloat(*this, *r);

			mgc.remPtr(r);
			return r;
		}
	}


  /* DOUBLEtoCDOUBLE */
	GPUtype * GPUtype::DOUBLEtoCDOUBLE() {
		// garbage collector
		MyGCObj<GPUtype> mgc;
		if (!this->isDouble()) {
			throw GPUexception(GPUmatError,
										ERROR_GPUTYPE_DOUBLETOCDOUBLE_DOUBLE);
		}

		if (this->isComplex()) {
			throw GPUexception(GPUmatError,
										ERROR_GPUTYPE_FLOATTOCOMPLEX_COMPLEX);
		}


		if (this->myScalar.isset==1) {
			DoubleComplex tmp = {this->getScalarD(),0.0};
			GPUtype *qtmp1 = new GPUtype(tmp,this->getGPUmanager());
			return qtmp1;
		} else {
			GPUtype *qtmp1 = new GPUtype(*this, 1);
			mgc.setPtr(qtmp1);
			qtmp1->setComplex();
			GPUopAllocVector(*qtmp1);
			GPUopZeros(*qtmp1,*qtmp1);
			// pack data
			GPUopRealImag(*qtmp1, *this, *this, 0, 1);
			//GPUopPackC2C(1, *this, *this, *qtmp1); //1 is for onlyreal
			mgc.remPtr(qtmp1);
			return qtmp1;
		}
	}


	/* DOUBLEtoCDOUBLE */
		GPUtype * GPUtype::DOUBLEtoCDOUBLE(GPUtype &im) {
		// garbage collector
		MyGCObj<GPUtype> mgc;
		if (!this->isDouble()) {
			throw GPUexception(GPUmatError,
										ERROR_GPUTYPE_DOUBLETOCDOUBLE_DOUBLE);
		}

		if (this->isComplex()) {
			throw GPUexception(GPUmatError,
										ERROR_GPUTYPE_FLOATTOCOMPLEX_COMPLEX);
		}

		// check size and number of elements
		if (this->getType()!= im.getType()) {
			throw GPUexception(GPUmatError,
									ERROR_GPUTYPE_FLOATTOCOMPLEX_SAMETYPE);
		}

		if (this->getNumel()!= im.getNumel()) {
			throw GPUexception(GPUmatError,
									ERROR_GPUTYPE_FLOATTOCOMPLEX_SAMENUMBER);
		}

		if (this->myScalar.isset==1) {
			DoubleComplex tmp = {this->getScalarD(),im.getScalarD()};
			GPUtype *qtmp1 = new GPUtype(tmp,this->getGPUmanager());
			return qtmp1;
		} else {
			GPUtype *qtmp1 = new GPUtype(*this, 1);
			mgc.setPtr(qtmp1);
			qtmp1->setComplex();
			GPUopAllocVector(*qtmp1);
			GPUopZeros(*qtmp1,*qtmp1);
			// pack data
			GPUopRealImag(*qtmp1, *this, im, 0, 0);
			//GPUopPackC2C(1, *this, *this, *qtmp1); //1 is for onlyreal
			mgc.remPtr(qtmp1);
			return qtmp1;
		}
	}





/*************** UTILS ***************/

/* dump */

void GPUtype::dump() {

	GPUmatResult_t status = GPUmatSuccess;
	if (this->getGPUptr() != NULL) {
		if (mytype == gpuFLOAT) {
			float *dest = (float*) Mymalloc(numel * mysize);
			status = GPUopCudaMemcpy(dest, this->getGPUptr(), mysize * numel,
					cudaMemcpyDeviceToHost, GPUman);

			for (int i = 0; i < numel; i++)
				myPrintf("%f \n", dest[i]);
      Myfree(dest);

		}

		if (mytype == gpuCFLOAT) {
			float *dest = (float*) Mymalloc(numel * mysize);
			status = GPUopCudaMemcpy(dest, this->getGPUptr(), mysize * numel,
					cudaMemcpyDeviceToHost, GPUman);

			for (int i = 0; i < numel * 2; i++)
				myPrintf("%f \n", dest[i]);
      
      Myfree(dest);

		}
	} else {
		myPrintf("Empty GPUtype \n");
	}

}

/* print */

void GPUtype::print() {
	int i;

	myPrintf("p.GPUptr    = %p \n", this->getGPUptr());
	myPrintf("p.trans     = %d \n", this->trans);
	myPrintf("p.numel     = %d \n", this->numel);
	myPrintf("p.mysize    = %d \n", this->mysize);

	myPrintf("p.size      = [ ");
	for (i = 0; i < this->ndims; i++)
		myPrintf(" %d", this->size[i]);
	myPrintf("]\n");

	myPrintf("p.ndims     = %d \n", this->ndims);
	myPrintf("p.stream    = %d \n", this->stream);
	myPrintf("p.iscompiled= %d \n", this->iscompiled);
	myPrintf("itsCounter  = %p \n", this->itsCounter);
	if (this->itsCounter)
		myPrintf("count       = %d \n", this->itsCounter->count);
	//mexPrintf("p.todelete  = %d \n", this->todelete);

}

/* print */

void GPUtype::printShort() {
	int i;
	myPrintf("this     = %p \n", this);
	myPrintf("p.GPUptr = %p \n", this->getGPUptr());
	myPrintf("p.ndims  = %d \n", this->ndims);

	myPrintf("p.size      = [ ");
	for (i = 0; i < this->ndims; i++)
		myPrintf(" %d", this->size[i]);
	myPrintf("]\n");

	myPrintf("p.stream    = %d \n", this->stream);
	//mexPrintf("p.todelete  = %d \n", this->todelete);

}

