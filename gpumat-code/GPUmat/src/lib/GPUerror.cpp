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

#define GPUerror_CPP
#include "GPUerror.hh"

static int freecounter = 0;
void *Mymalloc(size_t size, MyGC *m) {
#ifndef DEBUG
	void *p = malloc(size);
	if (m != 0) {
		m->setPtr(p);
	}
	return p;
#else
	void *p = malloc(size);
	if (m != 0) {
		m->setPtr(p);
	}
	FILE *fout=fopen("malloc.dat","a+");
	fprintf(fout,"%p\n",p);
	fclose(fout);
	return p;
#endif

}

void Myfree(void *p, MyGC *m) {
#ifndef DEBUG
	if (m != 0) {
		m->remPtr(p);
	}
	free(p);
#else
	freecounter++;
	if (freecounter==11475) {
	  int a = 10;
	}
  if (p==NULL) {
    FILE *fout=fopen("freeNULL.dat","a+");
    fprintf(fout,"%p\n",p);
    fclose(fout);
  }

	if (m != 0) {
		m->remPtr(p);
	}
	free(p);
	FILE *fout=fopen("free.dat","a+");
	fprintf(fout,"%p\n",p);
	fclose(fout);


#endif

}



// another GC
MyGC::MyGC() :
	ptr(NULL), mysize(10), idx(0) {

	ptr = (void **) Mymalloc(mysize * sizeof(void *));

	for (int i = 0; i < mysize; i++)
		ptr[i] = NULL;
}

void MyGC::setPtr(void *p) {
	if (idx == mysize) {
		// increase size
		int newmysize = mysize + 10;
		void **tmpptr = (void **) Mymalloc(newmysize * sizeof(void *));
		for (int i = 0; i < newmysize; i++)
			tmpptr[i] = NULL;

		memcpy(tmpptr, ptr, mysize * sizeof(void *));
		Myfree(ptr);
		mysize = newmysize;
		ptr = tmpptr;
	}
	ptr[idx] = p;
	idx++;
}

void MyGC::remPtr(void *p) {
	for (int i = mysize - 1; i >= 0; i--) {
		if (ptr[i] == p) {
			ptr[i] = NULL;
			break;
		}
	}
}

MyGC::~MyGC() {
	for (int i = 0; i < mysize; i++) {
		if (ptr[i] != NULL) {
			Myfree(ptr[i]);
		}
	}
	Myfree(ptr);

}

