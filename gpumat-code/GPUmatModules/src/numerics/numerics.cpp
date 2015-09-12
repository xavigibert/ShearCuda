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
#include <stdarg.h>

#include "mex.h"

#ifdef UNIX
#include <stdint.h>
#endif

// CUDA
#include "cuda.h"
#include "cuda_runtime.h"

#include "GPUmat.hh"
#include "numerics.hh"

/*************************************************
 * NUMERICS
 *************************************************/


/*************************************************
 * UTIL
 *************************************************/

void parseMxRange(int rhsdim, const mxArray *prhs, Range **outrg, GPUmat *gm,
    MyGCObj<Range> &mygc1) {

  Range *rg = new Range(0); // dummy start
  mygc1.setPtr(rg);
  Range * tmprg = rg; // used in loop

  // Index passed from Matlab can have more than 1 level,
  // but this function supports only 1 level
  if (mxGetNumberOfElements(prhs) > 1)
    mexErrMsgTxt("Two many indexes. Only one level is supported");

  // check Matlab documentation for subsref
  mxArray *field = mxGetField(prhs, 0, "type");

  // we manage also the condition where the range has more dimensions
  // than the RHS vector, but set to '1'
  // For example:
  // A = GPUsingle(rand(10,10,10));
  // A(1,1,1,1)
  // RHS dimensions are stored in rhsdim

  if (mxIsChar(field)) {
    char buffer[10];
    mxGetString(field, buffer, 10);
    if (strcmp(buffer, "()") == 0) {
      mxArray * subs = mxGetField(prhs, 0, "subs");
      int subsdim = mxGetNumberOfElements(subs);

      // subs is a cell array passed by Matlab to subsref
      // For example:
      // The syntax A(1:2,:) calls subsref(A,S) where S is a 1-by-1
      // structure with S.type='()' and S.subs={1:2,':'}.
      // A colon used as a subscript is passed as the string ':'.

      // subsdim = 0 means something like A(). In this case a copy
      // of the RHS must be returned, and we return an empty range
      if (subsdim!=0) {
        for (int i = 0; i < subsdim; i++) {
          mxArray *mx = mxGetCell(subs, i);

          if (mxGetClassID(mx) == mxCHAR_CLASS) {
            // we assume index is ':'
            if ((i>(rhsdim-1))) {
              // skip.
              // For example:
              // A = GPUsingle(rand(10,10,10));
              // A(1,1,1,:)
              // We do not generate the last index ':'
            } else {
              tmprg->next = new Range(1,1,END); //Matlab indexes
              // register Range in GC
              mygc1.setPtr(tmprg->next);
            }
          } else if (mxGetClassID(mx) == mxDOUBLE_CLASS) {

            // Do the following:
            // 1) Check if indexes are consecutive by scanning the array
            // 2) If consecutive, store only start/last index and stride
            // 3) If not consecutive store the entire array
            // 4) If i>rhsdim and index==1 do not create the Range

            int n = (int) mxGetNumberOfElements(mx);
            double *tmpidx = mxGetPr(mx);


            int delta = 0;
            if (n > 1) {
              delta = (int) (tmpidx[1] - tmpidx[0]);
            }

            int consecutive = 1;
            for (int jj = 1; jj < n; jj++) {
              int newdelta = (int) (tmpidx[jj] - tmpidx[jj - 1]);
              consecutive = (newdelta == delta) && consecutive &&(delta!=0);
              delta = newdelta;
            }

            if (consecutive) {
              if ((i>(rhsdim-1)) && (tmpidx[0]==1) && (n==1)) {
                // skip.
                // For example:
                // A = GPUsingle(rand(10,10,10));
                // A(1,1,1,1)
                // We do not generate the last index '1'
              } else {
                //mexPrintf("Cons\n");
                tmprg->next = new Range((int) tmpidx[0], delta, (int) tmpidx[n-1]);
                // register Range in GC
                mygc1.setPtr(tmprg->next);
              }
            } else {
              tmprg->next = new Range(n - 1,  tmpidx);
              // register Range in GC
              mygc1.setPtr(tmprg->next);
            }

          } else if ((mxIsClass(mx, "GPUdouble"))||(mxIsClass(mx, "GPUsingle"))) {
            GPUtype IN = gm->gputype.getGPUtype(mx);
            tmprg->next = new Range(IN);
            // register Range in GC
            mygc1.setPtr(tmprg->next);

          } else {
            mexErrMsgTxt("Unsupported index. Supported types are: ':', double, GPUtype.");
          }

          // we set to next only if tmprg->!=NULL
          // some conditions above may have prevented the creation of
          // tmprg->next
          if (tmprg->next!=NULL)
            tmprg = tmprg->next;
        }
      }

    } else if (strcmp(buffer, ".") == 0) {
      mexErrMsgTxt("Field name indexing not supported by GPUtype objects.");
    } else if (strcmp(buffer, "{}") == 0) {
      mexErrMsgTxt("Cell array indexing not supported by GPUtype objects.");
    }

  } else {
    mexErrMsgTxt("Unexpected input argument.");
  }


  *outrg = rg->next; //first is dummy
}


void parseRange(int nrhs, const mxArray *prhs[], Range **outrg,
    MyGCObj<Range> &mygc1) {

  Range *rg = new Range(0); // dummy start
  mygc1.setPtr(rg);
  Range * tmprg = rg; // used in loop

  // Loop through remaining parameters and construct the range
  for (int i = 0; i < nrhs; i++) {
    // I we find a Matlab array we generate a Range of type [inf:stride:sup]
    // If we find a Matlab cell we generate a Range of type
    // If we find a char, we assume it is ':' and generate 0:1:end

    if (mxGetClassID(prhs[i]) == mxDOUBLE_CLASS) {
      // prhs[i] can have 1 element or 3 elements
      int n = mxGetNumberOfElements(prhs[i]);
      if (n == 1) {
        // single element range
        tmprg->next = new Range((int) mxGetScalar(prhs[i]));
        // register Range in GC
        mygc1.setPtr(tmprg->next);
      } else if (n == 3) {
        // [inf:stride:sup] range
        double *idx = mxGetPr(prhs[i]);
        tmprg->next = new Range((int) idx[0], (int) idx[1], (int) idx[2]);
        // register Range in GC
        mygc1.setPtr(tmprg->next);
      } else {
        mexErrMsgTxt("Range array should have 1 or 3 elements.");
      }

    } else if (mxGetClassID(prhs[i]) == mxCHAR_CLASS) {
      // we assume index is ':'
      tmprg->next = new Range(1,1,END); //Matlab indexes
      // register Range in GC
      mygc1.setPtr(tmprg->next);

    } else if (mxGetClassID(prhs[i]) == mxCELL_CLASS) {
      // In this case I assume an array of indexes


      int nel = mxGetNumberOfElements(prhs[i]);
      if (nel > 1) {
        mexErrMsgTxt("Expected only 1 array in cell array range.");
      }
      mxArray *mx = mxGetCell(prhs[i], 0);
      // we expect an array of indexes

      if (mxGetClassID(mx) == mxINT32_CLASS) {
        // populate range
        int n = mxGetNumberOfElements(mx);
        tmprg->next = new Range(n - 1, (int *) mxGetPr(mx));
        // register Range in GC
        mygc1.setPtr(tmprg->next);
      } else if (mxGetClassID(mx) == mxSINGLE_CLASS) {
        // populate range
        int n = mxGetNumberOfElements(mx);
        tmprg->next = new Range(n - 1, (float *) mxGetPr(mx));
        // register Range in GC
        mygc1.setPtr(tmprg->next);

      } else if (mxGetClassID(mx) == mxDOUBLE_CLASS) {
        // populate range
        int n = mxGetNumberOfElements(mx);
        tmprg->next = new Range(n - 1, (double *) mxGetPr(mx));
        // register Range in GC
        mygc1.setPtr(tmprg->next);

      } else {
        mexErrMsgTxt("Cell array range must be SINGLE, DOUBLE or INT32 array.");
      }

    } else {
      mexErrMsgTxt("Unexpected range.");
    }
    tmprg = tmprg->next;
  }
  *outrg = rg->next; //first is dummy
}
