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

gmGPUtype gmGPUopAbsDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopAbsDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopAcosDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopAcosDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopAcoshDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopAcoshDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopAndDrv(const gmGPUtype &p, const gmGPUtype &q) {
  GPUtype *r;
  try {
    r = GPUopAndDrv(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r,gmDeleteGPUtype);
}
gmGPUtype gmGPUopAsinDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopAsinDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopAsinhDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopAsinhDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopAtanDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopAtanDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopAtanhDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopAtanhDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopCeilDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopCeilDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopConjDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopConjDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopCosDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopCosDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopCoshDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopCoshDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopCtransposeDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopCtransposeDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopEqDrv(const gmGPUtype &p, const gmGPUtype &q) {
  GPUtype *r;
  try {
    r = GPUopEqDrv(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r,gmDeleteGPUtype);
}
gmGPUtype gmGPUopExpDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopExpDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopFloorDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopFloorDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopGeDrv(const gmGPUtype &p, const gmGPUtype &q) {
  GPUtype *r;
  try {
    r = GPUopGeDrv(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r,gmDeleteGPUtype);
}
void gmGPUopAbs(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopAbs(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopAcos(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopAcos(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopAcosh(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopAcosh(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopAnd(const gmGPUtype &p, const gmGPUtype &q, const gmGPUtype &r) {
  try {
    GPUopAnd(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr), *((GPUtype *)r.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopAsin(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopAsin(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopAsinh(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopAsinh(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopAtan(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopAtan(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopAtanh(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopAtanh(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopCeil(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopCeil(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopConj(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopConj(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopCos(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopCos(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopCosh(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopCosh(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopCtranspose(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopCtranspose(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopEq(const gmGPUtype &p, const gmGPUtype &q, const gmGPUtype &r) {
  try {
    GPUopEq(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr), *((GPUtype *)r.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopExp(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopExp(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopFloor(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopFloor(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopGe(const gmGPUtype &p, const gmGPUtype &q, const gmGPUtype &r) {
  try {
    GPUopGe(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr), *((GPUtype *)r.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopGt(const gmGPUtype &p, const gmGPUtype &q, const gmGPUtype &r) {
  try {
    GPUopGt(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr), *((GPUtype *)r.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopImag(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopImag(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopLdivide(const gmGPUtype &p, const gmGPUtype &q, const gmGPUtype &r) {
  try {
    GPUopLdivide(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr), *((GPUtype *)r.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopLe(const gmGPUtype &p, const gmGPUtype &q, const gmGPUtype &r) {
  try {
    GPUopLe(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr), *((GPUtype *)r.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopLog(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopLog(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopLog10(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopLog10(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopLog1p(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopLog1p(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopLog2(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopLog2(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopLt(const gmGPUtype &p, const gmGPUtype &q, const gmGPUtype &r) {
  try {
    GPUopLt(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr), *((GPUtype *)r.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopMinus(const gmGPUtype &p, const gmGPUtype &q, const gmGPUtype &r) {
  try {
    GPUopMinus(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr), *((GPUtype *)r.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopMtimes(const gmGPUtype &p, const gmGPUtype &q, const gmGPUtype &r) {
  try {
    GPUopMtimes(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr), *((GPUtype *)r.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopNe(const gmGPUtype &p, const gmGPUtype &q, const gmGPUtype &r) {
  try {
    GPUopNe(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr), *((GPUtype *)r.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopNot(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopNot(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopOr(const gmGPUtype &p, const gmGPUtype &q, const gmGPUtype &r) {
  try {
    GPUopOr(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr), *((GPUtype *)r.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopPlus(const gmGPUtype &p, const gmGPUtype &q, const gmGPUtype &r) {
  try {
    GPUopPlus(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr), *((GPUtype *)r.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopPower(const gmGPUtype &p, const gmGPUtype &q, const gmGPUtype &r) {
  try {
    GPUopPower(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr), *((GPUtype *)r.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopRdivide(const gmGPUtype &p, const gmGPUtype &q, const gmGPUtype &r) {
  try {
    GPUopRdivide(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr), *((GPUtype *)r.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopReal(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopReal(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopRound(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopRound(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopSin(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopSin(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopSinh(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopSinh(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopSqrt(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopSqrt(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopTan(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopTan(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopTanh(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopTanh(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopTimes(const gmGPUtype &p, const gmGPUtype &q, const gmGPUtype &r) {
  try {
    GPUopTimes(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr), *((GPUtype *)r.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopTranspose(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopTranspose(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
void gmGPUopUminus(const gmGPUtype &p, const gmGPUtype &q) {
  try {
    GPUopUminus(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
}
gmGPUtype gmGPUopGtDrv(const gmGPUtype &p, const gmGPUtype &q) {
  GPUtype *r;
  try {
    r = GPUopGtDrv(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r,gmDeleteGPUtype);
}
gmGPUtype gmGPUopImagDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopImagDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopLdivideDrv(const gmGPUtype &p, const gmGPUtype &q) {
  GPUtype *r;
  try {
    r = GPUopLdivideDrv(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r,gmDeleteGPUtype);
}
gmGPUtype gmGPUopLeDrv(const gmGPUtype &p, const gmGPUtype &q) {
  GPUtype *r;
  try {
    r = GPUopLeDrv(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r,gmDeleteGPUtype);
}
gmGPUtype gmGPUopLogDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopLogDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopLog10Drv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopLog10Drv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopLog1pDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopLog1pDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopLog2Drv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopLog2Drv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopLtDrv(const gmGPUtype &p, const gmGPUtype &q) {
  GPUtype *r;
  try {
    r = GPUopLtDrv(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r,gmDeleteGPUtype);
}
gmGPUtype gmGPUopMinusDrv(const gmGPUtype &p, const gmGPUtype &q) {
  GPUtype *r;
  try {
    r = GPUopMinusDrv(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r,gmDeleteGPUtype);
}
gmGPUtype gmGPUopMtimesDrv(const gmGPUtype &p, const gmGPUtype &q) {
  GPUtype *r;
  try {
    r = GPUopMtimesDrv(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r,gmDeleteGPUtype);
}
gmGPUtype gmGPUopNeDrv(const gmGPUtype &p, const gmGPUtype &q) {
  GPUtype *r;
  try {
    r = GPUopNeDrv(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r,gmDeleteGPUtype);
}
gmGPUtype gmGPUopNotDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopNotDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopOrDrv(const gmGPUtype &p, const gmGPUtype &q) {
  GPUtype *r;
  try {
    r = GPUopOrDrv(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r,gmDeleteGPUtype);
}
gmGPUtype gmGPUopPlusDrv(const gmGPUtype &p, const gmGPUtype &q) {
  GPUtype *r;
  try {
    r = GPUopPlusDrv(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r,gmDeleteGPUtype);
}
gmGPUtype gmGPUopPowerDrv(const gmGPUtype &p, const gmGPUtype &q) {
  GPUtype *r;
  try {
    r = GPUopPowerDrv(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r,gmDeleteGPUtype);
}
gmGPUtype gmGPUopRdivideDrv(const gmGPUtype &p, const gmGPUtype &q) {
  GPUtype *r;
  try {
    r = GPUopRdivideDrv(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r,gmDeleteGPUtype);
}
gmGPUtype gmGPUopRealDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopRealDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopRoundDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopRoundDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopSinDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopSinDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopSinhDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopSinhDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopSqrtDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopSqrtDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopTanDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopTanDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopTanhDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopTanhDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopTimesDrv(const gmGPUtype &p, const gmGPUtype &q) {
  GPUtype *r;
  try {
    r = GPUopTimesDrv(*((GPUtype *)p.ptrCounter->ptr), *((GPUtype *)q.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r,gmDeleteGPUtype);
}
gmGPUtype gmGPUopTransposeDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopTransposeDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
gmGPUtype gmGPUopUminusDrv(const gmGPUtype &p) {
  GPUtype *r;
  try {
    r = GPUopUminusDrv(*((GPUtype *)p.ptrCounter->ptr));
  } catch (GPUexception ex) {
    mexErrMsgTxt(ex.getError());
  }
  return gmGPUtype(r, gmDeleteGPUtype);
}
