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

/*
* GPUmatNumerics struct
* Interface to different GPUmat Numerics functions
*/
typedef struct  {
/// AbsDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*AbsDrv)(const GPUtype & p);
/// AcosDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*AcosDrv)(const GPUtype & p);
/// AcoshDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*AcoshDrv)(const GPUtype & p);
/// AndDrv function
/**
* @param[in]  p 1st GPUtype input variable.
* @param[in]  q 2nd GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*AndDrv)(const GPUtype & p, const GPUtype & q);
/// AsinDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*AsinDrv)(const GPUtype & p);
/// AsinhDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*AsinhDrv)(const GPUtype & p);
/// AtanDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*AtanDrv)(const GPUtype & p);
/// AtanhDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*AtanhDrv)(const GPUtype & p);
/// CeilDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*CeilDrv)(const GPUtype & p);
/// ConjDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*ConjDrv)(const GPUtype & p);
/// CosDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*CosDrv)(const GPUtype & p);
/// CoshDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*CoshDrv)(const GPUtype & p);
/// CtransposeDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*CtransposeDrv)(const GPUtype & p);
/// EqDrv function
/**
* @param[in]  p 1st GPUtype input variable.
* @param[in]  q 2nd GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*EqDrv)(const GPUtype & p, const GPUtype & q);
/// ExpDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*ExpDrv)(const GPUtype & p);
/// FloorDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*FloorDrv)(const GPUtype & p);
/// GeDrv function
/**
* @param[in]  p 1st GPUtype input variable.
* @param[in]  q 2nd GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*GeDrv)(const GPUtype & p, const GPUtype & q);
/// Abs function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Abs)(const GPUtype & p, const GPUtype & q);
/// Acos function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Acos)(const GPUtype & p, const GPUtype & q);
/// Acosh function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Acosh)(const GPUtype & p, const GPUtype & q);
/// And function
/**
* @param[in] p 1st GPUtype input variable.
* @param[in] q 2nd GPUtype input variable.
* @param[out] r GPUtype used to store the result.
* @return void
*/
void (*And)(const GPUtype & p, const GPUtype & q, const GPUtype & r);
/// Asin function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Asin)(const GPUtype & p, const GPUtype & q);
/// Asinh function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Asinh)(const GPUtype & p, const GPUtype & q);
/// Atan function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Atan)(const GPUtype & p, const GPUtype & q);
/// Atanh function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Atanh)(const GPUtype & p, const GPUtype & q);
/// Ceil function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Ceil)(const GPUtype & p, const GPUtype & q);
/// Conj function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Conj)(const GPUtype & p, const GPUtype & q);
/// Cos function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Cos)(const GPUtype & p, const GPUtype & q);
/// Cosh function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Cosh)(const GPUtype & p, const GPUtype & q);
/// Ctranspose function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Ctranspose)(const GPUtype & p, const GPUtype & q);
/// Eq function
/**
* @param[in] p 1st GPUtype input variable.
* @param[in] q 2nd GPUtype input variable.
* @param[out] r GPUtype used to store the result.
* @return void
*/
void (*Eq)(const GPUtype & p, const GPUtype & q, const GPUtype & r);
/// Exp function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Exp)(const GPUtype & p, const GPUtype & q);
/// Floor function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Floor)(const GPUtype & p, const GPUtype & q);
/// Ge function
/**
* @param[in] p 1st GPUtype input variable.
* @param[in] q 2nd GPUtype input variable.
* @param[out] r GPUtype used to store the result.
* @return void
*/
void (*Ge)(const GPUtype & p, const GPUtype & q, const GPUtype & r);
/// Gt function
/**
* @param[in] p 1st GPUtype input variable.
* @param[in] q 2nd GPUtype input variable.
* @param[out] r GPUtype used to store the result.
* @return void
*/
void (*Gt)(const GPUtype & p, const GPUtype & q, const GPUtype & r);
/// Imag function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Imag)(const GPUtype & p, const GPUtype & q);
/// Ldivide function
/**
* @param[in] p 1st GPUtype input variable.
* @param[in] q 2nd GPUtype input variable.
* @param[out] r GPUtype used to store the result.
* @return void
*/
void (*Ldivide)(const GPUtype & p, const GPUtype & q, const GPUtype & r);
/// Le function
/**
* @param[in] p 1st GPUtype input variable.
* @param[in] q 2nd GPUtype input variable.
* @param[out] r GPUtype used to store the result.
* @return void
*/
void (*Le)(const GPUtype & p, const GPUtype & q, const GPUtype & r);
/// Log function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Log)(const GPUtype & p, const GPUtype & q);
/// Log10 function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Log10)(const GPUtype & p, const GPUtype & q);
/// Log1p function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Log1p)(const GPUtype & p, const GPUtype & q);
/// Log2 function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Log2)(const GPUtype & p, const GPUtype & q);
/// Lt function
/**
* @param[in] p 1st GPUtype input variable.
* @param[in] q 2nd GPUtype input variable.
* @param[out] r GPUtype used to store the result.
* @return void
*/
void (*Lt)(const GPUtype & p, const GPUtype & q, const GPUtype & r);
/// Minus function
/**
* @param[in] p 1st GPUtype input variable.
* @param[in] q 2nd GPUtype input variable.
* @param[out] r GPUtype used to store the result.
* @return void
*/
void (*Minus)(const GPUtype & p, const GPUtype & q, const GPUtype & r);
/// Mtimes function
/**
* @param[in] p 1st GPUtype input variable.
* @param[in] q 2nd GPUtype input variable.
* @param[out] r GPUtype used to store the result.
* @return void
*/
void (*Mtimes)(const GPUtype & p, const GPUtype & q, const GPUtype & r);
/// Ne function
/**
* @param[in] p 1st GPUtype input variable.
* @param[in] q 2nd GPUtype input variable.
* @param[out] r GPUtype used to store the result.
* @return void
*/
void (*Ne)(const GPUtype & p, const GPUtype & q, const GPUtype & r);
/// Not function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Not)(const GPUtype & p, const GPUtype & q);
/// Or function
/**
* @param[in] p 1st GPUtype input variable.
* @param[in] q 2nd GPUtype input variable.
* @param[out] r GPUtype used to store the result.
* @return void
*/
void (*Or)(const GPUtype & p, const GPUtype & q, const GPUtype & r);
/// Plus function
/**
* @param[in] p 1st GPUtype input variable.
* @param[in] q 2nd GPUtype input variable.
* @param[out] r GPUtype used to store the result.
* @return void
*/
void (*Plus)(const GPUtype & p, const GPUtype & q, const GPUtype & r);
/// Power function
/**
* @param[in] p 1st GPUtype input variable.
* @param[in] q 2nd GPUtype input variable.
* @param[out] r GPUtype used to store the result.
* @return void
*/
void (*Power)(const GPUtype & p, const GPUtype & q, const GPUtype & r);
/// Rdivide function
/**
* @param[in] p 1st GPUtype input variable.
* @param[in] q 2nd GPUtype input variable.
* @param[out] r GPUtype used to store the result.
* @return void
*/
void (*Rdivide)(const GPUtype & p, const GPUtype & q, const GPUtype & r);
/// Real function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Real)(const GPUtype & p, const GPUtype & q);
/// Round function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Round)(const GPUtype & p, const GPUtype & q);
/// Sin function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Sin)(const GPUtype & p, const GPUtype & q);
/// Sinh function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Sinh)(const GPUtype & p, const GPUtype & q);
/// Sqrt function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Sqrt)(const GPUtype & p, const GPUtype & q);
/// Tan function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Tan)(const GPUtype & p, const GPUtype & q);
/// Tanh function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Tanh)(const GPUtype & p, const GPUtype & q);
/// Times function
/**
* @param[in] p 1st GPUtype input variable.
* @param[in] q 2nd GPUtype input variable.
* @param[out] r GPUtype used to store the result.
* @return void
*/
void (*Times)(const GPUtype & p, const GPUtype & q, const GPUtype & r);
/// Transpose function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Transpose)(const GPUtype & p, const GPUtype & q);
/// Uminus function
/**
* @param[in] p GPUtype input variable.
* @param[out] q GPUtype used to store the result.
* @return void
*/
void (*Uminus)(const GPUtype & p, const GPUtype & q);
/// GtDrv function
/**
* @param[in]  p 1st GPUtype input variable.
* @param[in]  q 2nd GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*GtDrv)(const GPUtype & p, const GPUtype & q);
/// ImagDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*ImagDrv)(const GPUtype & p);
/// LdivideDrv function
/**
* @param[in]  p 1st GPUtype input variable.
* @param[in]  q 2nd GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*LdivideDrv)(const GPUtype & p, const GPUtype & q);
/// LeDrv function
/**
* @param[in]  p 1st GPUtype input variable.
* @param[in]  q 2nd GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*LeDrv)(const GPUtype & p, const GPUtype & q);
/// LogDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*LogDrv)(const GPUtype & p);
/// Log10Drv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*Log10Drv)(const GPUtype & p);
/// Log1pDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*Log1pDrv)(const GPUtype & p);
/// Log2Drv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*Log2Drv)(const GPUtype & p);
/// LtDrv function
/**
* @param[in]  p 1st GPUtype input variable.
* @param[in]  q 2nd GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*LtDrv)(const GPUtype & p, const GPUtype & q);
/// MinusDrv function
/**
* @param[in]  p 1st GPUtype input variable.
* @param[in]  q 2nd GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*MinusDrv)(const GPUtype & p, const GPUtype & q);
/// MtimesDrv function
/**
* @param[in]  p 1st GPUtype input variable.
* @param[in]  q 2nd GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*MtimesDrv)(const GPUtype & p, const GPUtype & q);
/// NeDrv function
/**
* @param[in]  p 1st GPUtype input variable.
* @param[in]  q 2nd GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*NeDrv)(const GPUtype & p, const GPUtype & q);
/// NotDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*NotDrv)(const GPUtype & p);
/// OrDrv function
/**
* @param[in]  p 1st GPUtype input variable.
* @param[in]  q 2nd GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*OrDrv)(const GPUtype & p, const GPUtype & q);
/// PlusDrv function
/**
* @param[in]  p 1st GPUtype input variable.
* @param[in]  q 2nd GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*PlusDrv)(const GPUtype & p, const GPUtype & q);
/// PowerDrv function
/**
* @param[in]  p 1st GPUtype input variable.
* @param[in]  q 2nd GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*PowerDrv)(const GPUtype & p, const GPUtype & q);
/// RdivideDrv function
/**
* @param[in]  p 1st GPUtype input variable.
* @param[in]  q 2nd GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*RdivideDrv)(const GPUtype & p, const GPUtype & q);
/// RealDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*RealDrv)(const GPUtype & p);
/// RoundDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*RoundDrv)(const GPUtype & p);
/// SinDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*SinDrv)(const GPUtype & p);
/// SinhDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*SinhDrv)(const GPUtype & p);
/// SqrtDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*SqrtDrv)(const GPUtype & p);
/// TanDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*TanDrv)(const GPUtype & p);
/// TanhDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*TanhDrv)(const GPUtype & p);
/// TimesDrv function
/**
* @param[in]  p 1st GPUtype input variable.
* @param[in]  q 2nd GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*TimesDrv)(const GPUtype & p, const GPUtype & q);
/// TransposeDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*TransposeDrv)(const GPUtype & p);
/// UminusDrv function
/**
* @param[in] p GPUtype input variable.
* @return GPUtype pointer
*/
GPUtype (*UminusDrv)(const GPUtype & p);
} GPUmatNumerics;
