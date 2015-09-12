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

#if !defined(GPUMATCOMPILERNUMERICS_HH_)
#define GPUMATCOMPILERNUMERICS_HH_
// Abs
#define GPUMAT_AbsDrv(OUT,IN)\
OUT = gm->numerics.AbsDrv(IN);

// Acos
#define GPUMAT_AcosDrv(OUT,IN)\
OUT = gm->numerics.AcosDrv(IN);

// Acosh
#define GPUMAT_AcoshDrv(OUT,IN)\
OUT = gm->numerics.AcoshDrv(IN);

// And
#define GPUMAT_AndDrv(OUT,IN1,IN2)\
OUT = gm->numerics.AndDrv(IN1,IN2);

// Asin
#define GPUMAT_AsinDrv(OUT,IN)\
OUT = gm->numerics.AsinDrv(IN);

// Asinh
#define GPUMAT_AsinhDrv(OUT,IN)\
OUT = gm->numerics.AsinhDrv(IN);

// Atan
#define GPUMAT_AtanDrv(OUT,IN)\
OUT = gm->numerics.AtanDrv(IN);

// Atanh
#define GPUMAT_AtanhDrv(OUT,IN)\
OUT = gm->numerics.AtanhDrv(IN);

// Ceil
#define GPUMAT_CeilDrv(OUT,IN)\
OUT = gm->numerics.CeilDrv(IN);

// Conj
#define GPUMAT_ConjDrv(OUT,IN)\
OUT = gm->numerics.ConjDrv(IN);

// Cos
#define GPUMAT_CosDrv(OUT,IN)\
OUT = gm->numerics.CosDrv(IN);

// Cosh
#define GPUMAT_CoshDrv(OUT,IN)\
OUT = gm->numerics.CoshDrv(IN);

// Ctranspose
#define GPUMAT_CtransposeDrv(OUT,IN)\
OUT = gm->numerics.CtransposeDrv(IN);

// Eq
#define GPUMAT_EqDrv(OUT,IN1,IN2)\
OUT = gm->numerics.EqDrv(IN1,IN2);

// Exp
#define GPUMAT_ExpDrv(OUT,IN)\
OUT = gm->numerics.ExpDrv(IN);

// Floor
#define GPUMAT_FloorDrv(OUT,IN)\
OUT = gm->numerics.FloorDrv(IN);

// Ge
#define GPUMAT_GeDrv(OUT,IN1,IN2)\
OUT = gm->numerics.GeDrv(IN1,IN2);

// Abs
#define GPUMAT_Abs(OUT,IN)\
gm->numerics.Abs(IN, OUT);

// Acos
#define GPUMAT_Acos(OUT,IN)\
gm->numerics.Acos(IN, OUT);

// Acosh
#define GPUMAT_Acosh(OUT,IN)\
gm->numerics.Acosh(IN, OUT);

// And
#define GPUMAT_And(OUT,IN1,IN2)\
gm->numerics.And(IN1, IN2, OUT);

// Asin
#define GPUMAT_Asin(OUT,IN)\
gm->numerics.Asin(IN, OUT);

// Asinh
#define GPUMAT_Asinh(OUT,IN)\
gm->numerics.Asinh(IN, OUT);

// Atan
#define GPUMAT_Atan(OUT,IN)\
gm->numerics.Atan(IN, OUT);

// Atanh
#define GPUMAT_Atanh(OUT,IN)\
gm->numerics.Atanh(IN, OUT);

// Ceil
#define GPUMAT_Ceil(OUT,IN)\
gm->numerics.Ceil(IN, OUT);

// Conj
#define GPUMAT_Conj(OUT,IN)\
gm->numerics.Conj(IN, OUT);

// Cos
#define GPUMAT_Cos(OUT,IN)\
gm->numerics.Cos(IN, OUT);

// Cosh
#define GPUMAT_Cosh(OUT,IN)\
gm->numerics.Cosh(IN, OUT);

// Ctranspose
#define GPUMAT_Ctranspose(OUT,IN)\
gm->numerics.Ctranspose(IN, OUT);

// Eq
#define GPUMAT_Eq(OUT,IN1,IN2)\
gm->numerics.Eq(IN1, IN2, OUT);

// Exp
#define GPUMAT_Exp(OUT,IN)\
gm->numerics.Exp(IN, OUT);

// Floor
#define GPUMAT_Floor(OUT,IN)\
gm->numerics.Floor(IN, OUT);

// Ge
#define GPUMAT_Ge(OUT,IN1,IN2)\
gm->numerics.Ge(IN1, IN2, OUT);

// Gt
#define GPUMAT_Gt(OUT,IN1,IN2)\
gm->numerics.Gt(IN1, IN2, OUT);

// Imag
#define GPUMAT_Imag(OUT,IN)\
gm->numerics.Imag(IN, OUT);

// Ldivide
#define GPUMAT_Ldivide(OUT,IN1,IN2)\
gm->numerics.Ldivide(IN1, IN2, OUT);

// Le
#define GPUMAT_Le(OUT,IN1,IN2)\
gm->numerics.Le(IN1, IN2, OUT);

// Log
#define GPUMAT_Log(OUT,IN)\
gm->numerics.Log(IN, OUT);

// Log10
#define GPUMAT_Log10(OUT,IN)\
gm->numerics.Log10(IN, OUT);

// Log1p
#define GPUMAT_Log1p(OUT,IN)\
gm->numerics.Log1p(IN, OUT);

// Log2
#define GPUMAT_Log2(OUT,IN)\
gm->numerics.Log2(IN, OUT);

// Lt
#define GPUMAT_Lt(OUT,IN1,IN2)\
gm->numerics.Lt(IN1, IN2, OUT);

// Minus
#define GPUMAT_Minus(OUT,IN1,IN2)\
gm->numerics.Minus(IN1, IN2, OUT);

// Mtimes
#define GPUMAT_Mtimes(OUT,IN1,IN2)\
gm->numerics.Mtimes(IN1, IN2, OUT);

// Ne
#define GPUMAT_Ne(OUT,IN1,IN2)\
gm->numerics.Ne(IN1, IN2, OUT);

// Not
#define GPUMAT_Not(OUT,IN)\
gm->numerics.Not(IN, OUT);

// Or
#define GPUMAT_Or(OUT,IN1,IN2)\
gm->numerics.Or(IN1, IN2, OUT);

// Plus
#define GPUMAT_Plus(OUT,IN1,IN2)\
gm->numerics.Plus(IN1, IN2, OUT);

// Power
#define GPUMAT_Power(OUT,IN1,IN2)\
gm->numerics.Power(IN1, IN2, OUT);

// Rdivide
#define GPUMAT_Rdivide(OUT,IN1,IN2)\
gm->numerics.Rdivide(IN1, IN2, OUT);

// Real
#define GPUMAT_Real(OUT,IN)\
gm->numerics.Real(IN, OUT);

// Round
#define GPUMAT_Round(OUT,IN)\
gm->numerics.Round(IN, OUT);

// Sin
#define GPUMAT_Sin(OUT,IN)\
gm->numerics.Sin(IN, OUT);

// Sinh
#define GPUMAT_Sinh(OUT,IN)\
gm->numerics.Sinh(IN, OUT);

// Sqrt
#define GPUMAT_Sqrt(OUT,IN)\
gm->numerics.Sqrt(IN, OUT);

// Tan
#define GPUMAT_Tan(OUT,IN)\
gm->numerics.Tan(IN, OUT);

// Tanh
#define GPUMAT_Tanh(OUT,IN)\
gm->numerics.Tanh(IN, OUT);

// Times
#define GPUMAT_Times(OUT,IN1,IN2)\
gm->numerics.Times(IN1, IN2, OUT);

// Transpose
#define GPUMAT_Transpose(OUT,IN)\
gm->numerics.Transpose(IN, OUT);

// Uminus
#define GPUMAT_Uminus(OUT,IN)\
gm->numerics.Uminus(IN, OUT);

// Gt
#define GPUMAT_GtDrv(OUT,IN1,IN2)\
OUT = gm->numerics.GtDrv(IN1,IN2);

// Imag
#define GPUMAT_ImagDrv(OUT,IN)\
OUT = gm->numerics.ImagDrv(IN);

// Ldivide
#define GPUMAT_LdivideDrv(OUT,IN1,IN2)\
OUT = gm->numerics.LdivideDrv(IN1,IN2);

// Le
#define GPUMAT_LeDrv(OUT,IN1,IN2)\
OUT = gm->numerics.LeDrv(IN1,IN2);

// Log
#define GPUMAT_LogDrv(OUT,IN)\
OUT = gm->numerics.LogDrv(IN);

// Log10
#define GPUMAT_Log10Drv(OUT,IN)\
OUT = gm->numerics.Log10Drv(IN);

// Log1p
#define GPUMAT_Log1pDrv(OUT,IN)\
OUT = gm->numerics.Log1pDrv(IN);

// Log2
#define GPUMAT_Log2Drv(OUT,IN)\
OUT = gm->numerics.Log2Drv(IN);

// Lt
#define GPUMAT_LtDrv(OUT,IN1,IN2)\
OUT = gm->numerics.LtDrv(IN1,IN2);

// Minus
#define GPUMAT_MinusDrv(OUT,IN1,IN2)\
OUT = gm->numerics.MinusDrv(IN1,IN2);

// Mtimes
#define GPUMAT_MtimesDrv(OUT,IN1,IN2)\
OUT = gm->numerics.MtimesDrv(IN1,IN2);

// Ne
#define GPUMAT_NeDrv(OUT,IN1,IN2)\
OUT = gm->numerics.NeDrv(IN1,IN2);

// Not
#define GPUMAT_NotDrv(OUT,IN)\
OUT = gm->numerics.NotDrv(IN);

// Or
#define GPUMAT_OrDrv(OUT,IN1,IN2)\
OUT = gm->numerics.OrDrv(IN1,IN2);

// Plus
#define GPUMAT_PlusDrv(OUT,IN1,IN2)\
OUT = gm->numerics.PlusDrv(IN1,IN2);

// Power
#define GPUMAT_PowerDrv(OUT,IN1,IN2)\
OUT = gm->numerics.PowerDrv(IN1,IN2);

// Rdivide
#define GPUMAT_RdivideDrv(OUT,IN1,IN2)\
OUT = gm->numerics.RdivideDrv(IN1,IN2);

// Real
#define GPUMAT_RealDrv(OUT,IN)\
OUT = gm->numerics.RealDrv(IN);

// Round
#define GPUMAT_RoundDrv(OUT,IN)\
OUT = gm->numerics.RoundDrv(IN);

// Sin
#define GPUMAT_SinDrv(OUT,IN)\
OUT = gm->numerics.SinDrv(IN);

// Sinh
#define GPUMAT_SinhDrv(OUT,IN)\
OUT = gm->numerics.SinhDrv(IN);

// Sqrt
#define GPUMAT_SqrtDrv(OUT,IN)\
OUT = gm->numerics.SqrtDrv(IN);

// Tan
#define GPUMAT_TanDrv(OUT,IN)\
OUT = gm->numerics.TanDrv(IN);

// Tanh
#define GPUMAT_TanhDrv(OUT,IN)\
OUT = gm->numerics.TanhDrv(IN);

// Times
#define GPUMAT_TimesDrv(OUT,IN1,IN2)\
OUT = gm->numerics.TimesDrv(IN1,IN2);

// Transpose
#define GPUMAT_TransposeDrv(OUT,IN)\
OUT = gm->numerics.TransposeDrv(IN);

// Uminus
#define GPUMAT_UminusDrv(OUT,IN)\
OUT = gm->numerics.UminusDrv(IN);

#endif
