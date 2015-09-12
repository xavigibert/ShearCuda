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

// Abs
gm.numerics.AbsDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("AbsDrv");

// Acos
gm.numerics.AcosDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("AcosDrv");

// Acosh
gm.numerics.AcoshDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("AcoshDrv");

// And
gm.numerics.AndDrv = (GPUtype (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("AndDrv");

// Asin
gm.numerics.AsinDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("AsinDrv");

// Asinh
gm.numerics.AsinhDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("AsinhDrv");

// Atan
gm.numerics.AtanDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("AtanDrv");

// Atanh
gm.numerics.AtanhDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("AtanhDrv");

// Ceil
gm.numerics.CeilDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("CeilDrv");

// Conj
gm.numerics.ConjDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("ConjDrv");

// Cos
gm.numerics.CosDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("CosDrv");

// Cosh
gm.numerics.CoshDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("CoshDrv");

// Ctranspose
gm.numerics.CtransposeDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("CtransposeDrv");

// Eq
gm.numerics.EqDrv = (GPUtype (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("EqDrv");

// Exp
gm.numerics.ExpDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("ExpDrv");

// Floor
gm.numerics.FloorDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("FloorDrv");

// Ge
gm.numerics.GeDrv = (GPUtype (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("GeDrv");

// Abs
gm.numerics.Abs    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Abs");

// Acos
gm.numerics.Acos    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Acos");

// Acosh
gm.numerics.Acosh    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Acosh");

// And
gm.numerics.And    = (void (*)(const GPUtype &, const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("And");

// Asin
gm.numerics.Asin    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Asin");

// Asinh
gm.numerics.Asinh    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Asinh");

// Atan
gm.numerics.Atan    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Atan");

// Atanh
gm.numerics.Atanh    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Atanh");

// Ceil
gm.numerics.Ceil    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Ceil");

// Conj
gm.numerics.Conj    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Conj");

// Cos
gm.numerics.Cos    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Cos");

// Cosh
gm.numerics.Cosh    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Cosh");

// Ctranspose
gm.numerics.Ctranspose    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Ctranspose");

// Eq
gm.numerics.Eq    = (void (*)(const GPUtype &, const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Eq");

// Exp
gm.numerics.Exp    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Exp");

// Floor
gm.numerics.Floor    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Floor");

// Ge
gm.numerics.Ge    = (void (*)(const GPUtype &, const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Ge");

// Gt
gm.numerics.Gt    = (void (*)(const GPUtype &, const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Gt");

// Imag
gm.numerics.Imag    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Imag");

// Ldivide
gm.numerics.Ldivide    = (void (*)(const GPUtype &, const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Ldivide");

// Le
gm.numerics.Le    = (void (*)(const GPUtype &, const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Le");

// Log
gm.numerics.Log    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Log");

// Log10
gm.numerics.Log10    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Log10");

// Log1p
gm.numerics.Log1p    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Log1p");

// Log2
gm.numerics.Log2    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Log2");

// Lt
gm.numerics.Lt    = (void (*)(const GPUtype &, const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Lt");

// Minus
gm.numerics.Minus    = (void (*)(const GPUtype &, const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Minus");

// Mtimes
gm.numerics.Mtimes    = (void (*)(const GPUtype &, const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Mtimes");

// Ne
gm.numerics.Ne    = (void (*)(const GPUtype &, const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Ne");

// Not
gm.numerics.Not    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Not");

// Or
gm.numerics.Or    = (void (*)(const GPUtype &, const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Or");

// Plus
gm.numerics.Plus    = (void (*)(const GPUtype &, const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Plus");

// Power
gm.numerics.Power    = (void (*)(const GPUtype &, const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Power");

// Rdivide
gm.numerics.Rdivide    = (void (*)(const GPUtype &, const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Rdivide");

// Real
gm.numerics.Real    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Real");

// Round
gm.numerics.Round    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Round");

// Sin
gm.numerics.Sin    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Sin");

// Sinh
gm.numerics.Sinh    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Sinh");

// Sqrt
gm.numerics.Sqrt    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Sqrt");

// Tan
gm.numerics.Tan    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Tan");

// Tanh
gm.numerics.Tanh    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Tanh");

// Times
gm.numerics.Times    = (void (*)(const GPUtype &, const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Times");

// Transpose
gm.numerics.Transpose    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Transpose");

// Uminus
gm.numerics.Uminus    = (void (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("Uminus");

// Gt
gm.numerics.GtDrv = (GPUtype (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("GtDrv");

// Imag
gm.numerics.ImagDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("ImagDrv");

// Ldivide
gm.numerics.LdivideDrv = (GPUtype (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("LdivideDrv");

// Le
gm.numerics.LeDrv = (GPUtype (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("LeDrv");

// Log
gm.numerics.LogDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("LogDrv");

// Log10
gm.numerics.Log10Drv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("Log10Drv");

// Log1p
gm.numerics.Log1pDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("Log1pDrv");

// Log2
gm.numerics.Log2Drv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("Log2Drv");

// Lt
gm.numerics.LtDrv = (GPUtype (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("LtDrv");

// Minus
gm.numerics.MinusDrv = (GPUtype (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("MinusDrv");

// Mtimes
gm.numerics.MtimesDrv = (GPUtype (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("MtimesDrv");

// Ne
gm.numerics.NeDrv = (GPUtype (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("NeDrv");

// Not
gm.numerics.NotDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("NotDrv");

// Or
gm.numerics.OrDrv = (GPUtype (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("OrDrv");

// Plus
gm.numerics.PlusDrv = (GPUtype (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("PlusDrv");

// Power
gm.numerics.PowerDrv = (GPUtype (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("PowerDrv");

// Rdivide
gm.numerics.RdivideDrv = (GPUtype (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("RdivideDrv");

// Real
gm.numerics.RealDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("RealDrv");

// Round
gm.numerics.RoundDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("RoundDrv");

// Sin
gm.numerics.SinDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("SinDrv");

// Sinh
gm.numerics.SinhDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("SinhDrv");

// Sqrt
gm.numerics.SqrtDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("SqrtDrv");

// Tan
gm.numerics.TanDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("TanDrv");

// Tanh
gm.numerics.TanhDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("TanhDrv");

// Times
gm.numerics.TimesDrv = (GPUtype (*)(const GPUtype &, const GPUtype &)) gmat->fun.getFunctionByName("TimesDrv");

// Transpose
gm.numerics.TransposeDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("TransposeDrv");

// Uminus
gm.numerics.UminusDrv = (GPUtype (*)(const GPUtype &)) gmat->fun.getFunctionByName("UminusDrv");

