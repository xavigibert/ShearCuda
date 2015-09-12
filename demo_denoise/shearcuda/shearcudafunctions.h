/*
     Copyright (C) 2013  University of Maryland
 
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

#include "cuda.h"
#include <string.h>
#include "datatypes.h"
#include "gputimes.h"
#include "shearcuda_global.h"

// This class encapsulates function pointers for all kernels used by
// the shearcuda module
class SHEARCUDASHARED_EXPORT ShearCudaFunctions
{
public:
    ShearCudaFunctions();
    
    bool LoadGpuFunctions( const CUmodule *drvmod );
    void setTimer(GpuTimes* gt) { m_timer = gt; }
    GpuTimes* timer() const { return m_timer; }

    bool supportsFloat;
    bool supportsDouble;

    // Function launchers
    void symExt(
            void* d_Output, int outputRows, int outputCols,
            void* d_Input, int inputRows, int inputCols,
            int topOffset, int leftOffset, gpuTYPE_t type_signal ) const;

    void atrousConvolution(
            void* d_Out, const void* d_Subsampled,
            void *d_Filter, int outW, int subW_padded,
            int filterLen, int level, gpuTYPE_t type_signal ) const;

    void atrousSubsample( void* d_Subsampled, const void* d_Signal,
            int subW, int signalW, int level, gpuTYPE_t type_signal ) const;

    void modulateConjAndNormalize(
            void* d_Dst, const void* d_Src, int fftW, int fftH, int numElem,
            int padding, gpuTYPE_t type_signal ) const;

    // Modulate and normalize a single image with a bank of filters
    void modulateAndNormalizeMany(
            void* d_Dst, const void* d_SrcData, const void* d_SrcKernel,
            int fftW, int fftH, int numElem, gpuTYPE_t type_signal ) const;

    // Add 2 vectors and save result on first one
    void addVector(
            void* d_Dst, const void* d_Src,
            int numElem, gpuTYPE_t type_signal ) const;

    // Add all components together (all components of d_Src are added toghether and result is saved into d_Dst)
    void sumVectors(
            void* d_Dst, const void* d_Src, int numElem,
            int numComponents, gpuTYPE_t type_signal ) const;

    // Multiply a vector by a scalar
    void mulMatrixByScalar(
            void* d_Dst, const void* d_Src, double scalar,
            int numElem, gpuTYPE_t type_signal ) const;

    void applyHardThreshold(
            void* d_Dst, const void* d_Src, int numElem,
            double th, gpuTYPE_t type_signal ) const;

    void applySoftThreshold(
        void* d_Dst, const void* d_Src, int numElem,
        double th, gpuTYPE_t type_signal ) const;

    // Redundant wavelet functions
    void mrdwtRow(
            const void* d_Signal, int numRows, int numCols,
            const void* d_Filter, int filterLen, int level,
            void* d_yl, void* d_yh, gpuTYPE_t type_signal ) const;

    void mrdwtCol(
            const void* d_Signal, int numRows, int numCols,
            const void* d_Filter, int filterLen, int level,
            void* d_yl, void* d_yh, gpuTYPE_t type_signal ) const;

    // Inverse redundant wavelet functions
    void mirdwtRow(
            const void* d_xinl, const void* d_xinh, int numRows, int numCols,
            const void* d_Filter, int filterLen, int level,
            void* d_xout, gpuTYPE_t type_signal ) const;

    void mirdwtCol(
            const void* d_xinl, const void* d_xinh, int numRows, int numCols,
            const void* d_Filter, int filterLen, int level,
            void* d_xout, gpuTYPE_t type_signal ) const;
    
    // Reduction functions
    // Calculate maximum absolute value within 256 samples (source can be either real
    // or complex, but result is always complex)
    void reduceMaxAbsVal256(
            void* d_Dst, const void* d_Src, int numElem, gpuTYPE_t type_signal ) const;

    // Reduce vector Lp norm
    void reduceNorm256(
            void* d_Dst, const void* d_Src, double p, int numElem, gpuTYPE_t type_signal ) const;

    // Reduce vector Lp norm of the difference between 2 vectors
    void reduceNormErr256(
            void* d_Dst, const void* d_SrcA, const void* d_SrcB, double p, int numElem, gpuTYPE_t type_signal ) const;

    // Reduce vector sum
    void reduceSum256(
            void* d_Dst, const void* d_Src, int numElem, gpuTYPE_t type_signal ) const;

    // Helper functions
    bool atrousConvolutionDevice(
            void* d_TempSubsampled, void* d_outArray, int O_SRowLength, int /* O_SColLength */,
            const void *d_SArray, int SRowLength, int SColLength,
            int nM, void* d_Filter, int filterLen, gpuTYPE_t type_signal ) const;

    // Convert real vector to complex
    void realToComplex(
            void* d_DstComplex, const void* d_SrcReal, int numElem, gpuTYPE_t type_signal ) const;

    // Convert complex vector to real
    void complexToReal(
            void* d_DstReal, const void* d_SrcComplex, int numElem, gpuTYPE_t type_signal ) const;

    // Zero padding
    void zeroPad(
            void* d_DstComplex, int dstDimX, int dstDimY, int dstDimZ, gpuTYPE_t type_dst,
            const void* d_SrcReal, int srcDimX, int srcDimY, int srcDimZ, gpuTYPE_t type_src ) const;

    void prepareMyerFilters(
            void *d_Filter, int size, int numDir, gpuTYPE_t data_type) const;

    // Image conversion
    bool convert(void* d_Dst, gpuTYPE_t type_dst, const void* d_Src, gpuTYPE_t type_src, int numElem) const;

    // Return sizes of different data types
    static int elemSize( gpuTYPE_t type_signal );

    // Return size of real part of different data types
    static int elemRealSize( gpuTYPE_t type_signal )
    {
        return
            (type_signal == gpuDOUBLE ? sizeof(double) :
             sizeof(float));
    }

private:
    CUfunction modulateConjAndNormalizeFC; // float complex
    CUfunction modulateConjAndNormalizeDC; // double complex
    CUfunction modulateAndNormalizeManyFC;
    CUfunction modulateAndNormalizeManyDC;
    CUfunction addVectorF;
    CUfunction addVectorD;
    CUfunction sumVectorsF;
    CUfunction sumVectorsD;
    CUfunction mulMatrixByScalarF;
    CUfunction mulMatrixByScalarD;
    CUfunction atrousSubsampleF;
    CUfunction atrousSubsampleD;
    CUfunction atrousUpsampleF;
    CUfunction atrousUpsampleD;
    CUfunction atrousConvolutionF;
    CUfunction atrousConvolutionD;
    CUfunction hardThresholdF;
    CUfunction hardThresholdD;
    CUfunction hardThresholdFC;
    CUfunction hardThresholdDC;
    CUfunction softThresholdF;
    CUfunction softThresholdD;
    CUfunction softThresholdFC;
    CUfunction softThresholdDC;
    CUfunction symExtF;
    CUfunction symExtD;
    CUfunction scalarVectorMulF;
    CUfunction scalarVectorMulD;
    CUfunction mrdwtRowF;
    CUfunction mrdwtRowD;
    CUfunction mrdwtColF;
    CUfunction mrdwtColD;
    CUfunction mirdwtRowF;
    CUfunction mirdwtRowD;
    CUfunction mirdwtColF;
    CUfunction mirdwtColD;
    CUfunction complexToRealF;
    CUfunction complexToRealD;
    CUfunction realToComplexF;
    CUfunction realToComplexD;
    CUfunction reduceMaxAbsVal256F;
    CUfunction reduceMaxAbsVal256D;
    CUfunction reduceMaxAbsVal256FC;
    CUfunction reduceMaxAbsVal256DC;
    CUfunction reduceNorm256F;
    CUfunction reduceNorm256D;
    CUfunction reduceNorm256FC;
    CUfunction reduceNorm256DC;
    CUfunction reduceNormErr256F;
    CUfunction reduceNormErr256D;
    CUfunction reduceNormErr256FC;
    CUfunction reduceNormErr256DC;
    CUfunction reduceSum256F;
    CUfunction reduceSum256D;
    CUfunction zeroPadF2FC;
    CUfunction zeroPadF2DC;
    CUfunction zeroPadD2FC;
    CUfunction zeroPadD2DC;
    CUfunction prepareMyerFiltersFC;
    CUfunction prepareMyerFiltersDC;
    CUfunction convert8UToF;
    CUfunction convert8UToD;
    CUfunction convertFTo8U;
    CUfunction convertDTo8U;

    GpuTimes* m_timer;
};

// Global instance of ShearCudaFunctions
extern ShearCudaFunctions g_ShearFunc;

//Round a / b to nearest higher integer value
int iDivUp(int a, int b);