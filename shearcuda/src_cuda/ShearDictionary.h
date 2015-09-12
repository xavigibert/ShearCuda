#pragma once

#include <vector>

#include "mex.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <string.h>
#include "GPUmat.hh"
#include "MexUtil.h"

class ShearDictionary
{
public:
    // Create empty object
    ShearDictionary();

    ~ShearDictionary(void);

    // Get buffer associated with a particular scale
    void* getBuffer(int scale) const;

    // Getters
    gpuTYPE_t dataType() const { return m_type; }
    int numScales() const { return m_anNumDirections.size(); }
    int filterLen() const { return m_nFilterLen; }
    int numDirections(int scale) const { return m_anNumDirections[scale]; }
    bool hasData() { return m_anNumDirections.size()>0; }
    cufftHandle fftPlanOne() const { return m_fftPlanOne; }
    cufftHandle fftPlanMany(int scale) const { return m_fftPlanMany[scale]; }
    cufftHandle ifftPlanMany(int scale) const { return m_ifftPlanMany[scale]; }

    // Return maximum number of directions across all scales
    int maxDirections() const;

    // Create object from MATLAB cell
    bool loadFromMx(const mxArray* pCell, GPUmat* gm);

protected:
    int m_nFilterLen;       // Filter width and height
    std::vector<int> m_anNumDirections;     // Number of directions per scale
    gpuTYPE_t m_type;       // Type of data
    std::vector<GPUtype> m_GPUtypeData;     // GPUtype objects containing the data
    cufftHandle* m_fftPlanMany;     // Plan for FFT (R2C) transformation of all directions
    cufftHandle* m_ifftPlanMany;    // Plan for IFFT (C2R) transformation of all directions
    cufftHandle m_fftPlanOne;       // Plan for FFT (R2C) transformation for single image
    GPUmat* m_gm;
};

