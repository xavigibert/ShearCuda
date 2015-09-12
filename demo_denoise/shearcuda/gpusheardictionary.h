#ifndef GPUSHEARDICTIONARY_H
#define GPUSHEARDICTIONARY_H

#include <vector>

#include "cuda.h"
#include "cuda_runtime.h"
#include <string.h>
#include "datatypes.h"
#include "cutil_inline_runtime.h"
#include "shearcuda_global.h"
#include "gpuimage.h"
#include "gpushearfilters.h"

class GpuShearFilters;
class GpuCellData;

class SHEARCUDASHARED_EXPORT GpuShearDictionary
{
public:
    // Create empty object
    GpuShearDictionary();

    ~GpuShearDictionary(void);

    // Getters
    gpuTYPE_t dataType() const { return m_type; }
    gpuTYPE_t realType() const { return m_pBaseFilters->dataType(); }
    int numScales() const { return m_GpuData.size(); }
    int filterLen() const { return m_nFilterLen; }
    int numDirections(int scale) const { return m_GpuData[scale].depth(); }
    const GpuImage* filters(int scale) const { return &m_GpuData[scale]; }
    bool hasData() const { return m_GpuData.size()>0; }
    cufftHandle fftPlanOne() const { return m_fftPlanOne; }
#if CUDART_VERSION >= 4000
    cufftHandle fftPlanMany(int scale) const { return m_fftPlanMany[scale]; }
#endif
    const GpuImage* h0() const { return m_pBaseFilters->h0(); }
    const GpuImage* h1() const { return m_pBaseFilters->h1(); }
    const GpuImage* g0() const { return m_pBaseFilters->g0(); }
    const GpuImage* g1() const { return m_pBaseFilters->g1(); }
    // Returns vector containing L2 norm of the transformation
    // (elements are accessessed as E[idxScale + idxDirection*(numScales+1)]
    double* norm() { return &m_dNorm[0]; }

    // Return maximum number of directions across all scales
    int maxDirections() const;

    // Prepare zero padded tranform filters from unpadded filters
    bool prepare(const GpuShearFilters* filters, int len);

    // Prepare temporary buffers (need one per stream)
    bool prepareTempBuffers(GpuCellData& temp) const;

protected:
    void prepareFilters( void* dst, int dstSize, int numDir, gpuTYPE_t data_type, const void* src, int srcSize, gpuTYPE_t real_type, cufftHandle fftPlan );

private:
    const GpuShearFilters* m_pBaseFilters;  // Shearlet filters
    int m_nFilterLen;       // Filter width and height
    gpuTYPE_t m_type;       // Type of data
    std::vector<GpuImage> m_GpuData;     // GPUtype objects containing the data
#if CUDART_VERSION >= 4000
    std::vector<cufftHandle> m_fftPlanMany;  // Plan for FFT/IFFT transformation of all directions
#endif
    cufftHandle m_fftPlanOne;    // Plan for FFT/IFFT transformation for single image
    std::vector<double> m_dNorm;
};

#endif // GPUSHEARDICTIONARY_H
