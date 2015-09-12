#ifndef GPUSHEARFILTERS_H
#define GPUSHEARFILTERS_H

#include "gpuimage.h"
#include "shearcuda_global.h"
#include <vector>

class SHEARCUDASHARED_EXPORT GpuShearFilters
{
public:
    //GpuShearFilters();
    GpuShearFilters(int version, gpuTYPE_t dataType, int numScales,
                    const int* sizeShearing, const int* numDir,
                    const void* const* filters, const void* const* atrousFilters,
                    const int* atrousLen);

    // Getters
    gpuTYPE_t dataType() const { return m_filter[0].type(); }
    int numScales() const { return m_filter.size(); }
    int numDirections(int scale) const { return m_filter[scale].depth(); }
    const GpuImage* filters(int scale) const { return &m_filter[scale]; }
    bool hasData() const { return m_filter.size()>0; }
    const GpuImage* h0() const { return &m_atrousFilters[0]; }
    const GpuImage* h1() const { return &m_atrousFilters[1]; }
    const GpuImage* g0() const { return &m_atrousFilters[2]; }
    const GpuImage* g1() const { return &m_atrousFilters[3]; }

private:
    int m_nVersion;         // Version of configuration object
    std::vector<GpuImage> m_filter;     // Filters
    GpuImage m_atrousFilters[4];        // GPUtype objects containing the filters at different scales
};

#endif // GPUSHEARFILTERS_H
