#ifndef GPUCELLDATA_H
#define GPUCELLDATA_H

#include "shearcuda.h"
#include <vector>
#include "gpuimage.h"

class SHEARCUDASHARED_EXPORT GpuCellData
{
public:
    GpuCellData();

    int numElem() const { return m_GpuData.size(); }
    void setNumElem(int n) { m_GpuData.resize(n); }
    GpuImage* elem(int idx) { return &m_GpuData[idx]; }
    const GpuImage* elem(int idx) const { return &m_GpuData[idx]; }

private:
    std::vector<GpuImage> m_GpuData;     // GPUtype objects containing the data
};

#endif // GPUCELLDATA_H
