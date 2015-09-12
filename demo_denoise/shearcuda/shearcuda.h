#ifndef SHEARCUDA_H
#define SHEARCUDA_H

#include "shearcuda_global.h"
#include "datatypes.h"
#include "cuda.h"

class GpuTimes;
class GpuImage;
class GpuShearDictionary;
class GpuCellData;

class SHEARCUDASHARED_EXPORT Shearcuda
{
public:
    Shearcuda();
    ~Shearcuda();

    bool init();
    void shutdown();

    bool addNoise( GpuImage& gpuImage, double sigma ) const;
    bool shearTrans( const GpuImage& gpuImage, const GpuShearDictionary& shearDict, GpuCellData& temp, GpuCellData& coeff ) const;
    bool hardThreshold(GpuCellData& coeff, double lambda, const double* E, const double* sc ) const;
    bool inverseShear( const GpuCellData& coeff, const GpuShearDictionary& shearDict, GpuCellData& temp, GpuImage& gpuImage ) const;
    bool computeShearNorm( GpuShearDictionary& shearDict, GpuCellData& temp ) const;

    bool atrousdec(const GpuImage& x, const GpuImage& h0, const GpuImage& h1, int numLevels,
                              GpuImage* y, GpuImage* tempBuffer) const;
    bool atrousrec( const GpuImage* y, const GpuImage& g0, const GpuImage& g1, int numLevels,
                   GpuImage& outputImage, GpuImage* tempBuffer) const;
    bool convolution(const GpuImage* y, const GpuShearDictionary& shear, GpuImage* tempBuffer, GpuImage* d ) const;
    double getNorm( const void* image_ptr, void* scratch_buf, double p, int numElem, gpuTYPE_t type_signal ) const;
    double getNormErr( const void* imageA_ptr, const void* imageB_ptr, void* scratch_buf, double p, int numElem, gpuTYPE_t type_signal ) const;
    bool applyfilters(const GpuImage* d, const GpuShearDictionary& shear, GpuImage* tempBuffer, GpuImage* y ) const;

    const GpuTimes* gt() { return m_times; }

    // Perform quick test to check that GPU works
    void quickTest() const;
    void startTimers() const;
    void stopTimers() const;

private:
    int m_nDevice;
    CUmodule m_ShearCudaModule;
    GpuTimes* m_times;
};

#endif // SHEARCUDA_H
