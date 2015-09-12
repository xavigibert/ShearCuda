#include "gpusheardictionary.h"
#include "gpuimage.h"
#include "shearcudafunctions.h"
#include "gpucelldata.h"
#include <iostream>

// Create empty object
GpuShearDictionary::GpuShearDictionary(void)
{
    m_pBaseFilters = NULL;
    m_nFilterLen = 0;
    m_type = gpuFLOAT;
    m_fftPlanOne = 0;
}

GpuShearDictionary::~GpuShearDictionary(void)
{
    if( m_GpuData.size() > 0 )
    {
#if CUDART_VERSION >= 4000
        for( int j = 0; j < numScales(); j++ )
        {
            // Check that plan has not been previously destroyed
            bool bFound = false;
            for( int k = 0; k < j; k++ )
            {
                if( m_fftPlanMany[j] == m_fftPlanMany[k] )
                    bFound = true;
            }
            if( !bFound )
                cufftSafeCall( cufftDestroy( m_fftPlanMany[j] ));
        }
        m_fftPlanMany.clear();
#endif
        cufftSafeCall( cufftDestroy( m_fftPlanOne ));
        m_GpuData.clear();
    }
}

int GpuShearDictionary::maxDirections() const
{
    int val = 1;

    for( int i = 0; i < numScales(); i++ )
    {
        if( numDirections(i) > val )
            val = numDirections(i);
    }

    return val;
}

// Prepare zero padded tranform filters from unpadded filters
bool GpuShearDictionary::prepare(const GpuShearFilters* filters, int len)
{
    if( hasData() )
    {
        std::cerr << "GpuShearDictionary::prepare() should only be called once" << std::endl;
        return false;
    }

    if( !filters->hasData() )
    {
        std::cerr << "GpuShearDictionary::prepare() has been called without valid filters" << std::endl;
        return false;
    }

    // Transfer configuration
    m_pBaseFilters = filters;
    m_nFilterLen = len;

    // Determine input data type
    gpuTYPE_t real_type = filters->dataType();
    m_type = (real_type == gpuFLOAT ? gpuCFLOAT : gpuCDOUBLE );

    // Set number of scales
    int numScales = filters->numScales();
    m_GpuData.resize( numScales );
#if CUDART_VERSION >= 4000
    m_fftPlanMany.resize( numScales );
#endif

    // Create single image plan                                                                                                                                       
    if( m_type == gpuCFLOAT )
        cufftSafeCall( cufftPlan2d( &m_fftPlanOne, len, len, CUFFT_C2C ) );
    else
        cufftSafeCall( cufftPlan2d( &m_fftPlanOne, len, len, CUFFT_Z2Z ) );

    // Allocate temporary storage
    std::vector<int> numDirections( numScales );

    // Prepare each scale
    for( int idxScale = 0; idxScale < numScales; idxScale++ )
    {
        const GpuImage* elem = m_pBaseFilters->filters( idxScale );
        numDirections[idxScale] = elem->depth();
        
        // Zero-padded dimensions
        int idims[3];
        idims[0] = len;
        idims[1] = len;
        idims[2] = numDirections[idxScale];

#if CUDART_VERSION >= 4000
        // Check if previous plan can be reused
        bool bFound = false;
        for( int idxOther = 0; idxOther < idxScale; idxOther++ )
        {
            if( numDirections[idxScale] == numDirections[idxOther] )
            {
                m_fftPlanMany[idxScale] = m_fftPlanMany[idxOther];

                bFound = true;
                break;
            }
        }

        // Create FFT plans
        if( !bFound )
        {
            if( m_type == gpuCFLOAT )
            {
                cufftSafeCall( cufftPlanMany( &m_fftPlanMany[idxScale], 2, idims, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, numDirections[idxScale] ) );
            }
            else
            {
                cufftSafeCall( cufftPlanMany( &m_fftPlanMany[idxScale], 2, idims, NULL, 0, 0, NULL, 0, 0, CUFFT_Z2Z, numDirections[idxScale] ) );
            }
        }
#endif

        // Prepare zero-padded filters
        m_GpuData[idxScale].allocate( m_type, 3, idims );
        void* dshear_ptr = m_GpuData[idxScale].getPtr();
        const void* w_s_ptr = m_pBaseFilters->filters(idxScale)->getPtr();
#if CUDART_VERSION >= 4000
        prepareFilters( dshear_ptr, len, numDirections[idxScale], m_type, w_s_ptr, elem->width(), real_type, m_fftPlanMany[idxScale] );
#else
        prepareFilters( dshear_ptr, len, numDirections[idxScale], m_type, w_s_ptr, elem->width(), real_type, m_fftPlanOne );
#endif
    }

    // Prepare the norm vector
    m_dNorm.resize((numScales + 1) * maxDirections());

    return true;
}

bool GpuShearDictionary::prepareTempBuffers(GpuCellData& temp) const
{
    // We need a total of numScales + 4 buffers
    // The first 3 elements are reserved for the submsampling buffers
    // The other numScales + 1 are reserved for the decomposition
    temp.setNumElem( numScales() + 4 );

    gpuTYPE_t real_type = m_pBaseFilters->dataType();

    // Preallocate big buffer for convolutions and subsampling
    temp.elem(0)->allocate( real_type, 2 * m_nFilterLen, m_nFilterLen * maxDirections() );
    temp.elem(1)->allocate( real_type, 2 * m_nFilterLen, m_nFilterLen * maxDirections() );
    temp.elem(2)->allocate( real_type, 2 * m_nFilterLen, 2 * m_nFilterLen );

    // Preallocate temporary buffers for decomposition
    for( int j = 0; j < numScales() + 1; j++ )
        temp.elem(3+j)->allocate( real_type, m_nFilterLen, m_nFilterLen );

    return true;
}


// Zero pad and prepare shearing filters
void GpuShearDictionary::prepareFilters( void* dst, int dstSize, int numDir, gpuTYPE_t data_type, const void* src, int srcSize, gpuTYPE_t real_type, cufftHandle fftPlan )
{
    g_ShearFunc.zeroPad(dst, dstSize, dstSize, numDir, data_type, src, srcSize, srcSize, numDir, real_type);

    g_ShearFunc.timer()->startTimer( GpuTimes::fftFwd );
#if CUDART_VERSION >= 4000
    if( data_type == gpuCFLOAT )
        cufftSafeCall( cufftExecC2C(fftPlan, (cufftComplex *)dst, (cufftComplex *)dst, CUFFT_FORWARD ));
    else
        cufftSafeCall( cufftExecZ2Z(fftPlan, (cufftDoubleComplex *)dst, (cufftDoubleComplex *)dst, CUFFT_FORWARD ));
#else
    for( int dir = 0; dir < numDir; dir++ )
    {
        int offset = dir * dstSize * dstSize;
        if( data_type == gpuCFLOAT )
            cufftSafeCall( cufftExecC2C(fftPlan, (cufftComplex *)dst + offset, (cufftComplex *)dst + offset, CUFFT_FORWARD ));
        else
            cufftSafeCall( cufftExecZ2Z(fftPlan, (cufftDoubleComplex *)dst + offset, (cufftDoubleComplex *)dst + offset, CUFFT_FORWARD ));
    }
#endif
    g_ShearFunc.timer()->stopTimer( GpuTimes::fftFwd );
    
    g_ShearFunc.prepareMyerFilters(dst, dstSize, numDir, data_type);
}
