#include "gpushearfilters.h"
#include <iostream>
#include <stdlib.h>
#include "shearcudafunctions.h"

//GpuShearFilters::GpuShearFilters()
//{
//}

GpuShearFilters::GpuShearFilters(int version, gpuTYPE_t dataType, int numScales,
                const int* sizeShearing, const int* numDir,
                const void* const* filters, const void* const* atrousFilters,
                const int* atrousLen)
{
    if( dataType == gpuDOUBLE && !g_ShearFunc.supportsDouble )
    {
        std::cerr << "DOUBLE precision requires compute capability >= 2.0" << std::endl;
        exit(1);
    }

    m_nVersion = version;
    m_filter.resize(numScales);

    // Allocate filters and transfer coefficientscoeff.elem(idxScale)->width()
    for( int j = 0; j < numScales; j++ )
    {
        m_filter[j].allocate(dataType, sizeShearing[j], sizeShearing[j], numDir[j]);
        m_filter[j].transferToDevice( filters[j] );
    }

    // Allocate a-trous filters and transfer coefficients
    for( int j = 0; j < 4; j++ )
    {
        m_atrousFilters[j].allocate(dataType, atrousLen[j], atrousLen[j]);
        m_atrousFilters[j].transferToDevice( atrousFilters[j] );
    }
}
