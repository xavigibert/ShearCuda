#include "shearconfig.h"
#include "shearcudafunctions.h"
#include <QFile>

ShearConfig::ShearConfig()
{
    m_nVersion = -1;
    m_dataType = gpuFLOAT;
    for( int i = 0; i < 4; i++ )
    {
        m_atrousFilters[i] = NULL;
        m_atrousLen[i] = 0;
    }
}

ShearConfig::~ShearConfig()
{
    for( int i = 0; i < 4; i++ )
    {
        free( m_atrousFilters[i] );
        m_atrousFilters[i] = NULL;
    }
}

// Read filters from file
bool ShearConfig::readFromFile( QString fileName )
{
    // Read data
    QFile file(fileName);

    if ( !file.open(QIODevice::ReadOnly) )
        return false;

    QByteArray bHeader = file.read(8);
    QString header( bHeader );
    if( header != QString("SHFM0002") )
        return false;

    m_nVersion = 1;

    unsigned short numScales = 0;
    unsigned short dataType = 0;
    if( file.read( (char*)&numScales, sizeof(unsigned short)) <= 0 ) return false;
    if( file.read( (char*)&dataType, sizeof(unsigned short)) <= 0 ) return false;

    m_dataType = (gpuTYPE_t)dataType;
    m_sizeShearing.resize( numScales );
    m_numDir.resize( numScales );
    m_filter.resize( numScales );
    int elem_size = ShearCudaFunctions::elemSize( m_dataType );

    // Read filters for each scale
    for( int j = 0; j < numScales; j++ )
    {
        unsigned short filterLen = 0;
        unsigned short numDir = 0;
        if( file.read( (char*)&filterLen, sizeof(unsigned short)) <= 0 ) return false;
        if( file.read( (char*)&numDir, sizeof(unsigned short)) <= 0 ) return false;
        m_sizeShearing[j] = filterLen;
        m_numDir[j] = numDir;
        int dataSize = numDir * filterLen * filterLen * elem_size;
        m_filter[j] = malloc( dataSize );
        if( file.read( (char*)m_filter[j], dataSize) < dataSize ) return false;
    }

    // Read a-trous decomposition filters
    for( int j = 0; j < 4; j++ )
    {
        unsigned short filterLen = 0;
        if( file.read( (char*)&filterLen, sizeof(unsigned short)) <= 0 ) return false;
        int dataSize = filterLen * filterLen * elem_size;
        m_atrousLen[j] = filterLen;
        m_atrousFilters[j] = malloc( dataSize );
        if( file.read( (char*)m_atrousFilters[j], dataSize) < dataSize ) return false;
    }

    return true;
}
