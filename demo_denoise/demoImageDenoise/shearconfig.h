#ifndef SHEARCONFIG_H
#define SHEARCONFIG_H

#include <vector>
#include "datatypes.h"
#include <QString>

// Shearlet configuration in CPU memory
class ShearConfig
{
public:
    ShearConfig();
    ~ShearConfig();

    // Read filters from file
    bool readFromFile( QString fileName );

    // Getters
    int version() const { return m_nVersion; }
    gpuTYPE_t dataType() const { return m_dataType; }
    int numScales() const { return m_sizeShearing.size(); }
    const int* sizeShearing() const { return &m_sizeShearing[0]; }
    const int* numDir() const { return &m_numDir[0]; }
    void* const* filters() const { return &m_filter[0]; }
    void* const* atrousFilters() const { return &m_atrousFilters[0]; }
    const int* atrousLen() const { return &m_atrousLen[0]; }

private:
    int m_nVersion;         // Version of configuration object
    gpuTYPE_t m_dataType;   // Filter data type
    std::vector<int> m_sizeShearing;    // Size of the shearig matrix w_c
    std::vector<int> m_numDir;          // Number of directions
    std::vector<void*> m_filter;        // Filters
    void* m_atrousFilters[4];
    int m_atrousLen[4];
};

#endif // SHEARCONFIG_H
