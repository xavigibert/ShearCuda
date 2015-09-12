#ifndef MATLABENGINE_H
#define MATLABENGINE_H

#include "gpuimage.h"
#include "gpusheardictionary.h"
#include "mat.h"
#include <deque>
#include <string>

class SHEARCUDASHARED_EXPORT Matlab
{
public:

    struct VarType
    {
        mxArray* data;
        std::string name;
    };

    Matlab();

    void writeMat(const char* fileName);
    void addToMat( mxArray* data, const char* varName);
    void addToMat(const  GpuImage* image, const char* varName);
    void addToMat(const  GpuShearDictionary* data, const char* varName);
    void addToMat(const  GpuCellData* data, const char* varName);
    mxArray* createArray(const  GpuImage* image);
private:
    std::deque<VarType> m_vars;
};

extern Matlab g_Matlab;

#endif // MATLABENGINE_H
