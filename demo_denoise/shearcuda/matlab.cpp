#include "matlab.h"
#include "gpuimage.h"
#include "gpucelldata.h"
#include "gpusheardictionary.h"
#include <iostream>

Matlab g_Matlab;

Matlab::Matlab()
{
}

void Matlab::writeMat(const char* fileName)
{
    std::deque<VarType>::iterator iter;

    MATFile *pmat = matOpen( fileName, "w");
    if( pmat == NULL )
    {
        std::cout << "Error creating file " << fileName << std::endl;
    }
    for( iter = m_vars.begin(); iter < m_vars.end(); ++iter )
    {
        VarType val = *iter;
        matPutVariable(pmat, val.name.c_str(), val.data);
        mxDestroyArray(val.data);
    }
    matClose( pmat );
    m_vars.clear();
}

void Matlab::addToMat( mxArray* data, const char* varName)
{
    VarType val;
    val.data = data;
    val.name = std::string(varName);
    m_vars.push_back(val);
}

void Matlab::addToMat(const  GpuImage* image, const char* varName)
{
    mxArray* data = createArray(image);
    addToMat( data, varName );
}

void Matlab::addToMat(const  GpuShearDictionary* data, const char* varName)
{
    mwSize dims[2] = { 1, data->numScales()};
    mxArray* mxCell = mxCreateCellArray( 2, dims );
    for( int i = 0; i < data->numScales(); i++ )
    {
        mxSetCell( mxCell, i, createArray( data->filters(i)));
    }
    addToMat( mxCell, varName );
}

void Matlab::addToMat(const  GpuCellData* data, const char* varName)
{
    mwSize dims[2] = { 1, data->numElem()};
    mxArray* mxCell = mxCreateCellArray( 2, dims );
    for( int i = 0; i < data->numElem(); i++ )
    {
        mxSetCell( mxCell, i, createArray( data->elem(i)));
    }
    addToMat( mxCell, varName );
}

mxArray* Matlab::createArray(const  GpuImage* image)
{
    mxClassID classID;
    mxComplexity cflag = mxREAL;
    int elem_size = 1;
    switch( image->type() )
    {
    case gpuFLOAT:
        classID = mxSINGLE_CLASS;
        elem_size = 4;
        break;
    case gpuCFLOAT:
        classID = mxSINGLE_CLASS;
        cflag = mxCOMPLEX;
        elem_size = 4;
        break;
    case gpuDOUBLE:
        classID = mxDOUBLE_CLASS;
        elem_size = 8;
        break;
    case gpuCDOUBLE:
        classID = mxDOUBLE_CLASS;
        cflag = mxCOMPLEX;
        elem_size = 8;
        break;
    case gpuINT32:
        classID = mxINT32_CLASS;
        elem_size = 4;
        break;
    case gpuUINT8:
        classID = mxUINT8_CLASS;
        elem_size = 1;
        break;
    default:
        return NULL;
    }
    mwSize dims[3];
    dims[0] = image->height();
    dims[1] = image->width();
    dims[2] = image->depth();

    int ndims = image->numDims();
    mxArray* mx = mxCreateNumericArray(ndims, dims, classID, cflag);
    void* mxReal = mxGetData(mx);
    int num = image->numElements();
    if( cflag == mxREAL )
    {
        if( classID == mxSINGLE_CLASS )
        {
            float* temp = new float[num];
            image->transferFromDevice( temp );
            for(int x=0; x<dims[1]; x++) {
                for(int y=0; y<dims[0]; y++) {
                    for(int z=0; z<dims[2]; z++) {
                        ((float*)mxReal)[(z*dims[1]+x)*dims[0]+y] = temp[(z*dims[0]+y)*dims[1]+x];
                    }
                }
            }
            delete temp;
        }
        else
        {
            double* temp = new double[num];
            image->transferFromDevice( temp );
            for(int x=0; x<dims[1]; x++) {
                for(int y=0; y<dims[0]; y++) {
                    for(int z=0; z<dims[2]; z++) {
                        ((double*)mxReal)[(z*dims[1]+x)*dims[0]+y] = temp[(z*dims[0]+y)*dims[1]+x];
                    }
                }
            }
            delete temp;
        }
        return mx;
    }

    void* mxImag = mxGetImagData(mx);
    if( classID == mxSINGLE_CLASS )
    {
        float* temp = new float[num * 2];
        image->transferFromDevice( temp );
        for(int x=0; x<dims[1]; x++) {
            for(int y=0; y<dims[0]; y++) {
                for(int z=0; z<dims[2]; z++) {
                    ((float*)mxReal)[(z*dims[1]+x)*dims[0]+y] = temp[((z*dims[0]+y)*dims[1]+x)*2];
                    ((float*)mxImag)[(z*dims[1]+x)*dims[0]+y] = temp[((z*dims[0]+y)*dims[1]+x)*2+1];
                }
            }
        }
        delete temp;
    }
    else
    {
        double* temp = new double[num * 2];
        image->transferFromDevice( temp );
        for(int x=0; x<dims[1]; x++) {
            for(int y=0; y<dims[0]; y++) {
                for(int z=0; z<dims[2]; z++) {
                    ((double*)mxReal)[(z*dims[1]+x)*dims[0]+y] = temp[((z*dims[0]+y)*dims[1]+x)*2];
                    ((double*)mxImag)[(z*dims[1]+x)*dims[0]+y] = temp[((z*dims[0]+y)*dims[1]+x)*2+1];
                }
            }
        }
        delete temp;
    }

    return mx;
}
