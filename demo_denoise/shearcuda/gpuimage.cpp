#include "gpuimage.h"
#include "shearcuda.h"
#include "shearcudafunctions.h"

// CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "cutil_inline_runtime.h"

#include <iostream>

GpuImage::GpuImage()
{
    m_type = gpuFLOAT;
    m_d_ptr = NULL;
    m_auto_delete = false;
    m_dims[0] = m_dims[1] = m_dims[2] = 0;
}

// Copy constructor makes a copy of the image
GpuImage::GpuImage(const GpuImage& src)
{
    m_type = src.m_type;
    m_dims[0] = src.m_dims[0];
    m_dims[1] = src.m_dims[1];
    m_dims[2] = src.m_dims[2];
    m_auto_delete = true;
    cutilSafeCall( cudaMalloc( &m_d_ptr, size() ) );
    cutilSafeCall( cudaMemcpy( m_d_ptr, src.m_d_ptr, size(), cudaMemcpyDeviceToDevice ));
}

GpuImage::GpuImage(gpuTYPE_t type, int ndims, const int* dims)
{
    m_type = type;
    int i;
    for( i = 0; i < ndims; i++ )
        m_dims[i] = dims[i];
    for( ; i < 3; i++ )
        m_dims[i] = 1;
    m_auto_delete = true;
    cutilSafeCall( cudaMalloc( &m_d_ptr, size() ) );
}

GpuImage::GpuImage(gpuTYPE_t type, int dimX, int dimY, int dimZ)
{
    m_type = type;
    m_dims[0] = dimX;
    m_dims[1] = dimY;
    m_dims[2] = dimZ;
    m_auto_delete = true;
    cutilSafeCall( cudaMalloc( &m_d_ptr, size() ) );
}

void GpuImage::allocate(gpuTYPE_t type, int ndims, const int* dims)
{
    if( m_auto_delete )
        freeData();
    m_type = type;
    int i;
    for( i = 0; i < ndims; i++ )
        m_dims[i] = dims[i];
    for( ; i < 3; i++ )
        m_dims[i] = 1;
    m_auto_delete = true;
    cutilSafeCall( cudaMalloc( &m_d_ptr, size() ) );
}

void GpuImage::allocate(gpuTYPE_t type, int dimX, int dimY, int dimZ)
{
    m_type = type;
    m_dims[0] = dimX;
    m_dims[1] = dimY;
    m_dims[2] = dimZ;
    m_auto_delete = true;
    cutilSafeCall( cudaMalloc( &m_d_ptr, size() ) );
}

GpuImage::~GpuImage()
{
    if( m_auto_delete )
        freeData();
}

// Set object to point to other data (disables auto-deletion)
void GpuImage::setData(gpuTYPE_t type, int dimX, int dimY, int dimZ, void *d_ptr)
{
    if( m_auto_delete )
        freeData();

    m_type = type;
    m_d_ptr = d_ptr;
    m_dims[0] = dimX;
    m_dims[1] = dimY;
    m_dims[2] = dimZ;
}

void GpuImage::freeData()
{
    cutilSafeCall( cudaFree( m_d_ptr ));
    m_d_ptr = NULL;
    m_auto_delete = false;
}

int GpuImage::size() const
{
    return m_dims[0] * m_dims[1] * m_dims[2] * ShearCudaFunctions::elemSize(m_type);
}

// Data transfer functions
void GpuImage::transferToDevice( const void* src )
{
    cutilSafeCall( cudaMemcpy( m_d_ptr, src, size(), cudaMemcpyHostToDevice ));
}

void GpuImage::transferFromDevice( void* dst ) const
{
    cutilSafeCall( cudaMemcpy( dst, m_d_ptr, size(), cudaMemcpyDeviceToHost ));
}

// Convert existing image to new data type
bool GpuImage::convert(gpuTYPE_t dstType)
{
    // Allocate destination buffer
    void* d_dst;
    cutilSafeCall( cudaMalloc( &d_dst, numElements() * ShearCudaFunctions::elemSize(dstType) ) );

    // Perform conversion
    if( !g_ShearFunc.convert(d_dst, dstType, m_d_ptr, m_type, numElements()) )
    {
        std::cerr << "Image conversion not supported" << std::endl;
        return false;
    }

    // Switch buffers
    cutilSafeCall( cudaFree( m_d_ptr ));
    m_d_ptr = d_dst;
    m_type = dstType;

    return true;
}
