#ifndef GPUIMAGE_H
#define GPUIMAGE_H

#include "datatypes.h"
#include "shearcuda_global.h"

// A very thin wrapper to an image in the GPU
class SHEARCUDASHARED_EXPORT GpuImage
{
public:
    GpuImage();     // Unallocated object
    GpuImage(const GpuImage& src);     // Copy constructor makes a copy of the image
    GpuImage(gpuTYPE_t type, int ndims, const int* dims);
    GpuImage(gpuTYPE_t type, int dimX, int dimY, int dimZ = 1);
    ~GpuImage();

    int size() const;    // Total size in bytes
    void* getPtr() const { return m_d_ptr; }
    int width() const { return m_dims[0]; }
    int height() const { return m_dims[1]; }
    int depth() const { return m_dims[2]; }
    const int* dims() const { return m_dims; }
    gpuTYPE_t type() const { return m_type; }
    // Number of dimensions
    int numDims() const { return m_dims[2] > 1 ? 3 : 2; }
    int numElements() const { return m_dims[0] * m_dims[1] * m_dims[2]; }

    // Set object to point to other data (disables auto-deletion)
    void setData(gpuTYPE_t type, int dimX, int dimY, int dimZ, void *d_ptr);

    // Allocate for automatic deletion
    void allocate(gpuTYPE_t type, int dimX, int dimY, int dimZ = 1);
    void allocate(gpuTYPE_t type, int ndims, const int* dims);

    // Data transfer functions
    void transferToDevice( const void* src );
    void transferFromDevice( void* dst) const;

    // Free data pointed by this object
    void freeData();

    // Convert existing image to new data type
    bool convert(gpuTYPE_t dstType);

private:
//    GpuImage(const GpuImage&);  // Disable copy constructor

    gpuTYPE_t m_type;     // Data type
    void* m_d_ptr;        // Device pointer
    bool m_auto_delete;   // True if delete on destruction
    int m_dims[3];        // Image dimensions
};

#endif // GPUIMAGE_H
