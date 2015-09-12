#include "shearcuda.h"
#include <iostream>
#include "cutil_inline_runtime.h"
#include "gputimes.h"
#include "shearcudafunctions.h"
#include "gpuimage.h"
#include "gpusheardictionary.h"
#include "gpucelldata.h"
#include <curand.h>
#include <vector>
#include <algorithm>

// TEMPORARY CODE
//#include "matlab.h"

const char *MODULE_BASE_NAME = "shear_cuda";

using namespace std;


Shearcuda::Shearcuda()
{
    // Default device
    m_nDevice = 0;
    m_ShearCudaModule = 0;
    m_times = NULL;
}

Shearcuda::~Shearcuda()
{
    if( m_times != NULL )
        shutdown();
}

bool Shearcuda::init()
{
    // Show information about runtime enviroment
    std::cout << "Compiled with CUDA runtime version " << (CUDART_VERSION/100)/10 << "." << (CUDART_VERSION%100)/10 << std::endl;

    // Select best device
    m_nDevice = cutGetMaxGflopsDeviceId();
    //cout << "cutGetMaxGflopsDeviceId() returned " << m_nDevice << endl;
    //if( cudaSetDevice(m_nDevice) != cudaSuccess )
    //{
    //    cerr << "cudaSetDevice() failed on device " << m_nDevice << "!" << endl;
    //    return false;
    //}

    char szName[256];
    CUdevice hcuDevice;
    CUresult res;
    
    if( ( res = cuInit(0) ) != CUDA_SUCCESS )
    {
        cerr << "cuInit(0) failed with error " << res << endl;
	return false;
    }
    if( ( res = cuDeviceGet( &hcuDevice, m_nDevice )) != CUDA_SUCCESS )
    {
        cerr << "cuDeviceGet() failed with error " << res << endl;
        return false;
    }

    if( cuDeviceGetName( szName, 256, hcuDevice ) != CUDA_SUCCESS )
    {
        cerr << "cuDeviceGetName() failed!" << endl;
        return false;
    }
    else
    {
        cout << "Using device " << m_nDevice << ": " << szName << endl;
    }

    // Create CUDA context
    CUcontext cuContext = 0;
    CUresult status = cuCtxCreate(&cuContext, 0, m_nDevice);
    if (status != CUDA_SUCCESS) {
        cerr << "cuContext() failed!" << endl;
        return false;
    }

    // Find modules
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties( &deviceProp, m_nDevice );
    std::cout << "CUDA device capability " << deviceProp.major << "." << deviceProp.minor << std::endl;
    char moduleName[256];
    sprintf(moduleName, "%s%d%d.cubin", MODULE_BASE_NAME, deviceProp.major, deviceProp.minor);

    CUresult cuRes;
    if( (cuRes = cuModuleLoad(&m_ShearCudaModule, moduleName)) != CUDA_SUCCESS )
    {
        cerr << "cuModuleLoad() failed on file " << moduleName << "!" << endl;
        return false;
    }

    if( !g_ShearFunc.LoadGpuFunctions( &m_ShearCudaModule ) )
    {
        cerr << "Failed to load module functions" << endl;
        return false;
    }

    m_times = new GpuTimes;
    m_times->disable();
    g_ShearFunc.setTimer( m_times );

    return true;
}

void Shearcuda::startTimers() const
{
    m_times->resetTimers();
    m_times->enable();
}

void Shearcuda::stopTimers() const
{
    m_times->disable();
}

// Perform quick test to check that GPU works
void Shearcuda::quickTest() const
{
    void* buffer;
    cutilSafeCall( cudaMalloc( &buffer, 1024*4 ));
    cutilSafeCall( cudaMemset( buffer, 0, 1024*4 ));
    g_ShearFunc.addVector( buffer, buffer, 1024, gpuFLOAT );
    cutilSafeCall( cudaFree( buffer ));
}

void Shearcuda::shutdown()
{
    m_times->displayTimers();
    delete m_times;
    m_times = NULL;

    cuModuleUnload(m_ShearCudaModule);
    cutilDeviceReset();
}

bool Shearcuda::addNoise( GpuImage& gpuImage, double sigma ) const
{
    curandGenerator_t generator;
    curandSafeCall( curandCreateGenerator( &generator, CURAND_RNG_PSEUDO_DEFAULT  ));
    curandSafeCall( curandSetPseudoRandomGeneratorSeed( generator, 1234ULL));

    // Allocate buffer for noise
    void* buffer;
    int numElems = gpuImage.numElements();
    int buffer_size = numElems * ShearCudaFunctions::elemSize(gpuImage.type());
    cutilSafeCall( cudaMalloc( &buffer, buffer_size ));

    // Generate Gaussian noise and add it to the image
    if( gpuImage.type() == gpuFLOAT )
        curandSafeCall( curandGenerateNormal( generator, (float*)buffer, numElems, 0, (float)sigma));
    else
        curandSafeCall( curandGenerateNormalDouble( generator, (double*)buffer, numElems, 0, (double)sigma));
    g_ShearFunc.addVector( gpuImage.getPtr(), buffer, numElems, gpuImage.type() );

    // Clean up
    cutilSafeCall( cudaFree( buffer ));
    curandSafeCall( curandDestroyGenerator( generator ));

    return true;
}

float median(std::vector<float>::iterator begin, std::vector<float>::iterator end)
{
    int size = end - begin;
    if( size % 2 == 0 )
    {
        std::vector<float>::iterator medIter1 = begin + size/2;
        std::vector<float>::iterator medIter2 = begin + (size/2 - 1);
        std::nth_element( begin, medIter1, end );
        std::nth_element( begin, medIter2, end );
        return ((*medIter1) + (*medIter2)) / 2;
    }
    else
    {
        std::vector<float>::iterator medIter = begin + size/2;
        std::nth_element( begin, medIter, end);
        return *medIter;
    }
}

double median(std::vector<double>::iterator begin, std::vector<double>::iterator end)
{
    int size = end - begin;
    if( size % 2 == 0 )
    {
        std::vector<double>::iterator medIter1 = begin + size/2;
        std::vector<double>::iterator medIter2 = begin + (size/2 - 1);
        std::nth_element( begin, medIter1, end );
        std::nth_element( begin, medIter2, end );
        return ((*medIter1) + (*medIter2)) / 2;
    }
    else
    {
        std::vector<double>::iterator medIter = begin + size/2;
        std::nth_element( begin, medIter, end);
        return *medIter;
    }
}

// Computes the L2 norm of the shearlet dictionary
bool Shearcuda::computeShearNorm( GpuShearDictionary& shearDict, GpuCellData& temp ) const
{
    // Prepare delta image
    gpuTYPE_t real_type = shearDict.realType();
    int filterLen = shearDict.filterLen();
    GpuImage x( real_type, filterLen, filterLen );

    // Generate random noise
    curandGenerator_t generator;
    curandSafeCall( curandCreateGenerator( &generator, CURAND_RNG_PSEUDO_DEFAULT  ));
    curandSafeCall( curandSetPseudoRandomGeneratorSeed( generator, 4321ULL));

    if( x.type() == gpuFLOAT )
        curandSafeCall( curandGenerateNormal( generator, (float*)x.getPtr(), x.numElements(), 0, 1.f) );
    else
        curandSafeCall( curandGenerateNormalDouble( generator, (double*)x.getPtr(), x.numElements(), 0, 1.0));

    // Transform test image
    GpuCellData Ct;
    if( !shearTrans( x, shearDict, temp, Ct ) )
        return false;
    gpuTYPE_t data_type = shearDict.dataType();
    int filter_size = g_ShearFunc.elemSize(data_type) * filterLen * filterLen;

    // Prepare host vector
    std::vector<float> imf;
    std::vector<double> imd;
    if( real_type == gpuFLOAT )
        imf.resize(x.numElements());
    else
        imd.resize(x.numElements());

    // Compute norm of each filter
    double* E = shearDict.norm();
    int numScales = Ct.numElem();

    for( int idxScale = 0; idxScale < numScales; idxScale++ )
    {
        char* base_ptr = (char*)Ct.elem(idxScale)->getPtr();
        for( int idxDirection = 0; idxDirection < Ct.elem(idxScale)->depth(); idxDirection++ )
        {
            if( real_type != Ct.elem(idxScale)->type() )
                g_ShearFunc.complexToReal( x.getPtr(), base_ptr + idxDirection * filter_size, x.numElements(), real_type );
            else
                cutilSafeCall( cudaMemcpy( x.getPtr(), base_ptr + idxDirection * filter_size, x.size(), cudaMemcpyDeviceToDevice ));
            // Do calculation on host
            if( real_type == gpuFLOAT )
            {
                x.transferFromDevice( &imf[0] );
                float med = median(imf.begin(), imf.end());
                for( int i = 0; i < (int)imf.size(); i++ )
                    imf[i] = fabs(imf[i] - med);
                E[idxScale + idxDirection*numScales] = median(imf.begin(), imf.end()) /.6745;
            }
            else
            {
                x.transferFromDevice( &imd[0] );
                double med = median(imd.begin(), imd.end());
                for( int i = 0; i < (int)imd.size(); i++ )
                    imd[i] = fabs(imd[i] - med);
                E[idxScale + idxDirection*numScales] = median(imd.begin(), imd.end()) /.6745;
            }
        }
    }

    return true;
}

bool Shearcuda::shearTrans( const GpuImage& gpuImage, const GpuShearDictionary& shearDict, GpuCellData& temp, GpuCellData& coeff ) const
{
    // Check parameter types and image size
    if( gpuImage.type() != shearDict.realType() )
    {
        std::cerr << "Data types mismatch in Shearcuda::shearTrans()" << std::endl;
        return false;
    }
    if( gpuImage.width() != shearDict.filterLen() || gpuImage.height() != shearDict.filterLen() )
    {
        std::cerr << "Image size mismatch in Shearcuda::shearTrans()" << std::endl;
        return false;
    }

    // Call atrousdec
    if( !atrousdec( gpuImage, *shearDict.h0(), *shearDict.h1(), shearDict.numScales(), temp.elem(3), temp.elem(0) ))
    {
        std::cerr << "Shearcuda::atrousdec() failed in Shearcuda::shearTrans()" << std::endl;
        return false;
    }

    // Allocate result
    if( coeff.numElem() != shearDict.numScales() + 1 )
        coeff.setNumElem( shearDict.numScales() + 1 );

    // Perform convolution
    if( !convolution( temp.elem(3), shearDict, temp.elem(0), coeff.elem(0) ))
    {
        std::cerr << "Shearcuda::convolution() failed in Shearcuda::shearTrans()" << std::endl;
        return false;
    }

//    g_Matlab.addToMat(&gpuImage, "image");
//    g_Matlab.addToMat(&shearDict, "cshear");
//    g_Matlab.addToMat(shearDict.h0(), "h0");
//    g_Matlab.addToMat(shearDict.h1(), "h1");
//    g_Matlab.addToMat(temp.elem(3), "y0");
//    g_Matlab.addToMat(temp.elem(4), "y1");
//    g_Matlab.addToMat(temp.elem(5), "y2");
//    g_Matlab.addToMat(temp.elem(6), "y3");
//    g_Matlab.addToMat(temp.elem(7), "y4");
//    g_Matlab.addToMat(&coeff, "coeff");
//    g_Matlab.writeMat("/scratch0/github/ShearCuda/shearlet_toolbox_1/debug1.mat");

    return true;
}

bool Shearcuda::hardThreshold( GpuCellData& coeff, double lambda, const double* E, const double* sc ) const
{
    int numScales = coeff.numElem();

    //g_Matlab.addToMat(&coeff, "coeff");

    // Apply thresholding
    for( int idxScale = 0; idxScale < numScales; idxScale++ )
    {
        int numDirections = coeff.elem(idxScale)->depth();
        int imageSize = coeff.elem(idxScale)->width() * coeff.elem(idxScale)->height();
        void* scale_ptr = coeff.elem(idxScale)->getPtr();
        gpuTYPE_t type_signal = coeff.elem(idxScale)->type();
        int elem_size = g_ShearFunc.elemSize( type_signal );

        for( int idxDirection = 0; idxDirection < numDirections; idxDirection++ )
        {
            // Apply hard threshold in-place
            double th, valE;
            valE = E[ idxScale + idxDirection * numScales ];
            if( idxScale == 0 )
                th = sc[0] * lambda * valE;
            else if( idxScale < numScales-1 )
                th = sc[1] * lambda * valE;
            else
                th = sc[2] * lambda * valE;

//            std::cout << "sc " << idxScale+1 << ", dir " << idxDirection+1 << ", th=" << th << std::endl;

            void* image_ptr = (char*)scale_ptr + idxDirection * imageSize * elem_size;
            g_ShearFunc.applyHardThreshold( image_ptr, image_ptr, imageSize, th, type_signal );
        }
    }

    //g_Matlab.addToMat(&coeff, "tcoeff");
    //g_Matlab.writeMat("/scratch0/github/ShearCuda/shearlet_toolbox_1/debug.mat");

    return true;
}

bool Shearcuda::inverseShear( const GpuCellData& coeff, const GpuShearDictionary& shearDict, GpuCellData& temp, GpuImage& gpuImage ) const
{
    // apply directional shearlet filters to decomposed images
    // for each scale j
    if( !applyfilters( coeff.elem(0), shearDict, temp.elem(0), temp.elem(3) ))
    {
        std::cerr << "Sheacuda::applyFilters() failed in Shearcuda::inverseShear()" << std::endl;
        return false;
    }

    // Prepare output
    if( gpuImage.numElements() == 0 )
        gpuImage.allocate( shearDict.realType(), shearDict.filterLen(), shearDict.filterLen() );

    // Call atrousrec
    if( !atrousrec( temp.elem(3), *shearDict.g0(), *shearDict.g1(), shearDict.numScales(), gpuImage, temp.elem(0)) )
    {
        std::cerr << "Shearcuda::atrousrec() failed in Shearcuda::inverseShear()" << std::endl;
        return false;
    }

//    g_Matlab.addToMat(&gpuImage, "image");
//    g_Matlab.addToMat(&shearDict, "cshear");
//    g_Matlab.addToMat(shearDict.h0(), "h0");
//    g_Matlab.addToMat(shearDict.h1(), "h1");
//    g_Matlab.addToMat(shearDict.g0(), "g0");
//    g_Matlab.addToMat(shearDict.g1(), "g1");
//    g_Matlab.addToMat(temp.elem(3), "y0");
//    g_Matlab.addToMat(temp.elem(4), "y1");
//    g_Matlab.addToMat(temp.elem(5), "y2");
//    g_Matlab.addToMat(temp.elem(6), "y3");
//    g_Matlab.addToMat(temp.elem(7), "y4");
//    g_Matlab.addToMat(&coeff, "coeff");
//    g_Matlab.writeMat("/scratch0/github/ShearCuda/shearlet_toolbox_1/debug.mat");

    return true;
}

// A trous decomposition
// INPUTS: x, h0, h1, level
// OUTPUT: y (must be an array of length level+1)
bool Shearcuda::atrousdec(const GpuImage& x, const GpuImage& h0, const GpuImage& h1, int numLevels,
                          GpuImage* y, GpuImage* tempBuffer) const
{
    // Get data size and dimensions
    int inputRows = x.height();
    int inputCols = x.width();
    gpuTYPE_t type_signal = x.type();
    void* d_InputImage = x.getPtr();

    // Get pointers to temporary buffers
    void* d_Temp = tempBuffer[0].getPtr();
    void* d_TempSubsampled = tempBuffer[1].getPtr();

    // Get length of analysis filters
    const int len_h0 = h0.width();
    const int len_h1 = h1.width();
    void* d_h0 = h0.getPtr();
    void* d_h1 = h1.getPtr();

    //// % First level
    //// shift = [1, 1]; % delay compensation
    //// y1 = conv2(symext(x,h1,shift),h1,'valid');
    int paddedRows = inputRows + len_h1 - 1;
    int paddedCols = inputCols + len_h1 - 1;
    void* d_y1 = y[numLevels].getPtr();
    g_ShearFunc.symExt( d_Temp, paddedRows, paddedCols, d_InputImage, inputRows, inputCols, len_h1/2, len_h1/2, type_signal );
    g_ShearFunc.atrousConvolutionDevice( d_TempSubsampled, d_y1, inputRows, inputCols, d_Temp, paddedRows, paddedCols, 1, d_h1, len_h1, type_signal );

    //// y0 = conv2(symext(x,h0,shift),h0,'valid');
    paddedRows = inputRows + len_h0 - 1;
    paddedCols = inputCols + len_h0 - 1;
    g_ShearFunc.symExt( d_Temp, paddedRows, paddedCols, d_InputImage, inputRows, inputCols, len_h0/2, len_h0/2, type_signal );
    void* d_y0 = y[0].getPtr();
    g_ShearFunc.atrousConvolutionDevice( d_TempSubsampled, d_y0, inputRows, inputCols, d_Temp, paddedRows, paddedCols, 1, d_h0, len_h0, type_signal );

    for( int i = 0; i < numLevels-1; i++ )
    {
        int shift = 1 - (1<<i);
        int m = 2<<i;

        void* d_x = d_y0;

        //// y1 = cuda_atrousc(symext(x,upsample2df(h1,i),shift),h1,I2 * L,'h1');
        //// y{Nlevels-i+1} = y1;
        paddedRows = inputRows + m * len_h1 - 1;
        paddedCols = inputCols + m * len_h1 - 1;
        int offsetRows = (m * len_h1)/2 - shift;
        int offsetCols = (m * len_h1)/2 - shift;
        d_y1 = y[numLevels-1-i].getPtr();
        g_ShearFunc.symExt( d_Temp, paddedRows, paddedCols, d_x, inputRows, inputCols, offsetRows, offsetCols, type_signal );
        g_ShearFunc.atrousConvolutionDevice( d_TempSubsampled, d_y1, inputRows, inputCols, d_Temp, paddedRows, paddedCols, m, d_h1, len_h1, type_signal );

        //// y0 = cuda_atrousc(symext(x,upsample2df(h0,i),shift),h0,I2 * L,'h0');
        //// x=y0;
        paddedRows = inputRows + m * len_h0 - 1;
        paddedCols = inputCols + m * len_h0 - 1;
        offsetRows = (m * len_h0)/2 - shift;
        offsetCols = (m * len_h0)/2 - shift;
        g_ShearFunc.symExt( d_Temp, paddedRows, paddedCols, d_x, inputRows, inputCols, offsetRows, offsetCols, type_signal );
        g_ShearFunc.atrousConvolutionDevice( d_TempSubsampled, d_y0, inputRows, inputCols, d_Temp, paddedRows, paddedCols, m, d_h0, len_h0, type_signal );
    }

    return true;
}

// A trous reconstruction
// INPUTS: y, g0, g1, level
// OUTPUT: outputImage
bool Shearcuda::atrousrec( const GpuImage* y, const GpuImage& g0, const GpuImage& g1, int numLevels,
               GpuImage& outputImage, GpuImage* tempBuffer) const
{
    // Get data size and dimensions
    int inputRows = y[0].height();
    int inputCols = y[0].width();
    gpuTYPE_t type_signal = y[0].type();
    //std::vector<void*> y_ptr( numLevels + 1 );
    //for( int i = 0; i < numLevels + 1; i++ )
    //    y_ptr[i] = const_cast<void*>( gm->gputype.getGPUptr( y[i] ));

    // Calculate dimensions and check temporary buffer
    int temp_size = (2*inputRows) * (2*inputCols);
    void* d_AtrousTemp = tempBuffer[0].getPtr();
    void* d_Temp = tempBuffer[1].getPtr();
    void* d_TempSubsampled = tempBuffer[2].getPtr();

    if( tempBuffer[0].numElements() < temp_size ||
        tempBuffer[0].numElements() < temp_size ||
        tempBuffer[0].numElements() < temp_size )
    {
        return false;
    }

    // Allocate results pointer
    if( outputImage.numElements() < inputRows * inputCols )
        return false;
    void* outputImage_ptr = outputImage.getPtr();

    // Get length of synthesis filters
    const int len_g0 = g0.width();
    const int len_g1 = g1.width();
    void* d_g0 = g0.getPtr();
    void* d_g1 = g1.getPtr();

    //// % First Nlevels - 1 levels
    void* d_x = y[0].getPtr();
    for( int i = numLevels - 2; i >= 0; i-- )
    {
        int shift = 1 - (1<<i);
        int m = 2<<i;

        void* d_y1 = y[numLevels-1-i].getPtr();

        //// x = cuda_atrousc(symext(x,upsample2df(g0,i),shift),g0,L*I2,'g0') + cuda_atrousc(symext(y1,upsample2df(g1,i),shift),g1,L*I2,'g1');
        int paddedRows = inputRows + m * len_g1 - 1;
        int paddedCols = inputCols + m * len_g1 - 1;
        int offsetRows = (m * len_g1)/2 - shift;
        int offsetCols = (m * len_g1)/2 - shift;
        g_ShearFunc.symExt( d_Temp, paddedRows, paddedCols, d_y1, inputRows, inputCols, offsetRows, offsetCols, type_signal );
        g_ShearFunc.atrousConvolutionDevice( d_TempSubsampled, d_AtrousTemp, inputRows, inputCols, d_Temp, paddedRows, paddedCols, m, d_g1, len_g1, type_signal );

        paddedRows = inputRows + m * len_g0 - 1;
        paddedCols = inputCols + m * len_g0 - 1;
        offsetRows = (m * len_g0)/2 - shift;
        offsetCols = (m * len_g0)/2 - shift;
        g_ShearFunc.symExt( d_Temp, paddedRows, paddedCols, d_x, inputRows, inputCols, offsetRows, offsetCols, type_signal );
        d_x = outputImage_ptr;
        g_ShearFunc.atrousConvolutionDevice( d_TempSubsampled, d_x, inputRows, inputCols, d_Temp, paddedRows, paddedCols, m, d_g0, len_g0, type_signal );
        g_ShearFunc.addVector( d_x, d_AtrousTemp, inputRows * inputCols, type_signal );
    }

    //// % Reconstruct first level
    //// x = conv2(symext(x,g0,shift),g0,'valid')+ conv2(symext(y{Nlevels+1},g1,shift),g1,'valid');
    int paddedRows = inputRows + len_g1 - 1;
    int paddedCols = inputCols + len_g1 - 1;
    g_ShearFunc.symExt( d_Temp, paddedRows, paddedCols, y[numLevels].getPtr(), inputRows, inputCols, len_g1/2, len_g1/2, type_signal );
    g_ShearFunc.atrousConvolutionDevice( d_TempSubsampled, d_AtrousTemp, inputRows, inputCols, d_Temp, paddedRows, paddedCols, 1, d_g1, len_g1, type_signal );

    paddedRows = inputRows + len_g0 - 1;
    paddedCols = inputCols + len_g0 - 1;
    g_ShearFunc.symExt( d_Temp, paddedRows, paddedCols, d_x, inputRows, inputCols, len_g0/2, len_g0/2, type_signal );
    g_ShearFunc.atrousConvolutionDevice( d_TempSubsampled, d_x, inputRows, inputCols, d_Temp, paddedRows, paddedCols, 1, d_g0, len_g0, type_signal );

    g_ShearFunc.addVector( d_x, d_AtrousTemp, inputRows * inputCols, type_signal );

    return true;
}

// Perform convolution
bool Shearcuda::convolution(const GpuImage* y, const GpuShearDictionary& shear, GpuImage* tempBuffer, GpuImage* d ) const
{
    // Calculate dimensions and allocate temporary buffer
    gpuTYPE_t type_signal = shear.dataType();
    gpuTYPE_t type_real = (type_signal == gpuCFLOAT ? gpuFLOAT : gpuDOUBLE );
    int filterLen = shear.filterLen();
    // Make sure that temp buffers are large enough
    if( tempBuffer[0].numElements() < filterLen * filterLen * 2 * shear.maxDirections() ||
        tempBuffer[1].numElements() < filterLen * filterLen * 2 )
        return false;
    // Fourier transform of the data
    void* d_DataSpectrum = tempBuffer[1].getPtr();
    // Fourier transform of the product
    void* d_TempSpectrum = tempBuffer[0].getPtr();

    // Dimensions for output matrices
    int dims[3];
    dims[0] = filterLen;
    dims[1] = filterLen;

    // Process scales
    for( int scale = 0; scale < shear.numScales(); scale++ )
    {
        int numDir = shear.numDirections(scale);

        // Get pointer to input
        const void* y_ptr = y[scale+1].getPtr();

        // Convert input to complex (we will use d_TempSpectrum as temporary storage)
        g_ShearFunc.realToComplex( d_TempSpectrum, y_ptr, filterLen * filterLen, type_signal );

        // Allocate output and get pointer
        dims[2] = numDir;
        if( d[scale+1].numElements() != dims[0] * dims[1] * dims[2] )
            d[scale+1].allocate( type_signal, 3, dims );
        void* d_ptr = d[scale+1].getPtr();

        g_ShearFunc.timer()->startTimer( GpuTimes::fftFwd );
        if( type_signal == gpuCFLOAT )
            cufftSafeCall( cufftExecC2C(shear.fftPlanOne(), (cufftComplex *)d_TempSpectrum, (cufftComplex *)d_DataSpectrum, CUFFT_FORWARD ));
        else
            cufftSafeCall( cufftExecZ2Z(shear.fftPlanOne(), (cufftDoubleComplex *)d_TempSpectrum, (cufftDoubleComplex *)d_DataSpectrum, CUFFT_FORWARD ));
        g_ShearFunc.timer()->stopTimer( GpuTimes::fftFwd );

        g_ShearFunc.modulateAndNormalizeMany( d_TempSpectrum, d_DataSpectrum, shear.filters(scale)->getPtr(), filterLen, filterLen, numDir, type_signal );

        g_ShearFunc.timer()->startTimer( GpuTimes::fftInv );
#if CUDART_VERSION >= 4000
        if( type_signal == gpuCFLOAT )
            cufftSafeCall( cufftExecC2C(shear.fftPlanMany(scale), (cufftComplex *)d_TempSpectrum, (cufftComplex *)d_ptr, CUFFT_INVERSE ));
        else
            cufftSafeCall( cufftExecZ2Z(shear.fftPlanMany(scale), (cufftDoubleComplex *)d_TempSpectrum, (cufftDoubleComplex *)d_ptr, CUFFT_INVERSE ));
#else
        for( int dir = 0; dir < numDir; dir++ )
        {
            int offset = dir * dims[0] * dims[1];
            if( type_signal == gpuCFLOAT )
                cufftSafeCall( cufftExecC2C(shear.fftPlanOne(), (cufftComplex *)d_TempSpectrum + offset, (cufftComplex *)d_ptr + offset, CUFFT_INVERSE ));
            else
                cufftSafeCall( cufftExecZ2Z(shear.fftPlanOne(), (cufftDoubleComplex *)d_TempSpectrum + offset, (cufftDoubleComplex *)d_ptr + offset, CUFFT_INVERSE ));
        }
#endif
        g_ShearFunc.timer()->stopTimer( GpuTimes::fftInv );

    }

    // Allocate base scale and transfer output
    if( d[0].numElements() != dims[0] * dims[1] )
        d[0].allocate( type_real, 2, dims );
    g_ShearFunc.timer()->startTimer( GpuTimes::cudaMemcpy );
    cutilSafeCall( cudaMemcpy( d[0].getPtr(), y[0].getPtr(), y[0].size(), cudaMemcpyDeviceToDevice ));
    g_ShearFunc.timer()->stopTimer( GpuTimes::cudaMemcpy );

    return true;
}

// Get Lp norm of a vector (Note: this function assumes that numElem > 1)
double Shearcuda::getNorm( const void* image_ptr, void* scratch_buf, double p, int numElem, gpuTYPE_t type_signal ) const
{
    int elem_size_real = (type_signal == gpuDOUBLE || type_signal == gpuCDOUBLE  ?
                          sizeof(double) : sizeof(float));

    const void* buffer_src = image_ptr;
    void* buffer_dst = scratch_buf;

    // The first step is calculate the norm of each element and reduce sum
    g_ShearFunc.reduceNorm256( buffer_dst, buffer_src, p, numElem, type_signal );

    buffer_src = buffer_dst;
    numElem = iDivUp(numElem, 256);
    buffer_dst = (char*)buffer_dst + numElem * elem_size_real;

    // We know that reduceNorm256 returns always real values
    if( type_signal == gpuCDOUBLE )
        type_signal = gpuDOUBLE;
    if( type_signal == gpuCFLOAT )
        type_signal = gpuFLOAT;

    // Subsequent reduction steps only involve summations
    while( numElem > 1 )
    {
        g_ShearFunc.reduceSum256( buffer_dst, buffer_src, numElem, type_signal );
        buffer_src = buffer_dst;
        numElem = iDivUp(numElem, 256);
        buffer_dst = (char*)buffer_dst + numElem * elem_size_real;
    }

    // Transfer result
    if( type_signal == gpuDOUBLE )
    {
        double h_Norm;
        cutilSafeCall( cudaMemcpy( &h_Norm, buffer_src, sizeof(h_Norm), cudaMemcpyDeviceToHost ));
        return pow( h_Norm, 1.0 / p );
    }
    else
    {
        float h_Norm;
        cutilSafeCall( cudaMemcpy( &h_Norm, buffer_src, sizeof(h_Norm), cudaMemcpyDeviceToHost ));
        return pow( double(h_Norm), 1.0 / p );
    }
}

// Get Lp norm of the difference between 2 vectors (Note: this function assumes that numElem > 1)
double Shearcuda::getNormErr( const void* imageA_ptr, const void* imageB_ptr, void* scratch_buf, double p, int numElem, gpuTYPE_t type_signal ) const
{
    int elem_size_real = (type_signal == gpuDOUBLE || type_signal == gpuCDOUBLE  ?
                          sizeof(double) : sizeof(float));

    const void* buffer_src = imageA_ptr;
    const void* buffer_srcB = imageB_ptr;
    void* buffer_dst = scratch_buf;

    // The first step is calculate the norm of each element and reduce sum
    g_ShearFunc.reduceNormErr256( buffer_dst, buffer_src, buffer_srcB, p, numElem, type_signal );

    buffer_src = buffer_dst;
    numElem = iDivUp(numElem, 256);
    buffer_dst = (char*)buffer_dst + numElem * elem_size_real;

    // We know that reduceNormErr256 returns always real values
    if( type_signal == gpuCDOUBLE )
        type_signal = gpuDOUBLE;
    if( type_signal == gpuCFLOAT )
        type_signal = gpuFLOAT;

    // Subsequent reduction steps only involve summations
    while( numElem > 1 )
    {
        g_ShearFunc.reduceSum256( buffer_dst, buffer_src, numElem, type_signal );
        buffer_src = buffer_dst;
        numElem = iDivUp(numElem, 256);
        buffer_dst = (char*)buffer_dst + numElem * elem_size_real;
    }

    // Transfer result
    if( type_signal == gpuDOUBLE )
    {
        double h_Norm;
        cutilSafeCall( cudaMemcpy( &h_Norm, buffer_src, sizeof(h_Norm), cudaMemcpyDeviceToHost ));
        return pow( h_Norm, 1.0 / p );
    }
    else
    {
        float h_Norm;
        cutilSafeCall( cudaMemcpy( &h_Norm, buffer_src, sizeof(h_Norm), cudaMemcpyDeviceToHost ));
        return pow( double(h_Norm), 1.0 / p );
    }
}

// apply directional shearlet filters to decomposed images for each scale
bool Shearcuda::applyfilters(const GpuImage* d, const GpuShearDictionary& shear, GpuImage* tempBuffer, GpuImage* y ) const
{
    // Calculate dimensions and check temporary buffer
    gpuTYPE_t type_signal = shear.dataType();
    gpuTYPE_t type_real = (type_signal == gpuCFLOAT ? gpuFLOAT : gpuDOUBLE );
    int filterLen = shear.filterLen();
    int elem_size = (type_signal == gpuCFLOAT ? sizeof(float) : sizeof(double));
    // Make sure that temp buffers are large enough
    if( tempBuffer[0].numElements() < filterLen * filterLen * 2 * shear.maxDirections() ||
        tempBuffer[1].numElements() < filterLen * filterLen * 2 * shear.maxDirections() )
        return false;
    void* d_DataSpectrum = tempBuffer[0].getPtr();
    void* d_Temp = tempBuffer[1].getPtr();

    // Allocate output image components
    int dims[2];
    dims[0] = filterLen;
    dims[1] = filterLen;

    for( int scale = 0; scale < shear.numScales(); scale++ )
    {
        int numDir = shear.numDirections(scale);

        // Get pointer to input
        const void* d_ptr = d[scale+1].getPtr();

        // Check size and zero out image component
        if( y[scale+1].width() != filterLen || y[scale+1].height() != filterLen )
            return false;
        //y[scale+1] = gm->gputype.create( type_signal, 2, dims, NULL );
        void* y_ptr = y[scale+1].getPtr();

        // Filter all directions
        if( type_signal == gpuCFLOAT )
        {
            g_ShearFunc.timer()->startTimer( GpuTimes::fftFwd );
#if CUDART_VERSION >= 4000
            cufftSafeCall( cufftExecC2C( shear.fftPlanMany(scale), (cufftComplex *)d_ptr, (cufftComplex *)d_DataSpectrum, CUFFT_FORWARD ));
#else
            for( int dir = 0; dir < numDir; dir++ )
            {
                int offset = dir * filterLen * filterLen;
                cufftSafeCall( cufftExecC2C( shear.fftPlanOne(), (cufftComplex *)d_ptr + offset, (cufftComplex *)d_DataSpectrum + offset, CUFFT_FORWARD ));
            }
#endif
            g_ShearFunc.timer()->stopTimer( GpuTimes::fftFwd );
            g_ShearFunc.modulateConjAndNormalize( d_DataSpectrum, shear.filters(scale)->getPtr(), filterLen, filterLen, numDir, 1, type_signal);
            g_ShearFunc.timer()->startTimer( GpuTimes::fftInv );
#if CUDART_VERSION >= 4000
            cufftSafeCall( cufftExecC2C( shear.fftPlanMany(scale), (cufftComplex *)d_DataSpectrum, (cufftComplex *)d_Temp, CUFFT_INVERSE ));
#else
            for( int dir = 0; dir < numDir; dir++ )
            {
                int offset = dir * filterLen * filterLen;
                cufftSafeCall( cufftExecC2C( shear.fftPlanOne(), (cufftComplex *)d_DataSpectrum + offset, (cufftComplex *)d_Temp + offset, CUFFT_INVERSE ));
            }
#endif
            g_ShearFunc.timer()->stopTimer( GpuTimes::fftInv );
        }
        else
        {
            g_ShearFunc.timer()->startTimer( GpuTimes::fftFwd );
#if CUDART_VERSION >= 4000
            cufftSafeCall( cufftExecZ2Z( shear.fftPlanMany(scale), (cufftDoubleComplex *)d_ptr, (cufftDoubleComplex *)d_DataSpectrum, CUFFT_FORWARD ));
#else
            for( int dir = 0; dir < numDir; dir++ )
            {
                int offset = dir * filterLen * filterLen;
                cufftSafeCall( cufftExecZ2Z( shear.fftPlanOne(), (cufftDoubleComplex *)d_ptr + offset, (cufftDoubleComplex *)d_DataSpectrum + offset, CUFFT_FORWARD ));
            }
#endif
            g_ShearFunc.timer()->stopTimer( GpuTimes::fftFwd );
            g_ShearFunc.modulateConjAndNormalize( d_DataSpectrum, shear.filters(scale)->getPtr(), filterLen, filterLen, numDir, 1, type_signal);
            g_ShearFunc.timer()->startTimer( GpuTimes::fftInv );
#if CUDART_VERSION >= 4000
            cufftSafeCall( cufftExecZ2Z( shear.fftPlanMany(scale), (cufftDoubleComplex *)d_DataSpectrum, (cufftDoubleComplex *)d_Temp, CUFFT_INVERSE ));
#else
            for( int dir = 0; dir < numDir; dir++ )
            {
                int offset = dir * filterLen * filterLen;
                cufftSafeCall( cufftExecZ2Z( shear.fftPlanOne(), (cufftDoubleComplex *)d_DataSpectrum + offset, (cufftDoubleComplex *)d_Temp + offset, CUFFT_INVERSE ));
            }
#endif
            g_ShearFunc.timer()->stopTimer( GpuTimes::fftInv );
        }
        // Convert data from complex to real (result goes to d_DataSpectrum)
        g_ShearFunc.complexToReal( d_DataSpectrum, d_Temp, filterLen * filterLen * numDir, type_signal );

        // Add all components together
        g_ShearFunc.sumVectors( y_ptr, d_DataSpectrum, filterLen * filterLen, shear.numDirections(scale), type_real);
    }

    // Allocate base scale and transfer output
    if( y[0].width() != filterLen || y[0].height() != filterLen )
        return false;
    const void* d0_ptr = d[0].getPtr();
    void* y0_ptr = y[0].getPtr();
    g_ShearFunc.timer()->startTimer( GpuTimes::cudaMemcpy );
    cutilSafeCall( cudaMemcpy( y0_ptr, d0_ptr, filterLen * filterLen * elem_size, cudaMemcpyDeviceToDevice ));
    g_ShearFunc.timer()->stopTimer( GpuTimes::cudaMemcpy );

    return true;
}
