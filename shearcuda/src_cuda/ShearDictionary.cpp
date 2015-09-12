#include "ShearDictionary.h"

// Create empty object
ShearDictionary::ShearDictionary(void)
{
    m_nFilterLen = 0;
    m_type = gpuFLOAT;
    m_fftPlanMany = NULL;
    m_ifftPlanMany = NULL;
    m_fftPlanOne = 0;
    m_gm = NULL;
}

ShearDictionary::~ShearDictionary(void)
{
}

// Get buffer associated with a particular scale
void* ShearDictionary::getBuffer(int scale) const
{
    return const_cast<void*>(m_gm->gputype.getGPUptr( m_GPUtypeData[scale] ));
}

// Return maximum number of directions across all scales
int ShearDictionary::maxDirections() const
{
    int val = 1;

    for( unsigned int i = 0; i < m_anNumDirections.size(); i++ )
    {
        if( m_anNumDirections[i] > val )
            val = m_anNumDirections[i];
    }

    return val;
}

// Create object from MATLAB struct variable.
// mx_shear is a struct containing the following fields:
// * fftPlanMany (array of forward FFT R2C plans, one multiplan per scale)
// * ifftPlanMany (array of inverse FFT C2R plans, one multiplan per scale)
// * fftPlanOne (one FFT R2C plan for a single image)
// * filter (cell array of filters, one cell element per scale)
bool ShearDictionary::loadFromMx(const mxArray* mx_shear, GPUmat* gm)
{
    if( hasData() )
    {
        mexErrMsgTxt( "Method ShearDictionary::loadFromMx() should only be called once" );
        return false;
    }

    // Get fields
    mxArray* mx_filter = mxGetField( mx_shear, 0, "filter" );
    mxArray* mx_fftPlanMany = mxGetField( mx_shear, 0, "fftPlanMany" );
    mxArray* mx_ifftPlanMany = mxGetField( mx_shear, 0, "ifftPlanMany" );
    mxArray* mx_fftPlanOne= mxGetField( mx_shear, 0, "fftPlanOne" );

    // Check size of first element
    mxArray* mx_filter_elem = mxGetCell(mx_filter, 0);

    // Get attributes
    GPUtype firstElem = gm->gputype.getGPUtype(mx_filter_elem);
    m_type = gm->gputype.getType(firstElem);
    if( m_type != gpuCFLOAT && m_type != gpuCDOUBLE )
    {
        mexErrMsgTxt("Filters should be complex of type GPUsingle or GPUdouble");
        return false;
    }
    m_nFilterLen = gm->gputype.getSize(firstElem)[0];

    // Check number of elements
    int numScales = (int)mxGetNumberOfElements(mx_filter);
    m_anNumDirections.resize(numScales);
    m_GPUtypeData.resize(numScales);

    // Check dimensions and get pointers
    for (int j = 0; j < numScales; j++)
    {
        mx_filter_elem = mxGetCell(mx_filter, (mwIndex)j);
        m_GPUtypeData[j] = gm->gputype.getGPUtype(mx_filter_elem);

        // Check filter type
        gpuTYPE_t type = gm->gputype.getType( m_GPUtypeData[j] );
        if( type != m_type )
        {
            mexErrMsgTxt("Filters should be of GPUsingle or GPUdouble and type should be consistent");
            return false;
        }

        // Check filter dimensions
        int numDims = gm->gputype.getNdims(m_GPUtypeData[j]);
        const int *dims = gm->gputype.getSize(m_GPUtypeData[j]);
        if (numDims>2)
            m_anNumDirections[j] = dims[2];
        else
            m_anNumDirections[j] = 1;
        if (dims[1] != (m_nFilterLen/2+1) || dims[0] != m_nFilterLen ) {
            mexErrMsgTxt("Filters should have same dimensions");
            return false;
        }
    }

    // Get pointers to FFT plans
    m_fftPlanMany = (cufftHandle*)mxGetData( mx_fftPlanMany );
    m_ifftPlanMany = (cufftHandle*)mxGetData( mx_ifftPlanMany );
    m_fftPlanOne = *(cufftHandle*)mxGetData( mx_fftPlanOne );

    // Set GPUmat environment
    m_gm = gm;

    return true;
}
