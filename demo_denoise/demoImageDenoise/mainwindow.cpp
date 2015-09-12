#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "imageview.h"
#include "shearcuda.h"
#include "shearconfig.h"
#include "gpushearfilters.h"
#include "gpusheardictionary.h"
#include "gpucelldata.h"
#include <iostream>
#include <QLabel>
#include <QGridLayout>
#include <math.h>
#include <QTime>

// TEMPORARY CODE
//#include "matlab.h"

using namespace std;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    m_pShearcuda(new Shearcuda),
    m_bDetailedTiming(false)
{
    ui->setupUi(this);
    this->setWindowTitle(QString("UMD Image denoising using Shearlets"));

    // We remove the status bar and toolbar since they are not currently used
    ui->statusBar->setVisible(false);
    ui->mainToolBar->setVisible(false);

    resize(512*3+50,512+50);

    // Initialize GUI
    m_pImageWidget = new ImageView;
    m_pNoisyWidget = new ImageView;
    m_pResultWidget = new ImageView;

    m_pImageLabel = makeNewLabel();
    m_pNoisyLabel = makeNewLabel();
    m_pResultLabel = makeNewLabel();

    // Place all widgets in the layout
    QGridLayout* layoutViews = new QGridLayout();
    layoutViews->setSpacing( 5 );
    layoutViews->addLayout( makeVBox( m_pImageLabel, m_pImageWidget ), 0, 0 );
    layoutViews->addLayout( makeVBox( m_pNoisyLabel, m_pNoisyWidget ), 0, 1 );
    layoutViews->addLayout( makeVBox( m_pResultLabel, m_pResultWidget ), 0, 2 );

    // Set layout
    ui->centralWidget->setLayout( layoutViews );
}

MainWindow::~MainWindow()
{
    delete ui;
    delete m_pShearcuda;
}

// Function that puts 2 widgets and a spacer on a QVBoxLayout
QVBoxLayout* MainWindow::makeVBox( QWidget* widget0, QWidget* widget1 )
{
    QVBoxLayout* box = new QVBoxLayout;
    box->setSpacing(0);
    box->addWidget( widget0 );
    widget1->setSizePolicy( QSizePolicy::Expanding, QSizePolicy::Expanding );
    box->addWidget( widget1 );
    //box->addItem( new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Expanding) );

    return box;
}

// Function that returns a new label
QLabel* MainWindow::makeNewLabel( Qt::Alignment flag, QString str )
{
    QLabel* label = new QLabel( str );
    label->setAlignment( flag );

    return label;
}

// Run demo
bool MainWindow::process()
{
    // Read input file
    if( !m_Image.load(m_ImageFile) )
    {
        cerr << "Cannot open image file " << (const char*)m_ImageFile.toAscii() << endl;
        return false;
    }

    // Display images
    m_pImageWidget->setImage(m_Image);

    // Initialize CUDA
    if( !m_pShearcuda->init() )
    {
        cerr << "Failed to initialize CUDA environment" << endl;
        return false;
    }

    // Load filters from file
    ShearConfig config;
    if( !config.readFromFile( m_FiltersFile ) )
    {
        cerr << "Failed to read shearlet configuration from file " << (const char*)m_FiltersFile.toAscii() << endl;
        return false;
    }

    // Run demo
    QTime t;
    t.start();
    if( !shearDenoise(config, m_Image, m_NoisyImage, m_ResultImage) )
    {
        cerr << "Failed to denoise image " << endl;
    }
    qDebug("Time elapsed: %d ms", t.elapsed());

    // Shutdown cuda
    m_pShearcuda->shutdown();

    // Display results
    m_pNoisyWidget->setImage( m_NoisyImage );
    m_pResultWidget->setImage( m_ResultImage );
    m_pImageLabel->setText( QString("Original image"));
    m_pNoisyLabel->setText( QString("Noisy image; PSNR = %1").arg(m_dNoisyPsnr,0,'f',2));
    m_pResultLabel->setText( QString("Reconstructed image; PSNR = %1").arg(m_dResultPsnr,0,'f',2));

    return true;
}

bool MainWindow::shearDenoise(ShearConfig& config, QImage& image, QImage& noisyImage, QImage& result)
{
    if( image.width() != image.height() )
    {
        cerr << "Non-square images are not supported" << endl;
        return false;
    }

    // Make sure that image is 8-bit gray
    QVector<QRgb> colorTable(256);
    for( int i = 0; i < 256; i++ ) colorTable[i] = qRgb(i,i,i);
//    cout << "image.isGrayscale() = " << image.isGrayscale() << endl;
//    cout << "image.format() = " << image.format() << endl;

    if( !image.isGrayscale() || image.format() != QImage::Format_Indexed8 )
    {
        if( image.format() == QImage::Format_RGB32)
        {
            QImage conv = QImage( image.width(), image.height(), QImage::Format_Indexed8 );
            unsigned char* pDst = conv.bits();
            QRgb* pSrc = (QRgb*)image.bits();
            int num = image.width() * image.height();
            for( int i = 0; i < num; i++ )
                pDst[i] = (unsigned char)qGray(pSrc[i]);
            image = conv;
        }
        else
            image.convertToFormat( QImage::Format_Indexed8, colorTable );
    }

    // Transfer filters to GPU
    GpuShearFilters gpuFilters(
                    config.version(), config.dataType(), config.numScales(),
                    config.sizeShearing(), config.numDir(), config.filters(),
                    config.atrousFilters(), config.atrousLen()
                );

    // Prepare filters for the given image size
    GpuShearDictionary gpuShear;
    gpuShear.prepare( &gpuFilters, image.width() );

    // Allocate temporary buffers
    GpuCellData gpuTemp;
    gpuShear.prepareTempBuffers( gpuTemp );

    // Determines via Monte Carlo the standard deviation of
    // the white Gaussian noise for each scale and
    // directional component when a white Gaussian noise of
    // standard deviation of 1 is feed through.
    m_pShearcuda->computeShearNorm( gpuShear, gpuTemp );
//    double* E = gpuShear.norm();
//    int ns = gpuShear.numScales() + 1;
//    int nd = gpuShear.maxDirections();
//    std::cout.precision(3);
//    for( int idxScale = 0; idxScale < ns; idxScale++ )
//    {
//        for( int idxDir = 0; idxDir < nd; idxDir++ )
//            std::cout << E[idxScale + idxDir * ns ] << "  ";
//        std::cout << std::endl;
//    }

    // Allocate memory for result and perform reconstruction
    GpuImage gpuResult( config.dataType(), image.width(), image.height() );

    // Transfer image to GPU
    GpuImage gpuImage( gpuUINT8, image.width(), image.height() );
    gpuImage.transferToDevice( image.bits() );

    // Convert image to same type as filters
    gpuImage.convert( config.dataType() );

    // Perform shearlet transformation
    GpuCellData gpuCoeff;      // Shearlet coefficients

    // Add noise
    GpuImage gpuNoisyImage( gpuImage );
    m_pShearcuda->addNoise( gpuNoisyImage, m_dSigma );

    // Calculate noise PSNR
    double noisyMse;
    m_dNoisyPsnr = calculatePsnr( gpuNoisyImage, gpuImage, gpuTemp.elem(0)->getPtr(), noisyMse );

    // Denoise image
    QTime t;
    cutilDeviceSynchronize();
    if( m_bDetailedTiming )
        m_pShearcuda->startTimers();
    t.start();
    m_pShearcuda->shearTrans( gpuNoisyImage, gpuShear, gpuTemp, gpuCoeff );

    //Tscalars determine the thresholding multipliers for
    //standard deviation noise estimates. Tscalars(1) is the
    //threshold scalar for the low-pass coefficients, Tscalars(2)
    //is the threshold scalar for the band-pass coefficients,
    //Tscalars(3) is the threshold scalar for the high-pass
    //coefficients.
    double Tscalars[3] = {0, 3, 4};

    m_pShearcuda->hardThreshold( gpuCoeff, m_dSigma, gpuShear.norm(), Tscalars );

    // Perform reconstruction
    m_pShearcuda->inverseShear( gpuCoeff, gpuShear, gpuTemp, gpuResult );
    cutilDeviceSynchronize();
    m_pShearcuda->stopTimers();
    qDebug("Denoise time elapsed: %d ms", t.elapsed());

    // Calculate resulting PSNR
    double resultMse;
    m_dResultPsnr = calculatePsnr( gpuResult, gpuImage, gpuTemp.elem(0)->getPtr(), resultMse );

    // Get noisy image for display
    noisyImage = QImage( image.width(), image.height(), QImage::Format_Indexed8 );
    noisyImage.setColorTable(colorTable);
    gpuNoisyImage.convert( gpuUINT8 );
    gpuNoisyImage.transferFromDevice( noisyImage.bits() );

    // Get results for display
    result = QImage( image.width(), image.height(), QImage::Format_Indexed8 );
    result.setColorTable(colorTable);
    gpuResult.convert( gpuUINT8 );
    gpuResult.transferFromDevice( result.bits() );

    // Report results
    std::cout.precision(4);
    std::cout << "Noisy image PSNR = " << m_dNoisyPsnr << " dB (MSE = " << noisyMse << ")" << std::endl;
    std::cout << "Denoised image PSNR = " << m_dResultPsnr << " dB (MSE = " << resultMse << ")" << std::endl;

    return true;
}

double MainWindow::calculatePsnr( QImage& im1, QImage& im2, double& mse )
{
    // Assume both images are 8-bit unsigned and have same dimensions
    const unsigned char* p1 = im1.bits();
    const unsigned char* p2 = im2.bits();
    int num = im1.width() * im2.height();
    double sqsum = 0.0;
    for( int i = 0; i < num; i++ )
    {
        double diff = double(p1[i]) - double(p2[i]) ;
        sqsum += diff * diff;
    }

    mse = sqsum / num;
    return 10 * log10(255.0 * 255.0 * num / sqsum);
}

double MainWindow::calculatePsnr( GpuImage& im1, GpuImage& im2, void* scratch_buf, double& mse )
{
    // Assume both images are 8-bit unsigned and have same dimensions
    int num = im1.numElements();
    double normDiff = m_pShearcuda->getNormErr( im1.getPtr(), im2.getPtr(), scratch_buf, 2.0, im1.numElements(), im1.type() );
    double sqsum = normDiff * normDiff;
    mse = sqsum / num;
    return 10 * log10(255.0 * 255.0 * num / sqsum);
}
