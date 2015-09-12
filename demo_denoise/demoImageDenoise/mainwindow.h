#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QString>

// Forward declarations
namespace Ui {
class MainWindow;
}
class ImageView;
class QLabel;
class QVBoxLayout;
class Shearcuda;
class ShearConfig;
class GpuImage;

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    // Run demo
    bool process();
    bool shearDenoise(ShearConfig& config, QImage& image, QImage& noisyImage, QImage& result);
    double calculatePsnr( QImage& im1, QImage& im2, double& mse );
    double calculatePsnr( GpuImage& im1, GpuImage& im2, void* scratch_buf, double& mse );

    // Command line parameters
    void setImageFile(const char* image_file) { m_ImageFile = image_file; }
    void setFiltersFile(const char* image_file) { m_FiltersFile = image_file; }
    void setSigma(double sigma) { m_dSigma = sigma; }
    void enableTiming(bool val) { m_bDetailedTiming = val; }

protected:
    // Function that puts 2 widgets and a spacer on a QVBoxLayout
    QVBoxLayout* makeVBox( QWidget* widget0, QWidget* widget1 );
    // Function that returns a new label
    QLabel* makeNewLabel( Qt::Alignment flag = Qt::AlignHCenter, QString str = tr("no data") );

private:
    Ui::MainWindow *ui;
    Shearcuda* m_pShearcuda;

    // GUI widgets
    ImageView* m_pImageWidget;
    ImageView* m_pNoisyWidget;
    ImageView* m_pResultWidget;
    QLabel* m_pImageLabel;
    QLabel* m_pNoisyLabel;
    QLabel* m_pResultLabel;

    // Demo settings
    QString m_ImageFile;
    QString m_FiltersFile;
    double m_dSigma;
    bool m_bDetailedTiming;

    // Demo data and results
    QImage m_Image;
    QImage m_NoisyImage;
    QImage m_ResultImage;
    double m_dNoisyPsnr;
    double m_dResultPsnr;
};

#endif // MAINWINDOW_H
