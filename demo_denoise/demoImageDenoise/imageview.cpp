#include "imageview.h"
#include <QPainter>
#include <qmath.h>

ImageView::ImageView(QWidget *parent) :
    QWidget(parent)
{
}

void ImageView::setImage(QImage& img)
{
    m_SrcImage = img;

    update();
}

// Display event
void ImageView::paintEvent( QPaintEvent* /* event */ )
{
    // Prepare painter
    QPainter painter(this);

    // Do not continue unless there is some data to show
    if( m_SrcImage.width() == 0 )
        return;

    // Calculate optimal display size
    double horResolution = double(size().width()) / m_SrcImage.width();
    double verResolution = double(size().height()) / m_SrcImage.height();
    double resolution = qMin( horResolution, verResolution );
    QSize actualViewSize( qCeil( resolution * m_SrcImage.width()),
                          qCeil( resolution * m_SrcImage.height()) );
    QPoint topLeft( (size().width() - actualViewSize.width()) / 2,
                    (size().height() - actualViewSize.height()) / 2 );

    // Draw image
    QRect target( topLeft, actualViewSize );
    QRect source( 0, 0, m_SrcImage.width(), m_SrcImage.height() );
    //painter.drawImage( target, m_SrcImage, source );
    painter.drawImage( source, m_SrcImage, source );
}
