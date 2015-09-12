#ifndef IMAGEVIEW_H
#define IMAGEVIEW_H

#include <QWidget>
#include <QImage>

class ImageView : public QWidget
{
    Q_OBJECT
public:
    explicit ImageView(QWidget *parent = 0);
    void setImage(QImage& img);

signals:
    
public slots:

protected:
    // Display events
    void paintEvent( QPaintEvent* event );
    
private:
    QImage m_SrcImage;
};

#endif // IMAGEVIEW_H
