#-------------------------------------------------
#
# Project created by QtCreator 2013-02-20T12:04:59
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = demoImageDenoise
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    imageview.cpp \
    shearconfig.cpp

HEADERS  += mainwindow.h \
    imageview.h \
    shearconfig.h

FORMS    += mainwindow.ui

INCLUDEPATH += ../shearcuda

INCLUDEPATH += /opt/common/cuda/cudatoolkit-4.2.9/include

unix{
    LIBS += -L/opt/common/cuda/cudatoolkit-4.2.9/lib64 -L/usr/lib64/nvidia -lcublas -lcuda -lcudart -lcufft -lcurand
# Use MATLAB for debugging
#    LIBS += -L/opt/common/cuda/cudatoolkit-4.2.9/lib64 -L/usr/lib64/nvidia -lcublas -lcuda -lcudart -lcufft -lcurand -L/opt/common/matlab-r2012b/bin/glnxa64 -lmat -lmex -lmx -lm

}

Debug:
    LIBS += -L../shearcuda-build-Desktop-Debug -lshearcuda

Release:
    LIBS += -L../shearcuda-build-Desktop-Release -lshearcuda
