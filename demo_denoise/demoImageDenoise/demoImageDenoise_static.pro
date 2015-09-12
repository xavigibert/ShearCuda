#-------------------------------------------------
#
# Project created by QtCreator 2013-02-20T12:04:59
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = demoImageDenoise
TEMPLATE = app
CONFIG += console

SOURCES += main.cpp\
        mainwindow.cpp \
    imageview.cpp \
    shearconfig.cpp \
    ../shearcuda/shearcuda.cpp \
    ../shearcuda/gpuimage.cpp \
    ../shearcuda/shearcudafunctions.cpp \
    ../shearcuda/gputimes.cpp \
    ../shearcuda/gpushearfilters.cpp \
    ../shearcuda/gpusheardictionary.cpp \
    ../shearcuda/gpucelldata.cpp

HEADERS  += mainwindow.h \
    imageview.h \
    shearconfig.h \
    ../shearcuda/shearcuda.h\
    ../shearcuda/shearcuda_global.h \
    ../shearcuda/datatypes.h \
    ../shearcuda/cutil_inline_runtime.h \
    ../shearcuda/gpuimage.h \
    ../shearcuda/shearcudafunctions.h \
    ../shearcuda/gputimes.h \
    ../shearcuda/gpushearfilters.h \
    ../shearcuda/gpusheardictionary.h \
    ../shearcuda/gpucelldata.h \
    ../shearcuda/cuda_common.h

FORMS    += mainwindow.ui

INCLUDEPATH += ../shearcuda $$(CUDA_INCLUDE)
#INCLUDEPATH += /opt/common/matlab-r2012b/extern/include

unix{
    LIBS += -L/usr/lib64/nvidia -L$$(CUDA_LIB) -lcublas -lcuda -lcudart -lcufft -lcurand
# Use MATLAB for debugging
#    LIBS += -L/usr/lib64/nvidia -L$$(CUDA_LIB) -lcublas -lcuda -lcudart -lcufft -lcurand -L/opt/common/matlab-r2012b/bin/glnxa64 -lmx -lmex -lmat -lm \
#            -Wl,--version-script,/opt/common/matlab-r2012b/extern/lib/glnxa64/mexFunction.map -Wl,--no-undefined -Wl,-rpath-link,/opt/common/matlab-r2012b/bin/glnxa64
}
