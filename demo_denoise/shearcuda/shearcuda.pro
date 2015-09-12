#-------------------------------------------------
#
# Project created by QtCreator 2013-02-20T12:01:03
#
#-------------------------------------------------

QT       -= core gui

TARGET = shearcuda
TEMPLATE = lib

DEFINES += SHEARCUDA_LIBRARY

SOURCES += shearcuda.cpp \
    gpusheartrans.cpp \
    gpuimage.cpp \
    shearcudafunctions.cpp \
    gputimes.cpp \
    gpushearfilters.cpp \
    gpusheardictionary.cpp \
    gpucelldata.cpp

HEADERS += shearcuda.h\
        shearcuda_global.h \
    datatypes.h \
    cutil_inline_runtime.h \
    gpusheartrans.h \
    gpuimage.h \
    shearcudafunctions.h \
    gputimes.h \
    gpushearfilters.h \
    gpusheardictionary.h \
    gpucelldata.h

INCLUDEPATH += /opt/common/cuda/cudatoolkit-4.2.9/include
# Use MATLAB for debugging
#INCLUDEPATH += /opt/common/cuda/cudatoolkit-4.2.9/include /opt/common/matlab-r2012b/extern/include/

LIBS += -L/opt/common/cuda/cudatoolkit-4.2.9/lib64 -L/usr/lib64/nvidia -lcublas -lcuda -lcudart -lcufft -lcurand
# Use MATLAB for debugging
#LIBS += -L/opt/common/cuda/cudatoolkit-4.2.9/lib64 -L/usr/lib64/nvidia -lcublas -lcuda -lcudart -lcufft -lcurand -L/opt/common/matlab-r2012b/bin/glnxa64 -lmat -lm
