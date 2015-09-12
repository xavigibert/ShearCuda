#-------------------------------------------------
#
# Project created by QtCreator 2013-01-22T20:16:30
#
#-------------------------------------------------

QT       -= core gui

TARGET = inverse_shear
TEMPLATE = lib

DEFINES += MATLAB_MEX_FILE _GNU_SOURCE UNIX MX_COMPAT_32

SOURCES += ../src_cuda/inverse_shear_cuda.cpp \
    ../src_cuda/atrousrec.cpp \
    ../src_cuda/ShearCudaFunctions.cpp \
    ../../GPUmat/modules/common/GPUmat.cpp \
    ../src_cuda/ShearDictionary.cpp \
    ../src_cuda/GpuTimes.cpp

HEADERS += ../src_cuda/ShearCudaFunctions.h \
    ../src_cuda/shear_cuda_common.h \
    ../src_cuda/ShearDictionary.h \
    ../src_cuda/GpuTimes.h

INCLUDEPATH += /opt/common/matlab-r2012b/extern/include/ \
    /opt/common/cuda/cudatoolkit-4.2.9/include/ \
    ../src_cuda/ \
    ../../GPUmat/modules/include/

QMAKE_CXXFLAGS += -fPIC -fno-omit-frame-pointer -pthread

TARGET_EXT = mexa64

LIBS += -L/opt/common/cuda/cudatoolkit-4.2.9/lib64 -L/usr/lib64/nvidia -lcublas -lcuda -lcudart -lcufft  -L/opt/common/matlab-r2012b/bin/glnxa64 -lmx -lmex -lmat -lm \
    -Wl,--version-script,/opt/common/matlab-r2012b/extern/lib/glnxa64/mexFunction.map -Wl,--no-undefined -Wl,-rpath-link,/opt/common/matlab-r2012b/bin/glnxa64 \
    -o "../inverse_shear_cuda.mexa64"

QMAKE_LN_SHLIB = echo QMAKE_LN_SHLIB >> /dev/null
