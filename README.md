ShearCuda
=========

This package contains ShearCuda, a CUDA port of the 2D and 3D Shearlet
transforms. If you use our code, please cite the following paper:

- X. Gibert, V. M. Patel, D. Labate, and R. Chellappa. "Discrete Shearlet
  Transform on GPU with Applications in Anomaly Detection and Denoising."
  EURASIP Journal on Advances in Signal Processing 2014 (1), 1-14.

Available at http://link.springer.com/article/10.1186%2F1687-6180-2014-64


Prerequisites
-------------

This library is based on NVIDIA CUDA and MATLAB. To use this library it is 
necessary to have a device that supports CUDA. The development has been
done in Linux (specifically RHEL 6), but the code should compile without
changes on Windows and Mac OS X.


Environment settings
--------------------

CUDA environment needs to be set up before starting MATLAB. In addition,
you need to define CUDA_PATH to the base path where CUDA is installed. For
example, a script to launch MATLAB in Linux could be:
relies on CUDA_PATH to determine the 

	export PATH=/opt/common/cuda/cuda-5.0.35/bin:$PATH
	export LD_LIBRARY_PATH=/opt/common/cuda/cuda-5.0.35/lib64:/usr/lib64/nvidia
	export LIBRARY_PATH=/usr/lib64/nvidia
	export CUDA_PATH=/opt/common/cuda/cuda-5.0.35
	/opt/common/matlab-r2013a/bin/matlab -desktop


Directory structure
-------------------

The code is organized into the following subdirectories:

* gpumat-code --  Source code of the GPUmat library. After compilation, the
    runtime code is copied into GPUmat.
* shearcuda -- Code for performing the 2D Shearlet transform as well as
    image separation.
* shearcuda3d -- Code for performing the 3D Shearlet transform.
* crack_detection -- Code to run crack detection using image separation.
* demo_denoise -- C++ code to run image denoising.


Compilation instructions
------------------------

ShearCuda relies on GPUmat. The instructions below allow compilation of
both GPUmat and ShearCuda. We first need to tell GPUmat what version of
CUDA it will target:

    edit gpumat-code/compile.m

Select CUDA version by changing `code(CUDAver = '5.0';)` to whatever
version of CUDA is installed. Then, type the following command to build
the whole thing:

    makeall

To avoid error in moduleinit, remove execution flag from NumericsModuleManager

    !chmod -x GPUmat/modules/numerics/NumericsModuleManager.mexa64

To verify that module works, run

    init


Instructions to generate results from SIAM paper
------------------------------------------------

The following scripts and project files reproduce the experiments reported
in the paper:

* demo_denoise/ (2D denoise demo, see section 5.1 of the paper)
* crack_detection/testall.m (crack detection demo, see section 5.2 of the paper)
* shearcuda3d/DenoiseDemo/main.m (3D denoise demo, see section 5.3 of the paper)

Before running any of the MATLAB scripts it is necessary to set up the GPU
by running

    init
