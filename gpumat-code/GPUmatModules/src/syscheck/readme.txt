CHECK SYSTEM CONFIGURATION

This packages allows the user  to check without starting GPUmat if the
system is properly configured.

- Quick start

Run   the  Matlab   script   'runme.m'  and   report   any  error   to
gp-you@gp-you.org.

>> runme

- Compilation

The  package  contains pre-compiled  kernels  (.cubin)  and mex  files
(.mex*).   If necessary, run  the following  commands to  compile .cpp
files and .cubin files respectively:

>> make cpp
>> make cubin

The .cubin compilation requires the  CUDA bin folder to be included in
your path (nvcc should be available from the command shell).

The .cpp compilation requires  Matlab MEX compilation to be configured
properly. Please run from Matlab command window the following:

>> mex -setup

and follow the instructions.
