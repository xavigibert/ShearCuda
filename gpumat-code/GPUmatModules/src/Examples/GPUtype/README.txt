GPUTYPE MODULE
---------------

- Implemented functions
 * gputype_properties
 * gputype_create1
 * gputype_create2
 * gputype_colon

- Module initialization
 * Run moduleinit from Matlab

COMPILATION
-----------

Include folder '../../util' in the MATLAB path.

- make cpp 
Compiles all .cpp files

- make cuda
Compiles .cu files into .cubin modules

- make all  
Compiles .cpp and .cu files and copy the compiled files to the release
folder

TESTING
-------

Run GPUtestInit before running tests

- runme
Run all the examples


