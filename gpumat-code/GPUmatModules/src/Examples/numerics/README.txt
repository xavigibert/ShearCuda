NUMERICS MODULE
---------------

- Implemented functions
 * mytimes
 * myplus
 * myexp
 * myslice1
 * myslice2

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

- testmyplus, testmytimes
Performs tests on myplus and mytimes functions

- testmyslice
Perform tests on myslice1, myslice2

