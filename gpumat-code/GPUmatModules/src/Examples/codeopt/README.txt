CODEOPT MODULE
---------------

- Implemented functions
 * forloop1

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

Folder:
 * Tests

Functions:
 * testforloop1
   Performs tests on forloop1
