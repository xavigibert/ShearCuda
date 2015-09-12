NUMERICS MODULE
---------------

- Implemented functions
 * GPUeye
 * eye
 * size
 * subsref
 * subsasgn
 * END
 * slice
 * assign
 * repmat
 * GPUfill
 * memCpyDtoD
 * memCpyHtoD
 * check manual for full reference

- Module initialization
 * Run moduleinit from Matlab

EXAMPLES
--------

Folder:
 * Examples

Functions:
 * IndexedReference
 * SliceAssign
 * GPUfill
 * memCpy
 
COMPILATION
-----------

Include folder '../util' in the MATLAB path.

- make cpp 
Compiles all .cpp files

- make cuda
Compiles .cu files into .cubin modules

- make all 
 
Compiles .cpp and .cu files and copy the compiled 
files to the release folder

TESTING
-------

Run GPUtestInit before running tests

Folder 
 * Tests

Functions
 * test_GPUeye
 * test_eye
 * test_slice
 * test_assign
 * test_repmat
 * test_GPUfill
 * test_memCpyHtoD
 * test_memCpyDtoD

