function test89
A = rand(5,GPUsingle);
isscalar(A)
A = GPUsingle(1);
isscalar(A)
A = GPUdouble(1);
isscalar(A)
