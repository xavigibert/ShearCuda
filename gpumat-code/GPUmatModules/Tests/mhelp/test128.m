function test128
A = GPUsingle([1 2 3 4 5]);
A = GPUdouble([1 2 3 4 5]);
idx = GPUsingle([1 2]);
B = A(idx)
