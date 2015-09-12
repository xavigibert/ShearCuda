function test112
A = ones(10,GPUsingle)
B = ones(10, 10,GPUsingle)
C = ones([10 10],GPUsingle)
A = ones(10,GPUdouble)
B = ones(10, 10,GPUdouble)
C = ones([10 10],GPUdouble)
