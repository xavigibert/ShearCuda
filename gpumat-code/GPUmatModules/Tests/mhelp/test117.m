function test117
A = rand(10,GPUsingle)
B = rand(10, 10,GPUsingle)
C = rand([10 10],GPUsingle)
A = rand(10,GPUdouble)
B = rand(10, 10,GPUdouble)
C = rand([10 10],GPUdouble)
