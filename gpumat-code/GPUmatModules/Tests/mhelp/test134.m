function test134
A = zeros(10,GPUsingle)
B = zeros(10, 10,GPUsingle)
C = zeros([10 10],GPUsingle)

A = zeros(10,GPUdouble)
B = zeros(10, 10,GPUdouble)
C = zeros([10 10],GPUdouble)
