function test118
A = randn(10,GPUsingle)
B = randn(10, 10,GPUsingle)
C = randn([10 10],GPUsingle)
A = randn(10,GPUdouble)
B = randn(10, 10,GPUdouble)
C = randn([10 10],GPUdouble)
