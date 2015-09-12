function test102
A = rand(10,GPUsingle);
B = rand(10,GPUsingle);
R = A * B
A = rand(10,GPUdouble);
B = rand(10,GPUdouble);
R = A * B
A = rand(10,GPUsingle)+i*rand(10,GPUsingle);
B = rand(10,GPUsingle)+i*rand(10,GPUsingle);
R = A * B
