function test101
A = rand(10,GPUsingle);
B = A / 5
A = rand(10,GPUdouble);
B = A / 5
