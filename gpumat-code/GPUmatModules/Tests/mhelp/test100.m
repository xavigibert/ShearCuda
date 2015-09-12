function test100
X = rand(10,GPUsingle);
Y = rand(10,GPUsingle);
R = Y - X
X = rand(10,GPUdouble);
Y = rand(10,GPUdouble);
R = Y - X
