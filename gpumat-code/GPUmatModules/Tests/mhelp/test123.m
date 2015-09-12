function test123
X = rand(10,GPUsingle);
R = sin(X)
X = rand(10,GPUdouble);
R = sin(X)
