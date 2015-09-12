function test129
X = rand(10,GPUsingle);
R = tan(X)
X = rand(10,GPUdouble);
R = tan(X)
