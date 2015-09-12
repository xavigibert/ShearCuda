function test133
X = rand(10,GPUsingle);
R = -X
R = uminus(X)
X = rand(10,GPUdouble);
R = -X
R = uminus(X)
