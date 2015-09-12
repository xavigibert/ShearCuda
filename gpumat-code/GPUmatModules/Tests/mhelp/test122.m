function test122
X = rand(10,GPUsingle);
R = round(X)
X = rand(10,GPUdouble);
R = round(X)
