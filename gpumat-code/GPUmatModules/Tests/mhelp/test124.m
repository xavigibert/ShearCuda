function test124
X = rand(10,GPUsingle);
R = sinh(X)
X = rand(10,GPUdouble);
R = sinh(X)
