function test130
X = rand(10,GPUsingle);
R = tanh(X)
X = rand(10,GPUdouble);
R = tanh(X)
