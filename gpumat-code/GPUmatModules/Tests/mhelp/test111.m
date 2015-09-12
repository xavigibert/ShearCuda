function test111
X = rand(10,GPUsingle);
numel(X)
X = rand(10,GPUdouble);
numel(X)
