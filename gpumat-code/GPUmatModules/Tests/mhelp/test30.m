function test30
X = rand(10,GPUsingle) + 1;
R = zeros(size(X), GPUsingle);
GPUacosh(X, R)
