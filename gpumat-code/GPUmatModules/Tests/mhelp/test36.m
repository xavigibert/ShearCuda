function test36
X = rand(10,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUceil(X, R)
