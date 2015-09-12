function test40
X = rand(10,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUcosh(X, R)
