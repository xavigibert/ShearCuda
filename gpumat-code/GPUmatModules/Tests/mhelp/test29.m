function test29
X = rand(10,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUacos(X, R)
