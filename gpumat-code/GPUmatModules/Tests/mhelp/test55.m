function test55
X = rand(10,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUlog2(X, R)
