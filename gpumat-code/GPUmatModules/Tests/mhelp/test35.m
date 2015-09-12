function test35
X = rand(10,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUatanh(X, R)
