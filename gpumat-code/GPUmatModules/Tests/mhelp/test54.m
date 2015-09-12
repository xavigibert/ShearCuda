function test54
X = rand(10,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUlog1p(X, R)
