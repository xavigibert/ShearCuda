function test80
X = rand(10,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUuminus(X, R)
