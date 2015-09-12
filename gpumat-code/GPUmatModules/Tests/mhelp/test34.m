function test34
X = rand(10,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUatan(X, R)
