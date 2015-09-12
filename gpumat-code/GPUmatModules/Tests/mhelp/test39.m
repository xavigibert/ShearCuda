function test39
X = rand(10,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUcos(X, R)
