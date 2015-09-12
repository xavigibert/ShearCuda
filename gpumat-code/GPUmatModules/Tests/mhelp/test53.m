function test53
X = rand(10,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUlog10(X, R)
