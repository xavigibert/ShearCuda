function test46
X = rand(1,5,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUfloor(X, R)
