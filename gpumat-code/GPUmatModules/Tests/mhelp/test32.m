function test32
X = rand(10,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUasin(X, R);
