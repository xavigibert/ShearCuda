function test73
X = rand(10,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUtan(X,R)
