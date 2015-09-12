function test71
X = rand(10,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUsinh(X,R)
