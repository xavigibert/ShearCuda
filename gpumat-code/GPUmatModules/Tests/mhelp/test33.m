function test33
X = rand(10,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUasinh(X, R)
