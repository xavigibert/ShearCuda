function test57
X = rand(10,GPUsingle);
Y = rand(10,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUminus(Y, X, R);
