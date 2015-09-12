function test69
X = rand(10,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUround(X,R);
