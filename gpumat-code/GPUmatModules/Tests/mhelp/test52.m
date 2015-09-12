function test52
X = rand(10,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUlog(X,R)
