function test70
X = rand(10,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUsin(X,R)
