function test72
X = rand(10,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUsqrt(X,R)
