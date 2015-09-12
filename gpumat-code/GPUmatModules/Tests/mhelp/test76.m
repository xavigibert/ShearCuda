function test76
X = rand(10,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUtranspose(X, R)
