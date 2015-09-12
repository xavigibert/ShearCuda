function test74
X = rand(10,GPUsingle);
R = zeros(size(X), GPUsingle);
GPUtanh(X, R)
