function test28
X = rand(1,5,GPUsingle)+i*rand(1,5,GPUsingle);
R = zeros(size(X),GPUsingle);
GPUabs(X, R)
