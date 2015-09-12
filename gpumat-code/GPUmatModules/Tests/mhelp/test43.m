function test43
X = rand(1,5,GPUsingle)+i*rand(1,5,GPUsingle);
R = complex(zeros(size(X), GPUsingle));
GPUexp(X, R)
