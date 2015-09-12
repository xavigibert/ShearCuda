function test41
X = rand(10,GPUsingle)+i*rand(10,GPUsingle);
R = complex(zeros(size(X), GPUsingle));
GPUctranspose(X, R)
