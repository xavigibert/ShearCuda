function test49
A = rand(10,GPUsingle) + sqrt(-1)*rand(10,GPUsingle);
R = zeros(size(A), GPUsingle);
GPUimag(A, R);
