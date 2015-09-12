function test68
A = rand(10,GPUsingle) + sqrt(-1)*rand(10,GPUsingle);
R = zeros(size(A), GPUsingle);
GPUreal(A, R);
