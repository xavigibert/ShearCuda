function test50
A = rand(10,GPUsingle);
B = rand(10,GPUsingle);
R = zeros(size(B), GPUsingle);
GPUldivide(A, B, R);
