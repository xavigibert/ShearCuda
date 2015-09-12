function test67
A = rand(10,GPUsingle);
B = rand(10,GPUsingle);
R = zeros(size(A), GPUsingle);
GPUrdivide(A, B, R);
