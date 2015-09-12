function test63
A = rand(10,GPUsingle);
B = rand(10,GPUsingle);
R = zeros(size(B), GPUsingle);
GPUplus(A, B, R);
