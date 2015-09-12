function test42
A = GPUsingle([1 2 0 4]);
B = GPUsingle([1 0 0 4]);
R = zeros(size(A), GPUsingle);
GPUeq(A, B, R);
