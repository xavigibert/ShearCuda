function test31
A = GPUsingle([1 3 0 4]);
B = GPUsingle([0 1 10 2]);
R = zeros(size(A), GPUsingle);
GPUand(A, B, R);
