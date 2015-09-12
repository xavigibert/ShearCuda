function test47
A = GPUsingle([1 2 0 4]);
B = GPUsingle([1 0 0 4]);
R = zeros(size(B),GPUsingle);
GPUge(A, B, R);

