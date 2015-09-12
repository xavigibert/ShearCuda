function test60
A = GPUsingle([1 2 0 4]);
R = zeros(size(A), GPUsingle);
GPUnot(A, R);

