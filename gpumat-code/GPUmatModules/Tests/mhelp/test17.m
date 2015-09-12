function test17
A = GPUsingle([1 2 0 4]);
B = GPUsingle([1 0 0 4]);
R = A == B;
single(R)
R = eq(A, B);
single(R)
