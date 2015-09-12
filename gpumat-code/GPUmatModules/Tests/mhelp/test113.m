function test113
A = GPUsingle([1 2 0 4]);
B = GPUsingle([1 0 0 4]);
R = A |  B;
single(R)
R = or(A, B);
single(R)
