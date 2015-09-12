function test3
A = GPUsingle([1 3 0 4]);
B = GPUsingle([0 1 10 2]);
R = A & B;
single(R)
