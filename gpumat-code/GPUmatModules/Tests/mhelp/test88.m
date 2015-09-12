function test88
A = rand(5,GPUsingle);
isreal(A)
A = rand(5,GPUsingle)+i*rand(5,GPUsingle);
isreal(A)

