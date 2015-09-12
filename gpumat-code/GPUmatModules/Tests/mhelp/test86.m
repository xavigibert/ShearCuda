function test86
A = rand(5,GPUsingle);
iscomplex(A)
A = rand(5,GPUsingle)+i*rand(5,GPUsingle);
iscomplex(A)

