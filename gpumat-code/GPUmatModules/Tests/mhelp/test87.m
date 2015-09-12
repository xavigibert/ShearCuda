function test87
A = GPUsingle();
isempty(A)
A = rand(5,GPUsingle)+i*rand(5,GPUsingle);
isempty(A)

