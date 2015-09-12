function test13
A = rand(1,5,GPUsingle) + i*rand(1,5,GPUsingle);
B = conj(A)
