function test114
A = rand(3,4,5,GPUsingle);
B = permute(A,[3 2 1]);
