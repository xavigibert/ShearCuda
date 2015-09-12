function test116
A = rand(10,GPUsingle);
B = 2;
R = A .^ B
A = rand(10,GPUsingle)+i*rand(10,GPUsingle);
R = A .^ B
