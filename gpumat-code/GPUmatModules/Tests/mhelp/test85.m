function test85
A = rand(10,GPUsingle) + sqrt(-1)*rand(10,GPUsingle);
R = imag(A);
