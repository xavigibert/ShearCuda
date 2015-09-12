function test75
A = rand(10,GPUsingle);
B = rand(10,GPUsingle);
R = zeros(size(A), GPUsingle);
GPUtimes(A, B, R);
