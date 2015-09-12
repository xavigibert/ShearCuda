function test58
A = rand(10,GPUsingle);
B = rand(10,GPUsingle);
R = zeros(size(A), GPUsingle);
GPUmtimes(A,  B, R);

