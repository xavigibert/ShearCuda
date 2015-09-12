function test_gmPlus
GPUtestLOG('Testing test_gmPlus',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = plus(A, B);
gmPlus(A, B, R);
compareCPUGPU(single(r), R);
end
