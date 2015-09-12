function test_gmRdivide
GPUtestLOG('Testing test_gmRdivide',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = rdivide(A, B);
gmRdivide(A, B, R);
compareCPUGPU(single(r), R);
end
