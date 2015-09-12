function test_gmLdivide
GPUtestLOG('Testing test_gmLdivide',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = ldivide(A, B);
gmLdivide(A, B, R);
compareCPUGPU(single(r), R);
end
