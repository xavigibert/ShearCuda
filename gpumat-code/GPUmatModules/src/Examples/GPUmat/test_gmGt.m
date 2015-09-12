function test_gmGt
GPUtestLOG('Testing test_gmGt',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = gt(A, B);
gmGt(A, B, R);
compareCPUGPU(single(r), R);
end
