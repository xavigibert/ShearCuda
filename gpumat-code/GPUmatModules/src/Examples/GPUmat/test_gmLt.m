function test_gmLt
GPUtestLOG('Testing test_gmLt',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = lt(A, B);
gmLt(A, B, R);
compareCPUGPU(single(r), R);
end
