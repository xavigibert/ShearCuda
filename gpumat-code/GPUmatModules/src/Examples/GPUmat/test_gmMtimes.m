function test_gmMtimes
GPUtestLOG('Testing test_gmMtimes',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = mtimes(A, B);
gmMtimes(A, B, R);
compareCPUGPU(single(r), R);
end
