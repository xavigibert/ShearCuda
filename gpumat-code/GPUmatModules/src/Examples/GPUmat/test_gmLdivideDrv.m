function test_gmLdivideDrv
GPUtestLOG('Testing test_gmLdivideDrv',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
r = ldivide(A, B);
R1 = gmLdivideDrv(A, B);
compareCPUGPU(single(r), R1);
end
