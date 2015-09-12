function test_gmRdivideDrv
GPUtestLOG('Testing test_gmRdivideDrv',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
r = rdivide(A, B);
R1 = gmRdivideDrv(A, B);
compareCPUGPU(single(r), R1);
end
