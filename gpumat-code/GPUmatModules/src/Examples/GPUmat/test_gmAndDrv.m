function test_gmAndDrv
GPUtestLOG('Testing test_gmAndDrv',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
r = and(A, B);
R1 = gmAndDrv(A, B);
compareCPUGPU(single(r), R1);
end
