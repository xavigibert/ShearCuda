function test_gmEqDrv
GPUtestLOG('Testing test_gmEqDrv',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
r = eq(A, B);
R1 = gmEqDrv(A, B);
compareCPUGPU(single(r), R1);
end
