function test_gmLtDrv
GPUtestLOG('Testing test_gmLtDrv',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
r = lt(A, B);
R1 = gmLtDrv(A, B);
compareCPUGPU(single(r), R1);
end
