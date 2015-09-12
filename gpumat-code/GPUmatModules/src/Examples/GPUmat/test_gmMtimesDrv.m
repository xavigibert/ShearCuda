function test_gmMtimesDrv
GPUtestLOG('Testing test_gmMtimesDrv',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
r = mtimes(A, B);
R1 = gmMtimesDrv(A, B);
compareCPUGPU(single(r), R1);
end
