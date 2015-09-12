function test_gmOrDrv
GPUtestLOG('Testing test_gmOrDrv',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
r = or(A, B);
R1 = gmOrDrv(A, B);
compareCPUGPU(single(r), R1);
end
