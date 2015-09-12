function test_gmAtanDrv
GPUtestLOG('Testing test_gmAtanDrv',0);
A = GPUsingle(rand(10));
r = atan(A);
R1 = gmAtanDrv(A);
compareCPUGPU(single(r), R1);
end
