function test_gmTanDrv
GPUtestLOG('Testing test_gmTanDrv',0);
A = GPUsingle(rand(10));
r = tan(A);
R1 = gmTanDrv(A);
compareCPUGPU(single(r), R1);
end
