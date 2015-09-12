function test_gmGtDrv
GPUtestLOG('Testing test_gmGtDrv',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
r = gt(A, B);
R1 = gmGtDrv(A, B);
compareCPUGPU(single(r), R1);
end
