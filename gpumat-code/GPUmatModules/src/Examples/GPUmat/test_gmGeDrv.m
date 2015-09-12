function test_gmGeDrv
GPUtestLOG('Testing test_gmGeDrv',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
r = ge(A, B);
R1 = gmGeDrv(A, B);
compareCPUGPU(single(r), R1);
end
