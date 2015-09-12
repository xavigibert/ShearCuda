function test_gmLeDrv
GPUtestLOG('Testing test_gmLeDrv',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
r = le(A, B);
R1 = gmLeDrv(A, B);
compareCPUGPU(single(r), R1);
end
