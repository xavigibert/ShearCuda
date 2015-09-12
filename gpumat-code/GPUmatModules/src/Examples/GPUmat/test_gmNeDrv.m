function test_gmNeDrv
GPUtestLOG('Testing test_gmNeDrv',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
r = ne(A, B);
R1 = gmNeDrv(A, B);
compareCPUGPU(single(r), R1);
end
