function test_gmSinhDrv
GPUtestLOG('Testing test_gmSinhDrv',0);
A = GPUsingle(rand(10));
r = sinh(A);
R1 = gmSinhDrv(A);
compareCPUGPU(single(r), R1);
end
