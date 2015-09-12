function test_gmSinDrv
GPUtestLOG('Testing test_gmSinDrv',0);
A = GPUsingle(rand(10));
r = sin(A);
R1 = gmSinDrv(A);
compareCPUGPU(single(r), R1);
end
