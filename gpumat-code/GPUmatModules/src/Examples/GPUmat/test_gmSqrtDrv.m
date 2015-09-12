function test_gmSqrtDrv
GPUtestLOG('Testing test_gmSqrtDrv',0);
A = GPUsingle(rand(10));
r = sqrt(A);
R1 = gmSqrtDrv(A);
compareCPUGPU(single(r), R1);
end
