function test_gmLog1pDrv
GPUtestLOG('Testing test_gmLog1pDrv',0);
A = GPUsingle(rand(10));
r = log1p(A);
R1 = gmLog1pDrv(A);
compareCPUGPU(single(r), R1);
end
