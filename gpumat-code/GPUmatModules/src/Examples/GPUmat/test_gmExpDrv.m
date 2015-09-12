function test_gmExpDrv
GPUtestLOG('Testing test_gmExpDrv',0);
A = GPUsingle(rand(10));
r = exp(A);
R1 = gmExpDrv(A);
compareCPUGPU(single(r), R1);
end
