function test_gmRealDrv
GPUtestLOG('Testing test_gmRealDrv',0);
A = GPUsingle(rand(10));
r = real(A);
R1 = gmRealDrv(A);
compareCPUGPU(single(r), R1);
end
