function test_gmPowerDrv
GPUtestLOG('Testing test_gmPowerDrv',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
r = power(A, B);
R1 = gmPowerDrv(A, B);
compareCPUGPU(single(r), R1);
end
