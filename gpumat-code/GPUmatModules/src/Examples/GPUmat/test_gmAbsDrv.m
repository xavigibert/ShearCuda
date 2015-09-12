function test_gmAbsDrv
GPUtestLOG('Testing test_gmAbsDrv',0);
A = GPUsingle(rand(10));
r = abs(A);
R1 = gmAbsDrv(A);
compareCPUGPU(single(r), R1);
end
