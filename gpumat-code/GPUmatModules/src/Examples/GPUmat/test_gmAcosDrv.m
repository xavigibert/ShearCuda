function test_gmAcosDrv
GPUtestLOG('Testing test_gmAcosDrv',0);
A = GPUsingle(rand(10));
r = acos(A);
R1 = gmAcosDrv(A);
compareCPUGPU(single(r), R1);
end
