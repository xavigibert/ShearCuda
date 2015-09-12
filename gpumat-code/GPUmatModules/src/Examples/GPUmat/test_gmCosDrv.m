function test_gmCosDrv
GPUtestLOG('Testing test_gmCosDrv',0);
A = GPUsingle(rand(10));
r = cos(A);
R1 = gmCosDrv(A);
compareCPUGPU(single(r), R1);
end
