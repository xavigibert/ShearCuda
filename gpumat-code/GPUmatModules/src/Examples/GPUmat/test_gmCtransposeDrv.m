function test_gmCtransposeDrv
GPUtestLOG('Testing test_gmCtransposeDrv',0);
A = GPUsingle(rand(10));
r = ctranspose(A);
R1 = gmCtransposeDrv(A);
compareCPUGPU(single(r), R1);
end
