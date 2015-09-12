function test_gmAsinDrv
GPUtestLOG('Testing test_gmAsinDrv',0);
A = GPUsingle(rand(10));
r = asin(A);
R1 = gmAsinDrv(A);
compareCPUGPU(single(r), R1);
end
