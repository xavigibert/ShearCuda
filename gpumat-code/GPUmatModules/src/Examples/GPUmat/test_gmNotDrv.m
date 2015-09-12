function test_gmNotDrv
GPUtestLOG('Testing test_gmNotDrv',0);
A = GPUsingle(rand(10));
r = not(A);
R1 = gmNotDrv(A);
compareCPUGPU(single(r), R1);
end
