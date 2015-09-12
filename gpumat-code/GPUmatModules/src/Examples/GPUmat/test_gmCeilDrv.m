function test_gmCeilDrv
GPUtestLOG('Testing test_gmCeilDrv',0);
A = GPUsingle(rand(10));
r = ceil(A);
R1 = gmCeilDrv(A);
compareCPUGPU(single(r), R1);
end
