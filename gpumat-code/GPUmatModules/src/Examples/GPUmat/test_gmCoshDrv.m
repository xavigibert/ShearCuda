function test_gmCoshDrv
GPUtestLOG('Testing test_gmCoshDrv',0);
A = GPUsingle(rand(10));
r = cosh(A);
R1 = gmCoshDrv(A);
compareCPUGPU(single(r), R1);
end
