function test_gmMinusDrv
GPUtestLOG('Testing test_gmMinusDrv',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
r = minus(A, B);
R1 = gmMinusDrv(A, B);
compareCPUGPU(single(r), R1);
end
