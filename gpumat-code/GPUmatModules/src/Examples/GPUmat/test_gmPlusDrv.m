function test_gmPlusDrv
GPUtestLOG('Testing test_gmPlusDrv',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
r = plus(A, B);
R1 = gmPlusDrv(A, B);
compareCPUGPU(single(r), R1);
end
