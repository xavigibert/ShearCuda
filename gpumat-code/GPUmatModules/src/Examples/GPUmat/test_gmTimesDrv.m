function test_gmTimesDrv
GPUtestLOG('Testing test_gmTimesDrv',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
r = times(A, B);
R1 = gmTimesDrv(A, B);
compareCPUGPU(single(r), R1);
end
