function test_gmLogDrv
GPUtestLOG('Testing test_gmLogDrv',0);
A = GPUsingle(rand(10));
r = log(A);
R1 = gmLogDrv(A);
compareCPUGPU(single(r), R1);
end
