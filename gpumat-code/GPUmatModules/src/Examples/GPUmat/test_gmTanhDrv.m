function test_gmTanhDrv
GPUtestLOG('Testing test_gmTanhDrv',0);
A = GPUsingle(rand(10));
r = tanh(A);
R1 = gmTanhDrv(A);
compareCPUGPU(single(r), R1);
end
