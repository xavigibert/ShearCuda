function test_gmAtanhDrv
GPUtestLOG('Testing test_gmAtanhDrv',0);
A = GPUsingle(rand(10));
r = atanh(A);
R1 = gmAtanhDrv(A);
compareCPUGPU(single(r), R1);
end
