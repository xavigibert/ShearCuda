function test_gmUminusDrv
GPUtestLOG('Testing test_gmUminusDrv',0);
A = GPUsingle(rand(10));
r = uminus(A);
R1 = gmUminusDrv(A);
compareCPUGPU(single(r), R1);
end
