function test_gmLog2Drv
GPUtestLOG('Testing test_gmLog2Drv',0);
A = GPUsingle(rand(10));
r = log2(A);
R1 = gmLog2Drv(A);
compareCPUGPU(single(r), R1);
end
