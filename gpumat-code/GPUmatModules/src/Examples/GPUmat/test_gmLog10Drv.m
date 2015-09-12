function test_gmLog10Drv
GPUtestLOG('Testing test_gmLog10Drv',0);
A = GPUsingle(rand(10));
r = log10(A);
R1 = gmLog10Drv(A);
compareCPUGPU(single(r), R1);
end
