function test_gmAsinhDrv
GPUtestLOG('Testing test_gmAsinhDrv',0);
A = GPUsingle(rand(10));
r = asinh(A);
R1 = gmAsinhDrv(A);
compareCPUGPU(single(r), R1);
end
