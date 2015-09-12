function test_gmTransposeDrv
GPUtestLOG('Testing test_gmTransposeDrv',0);
A = GPUsingle(rand(10));
r = transpose(A);
R1 = gmTransposeDrv(A);
compareCPUGPU(single(r), R1);
end
