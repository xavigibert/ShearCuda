function test_gmRoundDrv
GPUtestLOG('Testing test_gmRoundDrv',0);
A = GPUsingle(rand(10));
r = round(A);
R1 = gmRoundDrv(A);
compareCPUGPU(single(r), R1);
end
