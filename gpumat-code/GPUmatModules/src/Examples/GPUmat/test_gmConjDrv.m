function test_gmConjDrv
GPUtestLOG('Testing test_gmConjDrv',0);
A = GPUsingle(rand(10));
r = conj(A);
R1 = gmConjDrv(A);
compareCPUGPU(single(r), R1);
end
