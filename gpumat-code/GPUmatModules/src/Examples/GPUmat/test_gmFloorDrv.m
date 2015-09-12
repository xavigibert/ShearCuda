function test_gmFloorDrv
GPUtestLOG('Testing test_gmFloorDrv',0);
A = GPUsingle(rand(10));
r = floor(A);
R1 = gmFloorDrv(A);
compareCPUGPU(single(r), R1);
end
