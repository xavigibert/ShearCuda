function test_gmImagDrv
GPUtestLOG('Testing test_gmImagDrv',0);
A = GPUsingle(rand(10));
r = imag(A);
R1 = gmImagDrv(A);
compareCPUGPU(single(r), R1);
end
