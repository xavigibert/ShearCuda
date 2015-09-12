function test_gmCtranspose
GPUtestLOG('Testing test_gmCtranspose',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = ctranspose(A);
gmCtranspose(A, R);
compareCPUGPU(single(r), R);
end
