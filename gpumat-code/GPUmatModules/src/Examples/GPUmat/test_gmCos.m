function test_gmCos
GPUtestLOG('Testing test_gmCos',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = cos(A);
gmCos(A, R);
compareCPUGPU(single(r), R);
end
