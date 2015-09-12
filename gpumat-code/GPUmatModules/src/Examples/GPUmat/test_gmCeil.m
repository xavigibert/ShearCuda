function test_gmCeil
GPUtestLOG('Testing test_gmCeil',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = ceil(A);
gmCeil(A, R);
compareCPUGPU(single(r), R);
end
