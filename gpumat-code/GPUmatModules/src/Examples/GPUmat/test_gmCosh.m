function test_gmCosh
GPUtestLOG('Testing test_gmCosh',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = cosh(A);
gmCosh(A, R);
compareCPUGPU(single(r), R);
end
