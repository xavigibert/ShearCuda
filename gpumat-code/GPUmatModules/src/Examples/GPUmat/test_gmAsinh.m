function test_gmAsinh
GPUtestLOG('Testing test_gmAsinh',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = asinh(A);
gmAsinh(A, R);
compareCPUGPU(single(r), R);
end
