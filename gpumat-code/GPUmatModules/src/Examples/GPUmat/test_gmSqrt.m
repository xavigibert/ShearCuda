function test_gmSqrt
GPUtestLOG('Testing test_gmSqrt',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = sqrt(A);
gmSqrt(A, R);
compareCPUGPU(single(r), R);
end
