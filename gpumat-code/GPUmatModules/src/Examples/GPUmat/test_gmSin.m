function test_gmSin
GPUtestLOG('Testing test_gmSin',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = sin(A);
gmSin(A, R);
compareCPUGPU(single(r), R);
end
