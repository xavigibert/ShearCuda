function test_gmAbs
GPUtestLOG('Testing test_gmAbs',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = abs(A);
gmAbs(A, R);
compareCPUGPU(single(r), R);
end
