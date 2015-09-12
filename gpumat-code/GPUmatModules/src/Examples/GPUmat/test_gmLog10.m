function test_gmLog10
GPUtestLOG('Testing test_gmLog10',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = log10(A);
gmLog10(A, R);
compareCPUGPU(single(r), R);
end
