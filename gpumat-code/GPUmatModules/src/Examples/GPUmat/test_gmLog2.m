function test_gmLog2
GPUtestLOG('Testing test_gmLog2',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = log2(A);
gmLog2(A, R);
compareCPUGPU(single(r), R);
end
