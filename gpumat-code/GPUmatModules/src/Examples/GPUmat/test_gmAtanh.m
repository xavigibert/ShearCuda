function test_gmAtanh
GPUtestLOG('Testing test_gmAtanh',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = atanh(A);
gmAtanh(A, R);
compareCPUGPU(single(r), R);
end
