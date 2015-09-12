function test_gmSinh
GPUtestLOG('Testing test_gmSinh',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = sinh(A);
gmSinh(A, R);
compareCPUGPU(single(r), R);
end
