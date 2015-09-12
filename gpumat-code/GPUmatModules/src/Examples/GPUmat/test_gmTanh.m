function test_gmTanh
GPUtestLOG('Testing test_gmTanh',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = tanh(A);
gmTanh(A, R);
compareCPUGPU(single(r), R);
end
